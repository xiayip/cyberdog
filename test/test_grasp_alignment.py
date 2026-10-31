"""Off-axis balls must be aligned in a checked path, not rejected in a retry loop."""
import json
import math
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped

from cyberdog.behavior import Cyberdog
from cyberdog.ball_grasp_geometry import preview_waypoints,rotation
from cyberdog.senses import Senses
from cyberdog.table_scene import table_box
from test_behavior import graph,until,spin
from test_grasp_retry import lock_target


CASES=json.loads((Path(__file__).parent/'data/grasp_cone_cases.json').read_text())


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['snapshot'])
def test_recorded_45_degree_rejections_align_then_advance_without_retreat(case):
    tcp=np.array(case['tcp']['p']);q=case['tcp']['q'];rot=rotation(q)
    center=np.array(case['target']['center']);radius=case['target']['radius']
    relative=rot.T@(center-tcp)
    assert np.linalg.norm(relative[:2])>relative[2]>radius+.01
    waypoints=preview_waypoints(center,radius,orientation=q,tcp_position=tcp)
    assert [w['stage'] for w in waypoints]==['align','approach','pregrasp','grasp','lift']
    aligned=np.array(waypoints[0]['position'])
    assert rot.T@(aligned-tcp)==pytest.approx([relative[0],relative[1],0.],abs=1e-9)
    assert rot.T@(center-aligned)==pytest.approx([0.,0.,relative[2]],abs=1e-9)
    # The lateral segment preserves axial clearance from the sphere surface.
    for t in np.linspace(0.,1.,30):
        assert np.linalg.norm(center-(tcp+t*(aligned-tcp)))>=radius+.01
    previous=aligned
    for w in waypoints[1:4]:
        delta=rot.T@(np.array(w['position'])-previous)
        assert delta[:2]==pytest.approx([0.,0.],abs=1e-9) and delta[2]>0.
        assert w['orientation']==q
        previous=np.array(w['position'])
    assert previous==pytest.approx(center)
    assert waypoints[0]['gripper']=='open' and waypoints[3]['gripper']=='close'
    assert np.array(waypoints[4]['position'])-center==pytest.approx([0.,0.,.12])


def test_image_centre_is_not_sufficient_evidence_of_tcp_alignment():
    case=next(c for c in CASES if c['snapshot']=='snap0200')
    assert max(abs(x) for x in case['image_xy'])<.02
    assert case['angle_deg']>50.
    way=preview_waypoints(case['target']['center'],case['target']['radius'],
        orientation=case['tcp']['q'],tcp_position=case['tcp']['p'])
    assert way[0]['stage']=='align'


@pytest.mark.parametrize('relative',[[-.0221,.005,.0136],[-.0336,.0104,.0119],[0.,0.,.02]])
def test_recorded_near_ball_rejections_get_one_bounded_forward_correction(relative):
    tcp=np.array([.2,.1,.3]);q=[math.sqrt(.5),0.,0.,math.sqrt(.5)]
    rot=rotation(q);center=tcp+rot@relative
    way=preview_waypoints(center,.025,orientation=q,tcp_position=tcp,include_lift=False)
    assert [w['stage'] for w in way]==['grasp']
    assert way[0]['position']==pytest.approx(center) and way[0]['orientation']==q
    delta=rot.T@(np.array(way[0]['position'])-tcp)
    assert delta==pytest.approx(relative) and delta[2]>.005 and np.linalg.norm(delta)<.06
    assert way[0]['gripper']=='close'


@pytest.mark.parametrize('relative',[[-.0221,.005,.0136],[-.0336,.0104,.0119]])
def test_near_ball_scene_after_one_second_lock_plans_without_restarting_dwell(relative,monkeypatch):
    with graph() as ([g],e):
        seq=lock_target(g);seq.reject(ValueError('target unavailable'),transient=True)
        g.robot.mock=False;g.robot.fresh=lambda:True
        seq.feedback=([0.]*6,0.);seq.feedback_at=time.monotonic()
        seq.phase='scene';seq.future=Future()
        obj=table_box([0.,0.,0.],[.1,-.6],[.4,-.1],seq.table_id)
        seq.future.set_result(SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(collision_objects=[obj]))))
        tr=TransformStamped();tr.transform.rotation.w=1.
        tr.transform.translation.x,tr.transform.translation.y,tr.transform.translation.z=np.asarray(seq.frozen)-relative
        monkeypatch.setattr(seq.buffer,'lookup_transform',lambda *a:tr)
        plans=[];monkeypatch.setattr(seq,'request_plan',lambda w,p:plans.append((w,p)))
        seq.tick()
        assert seq.attempts==1 and seq.running and seq.retry_at is None
        assert len(plans)==1 and plans[0][1]=='full_plan'
        assert [w['stage'] for w in plans[0][0]]==['grasp']
        assert seq.execution_waypoints()==seq.waypoints and seq.execution_stage()=='approach'
        assert seq.geometry['first_step_in_tcp']==pytest.approx(relative)


def off_axis_waypoints(center,radius,*args,**kwargs):
    return preview_waypoints(center,radius,orientation=[0.,0.,0.,1.],
        tcp_position=(np.asarray(center)-[.08,0.,.04]).tolist())


def test_real_scene_branch_includes_alignment_in_full_plan_after_vision_loss(monkeypatch):
    with graph() as ([g],e):
        seq=lock_target(g);seq.reject(ValueError('target unavailable'),transient=True)
        g.robot.mock=False;g.robot.fresh=lambda:True
        seq.feedback=([0.]*6,0.);seq.feedback_at=time.monotonic()
        seq.phase='scene';seq.future=Future()
        obj=table_box([0.,0.,0.],[.1,-.6],[.4,-.1],seq.table_id)
        seq.future.set_result(SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(collision_objects=[obj]))))
        tr=TransformStamped();tr.transform.translation.x=.22;tr.transform.translation.z=.06;tr.transform.rotation.w=1.
        monkeypatch.setattr(seq.buffer,'lookup_transform',lambda *a:tr)
        plans=[]
        monkeypatch.setattr(seq,'request_plan',lambda waypoints,phase:plans.append((waypoints,phase)))
        g.robot.grip=lambda *a:pytest.fail('no gripper action before complete plan')
        seq.tick()
        assert seq.running and seq.attempts==1 and len(plans)==1 and plans[0][1]=='full_plan'
        assert [p['stage'] for p in plans[0][0]]==['align','approach','pregrasp','grasp']
        assert seq.geometry['first_stage']=='align' and seq.geometry['axis_angle_deg']>45.
        assert seq.geometry['first_step_in_tcp']==pytest.approx([.08,0.,0.])
        assert seq.retry_at is None


def test_incomplete_alignment_path_cannot_start_or_close(monkeypatch):
    with graph() as ([g],e):
        seq=lock_target(g);seq.waypoints=off_axis_waypoints(seq.frozen,seq.frozen_radius)
        g.robot.mock=False;g.robot.fresh=lambda:True
        seq.feedback=([0.]*6,0.);seq.feedback_at=time.monotonic()
        seq.phase='full_plan';seq.future=Future()
        seq.future.set_result(SimpleNamespace(error_code=SimpleNamespace(val=1),fraction=.99))
        g.robot.grip=lambda *a:pytest.fail('incomplete path must not open or close')
        seq.action.send_goal_async=lambda *a:pytest.fail('incomplete path must not execute')
        seq.tick()
        assert g.phase=='grasp_retry_wait' and seq.attempts==1
        assert not seq.closing_started


@pytest.mark.parametrize('event',['loss','stop','movement'])
def test_alignment_obeys_loss_stop_and_target_movement_rules(event,monkeypatch):
    monkeypatch.setattr('cyberdog.ball_sequence.preview_waypoints',off_axis_waypoints)
    with graph((Cyberdog,Senses),['mock_ball_visible:=true','ball_still_sec:=0.5']) as ([g,s],e):
        stages=[];closed=[];original=g.set_state;grip=g.robot.grip
        def record_grip(close,callback):
            if close:closed.append(g.phase)
            return grip(close,callback)
        g.robot.grip=record_grip
        def state(a,b,c):
            original(a,b,c);stages.append(b)
            if b=='grasp_approach':
                s.mock_ball_visible=False
                if event=='loss':g.ball_sequence.reject(ValueError('target unavailable'),transient=True)
                elif event=='stop':g.command('stop')
                else:
                    g.ball_sequence.accept(dict(center=[.33,0.,.1],radius=.025),g.get_clock().now().nanoseconds*1e-9)
                    g.ball_sequence.reject(ValueError('target unavailable'),transient=True)
        g.set_state=state
        g.command('reset')
        if event in ('loss','movement'):
            until(e,lambda:g.phase=='complete',timeout=6.)
            assert stages.index('grasp_approach')<stages.index('grasp_closing')
            assert 'grasp_align' not in stages and 'grasp_pregrasp' not in stages
            assert closed==['grasp_closing'] and g.ball_sequence.attempts==1
        else:
            until(e,lambda:'grasp_approach' in stages,timeout=6.)
            spin(e,.4)
            assert not closed and not g.robot.grip_closed
            assert g.state=='paused' and 'grasp_closing' not in stages
