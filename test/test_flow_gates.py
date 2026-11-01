import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import TransformStamped
from moveit_msgs.srv import GetPlanningScene
from std_msgs.msg import String

from cyberdog.behavior import Cyberdog
from cyberdog.senses import Senses
from cyberdog.choreography import TaughtDeparture,build_dance,collision_samples
from fake_hardware import FakeGameRobot,executor
from test_behavior import graph,reset,wake,until,spin
from test_dance import Scene,PACE,completed_round
from test_handshake import FakeGripper
from test_blind_grasp import FakeGraspPlanner
from test_grasp_retry import lock_target
from test_hardware_adapter import PARAMS


def test_visible_ball_cannot_skip_handshake_and_success_dances_once_then_relocks():
    with graph((Cyberdog,Senses),PACE+['mock_ball_visible:=true','dance_after_place:=true','ball_still_sec:=0.5']) as ([g,s],e):
        stages=[];original=g.set_state
        def state(a,b,c):stages.append(b);original(a,b,c)
        g.set_state=state
        reset(g,e);spin(e,.3)
        assert not s.watch_ball and g.state=='idle'
        # Neither an incoming observation nor a direct entry call bypasses wake-up.
        g.observation(String(data='{}'));g.ball_seen=time.monotonic();g.tick()
        assert g.state=='idle' and not g.enter_ball() and not g.command('dance')[0]
        g.robot.mock_pose[2]=.05
        until(e,lambda:'dance_running' in stages and g.state=='idle',timeout=12.)
        assert stages.index('handshake')<stages.index('tracking')<stages.index('grasp_closing')<stages.index('grasp_releasing')<stages.index('dance_running')
        assert stages.count('dance_running')==1 and g.robot.current==g.home
        assert not g.handshake_completed and not g.ball_sequence.dance_ready
        spin(e,.4);assert g.state=='idle' and not g.dance.auto_pending and not s.watch_ball
        assert not g.command('dance')[0]


@pytest.mark.parametrize('event',['close_failed','release_failed','stop_at_complete','reset_at_complete'])
def test_failure_or_operator_interrupt_never_queues_dance(event):
    with graph((Cyberdog,Senses),PACE+['mock_ball_visible:=true','dance_after_place:=true','ball_still_sec:=0.5']) as ([g,s],e):
        stages=[];original=g.set_state;grip=g.robot.grip
        def operation(closed,cb):
            if (event=='close_failed' and closed) or (event=='release_failed' and g.phase=='grasp_releasing'):
                return False,'test gripper failed'
            return grip(closed,cb)
        g.robot.grip=operation
        def state(a,b,c):
            stages.append(b);original(a,b,c)
            if b=='complete' and event.endswith('_at_complete'):g.command(event.split('_')[0])
        g.set_state=state;wake(g,e)
        until(e,lambda:g.state=='paused' or (event=='reset_at_complete' and 'complete' in stages and g.state=='idle'),timeout=8.)
        spin(e,.4)
        assert 'dance_running' not in stages and not g.dance.auto_pending and not g.ball_sequence.dance_ready
        assert not g.command('dance')[0]


def test_stop_and_reset_invalidate_previously_unlocked_dance():
    with graph(parameters=PACE) as ([g],e):
        reset(g,e);completed_round(g);g.command('stop')
        assert not g.command('dance')[0]
        completed_round(g);g.command('reset');until(e,lambda:g.state=='idle')
        assert not g.command('dance')[0] and not g.handshake_completed


@pytest.mark.parametrize('feedback',['held','empty','missing','release_stuck'])
def test_ros_measured_grasp_and_release_gate_auto_dance(feedback,monkeypatch):
    from cyberdog.table_scene import table_box
    class Robot(FakeGameRobot):
        def __init__(self):super().__init__();self.gripper_position=0.
    class Planner(FakeGraspPlanner):
        def __init__(self):super().__init__();self.scene_calls=0
        def scene(self,request,response):
            self.scene_calls+=1
            response.scene.world.collision_objects=[table_box(
                [.023508207260724053,-.009321865689088018,.003044323054376846],
                [.05558204049876611,-.6568967385542418],[.3985576776722882,-.0056047769641885264],'cyberdog_observed_table')]
            return response
    class Gripper(FakeGripper):
        def execute(self,handle):
            self.requests.append(handle.request);q=handle.request.command.position[0]
            closed=q==.95
            if closed:r.gripper_position=None if feedback=='missing' else .95 if feedback=='empty' else .7
            else:r.gripper_position=.7 if feedback=='release_stuck' and g.phase=='grasp_releasing' else q
            handle.succeed()
            return ParallelGripperCommand.Result(stalled=closed and feedback!='empty',reached_goal=not closed or feedback=='empty')
    params=PARAMS+PACE+['gripper_enabled:=true','allow_nominal_tcp_trial:=true','dance_after_place:=true',
                      'grasp_place_joint_pose:=[0.263,0.535,0.23,-0.599,-0.285,0.273]']
    with graph((Cyberdog,Robot,Gripper,Planner,Scene),params,executor) as ([g,r,gripper,planner,scene],e):
        stages=[];original=g.set_state
        def state(a,b,c):stages.append(b);original(a,b,c)
        g.set_state=state
        g.command('reset');until(e,lambda:g.state=='idle',timeout=8.)
        r.position[2]=.06;until(e,lambda:g.phase=='handshake',timeout=14.)
        g.enter_ball();until(e,lambda:g.phase=='tracking',timeout=8.)
        tr=TransformStamped();tr.transform.translation.x=.22;tr.transform.translation.z=.06;tr.transform.rotation.w=1.
        monkeypatch.setattr(g.ball_sequence.buffer,'lookup_transform',lambda *a:tr)
        before=len(r.requests);lock_target(g)
        until(e,lambda:('dance_running' in stages and g.state=='idle') if feedback=='held' else
              (g.state=='paused' and not g.robot.busy and not g.ball_sequence.busy),timeout=30.)
        spin(e,.2)
        if feedback=='held':
            assert stages.count('dance_running')==1 and len(r.requests)==before+3
            assert stages.index('grasp_releasing')<stages.index('complete')<stages.index('dance_running')
            assert g.robot.current==pytest.approx(g.home) and scene.calls
            assert planner.scene_calls>=2 and g.dance.departure.enabled
        else:
            assert 'dance_running' not in stages and not g.ball_sequence.dance_ready and not g.command('dance')[0]
            if feedback=='release_stuck':assert '张爪' in g.message
        assert not g.dance.auto_pending


def contact(depth=.0017,body='omni_picker_hand_wide3_Link'):
    return SimpleNamespace(contact_body_1=body,contact_body_2='cyberdog_observed_table',depth=depth)


def test_taught_departure_only_allows_known_outward_segment_before_clearance():
    q=TaughtDeparture.RELEASE.tolist();via=TaughtDeparture.CLEAR.tolist();table='cyberdog_observed_table'
    for contacts in ([],[contact(.0111)],[contact(body='link3')],[contact(float('nan'))]):
        assert not TaughtDeparture(True,q,via,table).accept(False,contacts,q)
    assert not TaughtDeparture(False,q,via,table).accept(False,[contact()],q)
    other=q[:];other[0]+=.01
    assert not TaughtDeparture(True,other,via,table).accept(False,[contact()],other)
    c=TaughtDeparture(True,q,via,table)
    assert c.accept(False,[contact()],q)
    turn=q[:];turn[0]+=.0001
    assert not c.accept(False,[contact(.001)],turn)
    outward=(TaughtDeparture.RELEASE+.02*(TaughtDeparture.CLEAR-TaughtDeparture.RELEASE)).tolist()
    assert c.accept(False,[contact(.003)],outward)
    assert not c.accept(False,[contact()],q)  # No reversal into the table.
    too_far=(TaughtDeparture.RELEASE+.06*(TaughtDeparture.CLEAR-TaughtDeparture.RELEASE)).tolist()
    assert not c.accept(False,[contact()],too_far)
    assert c.accept(True,[],q)
    assert not c.accept(False,[contact(.0001)],q)


def test_table_departure_raises_before_base_turn_and_preserves_final_home():
    start=[.263,.535,.23,-.599,-.285,.273];centre=[-1.5707,.6,.9,0.,0.,0.];home=[-1.5707,0.,0.,0.,0.,0.]
    via=[start[0]]+centre[1:]
    points,_=build_dance(start,centre,home,[.8,.8,.8,1.4,.8,1.4],[1.3,1.3,1.3,6.,1.3,6.],entry_via=via)
    first_turn=next(q for _,q,_,_ in points if abs(q[0]-start[0])>1e-8)
    assert first_turn[1:]==pytest.approx(centre[1:])
    assert points[-1][1]==pytest.approx(home)


def test_taught_departure_requires_the_unchanged_measured_table_in_both_ros_forms():
    from copy import deepcopy
    from geometry_msgs.msg import Pose
    from cyberdog.table_scene import table_box
    table=table_box([.023508207260724053,-.009321865689088018,.003044323054376846],
                    [.05558204049876611,-.6568967385542418],[.3985576776722882,-.0056047769641885264],'cyberdog_observed_table')
    assert TaughtDeparture.scene_matches(table)
    normalized=deepcopy(table);normalized.pose=normalized.primitive_poses[0]
    normalized.primitive_poses=[Pose()];normalized.primitive_poses[0].orientation.w=1.
    assert TaughtDeparture.scene_matches(normalized)
    for mode in ('height','size','frame','rotation'):
        changed=deepcopy(normalized)
        if mode=='height':changed.pose.position.z+=.001
        elif mode=='size':changed.primitives[0].dimensions[0]+=.001
        elif mode=='frame':changed.header.frame_id='world'
        else:changed.pose.orientation.x=.1
        assert not TaughtDeparture.scene_matches(changed)


@pytest.mark.parametrize('change',['unchanged','moved','removed','cancelled','error'])
def test_live_table_change_or_late_callback_cannot_keep_a_stale_departure_allowance(change):
    from cyberdog.table_scene import table_box
    table=table_box([.023508207260724053,-.009321865689088018,.003044323054376846],
                    [.05558204049876611,-.6568967385542418],[.3985576776722882,-.0056047769641885264],'cyberdog_observed_table')
    with graph() as ([g],e):
        d=g.dance;d.busy=True;d.preflight='table';token=d.token
        d.departure=TaughtDeparture(True,TaughtDeparture.RELEASE,TaughtDeparture.CLEAR,table.id)
        future=Future();future.set_running_or_notify_cancel();d.future=future
        result=GetPlanningScene.Response();result.scene.world.collision_objects=[table]
        if change=='moved':table.primitive_poses[0].position.z+=.001
        if change=='removed':result.scene.world.collision_objects=[]
        if change=='cancelled':g.command('stop')
        if change=='error':future.set_exception(RuntimeError('scene unavailable'))
        else:future.set_result(result)
        d.table_checked(future,token)
        if change in ('cancelled','error'):
            assert not d.busy and g.state=='paused' and d.preflight==''
        else:
            assert d.preflight=='feedback'
            assert bool(d.departure.enabled)==(change=='unchanged')
