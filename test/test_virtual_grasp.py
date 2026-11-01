"""Bounded air-grasp fallback, including isolated ROS actions and cancellation."""
import math
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped

from cyberdog.behavior import Cyberdog
from cyberdog.ball_grasp_geometry import virtual_grasp_waypoints,rotation
from fake_hardware import FakeGameRobot,executor
from test_behavior import graph,until,spin
from test_grasp_retry import lock_target
from test_blind_grasp import FakeGraspPlanner
from test_handshake import FakeGripper
from test_hardware_adapter import PARAMS


def exhaust(g):
    seq=lock_target(g);seq.attempts=seq.max_attempts
    seq.fail('incomplete target path',retryable=True)
    return seq


@pytest.mark.parametrize('q',[[0.,0.,0.,1.],[0.,math.sin(math.pi/4),0.,math.cos(math.pi/4)],
                            [math.sin(math.pi/4),0.,0.,math.cos(math.pi/4)]])
def test_air_grasp_uses_tool_forward_not_world_forward_and_returns_exactly(q):
    start=[.3,-.2,.1];waypoints=virtual_grasp_waypoints(start,q)
    assert rotation(q).T@(np.array(waypoints[0]['position'])-start)==pytest.approx([0.,0.,.05])
    assert waypoints[1]['position']==start and all(w['orientation']==q for w in waypoints)


def test_second_attempt_dwell_starts_fresh_and_reset_restores_first_dwell():
    with graph() as ([g],e):
        seq=exhaust(g);g.command('stop');seq.reset_attempts()
        assert seq.gate.dwell==1.
        seq.gate.update([.3,0.,.1],10.);seq.gate.update([.3,0.,.1],10.8)
        seq.attempts=1;seq.reset_tracking_gate()
        assert seq.status()['dwell_sec']==.5 and seq.gate.elapsed==0.
        assert not seq.gate.update([.3,0.,.1],11.)
        assert not seq.gate.update([.3,0.,.1],11.49)
        assert seq.gate.update([.3,0.,.1],11.5)
        g.command('reset');assert seq.gate.dwell==1. and not seq.virtual_attempted


@pytest.mark.parametrize('phase',['virtual_wait','virtual_settling','virtual_approach','virtual_closing','virtual_opening','virtual_return'])
def test_stop_at_every_virtual_stage_invalidates_all_followups(phase):
    with graph(parameters=['dance_after_place:=true']) as ([g],e):
        phases=[];original=g.set_state
        def record(state,stage,message):
            phases.append(stage);original(state,stage,message)
            if stage==phase:g.command('stop')
        g.set_state=record;seq=exhaust(g)
        until(e,lambda:g.state=='paused',timeout=5.);spin(e,.4)
        assert phase in phases and phases[-1]=='stopped'
        assert not seq.busy and not seq.virtual_pending and not seq.dance_ready and not g.dance.busy
        assert g.outcome!='virtual_grasp_only'


@pytest.mark.parametrize('closing,hard_fault,enabled',[(True,False,True),(False,True,True),(False,False,False)])
def test_no_virtual_fallback_after_close_hard_fault_or_when_disabled(closing,hard_fault,enabled):
    with graph(parameters=[f'grasp_virtual_after_retries:={str(enabled).lower()}']) as ([g],e):
        seq=lock_target(g);seq.attempts=3;seq.closing_started=closing;g.robot.grip_closed=closing
        seq.fail('feedback stale' if hard_fault else 'failed',retryable=not hard_fault)
        spin(e,.3)
        assert g.state=='paused' and not seq.virtual_attempted and not seq.retry_at
        assert g.robot.grip_closed==closing


def test_virtual_waits_for_late_acceptance_and_terminal_cancel_without_any_new_target():
    with graph() as ([g],e):
        seq=lock_target(g);seq.attempts=3;seq.submitting=True;g.robot.sequence_motion_busy=True
        terminal=Future();cancels=[]
        handle=SimpleNamespace(accepted=True,get_result_async=lambda:terminal,cancel_goal_async=lambda:cancels.append(True))
        accepted=Future();accepted.set_result(handle)
        seq.fail('replan',retryable=True);seq.tick()
        assert seq.virtual_pending and not seq.virtual
        seq.accepted(accepted);seq.tick()
        assert cancels and seq.virtual_pending and not seq.virtual
        terminal.set_result(SimpleNamespace(status=5,result=SimpleNamespace(error_code=0)))
        until(e,lambda:g.state=='paused')
        assert g.outcome=='virtual_grasp_only' and seq.attempts==3 and not seq.dance_ready
        assert not g.command('dance')[0]


@pytest.mark.parametrize('failure',[None,'partial','stale_tf','stale_feedback','stop_approach'])
def test_ros_virtual_forward_close_open_return_has_no_placement_or_dance(failure,monkeypatch):
    class Planner(FakeGraspPlanner):
        def __init__(self):super().__init__();self.origin=None
        def plan(self,request,response):
            self.requests.append(request)
            joints=request.start_state.joint_state
            start=[joints.position[joints.name.index('joint'+str(i))] for i in range(1,7)]
            if self.origin is None:self.origin=start[:]
            target=self.origin[:]
            if request.waypoints[-1].position.z>.13:target[3]+=.04
            response.error_code.val=1;response.fraction=.5 if failure=='partial' else 1.
            response.solution.joint_trajectory=self.path(request.start_state,target)
            return response
    params=PARAMS+['gripper_enabled:=true','allow_nominal_tcp_trial:=true','dance_after_place:=true']
    with graph((Cyberdog,FakeGameRobot,FakeGripper,Planner),params,executor) as ([g,r,gripper,planner],e):
        until(e,lambda:g.robot.motion.ready()=='',timeout=6.)
        switched=[];assert g.robot.mode('position',lambda ok,_:switched.append(ok),tracking=False)[0]
        until(e,lambda:bool(switched));assert switched==[True]
        seq=g.ball_sequence;phases=[];original=g.set_state
        def record(state,phase,message):
            phases.append(phase);original(state,phase,message)
        g.set_state=record
        def transform(*args):
            tr=TransformStamped();tr.header.stamp=g.get_clock().now().to_msg()
            if failure=='stale_tf':tr.header.stamp.sec-=1
            tr.transform.translation.x=.3;tr.transform.translation.z=.1;tr.transform.rotation.w=1.
            return tr
        monkeypatch.setattr(seq.buffer,'lookup_transform',transform)
        initial=r.position[:];seq=exhaust(g)
        if failure=='stale_feedback':
            until(e,lambda:seq.virtual)
            monkeypatch.setattr(g.robot,'fresh',lambda:False)
        if failure=='stop_approach':
            r.run_seconds=1.
            until(e,lambda:seq.handle is not None,timeout=10.)
            g.command('stop')
        until(e,lambda:g.state=='paused',timeout=20.)
        until(e,lambda:not seq.busy,timeout=5.);spin(e,.2)
        assert not seq.dance_ready and not seq.grasp_observed and not g.dance.busy
        assert not g.command('dance')[0] and not planner.transfers and 'grasp_place' not in phases
        assert all(req.avoid_collisions for req in planner.requests)
        assert all(not req.start_state.attached_collision_objects for req in planner.requests)
        if failure is None:
            assert g.outcome=='virtual_grasp_only',g.message
            assert len(r.requests)==2 and [len(req.waypoints) for req in planner.requests]==[2,1,1]
            assert r.position==pytest.approx(initial)
            assert [list(req.command.position) for req in gripper.requests]==[[0.],[.95],[0.]]
            assert [req.waypoints[-1].position.z for req in planner.requests]==pytest.approx([.1,.15,.1])
        elif failure=='stop_approach':
            assert r.cancelled>=1 and len(r.requests)==1
            assert [list(req.command.position) for req in gripper.requests]==[[0.]]
        else:
            assert not r.requests and not gripper.requests and g.outcome!='virtual_grasp_only'
            assert not seq.retry_at and seq.virtual_attempted

