"""Measured gripper loss, cancellation handover and the one-axis bow reaction."""
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import TransformStamped
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState

from cyberdog.behavior import Cyberdog
from cyberdog.motion import JOINTS
from fake_hardware import FakeGameRobot,executor
from test_behavior import graph,until,spin
from test_blind_grasp import FakeGraspPlanner
from test_grasp_retry import lock_target
from test_handshake import FakeGripper
from test_hardware_adapter import PARAMS


def feedback(g,position,age=0.):
    msg=JointState(name=[g.robot.gripper_joint],position=[position])
    ns=g.get_clock().now().nanoseconds-round(age*1e9)
    msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(ns,1_000_000_000)
    g.robot.gripper_joint_state(msg)
    return msg


def placing(g):
    seq=g.ball_sequence
    g.state='active';g.mode='ball';g.phase='grasp_place'
    seq.running=True;seq.phase='moving';seq.placement_started=True;seq.closing_started=True
    return seq


@pytest.mark.parametrize('value,age',[(.7,0.),(.92,0.),(.95,.4),(float('nan'),0.),(1.1,0.)])
def test_only_fresh_measured_position_strictly_above_threshold_interrupts(value,age):
    with graph() as ([g],e):
        seq=placing(g);g.robot.grip_target=.95
        assert not g.ball_taken.watch()  # Commanded .95 is never interpreted as state.
        feedback(g,value,age)
        assert not g.ball_taken.watch() and seq.running


def test_duplicate_gripper_feedback_cannot_refresh_an_old_position():
    with graph() as ([g],e):
        seq=placing(g);msg=feedback(g,.95)
        g.robot.gripper_position_at-=1.
        g.robot.gripper_joint_state(msg)
        assert g.robot.fresh_gripper_position() is None
        assert not g.ball_taken.watch() and seq.running


@pytest.mark.parametrize('stage',['approach','closing','releasing','paused'])
def test_empty_closed_gripper_does_not_interrupt_outside_placement(stage):
    with graph() as ([g],e):
        seq=placing(g);seq.placement_started=False
        if stage=='paused':seq.running=False
        seq.phase=stage;feedback(g,.95)
        assert not g.ball_taken.watch() and not g.ball_taken.busy


def test_late_accepted_place_must_terminate_before_bowing_from_the_stopped_pose():
    with graph() as ([g],e):
        seq=placing(g);seq.submitting=True;g.robot.sequence_motion_busy=True
        feedback(g,.95);assert g.ball_taken.watch()
        canceled=[];terminal=Future()
        handle=SimpleNamespace(accepted=True,get_result_async=lambda:terminal,
            cancel_goal_async=lambda:canceled.append(True))
        accepted=Future();accepted.set_result(handle);seq.accepted(accepted)
        g.ball_taken.tick()
        assert canceled and g.phase=='ball_taken_stopping' and not g.robot.motion.busy
        g.robot.mock_pose=[.13,.6,.8,.33,-.2,.4]
        terminal.set_result(SimpleNamespace(status=5,result=SimpleNamespace(error_code=0)))
        until(e,lambda:g.phase=='ball_taken')
        assert g.robot.current==pytest.approx([.13,.6,.8,-.12,-.2,.4])
        assert g.state=='paused' and not seq.running and seq.retry_at is None


def test_manual_stop_while_waiting_for_late_place_acceptance_prevents_bow():
    with graph() as ([g],e):
        seq=placing(g);seq.submitting=True;g.robot.sequence_motion_busy=True
        feedback(g,.95);assert g.ball_taken.watch()
        g.command('stop');before=list(g.robot.current)
        accepted=Future();accepted.set_result(SimpleNamespace(accepted=False));seq.accepted(accepted)
        spin(e,.15)
        assert g.state=='paused' and not g.ball_taken.busy and g.robot.current==before


def test_failed_cancel_request_keeps_motion_blocked_until_a_terminal_result():
    with graph() as ([g],e):
        seq=placing(g);before=list(g.robot.current)
        def fail_cancel():raise RuntimeError('transport unavailable')
        seq.handle=SimpleNamespace(cancel_goal_async=fail_cancel)
        g.robot.sequence_motion_busy=True
        feedback(g,.95);assert g.ball_taken.watch()
        g.ball_taken.tick()
        assert seq.busy and not g.robot.motion.busy and g.robot.current==before
        g.ball_taken.deadline=time.monotonic()-1.;g.ball_taken.tick()
        assert g.state=='paused' and not g.ball_taken.busy and seq.busy
        seq.handle=None;g.robot.sequence_motion_busy=False


def test_a_lost_ball_at_place_arrival_cannot_open_the_gripper():
    with graph() as ([g],e):
        seq=placing(g);seq.phase='arrival';seq.waypoints=[dict(stage='place')];seq.index=0
        g.robot.grip=lambda *args:pytest.fail('must not open after loss')
        feedback(g,.95);seq.arrived()
        assert g.phase=='ball_taken_stopping' and not seq.running


def test_j4_downward_limit_holds_without_reversing_the_bow():
    with graph() as ([g],e):
        placing(g);g.robot.mock_pose[3]=-1.55;before=list(g.robot.current)
        feedback(g,.95);assert g.ball_taken.watch()
        g.ball_taken.tick()
        assert g.state=='paused' and 'J4' in g.message
        assert g.robot.current==before and not g.robot.motion.busy


@pytest.mark.parametrize('event',['taken','collision','stop_bow'])
def test_ros_place_is_canceled_then_only_j4_bows_without_release(event,monkeypatch):
    class Robot(FakeGameRobot):
        traj='zephyr_arm_trajectory_controller'
        def __init__(self):
            super().__init__();self.gripper_position=0.
        def list(self,request,response):
            response=super().list(request,response)
            for c in response.controller:c.name=c.name.replace('rebot_arm_rs','zephyr_arm',1)
            return response
    class Gripper(FakeGripper):
        def execute(self,handle):
            self.requests.append(handle.request)
            closed=handle.request.command.position[0]>0.
            r.gripper_position=.7 if closed else 0.
            handle.succeed()
            return ParallelGripperCommand.Result(stalled=closed,reached_goal=not closed)
    class Planner(FakeGraspPlanner):
        def __init__(self):
            super().__init__();self.checks=[]
            self.create_service(GetStateValidity,'/check_state_validity',self.check)
        def check(self,req,res):
            self.checks.append(req);res.valid=event!='collision';return res
    params=PARAMS+['gripper_enabled:=true','allow_nominal_tcp_trial:=true','motion_limit_scale:=2.0',
                  'grasp_taken_bow_speed:=0.6','controller_prefix:=zephyr_arm']
    with graph((Cyberdog,Robot,Gripper,Planner),params,executor) as ([g,r,gripper,planner],e):
        until(e,lambda:g.robot.motion.ready()=='',timeout=6.)
        switched=[];g.robot.mode('position',lambda ok,reason:switched.append(ok),tracking=False)
        until(e,lambda:switched);assert switched==[True]
        tr=TransformStamped();tr.transform.translation.x=.22;tr.transform.translation.z=.06;tr.transform.rotation.w=1.
        monkeypatch.setattr(g.ball_sequence.buffer,'lookup_transform',lambda *args:tr)
        r.run_seconds=.6;seq=lock_target(g)
        until(e,lambda:g.phase=='grasp_place' and seq.handle is not None,timeout=12.)
        stopped=[.05,.6,.8,.32,.1,0.]
        r.position=list(stopped);r.gripper_position=.95
        if event=='stop_bow':
            until(e,lambda:g.phase=='ball_taken_bowing' and len(r.requests)==3,timeout=10.)
            g.command('stop')
        until(e,lambda:g.state=='paused' and not seq.busy and not g.robot.busy,timeout=12.)
        spin(e,.2)
        assert r.cancelled>=1 and seq.retry_at is None
        assert [list(req.command.position) for req in gripper.requests]==[[0.],[.95]]
        assert g.robot.grip_target==.95 and not seq.running
        assert planner.checks and all(not req.robot_state.attached_collision_objects for req in planner.checks)
        assert len(r.requests)==(2 if event=='collision' else 3)
        if event=='collision':
            assert '碰撞' in g.message and r.position==stopped
        else:
            target=list(stopped);target[3]-=.45
            assert list(r.requests[-1].trajectory.points[-1].positions)==pytest.approx(target)
            from cyberdog.game_core import path_points
            baseline=path_points(stopped,[target],.6,.65,[1.3]*6)
            assert [p.time_from_start.sec+p.time_from_start.nanosec*1e-9
                    for p in r.requests[-1].trajectory.points]==pytest.approx([t/3 for t,_ in baseline])
            for p in r.requests[-1].trajectory.points:
                assert [p.positions[i] for i in (0,1,2,4,5)]==pytest.approx([stopped[i] for i in (0,1,2,4,5)])
            if event=='taken':assert g.phase=='ball_taken' and r.position==pytest.approx(target)
            else:assert r.cancelled>=2 and g.phase=='stopped'
