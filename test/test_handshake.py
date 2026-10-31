import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionServer
from rclpy.node import Node

from cyberdog.behavior import Cyberdog
from cyberdog.game_core import inspection_poses
from cyberdog.touch import DampingMovementDetector
from fake_hardware import FakeGameRobot, executor
from test_behavior import graph, reset, until, spin
from test_hardware_adapter import PARAMS


def test_inspection_preserves_all_other_axes_and_checks_every_target_before_execution():
    start=[-.7,.1,.2,.3,-.4,.5]
    wrist,heads=inspection_poses(start,.25,[.25,-.25])
    assert wrist==[-.7,.1,.2,.55,-.4,.5]
    assert heads==[[-.7,.1,.2,.55,-.4,.75],[-.7,.1,.2,.55,-.4,.25]]
    assert start==[-.7,.1,.2,.3,-.4,.5]
    with pytest.raises(ValueError):inspection_poses([0.,0.,0.,1.5,0.,0.],.25,[.25])
    with pytest.raises(ValueError):inspection_poses([0.,0.,0.,0.,0.,3.1],.25,[.25])


@pytest.mark.parametrize('joint',range(6))
def test_damping_movement_detects_any_joint_once_after_settling_and_dwell(joint):
    d=DampingMovementDetector();q=[0.]*6
    for i in range(11):
        q[joint]=i*.01  # Controller handover settling is absorbed.
        assert not d.update(i*.05,q)
    q[joint]=.11
    assert not d.update(.55,q)  # Below threshold relative to the settled anchor.
    q[joint]=.16
    assert not d.update(.6,q)
    assert not d.update(.7,q)
    assert d.update(.76,q)
    assert d.joint==joint+1
    for i in range(10):assert not d.update(.8+i*.05,q)


def test_damping_noise_missing_feedback_and_short_impulses_do_not_trigger():
    d=DampingMovementDetector()
    for i in range(20):assert not d.update(i*.05,[.002*(i%2)]*6)
    assert not d.update(1.,[.06]*6)
    assert not d.update(1.05,[0.]*6)
    assert not d.update(1.1,[.06]*6)
    assert not d.update(1.15,None)
    assert not d.update(1.2,[.3]*6)
    assert not d.armed
    assert not d.update(2.,[.6]*6)  # Gap rearms from the new pose, no stale jump.


@pytest.mark.parametrize('phase',['wrist_lift','head_tilt','head_looking','lifting'])
def test_stop_during_inspection_cancels_remaining_moves_and_damping(phase):
    with graph(parameters=['mock_step_sec:=0.15']) as ([g],e):
        reset(g,e);g.robot.mock_pose[2]=.05
        until(e,lambda:g.phase==phase)
        g.command('stop');spin(e,.5)
        assert g.state=='paused' and not g.head_poses
        assert g.robot.modes.mode=='position' and not g.robot.busy


def test_mock_damping_motion_commands_point_seven_once_and_keeps_damping():
    with graph() as ([g],e):
        reset(g,e);g.robot.mock_pose[2]=.05
        until(e,lambda:g.phase=='handshake')
        until(e,lambda:g.damping_movement.armed)
        assert not g.handshake_grip_sent and g.robot.grip_target==0.
        g.robot.mock_pose[4]+=.06
        until(e,lambda:g.handshake_grip_sent and not g.robot.grip_busy)
        assert g.robot.grip_target==.7 and g.robot.modes.mode=='damping'
        g.robot.grip_position=lambda *a,**k:pytest.fail('must not send a second close')
        g.robot.mock_pose[0]+=.1;spin(e,.4)
        assert g.state=='active' and g.phase=='handshake'


class FakeGripper(Node):
    def __init__(self):
        super().__init__('cyberdog_test_gripper');self.requests=[]
        self.action=ActionServer(self,ParallelGripperCommand,'/rebot_arm_rs_gripper_controller/gripper_cmd',self.execute)
    def execute(self,handle):
        self.requests.append(handle.request);handle.succeed()
        result=ParallelGripperCommand.Result(reached_goal=True)
        result.state.position=list(handle.request.command.position)
        return result
    def destroy_node(self):
        self.action.destroy();return super().destroy_node()


def test_real_ros_damping_movement_sends_exact_point_seven_to_gripper_action():
    with graph((Cyberdog,FakeGameRobot,FakeGripper),PARAMS+['gripper_enabled:=true'],executor) as ([g,r,gripper],e):
        g.command('reset');until(e,lambda:g.state=='idle',timeout=8.)
        r.position[2]=.06;until(e,lambda:g.phase=='handshake',timeout=14.)
        until(e,lambda:g.damping_movement.armed)
        assert not gripper.requests
        requests=len(r.requests);switches=len(r.switch_requests)
        r.position[1]+=.06
        until(e,lambda:bool(gripper.requests) and not g.robot.grip_busy)
        assert len(gripper.requests)==1
        assert list(gripper.requests[0].command.position)==[.7]
        assert g.robot.modes.mode=='damping' and len(r.switch_requests)==switches
        assert len(r.requests)==requests and not r.direct_commands


@pytest.mark.parametrize('value',[-.1,1.1,float('nan'),True])
def test_invalid_partial_gripper_positions_do_not_submit(value):
    with graph() as ([g],e):
        assert not g.robot.grip_position(value,lambda *a:None)[0]
        assert not g.robot.grip_busy and g.robot.grip_target==0.


def test_partial_position_stall_is_not_claimed_as_arrival_but_full_grasp_stall_is_supported():
    with graph() as ([g],e):
        r=g.robot;outcomes=[]
        result=Future();result.set_result(SimpleNamespace(status=4,result=SimpleNamespace(reached_goal=False,stalled=True)))
        assert r.grip_position(.7,lambda ok,reason:outcomes.append(ok))[0]
        r.grip_result(result);assert outcomes==[False]
        assert r.grip(True,lambda ok,reason:outcomes.append(ok))[0]
        r.grip_result(result);assert outcomes==[False,True]
