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
    with graph(parameters=['handshake_gripper_enabled:=true']) as ([g],e):
        reset(g,e);g.robot.mock_pose[2]=.05
        until(e,lambda:g.phase=='handshake')
        until(e,lambda:g.damping_movement.armed)
        assert not g.handshake_grip_sent and g.robot.grip_target==.6
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


@pytest.mark.parametrize('enabled',[False,True])
def test_real_ros_damping_auto_grip_is_disabled_by_default_and_can_be_reenabled(enabled):
    params=PARAMS+['gripper_enabled:=true']+(['handshake_gripper_enabled:=true'] if enabled else [])
    with graph((Cyberdog,FakeGameRobot,FakeGripper),params,executor) as ([g,r,gripper],e):
        modes=[];original=g.robot.grip_position
        def record(*args,**kwargs):
            modes.append(g.robot.modes.mode);return original(*args,**kwargs)
        g.robot.grip_position=record
        g.command('reset');until(e,lambda:g.state=='idle',timeout=8.)
        assert not gripper.requests  # Reset/holding does not open the gripper.
        r.position[2]=.06;until(e,lambda:g.phase=='handshake',timeout=14.)
        if enabled:until(e,lambda:g.damping_movement.armed)
        else:spin(e,.8)
        assert [list(req.command.position) for req in gripper.requests]==[[.6]]
        assert modes==['damping'] and g.robot.grip_target==.6
        requests=len(r.requests);switches=len(r.switch_requests)
        r.position[1]+=.06
        if enabled:
            until(e,lambda:len(gripper.requests)==2 and not g.robot.grip_busy)
            assert [list(req.command.position) for req in gripper.requests]==[[.6],[.7]]
            assert modes==['damping','damping']
        else:
            spin(e,.5)
            assert len(gripper.requests)==1 and not g.handshake_grip_sent and not g.damping_movement.armed
            assert '自动闭爪已关闭' in g.message
        assert g.robot.modes.mode=='damping' and len(r.switch_requests)==switches
        assert len(r.requests)==requests and not r.direct_commands


@pytest.mark.parametrize('command',['stop','reset'])
def test_late_open_completion_cannot_resume_handshake_after_stop_or_reset(command):
    with graph() as ([g],e):
        callbacks=[]
        def delayed(position,callback,**kwargs):
            assert position==.6 and g.robot.modes.mode=='damping'
            callbacks.append(callback);return True,'accepted'
        g.robot.grip_position=delayed
        reset(g,e);g.robot.mock_pose[2]=.05
        until(e,lambda:g.phase=='handshake_opening')
        assert len(callbacks)==1
        g.command(command);callbacks[0](True,'late arrival')
        if command=='reset':until(e,lambda:g.state=='idle')
        else:spin(e,.15)
        assert g.phase==('holding' if command=='reset' else 'stopped')
        assert len(callbacks)==1 and not g.handshake_grip_sent


def test_partial_handshake_open_stall_pauses_without_claiming_ready():
    with graph(parameters=['mock_step_sec:=0.2']) as ([g],e):
        reset(g,e);g.robot.mock_pose[2]=.05
        until(e,lambda:g.phase=='handshake_opening' and g.robot.grip_busy)
        result=Future();result.set_result(SimpleNamespace(status=4,
            result=SimpleNamespace(reached_goal=False,stalled=True)))
        g.robot.grip_result(result);spin(e,.3)
        assert g.state=='paused' and g.phase=='stopped' and g.robot.modes.mode=='damping'
        assert g.robot.grip_target==.6 and not g.robot.grip_busy


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
