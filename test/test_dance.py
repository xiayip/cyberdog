import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from rclpy.node import Node
from moveit_msgs.srv import GetStateValidity

from cyberdog.behavior import Cyberdog
from cyberdog.choreography import DancePlan
from fake_hardware import FakeGameRobot,executor
from test_behavior import graph,reset,until,spin
from test_hardware_adapter import PARAMS

PACE=['motion_limit_scale:=2.0','handshake_inspection_acceleration:=6.0']


def completed_round(g):
    # This suite isolates dance submission; full grasp evidence is tested in test_flow_gates.
    g.handshake_completed=True
    seq=g.ball_sequence;seq.grasp_observed=True;seq.held_at_release=True;seq.release_completed=True;seq.dance_ready=True
    g.set_state('paused','complete','test: verified release')


def test_dance_requires_a_completed_grasp_place_round():
    with graph(parameters=PACE) as ([g],e):
        assert not g.command('dance')[0]
        reset(g,e);spin(e,.1);assert not g.dance.busy
        assert not g.command('dance')[0]
        g.handshake();until(e,lambda:g.phase=='handshake')
        assert not g.command('dance')[0]
        completed_round(g)
        assert g.command('dance')[0]
        assert not g.command('dance')[0]
        until(e,lambda:g.state=='idle',timeout=8.)
        assert g.robot.current==g.home and not g.dance.busy
        assert g.robot.modes.mode=='position' and not g.robot.servo
        g.mode='ball';g.set_state('active','tracking','test')
        assert not g.command('dance')[0]
        g.set_state('active','grasp_closing','test');assert not g.command('dance')[0]


def test_qualified_dance_does_not_operate_gripper_or_take_over_for_ball():
    with graph(parameters=PACE) as ([g],e):
        reset(g,e);completed_round(g)
        gripper=g.robot.grip_target
        assert g.command('dance')[0]
        until(e,lambda:g.phase=='dance_running',timeout=8.)
        g.ball_seen=time.monotonic();g.tick()
        assert g.mode=='dance' and g.phase=='dance_running'
        g.ball_seen=0.
        until(e,lambda:g.state=='idle',timeout=5.)
        assert g.robot.grip_target==gripper
        assert g.robot.current==g.home


@pytest.mark.parametrize('command',['stop','reset'])
def test_stop_or_reset_cancels_dance_and_late_callback_cannot_resume(command):
    with graph(parameters=PACE) as ([g],e):
        reset(g,e);completed_round(g);g.robot.mock_delay=2.
        assert g.command('dance')[0]
        until(e,lambda:g.phase=='dance_running',timeout=8.)
        late=g.robot.callback;assert g.command(command)[0]
        late(True,'late completion')
        assert not g.dance.busy and not g.robot.servo
        if command=='stop':assert g.state=='paused'
        else:
            until(e,lambda:g.state=='idle',timeout=8.)
            assert g.robot.current==g.home


def test_cancelled_scene_callback_and_timeout_never_start_a_trajectory():
    with graph(parameters=PACE) as ([g],e):
        reset(g,e);d=g.dance;d.busy=True;d.token+=1;token=d.token
        future=Future();future.set_running_or_notify_cancel();d.future=future
        g.command('stop');future.set_result(SimpleNamespace(valid=True));d.checked(future,token)
        assert g.state=='paused' and not g.robot.busy and not d.busy
        d.busy=True;d.deadline=time.monotonic()-1.;d.tick()
        assert not d.busy and '超时' in g.message


def test_late_planner_result_after_stop_cannot_start_dance():
    with graph(parameters=PACE) as ([g],e):
        reset(g,e);d=g.dance;d.busy=True
        work=Future();work.set_running_or_notify_cancel();d.work=work
        assert g.command('stop')[0]
        work.set_result(None);d.tick()
        assert d.work is None and not d.busy and not g.robot.busy and g.state=='paused'


def test_changed_start_reference_invalidates_checked_dance():
    with graph(parameters=PACE) as ([g],e):
        r=g.robot;plan=DancePlan(r.current,g.dance.centre,g.home,r.motion.velocity_limits(True),r.motion.acceleration_limits(True))
        r.mock_pose[0]+=.01
        ok,reason=r.dance(plan,lambda *args:None)
        assert not ok and 'reference moved' in reason and not r.busy


def test_submission_requires_immutable_plan_with_unchanged_limits():
    from dataclasses import FrozenInstanceError
    with graph(parameters=PACE) as ([g],e):
        r=g.robot;plan=DancePlan(r.current,g.dance.centre,g.home,r.motion.velocity_limits(True),r.motion.acceleration_limits(True))
        with pytest.raises(FrozenInstanceError):plan.points=()
        with pytest.raises(TypeError):plan.points[0][1][0]=0.
        assert not r.dance(list(plan.points),lambda *args:None)[0]
        r.motion.inspection_acceleration=3.
        ok,reason=r.dance(plan,lambda *args:None)
        assert not ok and 'limits changed' in reason and not r.busy


class Scene(Node):
    def __init__(self):
        super().__init__('fake_dance_scene');self.calls=[];self.valid=True
        self.service=self.create_service(GetStateValidity,'/check_state_validity',self.check)
    def check(self,request,response):
        self.calls.append(request);response.valid=self.valid;return response


@pytest.mark.parametrize('valid',[True,False])
def test_ros_scene_preflight_precedes_single_continuous_jtc_action(valid):
    with graph((Cyberdog,FakeGameRobot,Scene),PARAMS+PACE,executor) as ([g,r,scene],e):
        r.gripper_position=0.;scene.valid=valid
        g.command('reset');until(e,lambda:g.state=='idle',timeout=8.)
        completed_round(g)
        before=len(r.requests);assert g.command('dance')[0]
        until(e,lambda:(not g.dance.busy and g.state in ('idle','paused')),timeout=25.)
        assert scene.calls and all(req.group_name=='zephyr_arm_tcp' for req in scene.calls)
        if valid:
            assert g.state=='idle' and len(r.requests)==before+1
            assert len(scene.calls)==len(g.dance.samples)
            trajectory=r.requests[-1].trajectory
            assert len(trajectory.points)>300
            assert all(len(p.velocities)==len(p.accelerations)==6 for p in trajectory.points)
            assert list(trajectory.points[-1].positions)==pytest.approx(g.home)
            assert sum(max(abs(x) for x in p.velocities)>.01 for p in trajectory.points)>100
        else:
            assert g.state=='paused' and len(r.requests)==before
            assert '碰撞' in g.message
        assert not r.direct_commands and not r.impedance_goals
