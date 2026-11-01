"""Grasp transition timing: continuous rest, async feedback races, no hardware."""
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from controller_manager_msgs.msg import ControllerState
from geometry_msgs.msg import TransformStamped

import cyberdog.ball_sequence as module
from cyberdog.behavior import Cyberdog
from cyberdog.controller_modes import BASE,CHAINS,JOINTS_SETTLING,JTC_SETTLING
from rebot_arm_hmi_bridge.protocol import Mode
from fake_hardware import FakeGameRobot,executor
from test_behavior import graph,until
from test_grasp_retry import lock_target
from test_blind_grasp import FakeGraspPlanner
from test_handshake import FakeGripper
from test_hardware_adapter import PARAMS


def controllers(mode=Mode.POSITION):
    states={BASE:'active'}
    for chain in CHAINS.values():
        for name in chain:states[name]='inactive'
    for name in CHAINS[mode]:states[name]='active'
    return [ControllerState(name=name,state=state) for name,state in states.items()]


def real_ticks(g,monkeypatch):
    seq=lock_target(g);clock=[100.]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    monkeypatch.setattr(g.robot,'mock',False)
    monkeypatch.setattr(g.robot,'fresh',lambda:True)
    monkeypatch.setattr(g.robot,'stationary',lambda:True)
    seq.feedback=([0.]*6,0.);seq.deadline=105.;seq.attempt_deadline=220.
    def tick(at):
        clock[0]=at;seq.feedback_at=at;seq.tick()
    return seq,tick


def result_future(mode=Mode.POSITION):
    future=Future();future.set_result(SimpleNamespace(controller=controllers(mode)));return future


def test_continuous_settling_restarts_after_motion_instead_of_adding_separate_quiet_periods(monkeypatch):
    with graph() as ([g],e):
        seq,tick=real_ticks(g,monkeypatch);quiet=[True];queries=[]
        monkeypatch.setattr(g.robot,'stationary',lambda:quiet[0])
        monkeypatch.setattr(g.robot.modes,'list',SimpleNamespace(call_async=lambda request:queries.append(request) or Future()))
        tick(100.);tick(100.4)
        quiet[0]=False;tick(100.45)
        assert seq.settled_since is None and not queries
        quiet[0]=True;tick(100.6);tick(101.05)
        assert not queries and seq.phase=='settle'
        tick(101.11)
        assert len(queries)==1 and seq.phase=='guard_all' and seq.attempts==1


@pytest.mark.parametrize('phase',['guard_all','guard_segment'])
@pytest.mark.parametrize('reason',[JOINTS_SETTLING,JTC_SETTLING])
def test_async_transient_guard_failure_waits_with_the_same_target_attempt_and_deadline(phase,reason,monkeypatch):
    with graph() as ([g],e):
        seq,tick=real_ticks(g,monkeypatch);target=seq.frozen[:];generation=g.generation
        seq.phase=phase;seq.future=result_future();seq.settled_since=99.
        monkeypatch.setattr(g.robot.modes,'guard',lambda _:reason)
        tick(100.8)
        assert seq.phase==('settle' if phase=='guard_all' else 'segment_settle')
        assert seq.settled_since is None and seq.future is None and seq.deadline==105.
        assert seq.running and g.state=='active' and seq.attempts==1 and seq.frozen==target
        assert g.generation==generation and seq.retry_at is None and not g.robot.servo
        assert seq.status()['settling_reason']==reason
        # Recurring jitter must not extend the wait forever or open the gripper.
        seq.phase=phase;seq.future=result_future();tick(104.9)
        assert seq.deadline==105.
        tick(105.01)
        assert g.state=='paused' and '停稳超时' in g.message and seq.retry_at is None


def test_success_after_late_recovery_gets_time_to_read_scene_without_relocking_ball(monkeypatch):
    with graph() as ([g],e):
        seq,tick=real_ticks(g,monkeypatch);target=seq.frozen[:]
        reason=[JOINTS_SETTLING];queries=[]
        monkeypatch.setattr(g.robot.modes,'guard',lambda _:reason[0])
        monkeypatch.setattr(seq,'scene',SimpleNamespace(service_is_ready=lambda:True,
                           call_async=lambda request:queries.append(request) or Future()))
        seq.phase='guard_all';seq.future=result_future();tick(100.8)
        reason[0]='';seq.phase='guard_all';seq.future=result_future();tick(104.9)
        assert seq.phase=='scene' and queries and seq.deadline==pytest.approx(109.9)
        assert seq.frozen==target and seq.attempts==1 and not seq.settling_reason


@pytest.mark.parametrize('fault',['active_action','other_writer','wrong_mode','stale_joints','stale_controller','following_error'])
def test_only_transient_settling_is_recoverable_other_faults_still_pause(fault,monkeypatch):
    with graph() as ([g],e):
        seq,tick=real_ticks(g,monkeypatch)
        seq.phase='guard_all';seq.future=result_future(Mode.DAMPING if fault=='wrong_mode' else Mode.POSITION)
        reason={'active_action':'trajectory still active','other_writer':'another direct trajectory publisher exists'}.get(fault,'')
        monkeypatch.setattr(g.robot.modes,'guard',lambda _:reason)
        if fault=='stale_joints':monkeypatch.setattr(g.robot,'fresh',lambda:False)
        if fault=='following_error':seq.feedback=([0.]*6,.061)
        if fault=='stale_controller':seq.feedback_at=99.;module.time.monotonic=lambda:101.;seq.tick()
        else:tick(101.)
        assert g.state=='paused' and not seq.running and seq.retry_at is None
        assert not g.ball_sequence.dance_ready


def test_stop_cancels_the_settling_wait_and_a_late_list_result_cannot_restart(monkeypatch):
    with graph() as ([g],e):
        seq,tick=real_ticks(g,monkeypatch);future=Future();future.set_running_or_notify_cancel()
        seq.phase='guard_all';seq.future=future
        g.command('stop');future.set_result(SimpleNamespace(controller=controllers()));tick(101.)
        assert g.state=='paused' and not seq.running and seq.future is None
        assert seq.frozen is None and not g.robot.servo and not seq.retry_at


@pytest.mark.parametrize('disturb',['before_approach','before_place'])
def test_ros_joint_movement_during_list_response_recovers_same_blind_grasp(disturb,monkeypatch):
    class Robot(FakeGameRobot):
        def __init__(self):super().__init__();self.injected=False;self.joint_bump_at=0.
        def list(self,request,response):
            seq=getattr(g,'ball_sequence',None)
            phase='guard_all' if disturb=='before_approach' else 'guard_segment'
            placing=disturb=='before_approach' or seq.closing_started
            if seq and seq.running and seq.phase==phase and placing and not self.injected:
                self.injected=True;self.joint_bump_at=time.monotonic()
                self.position[3]+=.008
                self.telemetry()
                time.sleep(.12)  # Feedback changes while ListControllers is in flight.
            return super().list(request,response)
    params=PARAMS+['gripper_enabled:=true','allow_nominal_tcp_trial:=true']
    with graph((Cyberdog,Robot,FakeGripper,FakeGraspPlanner),params,executor) as ([g,r,gripper,planner],e):
        until(e,lambda:g.robot.motion.ready()=='',timeout=6.)
        switched=[];assert g.robot.mode('position',lambda ok,_:switched.append(ok),tracking=False)[0]
        until(e,lambda:bool(switched));assert switched==[True]
        tr=TransformStamped();tr.transform.translation.x=.22;tr.transform.translation.z=.06;tr.transform.rotation.w=1.
        monkeypatch.setattr(g.ball_sequence.buffer,'lookup_transform',lambda *a:tr)
        waits=[];wait=g.ball_sequence.wait_for_settling
        def record(reason):waits.append(reason);wait(reason)
        monkeypatch.setattr(g.ball_sequence,'wait_for_settling',record)
        seq=lock_target(g);target=seq.frozen[:]
        seq.reject(ValueError('target unavailable'),transient=True)
        until(e,lambda:g.state=='paused',timeout=18.)
        assert g.phase=='complete',g.message
        assert r.injected and JOINTS_SETTLING in waits and seq.attempts==1 and seq.retry_at is None
        assert len(r.requests)==2 and not r.switch_requests and not planner.transfers
        assert all([req.waypoints[-1].position.x,req.waypoints[-1].position.y,req.waypoints[-1].position.z]==pytest.approx(target)
                   for req in planner.requests)
        assert [list(req.command.position) for req in gripper.requests]==[[0.],[.95],[0.]]
        assert not g.robot.grip_closed
