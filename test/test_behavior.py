import json
import time
from contextlib import contextmanager
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from cyberdog.behavior import Cyberdog
from cyberdog.senses import Senses
from cyberdog.touch import TouchDetector
from cyberdog.ball_grasp_geometry import WorldStillGate, preview_waypoints


@contextmanager
def graph(classes=(Cyberdog,),parameters=(),executor_factory=SingleThreadedExecutor):
    args=['--ros-args','-p','mock_step_sec:=0.03','-p','ball_full_sequence:=true']
    for p in parameters:args+=['-p',p]
    rclpy.init(args=args)
    nodes=[];exe=executor_factory()
    try:
        for cls in classes:
            n=cls();nodes.append(n);exe.add_node(n)
        yield nodes,exe
    finally:
        for n in nodes:
            if isinstance(n,Cyberdog):n.prepare_shutdown()
        # Allow fake servers to deliver terminal results before destroying ROS waitables.
        end=time.monotonic()+.3
        while time.monotonic()<end:exe.spin_once(timeout_sec=.01)
        for n in nodes:
            for t in n.timers:t.cancel()
        exe.shutdown()
        for n in reversed(nodes):n.destroy_node()
        rclpy.shutdown()


def until(exe,condition,timeout=5.):
    deadline=time.monotonic()+timeout
    while not condition() and time.monotonic()<deadline:exe.spin_once(timeout_sec=.01)
    assert condition(),'condition timed out'


def spin(exe,seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:exe.spin_once(timeout_sec=.01)


def reset(g,e):
    assert g.command('reset')[0]
    until(e,lambda:g.state=='idle')


def test_starts_paused_reset_holds_exact_pose_without_calibration_override():
    with graph() as ([g],e):
        spin(e,.2)
        assert g.state=='paused' and not g.robot.busy
        reset(g,e)
        assert g.robot.current==[-1.5707,0.,0.,0.,0.,0.]
        assert g.robot.modes.mode=='position' and not g.robot.servo
        spin(e,.4)
        assert g.state=='idle' and g.robot.current==g.home


def test_only_j3_triggers_j4_lift_then_j6_inspection_then_handshake_pose_and_damping():
    with graph() as ([g],e):
        reset(g,e)
        g.robot.mock_pose[3]=.2
        spin(e,.35)
        assert g.state=='idle'
        g.robot.mock_pose[3]=0.;g.robot.mock_pose[2]=.05
        start=list(g.robot.current);moves=[];move=g.robot.move
        def record(targets,cb,*args):
            moves.append((g.phase,[list(p) for p in targets],g.robot.modes.mode))
            return move(targets,cb,*args)
        g.robot.move=record
        until(e,lambda:g.phase=='handshake')
        assert [p for p,_,_ in moves]==['wrist_lift','head_tilt','head_tilt','lifting']
        wrist=start[:];wrist[3]+=.25
        left=wrist[:];left[5]+=.25
        right=wrist[:];right[5]-=.25
        assert [targets for _,targets,_ in moves]==[[wrist],[left],[right],[g.lift]]
        assert all(mode=='position' for _,_,mode in moves)
        assert g.robot.current==g.lift
        assert g.robot.current[1:3]==[.6,.8]
        assert g.robot.current[3:]==[0.,0.,0.]
        assert g.robot.modes.mode=='damping'
        reset(g,e)
        assert g.robot.current==g.home and g.robot.modes.mode=='position'


def test_touch_noise_stale_samples_and_torque_baseline():
    d=TouchDetector('position')
    for i in range(20):assert not d.update(i*.05,.02,None,0.,True)
    assert not d.update(1.,.04,None,0.,True)
    assert not d.update(1.5,.04,None,0.,True)  # sample gap restarts dwell
    assert not d.update(1.6,.04,None,0.,True)
    assert not d.update(1.7,.04,None,0.,True)
    assert d.update(1.8,.04,None,0.,True)
    d=TouchDetector('torque')
    for i in range(30):assert not d.update(i*.05,0.,5.,0.,True)
    assert d.baseline==5.
    for i in range(7):trigger=d.update(1.5+i*.05,0.,6.2,0.,True)
    assert trigger
    d.update(2.,0.,None,0.,True)
    assert d.baseline is None


def test_ball_automatically_tracks_from_zero_pose_and_grabs_after_three_seconds():
    with graph((Cyberdog,Senses),['mock_ball_visible:=true']) as ([g,s],e):
        stages=[];original=g.set_state;events=[]
        def state(a,b,c):stages.append(b);original(a,b,c)
        g.set_state=state
        grip=g.robot.grip
        def record(closed,cb):events.append((closed,g.phase));return grip(closed,cb)
        g.robot.grip=record
        assert g.command('reset')[0]
        until(e,lambda:g.phase=='tracking')
        start=time.monotonic()
        spin(e,2.5)
        assert not any(closed for closed,_ in events)
        until(e,lambda:g.phase=='complete',timeout=6.)
        assert time.monotonic()-start>=3.
        assert [p for p in stages if p.startswith('grasp_')]==[
            'grasp_settling','grasp_approach','grasp_pregrasp','grasp_grasp','grasp_closing','grasp_lift']
        assert [(closed,phase) for closed,phase in events if closed]==[(True,'grasp_closing')]
        assert g.robot.grip_closed and g.state=='paused' and not g.ball_sequence.busy


def test_ball_detected_in_damping_enters_tracking_without_gesture():
    with graph((Cyberdog,Senses)) as ([g,s],e):
        reset(g,e);g.robot.mock_pose[2]=.05
        until(e,lambda:g.phase=='handshake')
        s.mock_ball_visible=True
        until(e,lambda:g.phase=='tracking')
        assert g.robot.modes.mode=='position'


@pytest.mark.parametrize('stop',['stop','loss'])
def test_stop_or_loss_during_approach_never_closes(stop):
    with graph((Cyberdog,Senses),['mock_ball_visible:=true','ball_still_sec:=0.5','mock_step_sec:=0.25',
                                'grasp_continue_on_loss:=false','grasp_max_attempts:=1']) as ([g,s],e):
        g.command('reset');until(e,lambda:g.phase=='grasp_approach')
        if stop=='stop':g.command('stop')
        else:s.mock_ball_visible=False
        until(e,lambda:g.state=='paused')
        spin(e,.5)
        assert not g.robot.grip_closed and not g.ball_sequence.running


def test_still_gate_resets_when_ball_moves_or_frames_are_missing():
    gate=WorldStillGate(3.)
    for i in range(30):assert not gate.update([.3,0.,.1],i*.1)
    assert gate.update([.3,0.,.1],3.)
    assert not gate.update([.32,0.,.1],3.1)
    assert not gate.update([.32,0.,.1],4.)
    assert not gate.update([float('nan'),0.,.1],4.1)


@pytest.mark.parametrize('fraction',[.76,float('nan')])
def test_incomplete_plan_cannot_send_or_close(fraction):
    with graph() as ([g],e):
        seq=g.ball_sequence;g.mode='ball';seq.running=True;seq.phase='full_plan'
        seq.attempts=1
        seq.deadline=time.monotonic()+5;seq.target=dict(center=[.3,0.,.1]);seq.frozen=[.3,0.,.1]
        seq.target_at=time.monotonic();seq.feedback=([0.,.6,.8,0.,0.,0.],0.);seq.feedback_at=time.monotonic()
        g.robot.mock=False;g.robot.fresh=lambda:True
        seq.future=Future();seq.future.set_result(SimpleNamespace(error_code=SimpleNamespace(val=1),fraction=fraction))
        g.robot.grip=lambda *a:pytest.fail('must not grip')
        seq.action.send_goal_async=lambda *a:pytest.fail('must not execute')
        seq.tick()
        assert g.state=='active' and g.phase=='grasp_retry_wait' and '路径不完整' in g.message
        assert seq.retry_at is not None and seq.attempts==1


def test_stop_cancels_late_accepted_goal_and_blocks_reset_until_terminal():
    with graph() as ([g],e):
        seq=g.ball_sequence;cancels=[];terminal=Future()
        handle=SimpleNamespace(accepted=True,get_result_async=lambda:terminal,cancel_goal_async=lambda:cancels.append(True))
        future=Future();future.set_result(handle)
        seq.running=True;seq.submitting=True;g.robot.sequence_motion_busy=True
        g.command('reset');seq.accepted(future)
        spin(e,.2)
        assert cancels and seq.busy and g.state=='resetting'
        terminal.set_result(SimpleNamespace(status=4,result=SimpleNamespace(error_code=0)))
        until(e,lambda:g.state=='idle')
        assert g.robot.current==g.home


def test_close_failure_never_lifts():
    with graph((Cyberdog,Senses),['mock_ball_visible:=true','ball_still_sec:=0.5']) as ([g,s],e):
        grip=g.robot.grip
        g.robot.grip=lambda closed,cb:(False,'close failed') if closed else grip(closed,cb)
        g.command('reset');until(e,lambda:g.state=='paused')
        assert g.message=='close failed'
        assert g.ball_sequence.index==2


def test_zero_reset_axes_remain_fixed_during_tracking():
    with graph() as ([g],e):
        reset(g,e);g.begin_tracking();until(e,lambda:g.robot.servo)
        for _ in range(100):
            target=g.robot.reference[:];target[3]=.2;target[4]=.3
            assert g.robot.track(target,.025)
            assert g.robot.current[:3]==g.home[:3]
            assert not g.robot.fault


def test_short_dropouts_allow_three_second_grasp_only_after_fresh_reconfirmation(monkeypatch):
    with graph() as ([g],e):
        g.mode='ball';g.state='active';s=g.ball_sequence
        base=g.get_clock().now().nanoseconds*1e-9
        with monkeypatch.context() as m:
            stamp=[base]
            m.setattr(g,'get_clock',lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(stamp[0]*1e9)+1000)))
            for i in range(31):
                stamp[0]=base+i*.1
                missing=i in (7,14,21,28)
                s.observe(dict(stamp=stamp[0],valid=True,source='mock',ball=None if missing else {'x':0.}))
                if missing:
                    assert not s.stable and not s.target_valid()
                    s.begin();assert not s.running
            assert s.stable and s.target_valid()
            s.begin();assert s.running and g.phase=='grasp_settling'


def test_long_dropout_or_real_movement_restarts_dwell():
    with graph() as ([g],e):
        s=g.ball_sequence;base=g.get_clock().now().nanoseconds*1e-9
        target=dict(center=[.3,0.,.1],radius=.025)
        for i in range(29):s.accept(target,base+i*.1)
        s.reject(ValueError('target unavailable'),transient=True)
        s.accept(target,base+3.1)
        assert not s.stable and s.still_sec==0
        s.accept(dict(target,center=[.32,0.,.1]),base+3.2)
        assert not s.stable and s.still_sec==0


def test_wider_tracking_envelope_is_applied_in_both_layers_and_respects_limits():
    from cyberdog.ball_prediction import BallPrediction
    from ghost_game_orchestrator.face_tracking import make_face_sample
    with graph() as ([g],e):
        r=g.robot;r.servo=True;r.set_tracking_profile(True)
        r.reference=[-1.5707,0.,0.,.8,.8,0.];r.mock_pose=list(r.reference)
        r.tracking_anchor=list(r.reference);g.anchor=list(r.reference)
        for direction in (-1.,1.):
            for i in range(500):
                now=time.monotonic()
                g.sample=make_face_sample(now,1.+direction,1.+direction,.2,.2,2,2,.9)
                g.ball_prediction.reset()
                g.tracking_tick(now,.025)
                assert not r.fault
                assert -.2-1e-8<=r.reference[3]<=1.52+1e-8
                assert -.4-1e-8<=r.reference[4]<=1.52+1e-8
                assert r.reference[:3]==[-1.5707,0.,0.] and r.reference[5]==0.
            if direction<0:
                assert r.reference[3]>1.5 and r.reference[4]>1.5
            else:
                assert r.reference[3]<-.19 and r.reference[4]<-.39


def test_tracking_can_retreat_from_a_start_inside_joint_margin():
    with graph() as ([g],e):
        r=g.robot;r.servo=True;r.set_tracking_profile(True)
        r.reference=[-1.5707,0.,0.,1.55,1.55,0.];r.mock_pose=list(r.reference)
        r.tracking_anchor=list(r.reference)
        for _ in range(100):
            target=list(r.reference);target[3]=1.4;target[4]=1.4
            assert r.track(target,.025)
            assert max(r.reference[3:5])<=1.55
        assert max(r.reference[3:5])<1.52 and not r.fault
