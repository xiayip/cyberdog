"""Noisy stationary targets, bounded dropouts and fresh-only grasp triggers."""
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time as Stamp
from geometry_msgs.msg import TransformStamped

from cyberdog.ball_grasp_geometry import BallFitError,WorldStillGate
from test_behavior import graph


def test_jitter_and_one_bad_fit_do_not_restart_one_second_dwell():
    gate=WorldStillGate()
    for i in range(11):
        p=[.3+(.012 if i%2 else -.012),.002*(i%3),.1]
        if i==5:p[0]+=.08
        stable=gate.update(p,i*.1)
        if i<10:assert not stable
    assert stable  # Fresh reconfirmation credits the brief outlier interval.
    assert gate.elapsed==pytest.approx(1.)


def test_outlier_at_trigger_time_is_not_a_grasp_target():
    gate=WorldStillGate()
    for i in range(10):assert not gate.update([.3,0.,.1],i*.1)
    assert not gate.update([.4,0.,.1],1.)
    assert gate.reason=='position_outlier'
    assert gate.elapsed==pytest.approx(.9)
    assert gate.update([.3,0.,.1],1.1)


@pytest.mark.parametrize('speed',[.06,.1,-.08])
def test_median_filter_does_not_turn_continuous_motion_into_stationarity(speed):
    gate=WorldStillGate()
    for i in range(80):assert not gate.update([.3+speed*i*.1,0.,.1],i*.1)


def test_confirmed_step_restarts_and_long_loss_discards_history():
    gate=WorldStillGate()
    for i in range(8):assert not gate.update([.3,0.,.1],i*.1)
    assert not gate.update([.36,0.,.1],.8)
    assert not gate.update([.36,0.,.1],.9)
    assert gate.reason=='movement_pending' and gate.elapsed==pytest.approx(.7)
    assert not gate.update([.36,0.,.1],1.)
    assert not gate.update([.36,0.,.1],1.1)
    assert gate.elapsed==0. and gate.reason=='target_moving'
    assert not gate.update([.36,0.,.1],2.7)
    assert gate.elapsed==0.
    assert gate.reset_reason=='sample_gap'
    assert not gate.update([.36,0.,.1],2.7)  # Duplicate stamps add no time.


def test_reconfirmed_short_gaps_count_the_full_interval():
    gate=WorldStillGate()
    for stamp in (0.,.45,.9):assert not gate.update([.3,0.,.1],stamp)
    assert gate.elapsed==pytest.approx(.9)
    assert gate.update([.3,0.,.1],1.05)


def test_hand_jitter_and_intermittent_fits_pass_after_one_second():
    gate=WorldStillGate()
    for stamp,x in [(0.,.3),(.3,.325),(.65,.29)]:
        assert not gate.update([x,0.,.1],stamp)
    assert gate.update([.32,0.,.1],1.05)


def test_a_long_gap_or_a_new_position_cannot_inherit_old_progress():
    for missing in (True,False):
        gate=WorldStillGate()
        for i in range(9):assert not gate.update([.3,0.,.1],i*.1)
        if missing:
            assert not gate.update([.3,0.,.1],2.4)
        else:
            for t in (1.,1.1,1.2,1.3):assert not gate.update([.4,0.,.1],t)
        assert gate.elapsed==0.


def test_short_dropout_survives_timer_but_needs_fresh_reconfirmation(monkeypatch):
    with graph() as ([g],executor):
        g.mode='ball';g.state='active';g.handshake_completed=True;seq=g.ball_sequence
        base=g.get_clock().now().nanoseconds*1e-9;stamp=[base]
        monkeypatch.setattr(g,'get_clock',lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=round(stamp[0]*1e9))))
        for i in range(7):
            stamp[0]=base+i*.1
            seq.observe(dict(stamp=stamp[0],valid=True,source='mock',ball={'x':0.}))
        stamp[0]=base+.7
        seq.observe(dict(stamp=stamp[0],valid=False,source='mock'))
        assert seq.still_sec==pytest.approx(.6,abs=1e-5)
        stamp[0]=base+.9;seq.tick();seq.begin()
        assert not seq.running and not seq.stable and not seq.target_valid()
        assert seq.still_sec==pytest.approx(.6,abs=1e-5)
        for t in (.95,1.05,1.15):
            stamp[0]=base+t
            seq.observe(dict(stamp=stamp[0],valid=True,source='mock',ball={'x':0.}))
        assert seq.stable and seq.target_valid()
        seq.begin();assert seq.running


@pytest.mark.parametrize('reconfirm',[True,False])
def test_looser_gap_handles_capture_latency_but_never_triggers_without_fresh_fit(reconfirm,monkeypatch):
    with graph() as ([g],executor):
        g.mode='ball';g.state='active';g.handshake_completed=True;seq=g.ball_sequence
        base=g.get_clock().now().nanoseconds*1e-9;now=[base]
        monkeypatch.setattr(g,'get_clock',lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=round(now[0]*1e9))))
        for t in (0.,.2,.4):
            now[0]=base+t
            seq.observe(dict(stamp=now[0],valid=True,source='mock',ball={'x':0.}))
        now[0]=base+.5
        seq.observe(dict(stamp=now[0],valid=False,source='mock'))
        now[0]=base+1.9;seq.tick();seq.begin()
        assert not seq.running and seq.target is None
        assert seq.still_sec==pytest.approx(.4,abs=1e-5)
        if reconfirm:
            seq.observe(dict(stamp=base+1.8,valid=True,source='mock',ball={'x':0.}))
            assert seq.stable and seq.target_valid()
            seq.begin();assert seq.running
        else:
            now[0]=base+2.15;seq.tick();seq.begin()
            assert not seq.running and seq.still_sec==0.


@pytest.mark.parametrize('fit_error',[True,False])
def test_depth_fit_failure_preserves_progress_but_calibration_error_resets(fit_error,monkeypatch):
    with graph() as ([g],executor):
        g.mode='ball';g.state='active';g.handshake_completed=True;seq=g.ball_sequence
        base=g.get_clock().now().nanoseconds*1e-9;now=base+.7
        monkeypatch.setattr(g,'get_clock',lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=round(now*1e9))))
        for i in range(7):seq.accept(dict(center=[.3,0.,.1],radius=.025),base+i*.1)
        seconds,nanos=divmod(round(now*1e9),1_000_000_000)
        header=SimpleNamespace(frame_id='camera',stamp=Stamp(sec=seconds,nanosec=nanos))
        seq.info=seq.depth_info=SimpleNamespace(header=header,distortion_model='plumb_bob',height=2,width=2,k=[1.]*9,d=[])
        seq.depth.append(SimpleNamespace(header=header));seq.pending.append(dict(stamp=now,ball={}))
        transform=TransformStamped();transform.transform.rotation.w=1.
        monkeypatch.setattr(seq.buffer,'lookup_transform',lambda *args:transform)
        monkeypatch.setattr('cyberdog.ball_sequence.decode_depth',lambda *args:np.ones((2,2)))
        def fail(*args,**kwargs):
            raise BallFitError('sphere fit does not reliably describe selected ball') if fit_error else ValueError('invalid camera intrinsics')
        monkeypatch.setattr('cyberdog.ball_sequence.locate_ball',fail)
        seq.locate_pending();seq.begin()
        assert not seq.running and not seq.stable and seq.target is None
        assert seq.still_sec==pytest.approx(.6 if fit_error else 0.,abs=1e-5)


def test_ready_fit_locks_before_a_dropout_can_erase_it_between_control_ticks(monkeypatch):
    with graph() as ([g],executor):
        g.mode='ball';g.state='active';g.handshake_completed=True;g.phase='tracking';g.robot.servo=True
        seq=g.ball_sequence;base=g.get_clock().now().nanoseconds*1e-9;now=[base]
        monkeypatch.setattr(g,'get_clock',lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=round(now[0]*1e9))))
        for i in range(11):
            now[0]=base+i*.1
            seq.observe(dict(stamp=now[0],valid=True,source='mock',ball={'x':0.}))
            if i<10:assert not seq.running
        # No timer has run: locking must consume the qualifying fit itself.
        assert seq.running and seq.vision_locked and seq.attempts==1
        assert g.phase=='grasp_settling' and not g.robot.servo
        frozen=list(seq.frozen)
        now[0]+=.03
        seq.observe(dict(stamp=now[0],valid=False,source='mock'))
        assert seq.running and seq.frozen==frozen and seq.attempts==1
        g.pause('operator pause')
        assert not seq.running and seq.frozen is None and seq.gate.elapsed==0.


@pytest.mark.parametrize('block',['paused','transition','retry','servo_off','busy','fault','stale','before_handshake'])
def test_qualifying_fit_cannot_start_outside_available_tracking(block,monkeypatch):
    with graph() as ([g],executor):
        g.mode='ball';g.state='active';g.handshake_completed=True;g.phase='tracking';g.robot.servo=True
        seq=g.ball_sequence;base=g.get_clock().now().nanoseconds*1e-9;now=[base]
        monkeypatch.setattr(g,'get_clock',lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=round(now[0]*1e9))))
        if block in ('paused','transition'):g.state=block
        elif block=='before_handshake':g.handshake_completed=False
        elif block=='retry':g.phase='grasp_retry_wait'
        elif block=='servo_off':g.robot.servo=False
        elif block=='busy':g.robot.sequence_motion_busy=True
        elif block=='fault':g.robot.fault='feedback lost'
        for i in range(11):
            stamp=base+i*.1;now[0]=stamp+(.25 if block=='stale' else 0.)
            seq.accept(dict(center=[.3,0.,.1],radius=.025),stamp)
        assert not seq.running and seq.attempts==0


@pytest.mark.parametrize('offset',[-.06,.06])
def test_brief_cluster_of_jump_points_preserves_progress_until_reconfirmed(offset):
    gate=WorldStillGate()
    for i in range(12):
        x=.3+(offset if i in (8,9) else 0.)
        stable=gate.update([x,0.,.1],i*.1)
        if i<11:assert not stable
        if i in (8,9,10):assert gate.elapsed==pytest.approx(.7)
    assert stable and gate.elapsed==pytest.approx(1.1)


def test_intermittent_depth_with_one_second_gap_can_resume_but_needs_three_fits():
    gate=WorldStillGate()
    assert not gate.update([.3,0.,.1],0.)
    assert not gate.update([.301,0.,.1],1.1)  # Two fits alone are insufficient.
    assert gate.update([.299,0.,.1],1.2)


def test_confirmed_new_position_must_complete_its_own_dwell():
    gate=WorldStillGate()
    for i in range(17):
        x=.3 if i<8 else .36
        assert not gate.update([x,0.,.1],i*.1)
    assert gate.reset_reason=='target_moving'
    assert not gate.update([.36,0.,.1],2.)
    assert gate.update([.36,0.,.1],2.1)


def test_long_sequence_of_outliers_cannot_keep_old_progress_alive():
    gate=WorldStillGate()
    for t in (0.,.1,.2):assert not gate.update([.3,0.,.1],t)
    jumps=([.5,0.,.1],[.3,.2,.1],[.3,0.,.3])
    for i in range(3,20):assert not gate.update(jumps[i%3],i*.1)
    assert not gate.update([.3,0.,.1],2.)
