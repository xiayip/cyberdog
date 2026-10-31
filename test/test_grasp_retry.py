"""Committed target, bounded recovery and cancellation-race regressions."""
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from cyberdog.behavior import Cyberdog
from cyberdog.senses import Senses
from cyberdog.table_scene import table_box
from test_behavior import graph, until, spin


def lock_target(game):
    game.state='active';game.mode='ball'
    seq=game.ball_sequence
    seq.accept(dict(center=[.3,0.,.1],radius=.025),game.get_clock().now().nanoseconds*1e-9)
    seq.stable=True;seq.begin()
    assert seq.running and seq.attempts==1
    return seq


@pytest.mark.parametrize('lost_phase',['grasp_settling','grasp_approach'])
def test_loss_after_lock_completes_one_attempt_including_close_and_direct_place(lost_phase):
    with graph((Cyberdog,Senses),['mock_ball_visible:=true','ball_still_sec:=0.5']) as ([g,s],e):
        original=g.set_state;stages=[];closed=[]
        grip=g.robot.grip
        def record_grip(close,cb):
            if close:closed.append(g.phase)
            return grip(close,cb)
        g.robot.grip=record_grip
        def state(a,b,c):
            original(a,b,c);stages.append(b)
            if b==lost_phase:
                s.mock_ball_visible=False
                # Same race as the report: disappearance immediately after
                # settling starts, before mock/real path generation.
                g.ball_sequence.reject(ValueError('target unavailable'),transient=True)
                assert g.ball_sequence.status()['using_frozen_target']
        g.set_state=state
        g.command('reset');until(e,lambda:g.phase=='complete',timeout=6.)
        assert closed==['grasp_closing'] and 'grasp_place' in stages and 'grasp_lift' not in stages
        assert g.ball_sequence.attempts==1 and not g.robot.grip_closed
        assert 'grasp_retry_wait' not in stages


def test_fit_failure_or_expired_image_does_not_erase_locked_geometry():
    with graph() as ([g],e):
        seq=lock_target(g);center=seq.frozen[:];radius=seq.frozen_radius
        seq.reject(ValueError('sphere fit does not reliably describe selected ball'))
        seq.target_at-=10.
        seq.check_target()
        seq.tick()  # No target['radius'] dereference after detector rejection.
        assert seq.running and seq.frozen==center and seq.frozen_radius==radius
        assert seq.phase=='opening'


def test_missing_target_cannot_start_first_or_retry_attempt():
    with graph() as ([g],e):
        g.state='active';g.mode='ball';seq=g.ball_sequence
        seq.stable=True;seq.begin()
        assert seq.attempts==0 and not seq.running
        seq=lock_target(g);seq.fail('replan',retryable=True)
        seq.retry_at=time.monotonic()-1
        seq.tick();until(e,lambda:g.phase=='tracking')
        spin(e,.3)
        assert seq.attempts==1 and not seq.running and seq.frozen is None


@pytest.mark.parametrize('eventual_success',[True,False])
def test_three_attempts_reacquire_with_new_dwell_then_complete_or_stop(eventual_success):
    with graph((Cyberdog,Senses),['mock_ball_visible:=true','ball_still_sec:=0.5','grasp_retry_delay_sec:=0.2']) as ([g,s],e):
        original=g.set_state;attempts=[];tracking_at=[];starts=[]
        def state(a,b,c):
            original(a,b,c)
            if b=='tracking':tracking_at.append(time.monotonic())
            if b=='grasp_settling':
                attempts.append(g.ball_sequence.attempts);starts.append(time.monotonic())
                if not eventual_success or g.ball_sequence.attempts<3:
                    g.ball_sequence.fail('抓取路径不完整',retryable=True)
        g.set_state=state
        g.command('reset')
        until(e,lambda:g.state=='paused',timeout=8.)
        assert attempts==[1,2,3] and len(tracking_at)==3
        assert all(start-track>=.5 for start,track in zip(starts,tracking_at))
        assert g.ball_sequence.retry_at is None
        if eventual_success:
            assert g.phase=='complete' and not g.robot.grip_closed
        else:
            assert '3/3' in g.message and not g.robot.grip_closed
            spin(e,.4);assert attempts==[1,2,3]
        g.command('reset');assert g.ball_sequence.attempts==0


def test_explicit_movement_survives_next_missing_sample_and_prevents_close():
    with graph(parameters=['grasp_blind_after_lock:=false']) as ([g],e):
        seq=lock_target(g)
        seq.waypoints=[dict(stage='grasp')];seq.index=0
        seq.accept(dict(center=[.33,0.,.1],radius=.025),g.get_clock().now().nanoseconds*1e-9)
        seq.reject(ValueError('target unavailable'),transient=True)
        g.robot.grip=lambda *a:pytest.fail('moved target must not close')
        seq.arrived()  # Can run before the next timer tick.
        assert g.phase=='grasp_retry_wait' and seq.retry_at is not None
        assert not seq.running and not seq.closing_started


def test_retry_waits_for_late_goal_acceptance_terminal_result_and_new_dwell():
    with graph() as ([g],e):
        seq=lock_target(g);cancels=[];terminal=Future()
        handle=SimpleNamespace(accepted=True,get_result_async=lambda:terminal,cancel_goal_async=lambda:cancels.append(True))
        accepted=Future();accepted.set_result(handle)
        seq.submitting=True;g.robot.sequence_motion_busy=True
        seq.fail('球移动',retryable=True);seq.retry_at=time.monotonic()-1
        seq.tick();assert g.phase=='grasp_retry_wait' and not g.robot.servo
        seq.accepted(accepted)
        seq.tick();assert cancels and seq.busy and not g.robot.servo
        terminal.set_result(SimpleNamespace(status=5,result=SimpleNamespace(error_code=0)))
        seq.tick();until(e,lambda:g.phase=='tracking')
        assert seq.attempts==1 and seq.target is None and not seq.stable
        assert not seq.running and seq.frozen is None


def test_stop_cancels_pending_retry_and_never_restarts():
    with graph() as ([g],e):
        seq=lock_target(g);seq.fail('replan',retryable=True)
        assert seq.retry_at is not None
        g.command('stop');spin(e,.4)
        assert g.state=='paused' and seq.retry_at is None and not seq.busy and not g.robot.servo


def test_long_loss_is_bounded_by_attempt_deadline_and_does_not_reuse_old_target():
    with graph() as ([g],e):
        seq=lock_target(g)
        seq.reject(ValueError('target unavailable'),transient=True)
        seq.attempt_deadline=time.monotonic()-1;seq.tick()
        assert g.phase=='grasp_retry_wait' and seq.frozen is None and seq.target is None
        seq.retry_at=time.monotonic()-1;seq.tick();until(e,lambda:g.phase=='tracking')
        seq.stable=True;seq.begin()
        assert seq.attempts==1 and not seq.running


def test_hardware_fault_and_closing_failure_do_not_auto_retry_or_open_gripper():
    with graph() as ([g],e):
        seq=lock_target(g);seq.reject(ValueError('target unavailable'),transient=True)
        g.robot.mock=False;g.robot.fresh=lambda:False
        seq.tick()
        assert g.state=='paused' and seq.retry_at is None
        assert 'feedback stale' in g.message
        g.robot.mock=True;g.robot.fresh=lambda:True
        seq.reset_attempts();seq=lock_target(g)
        seq.closing_started=True;g.robot.grip_closed=True
        seq.fail('timeout',retryable=True)
        assert g.state=='paused' and seq.retry_at is None and g.robot.grip_closed


def test_real_scene_branch_plans_from_frozen_center_and_radius_after_loss(monkeypatch):
    from geometry_msgs.msg import TransformStamped
    with graph() as ([g],e):
        seq=lock_target(g);seq.reject(ValueError('target unavailable'),transient=True)
        g.robot.mock=False;g.robot.fresh=lambda:True
        seq.feedback=([0.]*6,0.);seq.feedback_at=time.monotonic()
        seq.phase='scene';seq.future=Future()
        obj=table_box([0.,0.,0.],[.1,-.6],[.4,-.1],seq.table_id)
        seq.future.set_result(SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(collision_objects=[obj]))))
        tr=TransformStamped();tr.transform.translation.x=.3;tr.transform.rotation.w=1.
        monkeypatch.setattr(seq.buffer,'lookup_transform',lambda *a:tr)
        plans=[]
        monkeypatch.setattr(seq,'request_plan',lambda waypoints,phase:plans.append((waypoints,phase)))
        seq.tick()
        assert seq.running and len(plans)==1 and plans[0][1]=='full_plan'
        assert plans[0][0][2]['position']==[.3,0.,.1]
        assert seq.target is None and seq.frozen_radius==.025


def test_callback_from_cancelled_attempt_cannot_close_or_advance_during_retry():
    with graph() as ([g],e):
        seq=lock_target(g);seq.waypoints=[dict(stage='grasp')];seq.index=0
        old_callback=g.callback(seq.arrived)
        seq.fail('replan',retryable=True)
        g.robot.grip=lambda *a:pytest.fail('late callback must not close')
        old_callback(True,'late success');seq.arrived()
        assert g.phase=='grasp_retry_wait' and not seq.closing_started
