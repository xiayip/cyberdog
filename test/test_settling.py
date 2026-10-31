import math
import pytest
from cyberdog.settling import SettlingWindow


def sample(window, t, q=0., v=0., stamp=None):
    return window.update(t,[q]*6,[v]*6,1+t if stamp is None else stamp)


def test_noise_is_not_motion_but_warmup_is_required():
    w=SettlingWindow()
    for i in range(45):assert not sample(w,i*.01,.0002*(i%2),.2*(-1)**i)
    assert sample(w,.45,.0002,-.2)
    assert sample(w,.46,0.,.2)


@pytest.mark.parametrize('signal', ['drift','oscillation','velocity'])
def test_real_motion_or_inconsistent_velocity_cannot_average_to_rest(signal):
    w=SettlingWindow()
    for i in range(100):
        t=i*.01
        q=.02*t if signal=='drift' else .004*math.sin(40*t) if signal=='oscillation' else 0.
        v=.04 if signal=='velocity' else 0.
        assert not sample(w,t,q,v)


@pytest.mark.parametrize('fault', ['gap','duplicate','backward','nan','spike'])
def test_bad_feedback_restarts_evidence_window(fault):
    w=SettlingWindow()
    for i in range(60):sample(w,i*.01)
    assert sample(w,.6)
    if fault=='gap':assert not sample(w,.9)
    elif fault=='duplicate':assert not sample(w,.61,stamp=1.6)
    elif fault=='backward':assert not sample(w,.61,stamp=1.5)
    elif fault=='nan':assert not sample(w,.61,float('nan'))
    else:assert not sample(w,.61,v=.6)


def test_sparse_samples_do_not_prove_stationary():
    w=SettlingWindow()
    for i in range(20):assert not sample(w,i*.2)


def test_position_only_samples_keep_displacement_guard():
    w=SettlingWindow()
    for i in range(60):quiet=w.update(i*.01,[0.]*6,None)
    assert quiet
    assert not w.update(.61,[.01]*6,None)


def test_position_only_controller_feedback_is_explicitly_scoped(monkeypatch):
    from types import SimpleNamespace
    from cyberdog.controller_modes import ControllerModes
    from control_msgs.msg import JointTrajectoryControllerState
    from cyberdog.motion import JOINTS
    modes=ControllerModes.__new__(ControllerModes)
    modes.pos='zephyr_arm_trajectory_controller'
    imp='zephyr_arm_impedance_trajectory_controller'
    modes.settling={k:(SettlingWindow(),SettlingWindow()) for k in (modes.pos,imp)}
    modes.telemetry={}
    modes.node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=0)))
    msg=JointTrajectoryControllerState(joint_names=JOINTS)
    msg.reference.positions=msg.feedback.positions=[0.]*6
    msg.reference.velocities=[0.]*6
    for i in range(60):
        monkeypatch.setattr('cyberdog.controller_modes.time.monotonic',lambda:i*.01)
        modes.on_state(modes.pos,msg)
        modes.on_state(imp,msg)
    assert modes.telemetry[modes.pos][1]
    assert not modes.telemetry[imp][1]
