"""Boundary feedback regressions; ROS integration uses fake controllers only."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from cyberdog.behavior import Cyberdog
from cyberdog.game_core import inspection_poses, path_points, pose
from cyberdog.motion import LOWER, UPPER, START_POSITION_TOLERANCE, Motion, bounded_start
from fake_hardware import FakeGameRobot, executor
from test_behavior import graph, until
from test_hardware_adapter import PARAMS


@pytest.mark.parametrize('joint', range(6))
@pytest.mark.parametrize('side', [-1, 1])
def test_only_small_boundary_errors_are_projected_without_altering_feedback(joint, side):
    q=[0., .6, .8, 0., 0., 0.]
    limit=(LOWER if side<0 else UPPER)[joint]
    for error in (.000383, START_POSITION_TOLERANCE):
        q[joint]=limit+side*error
        result=bounded_start(q)
        assert result[joint]==limit
        assert q[joint]==limit+side*error
        assert all(result[i]==q[i] for i in range(6) if i!=joint)
    q[joint]=limit+side*(START_POSITION_TOLERANCE+.00001)
    with pytest.raises(ValueError, match=f'joint{joint+1} measured position'):
        bounded_start(q)


@pytest.mark.parametrize('value', [None, [], [0.]*5, [0.,float('nan'),0.,0.,0.,0.],
                                 [0.,float('inf'),0.,0.,0.,0.], [0.,True,0.,0.,0.,0.]])
def test_invalid_start_never_becomes_a_command(value):
    with pytest.raises(ValueError):bounded_start(value)


def test_inspection_absorbs_zero_noise_but_targets_keep_strict_limits():
    start=[-1.5707,-.000383,.06,0.,0.,0.]
    wrist,heads=inspection_poses(start,.25,[.25,-.25])
    assert wrist==[-1.5707,0.,.06,.25,0.,0.]
    assert [q[5] for q in heads]==[.25,-.25]
    assert all(q[1]==0. for q in heads)
    assert start[1]==-.000383
    # The same small negative value is forbidden as an explicit target.
    with pytest.raises(ValueError):pose(start)
    with pytest.raises(ValueError):path_points(wrist,[start])
    with pytest.raises(ValueError):inspection_poses([0.,0.,0.,1.32,0.,0.],.250383,[.25])


def preflight(measured, reference, factory=None):
    """Run the actual async submission callback with a newly arrived sample."""
    m=Motion.__new__(Motion)
    sent=[];outcomes=[]
    m.busy=True;m.cancel_requested=False;m.phase='preflight'
    m.required=['position_jtc'];m.backend='trajectory'
    m.ready=lambda:''
    m.pose=measured;m.reference_pose=reference
    m.path_factory=factory or (lambda anchor:path_points(anchor,[[-1.5707,0.,.06,.25,0.,0.]]))
    m.limit_scale=1.;m.timeout=12.
    m.report=lambda ok,reason:outcomes.append((ok,reason))
    def send(goal):
        sent.append(goal)
        return Future()
    m.traj=SimpleNamespace(server_is_ready=lambda:True,send_goal_async=send)
    future=Future()
    future.set_result(SimpleNamespace(controller=[SimpleNamespace(name='position_jtc',state='active')]))
    m.preflight_done(future)
    return m,sent,outcomes


@pytest.mark.parametrize('reference_j2', [0., -.000383])
def test_submission_rechecks_new_zero_noise_and_sends_only_bounded_points(reference_j2):
    measured=[-1.5707,-.000383,.06,0.,0.,0.]
    reference=[-1.5707,reference_j2,.06,0.,0.,0.]
    m,sent,outcomes=preflight(measured,reference)
    assert len(sent)==1 and m.phase=='sending' and not outcomes
    assert sent[0].trajectory.points[0].positions[1]==0.
    assert measured[1]==-.000383 and reference[1]==reference_j2
    for point in sent[0].trajectory.points:
        assert all(lo<=q<=hi for q,lo,hi in zip(point.positions,LOWER,UPPER))


@pytest.mark.parametrize('source', ['measured','reference','target'])
def test_submission_still_rejects_real_limit_violations(source):
    measured=[-1.5707,0.,.06,0.,0.,0.];reference=list(measured)
    factory=None
    if source=='measured':measured[1]=-.003
    elif source=='reference':reference[1]=-.003
    else:
        target=list(measured);target[1]=-.000383
        factory=lambda anchor:[(.2,anchor),(1.,target)]
    m,sent,outcomes=preflight(measured,reference,factory)
    assert not sent and not m.busy and len(outcomes)==1 and not outcomes[0][0]


def test_ros_zero_feedback_noise_completes_wrist_head_and_damping_sequence():
    with graph((Cyberdog,FakeGameRobot),PARAMS,executor) as ([g,r],e):
        r.measured_offset[1]=-.000383
        assert g.command('reset')[0]
        until(e,lambda:g.state=='idle',timeout=8.)
        assert r.position==g.home
        assert g.robot.current[1]==pytest.approx(-.000383)
        before=len(r.requests)
        r.position[2]=.06
        until(e,lambda:g.phase=='handshake',timeout=14.)
        goals=r.requests[before:]
        assert len(goals)==4
        targets=[list(goal.trajectory.points[-1].positions) for goal in goals]
        assert targets[0]==[-1.5707,0.,.06,.25,0.,0.]
        assert [q[5] for q in targets[:3]]==[0.,.25,-.25]
        assert targets[-1]==g.lift and g.robot.modes.mode=='damping'
        assert not r.impedance_goals and not r.direct_commands
        for goal in r.requests:
            for point in goal.trajectory.points:
                assert all(lo<=q<=hi for q,lo,hi in zip(point.positions,LOWER,UPPER))


def test_robot_move_never_accepts_out_of_limit_targets_even_with_valid_noisy_start():
    with graph() as ([g],e):
        g.robot.mock_pose=[-1.5707,-.000383,.06,0.,0.,0.]
        assert not g.robot.move([g.robot.current],lambda *args:None)[0]
        assert not g.robot.busy
