import math
import pytest
from cyberdog.game_core import inspection_poses,path_points
from test_behavior import graph
from test_joint_boundary import preflight


def test_short_wrist_is_faster_but_other_joints_keep_their_acceleration_limits():
    start=[0.,.6,.8,0.,0.,0.]
    wrist=start[:];wrist[3]=.25
    ordinary=[.975]*6;inspection=ordinary[:];inspection[3]=inspection[5]=1.5
    duration=lambda points:points[-1][0]-points[0][0]
    previous=duration(path_points(start,[wrist],.48,1.,ordinary))
    faster=duration(path_points(start,[wrist],.6,1.,inspection))
    assert faster<previous*.83
    # A concurrent shoulder correction must still use the ordinary bound.
    wrist[1]+=.25
    constrained=duration(path_points(start,[wrist],.6,1.,inspection))
    assert constrained>faster
    assert (10/math.sqrt(3))*.25/constrained**2<=.975+1e-9
    for invalid in ([1.5]*5,[.975,.975,.975,float('nan'),.975,1.5],[True]*6):
        with pytest.raises(ValueError):path_points(start,[wrist],.6,1.,invalid)


@pytest.mark.parametrize('j3',[0.,.06,.75])
def test_lift_halves_entire_command_time_with_bounded_quintic_peaks(j3):
    start=[-1.5707,0.,j3,.4,0.,.25];target=[-1.5707,.6,.8,0.,0.,0.]
    baseline=path_points(start,[target],.48,1.,[.975]*6)
    faster=path_points(start,[target],.48,1.,[.975]*6,time_scale=.5)
    assert [q for _,q in faster]==[q for _,q in baseline]
    assert [t for t,_ in faster]==pytest.approx([t*.5 for t,_ in baseline])
    duration=faster[-1][0]-faster[0][0]
    for a,b in zip(start,target):
        assert 1.875*abs(a-b)/duration<=.96+1e-9
        assert (10/math.sqrt(3))*abs(a-b)/duration**2<=3.9+1e-9


@pytest.mark.parametrize('scale',[0.,.49,1.01,float('nan'),True])
def test_invalid_time_scale_never_starts_motion(scale):
    with graph() as ([g],e):
        r=g.robot
        assert not r.move([g.lift],lambda *args:None,time_scale=scale)[0]
        assert not r.motion.start_path(lambda _:[],time_scale=scale)[0]
        assert not r.busy and r.motion.path_time_scale==1.


def test_scaled_lift_cannot_stack_with_inspection_or_exceed_effective_speed_cap():
    with graph(parameters=['motion_limit_scale:=2.0']) as ([g],e):
        r=g.robot
        assert not r.move([g.lift],lambda *args:None,speed=.7,time_scale=.5)[0]
        assert not r.move([g.lift],lambda *args:None,inspection=True,time_scale=.5)[0]
        assert not r.motion.start_path(lambda _:[],inspection=True,time_scale=.5)[0]
        assert not r.busy and r.motion.path_time_scale==1.
        assert r.move([g.lift],lambda *args:None,speed=.6,time_scale=.5)[0]
        assert r.motion.path_time_scale==.5
        r.motion.stop()
        assert r.motion.path_time_scale==1.
        assert r.move([g.home],lambda *args:None,speed=.24)[0]
        assert r.motion.path_time_scale==1.


@pytest.mark.parametrize('scale,velocity,acceleration,accepted',[
    (.5,.7,2.,True),(1.,.7,2.,False),(.5,.81,0.,False),(.5,0.,2.61,False)])
def test_submission_checks_scaled_limits_and_restores_ordinary_limits(scale,velocity,acceleration,accepted):
    q=[0.,.6,.8,0.,0.,0.];v=[0.]*6;a=[0.]*6;v[2]=velocity;a[2]=acceleration
    factory=lambda anchor:[(.1,anchor,[0.]*6,[0.]*6),(1.5,q,v,a)]
    m,sent,outcomes=preflight(q,q,factory,time_scale=scale)
    assert bool(sent)==accepted
    if accepted:m.finish(True,'done')
    assert m.path_time_scale==1.


def test_larger_wrist_and_head_segments_finish_sooner_with_the_published_profile():
    start=[-1.5707,0.,.05,0.,0.,0.]
    old_wrist,old_heads=inspection_poses(start,.4,[.25])
    wrist,heads=inspection_poses(start,.55,[.4])
    old_acc=[1.3]*6;old_acc[3]=old_acc[5]=2.4
    acceleration=[1.3]*6;acceleration[3]=acceleration[5]=6.
    limits=[.8]*6;limits[3]=limits[5]=1.4
    for a,b,old_a,old_b in ((start,wrist,start,old_wrist),(wrist,heads[0],old_wrist,old_heads[0])):
        previous=path_points(old_a,[old_b],.8,.65,old_acc)
        points=path_points(a,[b],1.4,.45,acceleration,velocity_limits=limits)
        assert points[-1][0]<previous[-1][0]*.85
        duration=points[-1][0]-points[0][0]
        for i in range(6):
            assert 1.875*abs(b[i]-a[i])/duration<=limits[i]+1e-9
            assert 10/math.sqrt(3)*abs(b[i]-a[i])/duration**2<=acceleration[i]+1e-9


def test_fast_inspection_still_slows_for_shoulder_corrections_and_restores_normal_speed():
    with graph(parameters=['motion_limit_scale:=2.0','handshake_inspection_speed:=1.4',
                           'handshake_inspection_acceleration:=6.0']) as ([g],e):
        r=g.robot;start=r.current;target=start[:];target[1]+=.4;target[3]+=.55
        assert r.move([target],lambda *args:None,speed=1.4,inspection=True,minimum=.45)[0]
        points=r.motion.path_factory(start);duration=points[-1][0]-points[0][0]
        assert 1.875*.4/duration<=.8+1e-9
        assert 10/math.sqrt(3)*.4/duration**2<=1.3+1e-9
        assert r.motion.velocity_limits(True)==[.8,.8,.8,1.4,.8,1.4]
        r.motion.stop()
        assert not r.move([target],lambda *args:None,speed=1.4)[0]
        assert not r.move([target],lambda *args:None,speed=1.4,inspection=True,time_scale=.5)[0]
        assert r.move([target],lambda *args:None,speed=.8)[0]
        assert not r.motion.path_inspection


@pytest.mark.parametrize('joint,duration,accepted',[(3,1.,True),(5,1.,True),(1,1.,False),(3,.2,False)])
def test_inspection_submission_checks_quintic_peaks_between_stopped_endpoints(joint,duration,accepted):
    start=[0.,.6,.8,0.,0.,0.];target=start[:];target[joint]+=.25
    # Fixture uses limit_scale=1: wrist/head .7 rad/s, other joints .4 rad/s.
    m,sent,outcomes=preflight(start,start,lambda anchor:[(.2,anchor),(.2+duration,target)],inspection=True)
    assert bool(sent)==accepted
    if accepted:m.finish(True,'done')
    assert not m.path_inspection


@pytest.mark.parametrize('limits',[[1.4]*6,[.8]*5,[.8,.8,.8,1.41,.8,1.4],
                                 [.8,.8,.8,float('nan'),.8,1.4],[True]*6])
def test_custom_velocity_limits_cannot_raise_other_axes_or_accept_invalid_values(limits):
    start=[0.,.6,.8,0.,0.,0.];target=start[:];target[3]=.55
    with pytest.raises(ValueError):path_points(start,[target],1.4,.45,[1.3]*6,velocity_limits=limits)
