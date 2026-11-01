"""Threefold time scaling applies only to a bounded downward J4 quintic."""
import math

import pytest

from cyberdog.game_core import bow_points,path_points
from cyberdog.motion import MAX_BOW_SPEED,MAX_BOW_ACCELERATION
from test_behavior import graph
from test_joint_boundary import preflight


@pytest.mark.parametrize('distance',[.005,.05,.25,.45,.5,.75])
def test_threefold_bow_keeps_geometry_and_checks_interior_quintic_extrema(distance):
    start=[.2,.6,.8,.3,-.1,0.];target=start[:];target[3]-=distance
    previous=path_points(start,[target],.6,.65,[1.3]*6)
    faster=bow_points(start,target,.6,[1.3]*6,3.)
    assert [q for _,q in faster]==[q for _,q in previous]
    assert [t for t,_ in faster]==pytest.approx([t/3 for t,_ in previous])
    duration=faster[-1][0]-faster[0][0]
    assert 1.875*distance/duration<=1.8+1e-9
    assert 10/math.sqrt(3)*distance/duration**2<=11.7+1e-9
    assert MAX_BOW_SPEED<2.5 and MAX_BOW_ACCELERATION<20.
    m,sent,_=preflight(start,start,lambda _:faster,bow=True)
    assert sent and m.path_bow
    m.finish(True,'done');assert not m.path_bow


@pytest.mark.parametrize('change',['other_joint','up','excess_distance','excess_speed','excess_acceleration','derivatives'])
def test_submission_rejects_inappropriate_fast_bow_even_with_zero_endpoint_derivatives(change):
    start=[0.,.6,.8,.3,0.,0.];target=start[:];target[3]-=.45
    points=bow_points(start,target,.6,[1.3]*6)
    if change=='other_joint':target[0]+=.01
    elif change=='up':target[3]=start[3]+.45
    elif change=='excess_distance':target[3]=start[3]-.76
    elif change=='excess_speed':points=[(.05,start),(.15,target)]
    elif change=='excess_acceleration':points=[(.05,start),(.5,target)]
    elif change=='derivatives':points=[(t,q,[0.]*6,[0.]*6) for t,q in points]
    if change in ('other_joint','up','excess_distance'):points=[(points[0][0],start),(points[1][0],target)]
    if change=='excess_distance':
        # Keep speed/acceleration well below their envelopes so only the new
        # amplitude bound can reject this segment.
        points=[(.2,start),(2.,target)]
    m,sent,outcomes=preflight(start,start,lambda _:points,bow=True)
    assert not sent and not m.busy and not m.path_bow
    assert outcomes and not outcomes[0][0]


@pytest.mark.parametrize('scale',[.9,3.01,float('nan'),True])
def test_invalid_speed_scale_never_starts_bow(scale):
    with graph() as ([g],e):
        target=g.robot.current;target[3]-=.45
        assert not g.robot.bow(target,lambda *args:None,speed=.4,speed_scale=scale)[0]
        assert not g.robot.busy and not g.robot.motion.path_bow


def test_bow_cannot_stack_profiles_or_raise_ordinary_motion_limits():
    with graph(parameters=['motion_limit_scale:=2.0']) as ([g],e):
        r=g.robot;target=r.current;target[3]-=.45
        assert not r.motion.start_path(lambda _:[],bow=True,inspection=True)[0]
        assert not r.motion.start_path(lambda _:[],bow=True,time_scale=.5)[0]
        assert not r.move([target],lambda *args:None,speed=1.8)[0]
        assert r.bow(target,lambda *args:None)[0] and r.motion.path_bow
        r.motion.stop();assert not r.motion.path_bow
        assert r.move([target],lambda *args:None,speed=.6)[0]
        assert not r.motion.path_bow and r.motion.path_time_scale==1.
