"""Deterministic gameplay helpers; no ROS or hardware side effects."""
import math
from ghost_game_orchestrator.mock_trajectory import minimum_quintic_duration
from .common import finite
from .motion import JOINTS, LOWER, UPPER, MAX_PATH_SPEED, MAX_INSPECTION_SPEED, bounded_start, validate_bow_path


def pose(value):
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError('pose must contain six joint positions')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           or not lo <= x <= hi for x, lo, hi in zip(value, LOWER, UPPER)):
        raise ValueError('pose outside Rebot limits')
    return [float(x) for x in value]


def inspection_poses(start,wrist_delta,head_offsets):
    """Lift only J4, then tilt only J6 relative to the captured starting pose."""
    start=bounded_start(start)
    wrist=list(start);wrist[3]+=wrist_delta;wrist=pose(wrist)
    heads=[]
    for offset in head_offsets:
        head=list(wrist);head[5]=start[5]+offset;heads.append(pose(head))
    return wrist,heads


def path_points(start, targets, speed=.12, minimum=1., acceleration=None, *, time_scale=1., velocity_limits=None):
    start = pose(start)
    time_scale=finite(time_scale,.5,1.,'path time scale')
    speed_limits=[MAX_PATH_SPEED]*6
    if velocity_limits is not None:
        caps=[MAX_PATH_SPEED]*6;caps[3]=caps[5]=MAX_INSPECTION_SPEED
        if not isinstance(velocity_limits,(list,tuple)) or len(velocity_limits)!=6:
            raise ValueError('invalid joint velocity limits')
        speed_limits=[finite(v,.01,cap,'joint velocity limit') for v,cap in zip(velocity_limits,caps)]
    if not targets or not 0 < speed <= max(speed_limits) or not math.isfinite(minimum) or minimum < .2:
        raise ValueError('invalid path settings')
    speed_limits=[min(speed,limit) for limit in speed_limits]
    if acceleration is not None:
        limits=list(acceleration) if isinstance(acceleration,(list,tuple)) else [acceleration]*6
        if len(limits)!=6 or any(isinstance(a,bool) or not isinstance(a,(int,float))
                                or not math.isfinite(a) or a<=0 for a in limits):
            raise ValueError('invalid acceleration limit')
    points, elapsed, previous = [(.2, start)], .2, start
    for target in targets:
        target = pose(target)
        duration=minimum_quintic_duration(previous, target, speed_limits, minimum)
        if acceleration is not None:
            # Zero endpoint velocity/acceleration gives a quintic segment whose
            # peak acceleration is (10/sqrt(3))*distance/duration**2.
            duration=max(duration,*(math.sqrt((10/math.sqrt(3))*abs(a-b)/limit)
                                   for a,b,limit in zip(previous,target,limits)))
        elapsed += duration
        points.append((elapsed, target))
        previous = target
    if elapsed > 60.:
        raise ValueError('path longer than 60 seconds; split into separate stages')
    # Time scaling the zero-derivative endpoints also scales the JTC's entire
    # quintic segment: peak velocity / scale, peak acceleration / scale**2.
    return [(at*time_scale, target) for at,target in points]


def bow_points(start,target,speed,acceleration,speed_scale=3.):
    """Speed up a J4-only bow by scaling its complete quintic time axis."""
    speed_scale=finite(speed_scale,1.,3.,'bow speed scale')
    points=path_points(start,[target],speed,.65,acceleration)
    points=[(at/speed_scale,q) for at,q in points]
    validate_bow_path(points,start)
    return points
