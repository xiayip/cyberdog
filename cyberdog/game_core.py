"""Deterministic gameplay helpers; no ROS or hardware side effects."""
import math
from ghost_game_orchestrator.mock_trajectory import minimum_quintic_duration
from .motion import JOINTS, LOWER, UPPER, bounded_start


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


def path_points(start, targets, speed=.12, minimum=1., acceleration=None):
    start = pose(start)
    if not targets or not 0 < speed <= .6 or not math.isfinite(minimum) or minimum < .2:
        raise ValueError('invalid path settings')
    if acceleration is not None and (not math.isfinite(acceleration) or acceleration<=0):
        raise ValueError('invalid acceleration limit')
    points, elapsed, previous = [(.2, start)], .2, start
    for target in targets:
        target = pose(target)
        duration=minimum_quintic_duration(previous, target, [speed]*6, minimum)
        if acceleration is not None:
            # Zero endpoint velocity/acceleration gives a quintic segment whose
            # peak acceleration is (10/sqrt(3))*distance/duration**2.
            distance=max(abs(a-b) for a,b in zip(previous,target))
            duration=max(duration,math.sqrt((10/math.sqrt(3))*distance/acceleration))
        elapsed += duration
        points.append((elapsed, target))
        previous = target
    if elapsed > 60.:
        raise ValueError('path longer than 60 seconds; split into separate stages')
    return points
