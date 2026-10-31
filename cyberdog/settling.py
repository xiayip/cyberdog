"""Stationary evidence from a recent position/velocity window, not one noisy sample."""
from collections import deque
import math


class SettlingWindow:
    def __init__(self):
        self.samples = deque()
        self.last_stamp = None

    def reset(self):
        self.samples.clear()
        self.last_stamp = None

    def update(self, now, positions, velocities, stamp=0.):
        # Position-only JTC telemetry is supported explicitly; it does not report
        # a measured velocity. Callers must also require fresh joint-state evidence.
        values = [] if velocities is None else velocities
        if (len(positions) != 6 or (velocities is not None and len(velocities) != 6)
                or not all(math.isfinite(v) for v in [now, stamp, *positions, *values])
                or any(abs(v) > .5 for v in values)):
            self.reset()
            return False
        if stamp > 0 and self.last_stamp is not None and stamp <= self.last_stamp:
            self.reset()
            return False
        if self.samples and (now <= self.samples[-1][0] or now-self.samples[-1][0] > .15):
            self.reset()
        if self.samples and ((self.samples[-1][2] is None) != (velocities is None)):
            self.reset()
        self.last_stamp = stamp if stamp > 0 else None
        self.samples.append((now, tuple(positions), None if velocities is None else tuple(velocities)))
        while len(self.samples) > 1 and self.samples[1][0] < now-.5:
            self.samples.popleft()
        if len(self.samples) < 5 or now-self.samples[0][0] < .45:
            return False
        rows = list(self.samples)
        duration = rows[-1][0]-rows[0][0]
        for axis in range(6):
            q = [r[1][axis] for r in rows]
            # A drifting or oscillating position cannot be hidden by averaging velocity.
            if max(q)-min(q) > .003:
                return False
            if abs(q[-1]-q[0])/duration > .01:
                return False
            if velocities is None:
                continue
            integral = sum((b[0]-a[0])*(a[2][axis]+b[2][axis])*.5
                           for a,b in zip(rows,rows[1:]))
            if abs(integral/duration) >= .03:
                return False
        return True
