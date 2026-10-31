"""J3 touch intent and six-joint movement after entering damping."""
import math


class TouchDetector:
    def __init__(self,signal='either',position=.03,torque=1.,dwell=.25):
        if signal not in ('position','torque','either'):raise ValueError('invalid touch_signal')
        self.signal,self.position,self.torque,self.dwell=signal,position,torque,dwell
        self.reset()

    def reset(self):
        self.baseline=None;self.learn_since=None;self.over_since=None;self.last=None
        self.position_error=0.;self.torque_error=0.

    def update(self,now,q,effort,target,stationary):
        if self.last is not None and now-self.last>.2:self.reset()
        self.last=now
        if q is None or not math.isfinite(q):self.reset();return False
        self.position_error=abs(q-target)
        valid_effort=effort is not None and math.isfinite(effort)
        if not valid_effort:self.baseline=None;self.learn_since=None
        elif self.baseline is None:
            if stationary and self.position_error<self.position/2:
                if self.learn_since is None:self.learn_since=now
                elif now-self.learn_since>=1.:self.baseline=effort
            else:self.learn_since=None
        self.torque_error=abs(effort-self.baseline) if valid_effort and self.baseline is not None else 0.
        over=((self.signal in ('position','either') and self.position_error>=self.position) or
              (self.signal in ('torque','either') and self.torque_error>=self.torque))
        if not over:self.over_since=None;return False
        if self.over_since is None:self.over_since=now
        return now-self.over_since>=self.dwell

    def status(self):
        return dict(signal=self.signal,j3_position_error=self.position_error,
                    j3_torque_error=self.torque_error,torque_baseline=self.baseline)


class DampingMovementDetector:
    """One trigger per damping session, after handover settling, from six joints."""
    def __init__(self,threshold=.04,dwell=.15,settle=.5):
        self.threshold,self.dwell,self.settle=threshold,dwell,settle
        self.reset()

    def reset(self):
        self.anchor=None;self.last=None;self.started=None;self.over_since=None
        self.armed=False;self.triggered=False;self.displacement=0.;self.joint=0

    def update(self,now,q):
        if q is None or len(q)!=6 or not all(math.isfinite(v) for v in q) or not math.isfinite(now):
            self.reset();return False
        if self.last is not None and (now<=self.last or now-self.last>.2):self.reset()
        self.last=now
        if self.started is None:self.started=now;self.anchor=list(q)
        if now-self.started<self.settle:
            self.anchor=list(q);return False
        self.armed=True
        differences=[abs(a-b) for a,b in zip(q,self.anchor)]
        self.displacement=max(differences);self.joint=differences.index(self.displacement)+1
        if self.triggered:return False
        if self.displacement<self.threshold:self.over_since=None;return False
        if self.over_since is None:self.over_since=now
        if now-self.over_since<self.dwell:return False
        self.triggered=True;return True

    def status(self):
        return dict(armed=self.armed,triggered=self.triggered,joint=self.joint,
                    displacement=self.displacement,threshold=self.threshold)
