"""World-space target velocity, projected onto the two tracking joints.

Camera motion is removed by the timestamped base_link transforms used for RGB-D
localization. Image velocity alone must not be fed forward as target motion.
"""
from collections import deque
import math

import numpy as np


def vector(value):
    value=np.asarray(value,dtype=float)
    if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('invalid velocity geometry')
    return value


def joint_velocity(center,camera,origins,axes,velocity):
    """Cancel transverse target motion, including the camera's lever arm.

    For a revolute joint, the apparent target velocity is
    v_target - axis x (target - joint_origin) * q_dot. Project out motion along
    the sight line, then solve a damped two-axis least-squares problem.
    """
    center,camera,velocity=map(vector,(center,camera,velocity))
    sight=center-camera;distance=float(np.linalg.norm(sight))
    if not .08<=distance<=2.:raise ValueError('target distance outside feedforward range')
    if len(origins)!=2 or len(axes)!=2:raise ValueError('expected J4/J5 geometry')
    sight/=distance;transverse=np.eye(3)-np.outer(sight,sight)
    columns=[]
    for origin,axis in zip(origins,axes):
        origin,axis=vector(origin),vector(axis)
        if not .99<=np.linalg.norm(axis)<=1.01:raise ValueError('invalid joint axis')
        columns.append(transverse@np.cross(axis,center-origin))
    jacobian=np.column_stack(columns)
    u,s,vh=np.linalg.svd(jacobian,full_matrices=False)
    if s[-1]<.025 or s[0]/s[-1]>20.:raise ValueError('singular tracking geometry')
    return vh.T@((s/(s*s+.01**2))*(u.T@(transverse@velocity)))


class BallVelocity:
    """Short robust fit in metres/second; bounded, expiring J4/J5 feedforward."""
    def __init__(self,enabled=True,gain=.85,max_speed=.6):
        if not isinstance(enabled,bool):raise ValueError('invalid target feedforward switch')
        if not math.isfinite(gain) or not 0<=gain<=1.:raise ValueError('invalid feedforward gain')
        if not math.isfinite(max_speed) or not .01<=max_speed<=.8:raise ValueError('invalid feedforward speed')
        self.enabled,self.gain,self.max_speed=enabled,gain,max_speed
        self.reset()

    def reset(self,reason='waiting for world velocity'):
        self.samples=deque(maxlen=6);self.stamp=None
        self.invalidate(reason)

    def invalidate(self,reason):
        # Keep a short fit history across a missed frame, but send no stale FF.
        self.world=np.zeros(3);self.joints=np.zeros(2);self.reason=reason

    def update(self,stamp,center,camera,origins,axes):
        if not self.enabled:self.reset('disabled');return
        try:
            center=vector(center)
            if not math.isfinite(stamp):raise ValueError('invalid capture stamp')
            if self.stamp is not None and stamp<=self.stamp:return
            if self.stamp is not None and stamp-self.stamp>.15:self.reset('capture gap')
            if self.samples:
                dt=stamp-self.samples[-1][0]
                if dt<.01:return
                if np.linalg.norm(center-self.samples[-1][1])>1.5*dt+.02:
                    self.reset('world position jump')
            self.stamp=stamp;self.samples.append((stamp,center.copy()))
            while stamp-self.samples[0][0]>.18:self.samples.popleft()
            self.invalidate('warming up')
            if len(self.samples)<3 or stamp-self.samples[0][0]<.045:return
            times=np.array([t-stamp for t,p in self.samples]);points=np.array([p for t,p in self.samples])
            # Median pairwise slopes reject a single bad sphere fit better than
            # differentiation of adjacent frames, without a long smoothing lag.
            slopes=[(points[j]-points[i])/(times[j]-times[i])
                    for j in range(len(times)) for i in range(j) if times[j]-times[i]>=.025]
            velocity=np.median(slopes,axis=0);speed=float(np.linalg.norm(velocity))
            intercept=np.median(points-times[:,None]*velocity,axis=0)
            residual=np.linalg.norm(points-(intercept+times[:,None]*velocity),axis=1)
            recent=(points[-1]-points[-3])/(times[-1]-times[-3])
            if speed>.05 and np.linalg.norm(recent)>.05 and np.dot(recent,velocity)<0:
                latest=list(self.samples)[-2:];self.reset('direction changed');self.samples.extend(latest);self.stamp=stamp
                return
            if speed<.05 or speed*(times[-1]-times[0])<.006:
                self.invalidate('stationary');return
            if speed>1.5 or float(np.max(residual))>.012:
                self.invalidate('noisy world velocity');return
            # A recent stop must not coast on the older part of the fit window.
            if np.linalg.norm(recent)<.025:self.invalidate('stationary');return
            qdot=joint_velocity(center,camera,origins,axes,velocity)*self.gain
            qdot*=min(1.,self.max_speed/max(1e-12,float(np.max(np.abs(qdot)))))
            self.world,self.joints,self.reason=velocity,qdot,'tracking'
        except (ValueError,TypeError,np.linalg.LinAlgError):
            self.reset('invalid velocity or geometry')

    def output(self,now):
        age=None if self.stamp is None else now-self.stamp
        if not self.enabled or age is None or not math.isfinite(age) or not 0<=age<=.15:return [0.,0.]
        return self.joints.tolist()

    def status(self,now):
        age=None if self.stamp is None else now-self.stamp
        fresh=age is not None and math.isfinite(age) and 0<=age<=.15
        return dict(enabled=self.enabled,active=fresh and self.reason=='tracking',
            reason=self.reason if fresh or self.stamp is None else 'stale world velocity',
            age_sec=round(age,3) if age is not None and math.isfinite(age) else None,
            world_mps=self.world.tolist() if fresh else [0.,0.,0.],
            joint_radps=self.output(now),gain=self.gain,max_speed=self.max_speed)
