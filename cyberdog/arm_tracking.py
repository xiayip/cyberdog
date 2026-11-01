"""Calibrated, redundant image servo for Zephyr J1..J5; J6 keeps its roll."""
import math
import numpy as np

from .ball_velocity import vector


def image_jacobian(center,camera,orientation,origins,axes,image_scale):
    center,camera=vector(center),vector(camera)
    orientation=np.asarray(orientation,dtype=float)
    if (orientation.shape!=(3,3) or not np.isfinite(orientation).all()
            or not np.allclose(orientation.T@orientation,np.eye(3),atol=1e-5)
            or np.linalg.det(orientation)<.99):raise ValueError('invalid camera orientation')
    local=orientation.T@(center-camera);x,y,z=local
    scale=np.asarray(image_scale,dtype=float)
    if scale.shape!=(2,) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('invalid camera calibration')
    if not .08<=z<=2.:raise ValueError('target depth outside coordinated tracking range')
    projection=np.diag(scale)@np.array([[1/z,0.,-x/z**2],[0.,1/z,-y/z**2]])@orientation.T
    if len(origins)!=5 or len(axes)!=5:raise ValueError('expected J1..J5 geometry')
    columns=[]
    for origin,axis in zip(origins,axes):
        origin,axis=vector(origin),vector(axis)
        if not .99<=np.linalg.norm(axis)<=1.01:raise ValueError('invalid joint axis')
        columns.append(-projection@np.cross(axis,center-origin))
    jacobian=np.column_stack(columns)
    if np.linalg.svd(jacobian,compute_uv=False)[-1]<.03:raise ValueError('singular tracking geometry')
    return jacobian,projection


def distribute(jacobian,desired,position,anchor,lower,upper,speed,posture_gain=.15):
    """Weighted damped solve, reassigning work when a joint hits a bound.

    The soft posture preference acts in the redundant directions: it prevents
    slow shoulder drift while the main task keeps the ball centred.
    """
    q,anchor,lo,hi,vmax=[np.asarray(x,dtype=float)[:5] for x in (position,anchor,lower,upper,speed)]
    if any(x.shape!=(5,) or not np.isfinite(x).all() for x in (q,anchor,lo,hi,vmax)):
        raise ValueError('invalid coordinated joint state')
    if np.any(vmax<=0) or np.any(lo>hi):raise ValueError('invalid coordinated bounds')
    # Gradually surrender the task to other joints near an anchor/hard limit.
    vlow=-np.minimum(vmax,4*np.maximum(0.,q-lo))
    vhigh=np.minimum(vmax,4*np.maximum(0.,hi-q))
    weights=np.array([1.2,.8,.8,.6,.6])**2
    preference=posture_gain*(anchor-q)
    result=np.zeros(5);free=list(range(5));fixed=[]
    for _ in range(6):
        if not free:break
        j=jacobian[:,free];w=weights[free];preferred=preference[free]
        residual=desired-(jacobian[:,fixed]@result[fixed] if fixed else 0.)
        candidate=preferred+(w[:,None]*j.T)@np.linalg.solve(
            (j*w)@j.T+.05**2*np.eye(2),residual-j@preferred)
        result[free]=candidate
        excess=[i for i in free if result[i]<vlow[i] or result[i]>vhigh[i]]
        if not excess:break
        for i in excess:
            result[i]=np.clip(result[i],vlow[i],vhigh[i]);free.remove(i);fixed.append(i)
    return result


class ArmTracking:
    def __init__(self,enabled=False):
        self.enabled=enabled;self.reset()

    def reset(self,reason='waiting for calibrated geometry'):
        self.stamp=None;self.jacobian=None;self.projection=None
        self.reason=reason;self.rates=[0.]*6;self.active=False

    def update(self,stamp,center,camera,orientation,origins,axes,image_scale):
        if not self.enabled:return
        try:
            if not math.isfinite(stamp):raise ValueError('invalid geometry stamp')
            if self.stamp is not None and stamp<=self.stamp:return
            self.jacobian,self.projection=image_jacobian(center,camera,orientation,origins,axes,image_scale)
            self.stamp=stamp;self.reason='geometry ready'
        except (ValueError,TypeError,np.linalg.LinAlgError):self.reset('invalid coordinated geometry')

    def target(self,now,error,reference,anchor,lower,upper,speed,dt,gain,world_velocity,ff_gain,ff_speed):
        self.active=False;self.rates=[0.]*6
        if not self.enabled:self.reason='disabled';return None
        if self.stamp is None or not 0<=now-self.stamp<=.15:
            self.reason='coordinated geometry unavailable or stale';return None
        try:
            e=np.asarray(error,dtype=float)
            if e.shape!=(2,) or not np.isfinite(e).all() or not 0<dt<=.05:raise ValueError('invalid image servo input')
            e=np.where(np.abs(e)<=.08,0.,e)
            # Stop completely when centred and the target is stationary; do not
            # move the shoulders solely to improve the redundant posture.
            moving=ff_gain>0 and np.linalg.norm(world_velocity)>0
            if not np.any(e) and not moving:
                self.active=True;self.reason='centred';return list(reference)
            # The existing gain is rad/s per normalized image error, not an
            # image-space gain. Match its local wrist sensitivity before
            # redistributing the task, otherwise extra joints slow feedback.
            sensitivity=np.clip(np.linalg.norm(self.jacobian[:,3:5],axis=1),.75,3.)
            positional=distribute(self.jacobian,-gain*sensitivity*e,reference,anchor,lower,upper,speed)
            feedforward=distribute(self.jacobian,-ff_gain*(self.projection@vector(world_velocity)),
                reference,anchor,lower,upper,np.minimum(speed,ff_speed),posture_gain=0.)
            rate=positional+feedforward
            rate/=max(1.,float(np.max(np.abs(rate)/np.asarray(speed)[:5])))
            target=list(reference)
            for i in range(5):target[i]=float(np.clip(reference[i]+rate[i]*dt,lower[i],upper[i]))
            self.rates=[(target[i]-reference[i])/dt for i in range(6)]
            self.active=True;self.reason='tracking with J1..J5'
            return target
        except (ValueError,TypeError,np.linalg.LinAlgError):
            self.reset('coordinated solve unavailable');return None

    def status(self,now):
        fresh=self.stamp is not None and math.isfinite(now) and 0<=now-self.stamp<=.15
        return dict(enabled=self.enabled,active=self.active and fresh,
            reason=self.reason if fresh or self.stamp is None else 'stale coordinated geometry',
            joint_radps=self.rates if fresh else [0.]*6)
