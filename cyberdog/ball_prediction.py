"""Bounded image-space prediction using capture timestamps, never stale frames."""
import math


class BallPrediction:
    def __init__(self,lead=.04,max_horizon=.12,max_offset=.10):
        if not 0<=lead<=.08 or not lead<=max_horizon<=.15 or not 0<max_offset<=.15:
            raise ValueError('invalid prediction bounds')
        self.lead,self.max_horizon,self.max_offset=lead,max_horizon,max_offset
        self.reset()

    def reset(self):
        self.stamp=None;self.point=None;self.velocity=[0.,0.];self.frames=0

    def update(self,stamp,x,y,score):
        if not all(math.isfinite(v) for v in (stamp,x,y,score)) or not -1<=x<=1 or not -1<=y<=1:
            self.reset();return False
        if self.stamp is not None and stamp<=self.stamp:return False
        point=[x,y]
        if self.stamp is None or stamp-self.stamp>.15 or score<.75:
            self.velocity=[0.,0.];self.frames=1
        else:
            dt=stamp-self.stamp
            if dt<.01:return False
            observed=[(a-b)/dt for a,b in zip(point,self.point)]
            # An implausible jump or a clear reversal invalidates the old estimate.
            jump=math.hypot(x-self.point[0],y-self.point[1])>.20
            reversing=any(v*u<0 and abs(v)>.15 and abs(u)>.15 for v,u in zip(observed,self.velocity))
            if jump or reversing:
                self.velocity=[0.,0.];self.frames=1
            else:
                self.frames+=1
                self.velocity=[.55*v+.45*max(-2.5,min(2.5,u)) for v,u in zip(self.velocity,observed)]
        self.stamp,self.point=stamp,point
        return True

    def predict(self,now):
        if self.point is None:return None
        age=now-self.stamp
        # Old positions are not revived by extrapolation. The game independently
        # brakes on capture-age expiry and on invalid observations.
        if not math.isfinite(age) or age<0 or age>.12:return tuple(self.point)
        if self.frames<3:return tuple(self.point)
        horizon=min(self.max_horizon,age+self.lead)
        dx,dy=[v*horizon for v in self.velocity]
        size=math.hypot(dx,dy)
        if size>self.max_offset:dx*=self.max_offset/size;dy*=self.max_offset/size
        return tuple(max(-1.,min(1.,q+d)) for q,d in zip(self.point,(dx,dy)))
