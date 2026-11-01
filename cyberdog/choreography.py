"""A continuous lamp-like groove: wide nods, upright wrist and side sway.

The reference is a qualitative choreography, not joint-angle reconstruction
from the single-view video. All times are retimed to the commissioned limits.
"""
import math
from dataclasses import dataclass
import numpy as np

from .common import finite
from .game_core import pose


SECTIONS=((0.,'大幅点头'),(4.,'抬起弹跳'),(8.,'抬腕竖起'),(10.,'竖腕左右摇摆'))


def ramp(b,start,end):
    if b<=start:return np.array([0.,0.,0.])
    if b>=end:return np.array([1.,0.,0.])
    u=(b-start)/(end-start);d=end-start
    return np.array([10*u**3-15*u**4+6*u**5,(30*u**2-60*u**3+30*u**4)/d,
                     (60*u-180*u**2+120*u**3)/d**2])


def multiply(a,b):
    return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])


def wave(b,frequency,phase=0.):
    angle=frequency*b+phase
    return np.array([math.sin(angle),frequency*math.cos(angle),-frequency**2*math.sin(angle)])


def groove(beat):
    """Joint offsets and their first/second derivatives with respect to beats."""
    one=np.array([1.,0.,0.])
    envelope=multiply(ramp(beat,0.,2.),one-ramp(beat,14.,16.))
    rise=multiply(ramp(beat,4.,6.),one-ramp(beat,12.,14.))
    # Finish raising J4 before the second eight-count's lateral movement.
    sway=ramp(beat,10.,12.)
    body=wave(beat,math.pi,math.pi/2)
    # Add a slower four-beat rise/fall to J3, under the same smooth envelope.
    j3_bob=wave(beat,math.pi/2,math.pi/2)
    side=multiply(sway,wave(beat,math.pi/2))
    tilt=multiply(sway,wave(beat,math.pi/2,-.35))
    nod_envelope=multiply(ramp(beat,0.,2.),one-ramp(beat,6.,8.))
    # One broad pulse per two beats gives visible motion without the old
    # small double-frequency flick. Stay upright through beat 16; only the
    # return-home segment lowers the wrist again.
    wrist=multiply(nod_envelope,.26*wave(beat,math.pi,-.5))+1.2*ramp(beat,8.,10.)
    rows=[.18*side,
          -.035*body+.03*rise,
          .045*body+.07*j3_bob+.06*rise,
          np.zeros(3),
          -.10*side,
          .14*tilt+.025*wave(beat,math.pi,-1.)]
    result=np.column_stack([multiply(envelope,row) for row in rows])
    result[:,3]=wrist
    return result


def quintic(start,end,duration,t):
    s=ramp(t,0.,duration);delta=np.asarray(end)-start
    return np.asarray(start)+s[0]*delta,s[1]*delta,s[2]*delta


def coefficients(left,right):
    """Normalized-time quintic used by JTC when position/velocity/acceleration exist."""
    t0,q0,v0,a0=left;t1,q1,v1,a1=right;dt=t1-t0
    c0=np.asarray(q0);c1=np.asarray(v0)*dt;c2=np.asarray(a0)*dt*dt/2
    d=np.asarray(q1)-c0-c1-c2;v=np.asarray(v1)*dt-c1-2*c2;a=np.asarray(a1)*dt*dt-2*c2
    return np.array([c0,c1,c2,10*d-4*v+a/2,-15*d+7*v-a,6*d-3*v+a/2])


def interpolate(left,right,t):
    c=coefficients(left,right);u=(t-left[0])/(right[0]-left[0])
    return np.polynomial.polynomial.polyval(u,c).tolist()


def validate(points,speeds,accelerations):
    """Check continuous quintic extrema, not just the transmitted samples."""
    from .motion import LOWER,UPPER
    if len(points)<2:raise ValueError('empty dance')
    for t,q,v,a in points:
        pose(q)
        if not math.isfinite(t) or not 0<t<=60.:raise ValueError('invalid dance duration')
        if any(len(x)!=6 or not np.isfinite(x).all() for x in (q,v,a)):raise ValueError('invalid dance sample')
    if any(abs(x)>1e-9 for row in (points[0][2],points[0][3],points[-1][2],points[-1][3]) for x in row):
        raise ValueError('dance must start and end at rest')
    for left,right in zip(points,points[1:]):
        dt=right[0]-left[0]
        if dt<=0:raise ValueError('dance timestamps must increase')
        c=coefficients(left,right)
        for j in range(6):
            polynomial=np.polynomial.Polynomial(c[:,j])
            for order,low,high in ((0,LOWER[j],UPPER[j]),(1,-speeds[j],speeds[j]),(2,-accelerations[j],accelerations[j])):
                derivative=polynomial.deriv(order)
                roots=derivative.deriv().roots()
                samples=[0.,1.]+[float(x.real) for x in roots if abs(x.imag)<1e-7 and 0<x.real<1]
                values=derivative(samples)/dt**order
                if min(values)<low-1e-7 or max(values)>high+1e-7:raise ValueError('dance interpolation exceeds joint envelope')


def build_dance(start,centre,finish,speeds,accelerations,bpm=90.,amplitude=1.,entry_via=None):
    start,centre,finish=map(pose,(start,centre,finish))
    bpm=finite(bpm,60.,110.,'dance tempo');amplitude=finite(amplitude,.4,1.2,'dance amplitude')
    if (len(speeds)!=6 or len(accelerations)!=6
            or not all(math.isfinite(x) and x>0 for x in list(speeds)+list(accelerations))):raise ValueError('invalid dance limits')
    dense=np.array([groove(b) for b in np.linspace(0.,16.,3201)])*amplitude
    beat=max(60./bpm,float(np.max(np.abs(dense[:,1,:])/speeds))*1.03,
             math.sqrt(float(np.max(np.abs(dense[:,2,:])/accelerations)))*1.03)
    def travel(a,b):
        delta=np.abs(np.array(b)-a)
        return max(.8,float(np.max(1.875*delta/np.minimum(speeds,.65)))*1.03,
                   math.sqrt(float(np.max((10/math.sqrt(3))*delta/accelerations)))*1.03)
    entry_poses=[start]+([] if entry_via is None else [pose(entry_via)])+[centre]
    entry_segments=[];offset=.2
    for a,b in zip(entry_poses,entry_poses[1:]):
        duration=travel(a,b)
        entry_segments.append((duration,offset,lambda t,a=a,b=b,d=duration:quintic(a,b,d,t)))
        offset+=duration
    core_finish=(np.array(centre)+amplitude*groove(16.)[0]).tolist()
    entry=offset-.2;exit=travel(core_finish,finish)
    core=16*beat;points=[(.2,start,[0.]*6,[0.]*6)]
    for duration,offset,fn in entry_segments+[
        (core,.2+entry,lambda t:(np.array(centre)+amplitude*groove(t/beat)[0],
                                  amplitude*groove(t/beat)[1]/beat,amplitude*groove(t/beat)[2]/beat**2)),
        (exit,.2+entry+core,lambda t:quintic(core_finish,finish,exit,t))]:
        for t in np.linspace(0.,duration,math.ceil(duration/.04)+1)[1:]:
            q,v,a=fn(float(t));points.append((offset+float(t),list(q),list(v),list(a)))
    validate(points,speeds,accelerations)
    return points,dict(beat_sec=beat,actual_bpm=60/beat,core_start=.2+entry,core_end=.2+entry+core,
                       duration=points[-1][0],requested_bpm=bpm,amplitude=amplitude)


def collision_samples(points):
    """Sample the actual JTC interpolation at <=20 ms, including transitions."""
    samples=[list(points[0][1])]
    for left,right in zip(points,points[1:]):
        count=max(1,math.ceil((right[0]-left[0])/.02))
        samples.extend(interpolate(left,right,left[0]+(right[0]-left[0])*i/count) for i in range(1,count+1))
    return samples


@dataclass(frozen=True,init=False)
class DancePlan:
    """Immutable output of the continuously validated planner.

    Construction belongs on a worker. Submission checks the limits and anchor
    again, without rerunning polynomial roots inside a ROS feedback callback.
    """
    points: tuple
    speeds: tuple
    accelerations: tuple
    metadata: tuple

    def __init__(self,start,centre,finish,speeds,accelerations,bpm=90.,amplitude=1.,entry_via=None):
        points,meta=build_dance(start,centre,finish,speeds,accelerations,bpm,amplitude,entry_via)
        object.__setattr__(self,'points',tuple((t,tuple(q),tuple(v),tuple(a)) for t,q,v,a in points))
        object.__setattr__(self,'speeds',tuple(speeds))
        object.__setattr__(self,'accelerations',tuple(accelerations))
        object.__setattr__(self,'metadata',tuple(meta.items()))


class TaughtDeparture:
    """Narrow exemption for departure from the operator-confirmed release pose.

    MoveIt FK + the complete finger mesh verify increasing minimum height on
    the first 5% of this exact joint segment (0.019 -> 0.036 m). The recorded
    table top is 0.029 m; per-triangle contact.depth is not maximum penetration.
    Keep the base fixed, progress only outward, allow only this known pair and
    <=11 mm contact depth. Once clear, require strict validity for all samples.
    Other taught poses/entry paths get no exemption. No global scene is changed.
    """
    RELEASE=np.array([.263,.535,.23,-.599,-.285,.273])
    CLEAR=np.array([.263,.6,.9,0.,0.,0.])

    @staticmethod
    def scene_matches(table):
        if (table is None or table.id!='cyberdog_observed_table' or table.header.frame_id!='base_link'
                or len(table.primitives)!=1 or len(table.primitive_poses)!=1 or table.meshes or table.planes):return False
        if table.primitives[0].type!=1:return False
        lo=[.05558204049876611,-.6568967385542418];hi=[.3985576776722882,-.0056047769641885264]
        top=max(.023508207260724053*x-.009321865689088018*y+.003044323054376846
                for x in (lo[0],hi[0]) for y in (lo[1],hi[1]))+.01
        p=table.primitive_poses[0];origin=table.pose
        root_q=[origin.orientation.x,origin.orientation.y,origin.orientation.z,abs(origin.orientation.w)]
        root_p=[origin.position.x,origin.position.y,origin.position.z]
        # GetPlanningScene normalizes the primitive's transform into object.pose.
        identity=np.allclose(root_q,[0,0,0,1],rtol=0.,atol=1e-6)
        unspecified=np.allclose(root_q+root_p,[0]*7,rtol=0.,atol=1e-6)
        if not identity and not unspecified:return False
        return (np.allclose(table.primitives[0].dimensions,[hi[0]-lo[0],hi[1]-lo[1],.1],rtol=0.,atol=1e-6)
                and np.allclose(np.array(root_p)+[p.position.x,p.position.y,p.position.z],[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,top-.05],rtol=0.,atol=1e-6)
                and np.allclose([p.orientation.x,p.orientation.y,p.orientation.z,abs(p.orientation.w)],[0,0,0,1],rtol=0.,atol=1e-6))

    def __init__(self,enabled,start,via,table):
        self.enabled=(enabled and table=='cyberdog_observed_table'
                      and np.allclose(start,self.RELEASE,rtol=0.,atol=1e-5)
                      and np.allclose(via,self.CLEAR,rtol=0.,atol=1e-5))
        self.table=table;self.clear=False;self.progress=0.

    def accept(self,valid,contacts,q):
        if valid:self.clear=True;return True
        if not self.enabled or self.clear or not contacts:return False
        delta=self.CLEAR-self.RELEASE;u=(q[2]-self.RELEASE[2])/delta[2]
        if not self.progress-1e-9<=u<=.05 or not np.allclose(q,self.RELEASE+u*delta,rtol=0.,atol=1e-7):return False
        pair={self.table,'omni_picker_hand_wide3_Link'}
        if any({c.contact_body_1,c.contact_body_2}!=pair or not math.isfinite(c.depth)
               or c.depth<0 or c.depth>.011 for c in contacts):return False
        self.progress=u
        return True
