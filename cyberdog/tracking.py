"""Velocity/acceleration bounded visual-servo integration with braking room."""
import math


def add_velocity_feedforward(reference,target,joints,dt,max_speed,lower,upper):
    """Blend target motion with image feedback; Robot.track still limits braking."""
    result=list(target)
    for index,ff in zip((3,4),joints):
        speed=(target[index]-reference[index])/dt+ff
        speed=max(-max_speed,min(max_speed,speed))
        result[index]=max(lower[index],min(upper[index],reference[index]+speed*dt))
    return result


def tracking_step(reference, velocity, target, dt, max_speed, max_acceleration, lower, upper):
    if not math.isfinite(dt) or not 0 < dt <= .05:
        raise ValueError('tracking step requires 0 < dt <= 0.05')
    if not all(len(row)==6 for row in (reference,velocity,target,lower,upper)):
        raise ValueError('tracking requires six axes')
    if not all(math.isfinite(x) for row in (reference,velocity,target,lower,upper) for x in row):
        raise ValueError('non-finite tracking input')
    speeds=list(max_speed) if isinstance(max_speed,(list,tuple)) else [max_speed]*6
    accelerations=list(max_acceleration) if isinstance(max_acceleration,(list,tuple)) else [max_acceleration]*6
    if (len(speeds)!=6 or len(accelerations)!=6
            or not all(math.isfinite(x) and x>0 for x in speeds+accelerations)):
        raise ValueError('invalid tracking bounds')
    q_out,v_out=[],[]
    for q,v,goal,lo,hi,max_speed,max_acceleration in zip(reference,velocity,target,lower,upper,speeds,accelerations):
        if not lo-1e-9<=q<=hi+1e-9:raise ValueError('tracking reference outside workspace')
        # Reserve one full command interval plus continuous stopping distance.
        def braking_speed(distance):
            adt=max_acceleration*dt
            return max(0.,math.sqrt(adt*adt+2*max_acceleration*max(0.,distance))-adt)
        v_lo=-min(max_speed,braking_speed(q-lo))
        v_hi=min(max_speed,braking_speed(hi-q))
        wanted=max(v_lo,min(v_hi,(goal-q)/dt))
        step=max_acceleration*dt
        new_v=max(v-step,min(v+step,wanted))
        if not v_lo-1e-8<=new_v<=v_hi+1e-8:
            raise ValueError('insufficient braking distance')
        new_q=q+new_v*dt
        if not lo-1e-8<=new_q<=hi+1e-8:raise ValueError('tracking workspace boundary')
        q_out.append(new_q);v_out.append(new_v)
    return q_out,v_out
