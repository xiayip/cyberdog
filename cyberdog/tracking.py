"""Velocity/acceleration bounded visual-servo integration with braking room."""
import math


def tracking_step(reference, velocity, target, dt, max_speed, max_acceleration, lower, upper):
    if not math.isfinite(dt) or not 0 < dt <= .05:
        raise ValueError('tracking step requires 0 < dt <= 0.05')
    if not all(len(row)==6 for row in (reference,velocity,target,lower,upper)):
        raise ValueError('tracking requires six axes')
    if not all(math.isfinite(x) for row in (reference,velocity,target,lower,upper) for x in row):
        raise ValueError('non-finite tracking input')
    if not math.isfinite(max_speed) or not math.isfinite(max_acceleration) or min(max_speed,max_acceleration)<=0:
        raise ValueError('invalid tracking bounds')
    q_out,v_out=[],[]
    for q,v,goal,lo,hi in zip(reference,velocity,target,lower,upper):
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
