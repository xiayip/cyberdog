import math
import pytest
from cyberdog.tracking import tracking_step


def test_tracking_reversal_braking_and_dropout_respect_bounds():
    q=[0.]*6;v=[0.]*6;lo=[-.25]*6;hi=[.25]*6
    for i in range(600):
        dt=[.025,.04,.01][i%3]
        wanted=.3 if i<180 else -.3 if i<360 else 0.
        target=[p+wanted*dt for p in q]
        old=list(v)
        q,v=tracking_step(q,v,target,dt,.3,.975,lo,hi)
        assert max(abs(x) for x in q)<=.25
        assert max(abs(x) for x in v)<=.3+1e-9
        assert max(abs(a-b)/dt for a,b in zip(v,old))<=.975+1e-8
    assert max(abs(x) for x in v)<1e-9


@pytest.mark.parametrize('dt',[0.,-.1,.06,math.nan])
def test_tracking_rejects_invalid_timestep(dt):
    with pytest.raises(ValueError):tracking_step([0.]*6,[0.]*6,[0.]*6,dt,.3,.975,[-1.]*6,[1.]*6)


def test_fast_profile_brakes_on_reversal_dropout_and_near_edges():
    for dt in (.01,.025,.05):
        q=[0.]*6;v=[0.]*6
        lo=[0.,0.,0.,-.25,-.35,0.];hi=[0.,0.,0.,.25,.35,0.]
        for i in range(900):
            speed=.9 if i<250 else -.9 if i<500 else 0.
            target=q[:]
            for axis in (3,4):target[axis]+=speed*dt
            old=v[:]
            q,v=tracking_step(q,v,target,dt,.9,1.5,lo,hi)
            assert all(a-1e-9<=x<=b+1e-9 for x,a,b in zip(q,lo,hi))
            assert max(abs(x) for x in v)<=.9+1e-9
            assert max(abs(a-b)/dt for a,b in zip(v,old))<=1.5+1e-8
        assert max(abs(x) for x in v)<1e-9
