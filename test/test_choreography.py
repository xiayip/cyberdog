import math
import numpy as np
import pytest

from cyberdog.choreography import build_dance,groove,validate,collision_samples,interpolate

HOME=[-1.5707,0.,0.,0.,0.,0.]
CENTRE=[-1.5707,.6,.9,0.,0.,0.]
SPEED=[.8,.8,.8,1.4,.8,1.4]
ACC=[1.3,1.3,1.3,6.,1.3,6.]


def test_analytic_wave_derivatives_and_c2_phase_boundaries():
    h=1e-5
    # Difference derivatives away from piecewise knots; check C2 continuity
    # at knots below (the jerk can legitimately jump there).
    for b in np.linspace(.1,15.9,60):
        q,v,a=groove(b)
        assert np.allclose((groove(b+h)[0]-groove(b-h)[0])/(2*h),v,atol=1e-7)
        assert np.allclose((groove(b+h)[1]-groove(b-h)[1])/(2*h),a,atol=1e-6)
    for b in (0.,2.,4.,6.,8.,10.,12.,14.,16.):
        assert np.max(np.abs(groove(b-h)-groove(b+h)))<.002
    assert np.allclose(groove(0.),0.)
    assert groove(16.)[0]==pytest.approx([0.,0.,0.,1.2,0.,0.])
    assert np.allclose(groove(16.)[1:],0.)


def test_first_eight_have_broad_wrist_nods_then_raise_before_side_sway():
    first=np.array([groove(b)[0] for b in np.linspace(0.,8.,1601)])
    assert np.ptp(first[:,3])>.51 and np.min(first[:,3])<-.25 and np.max(first[:,3])>.25
    assert np.max(np.abs(first[:,[0,4]]))==0.
    raised=np.array([groove(b) for b in np.linspace(8.,10.,401)])
    assert np.all(np.diff(raised[:,0,3])>=0.)
    assert np.max(np.abs(raised[:,0,[0,4]]))==0.
    sway=np.array([groove(b) for b in np.linspace(10.,16.,1201)])
    assert np.allclose(sway[:,0,3],1.2) and np.allclose(sway[:,1:,3],0.)
    assert np.ptp(sway[:,0,0])>.3 and np.ptp(sway[:,0,4])>.16


@pytest.mark.parametrize('bpm,amplitude,scale',[(60.,.4,1.),(90.,1.,1.),(110.,1.2,1.),(90.,1.,.5)])
def test_dance_is_continuous_retimed_bounded_and_returns_home(bpm,amplitude,scale):
    speeds=[x*scale for x in SPEED];acc=[x*scale for x in ACC]
    points,meta=build_dance(HOME,CENTRE,HOME,speeds,acc,bpm,amplitude)
    assert points[0][1]==HOME and np.allclose(points[-1][1],HOME)
    assert meta['actual_bpm']<=bpm and meta['duration']<60.
    assert points[-1][2]==pytest.approx([0.]*6) and points[-1][3]==pytest.approx([0.]*6)
    core=[p for p in points if meta['core_start']<p[0]<meta['core_end']]
    # The wrist stays raised on the last beat; return starts from that exact
    # pose, not the old neutral centre (which would make the interpolator snap).
    end=next(i for i,p in enumerate(points) if abs(p[0]-meta['core_end'])<1e-8)
    assert points[end][1][3]==pytest.approx(1.2*amplitude)
    assert points[end][2]==pytest.approx([0.]*6,abs=1e-9)
    assert points[end][3]==pytest.approx([0.]*6,abs=1e-9)
    assert abs(points[end+1][1][3]-points[end][1][3])<.001
    assert all(np.ptp([p[1][j] for p in core])>.015*amplitude for j in range(6))
    assert sum(max(abs(x) for x in p[2])>.01 for p in core)>.85*len(core)
    for left,right in zip(points,points[1:]):
        assert interpolate(left,right,left[0])==pytest.approx(left[1],abs=1e-9)
        assert interpolate(left,right,right[0])==pytest.approx(right[1],abs=1e-9)
    samples=collision_samples(points)
    assert samples[0]==HOME and samples[-1]==pytest.approx(HOME)
    assert len(samples)>len(points)


def test_interpolation_overspeed_cannot_hide_between_stopped_endpoints():
    target=HOME[:];target[0]+=.4
    with pytest.raises(ValueError,match='interpolation'):
        validate([(.2,HOME,[0.]*6,[0.]*6),(.4,target,[0.]*6,[0.]*6)],SPEED,ACC)


@pytest.mark.parametrize('bpm,amp',[(0.,1.),(111.,1.),(90.,1.21),(math.nan,1.),(90.,False)])
def test_invalid_tempo_and_amplitude_are_rejected(bpm,amp):
    with pytest.raises(ValueError):build_dance(HOME,CENTRE,HOME,SPEED,ACC,bpm,amp)
