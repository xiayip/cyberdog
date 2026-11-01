import math

import numpy as np
import pytest

from cyberdog.ball_velocity import BallVelocity, joint_velocity
from cyberdog.tracking import add_velocity_feedforward, tracking_step


CAMERA=[0.,0.,0.]
ORIGINS=[[0.,0.,0.],[0.,0.,0.]]
AXES=[[1.,0.,0.],[0.,1.,0.]]


def observe(fit,t,p):
    fit.update(t,p,CAMERA,ORIGINS,AXES)


def test_world_velocity_vector_has_correct_sign_and_freshness():
    fit=BallVelocity()
    for i in range(6):
        t=10+i/30
        observe(fit,t,[.2*(t-10),-.1*(t-10),.5])
    assert fit.status(t)['active']
    assert np.allclose(fit.world,[.2,-.1,0.])
    assert np.allclose(fit.output(t),[.17,.34],atol=.004)
    assert fit.output(t+.151)==[0.,0.]
    assert fit.output(t-.001)==[0.,0.]
    assert fit.output(math.nan)==[0.,0.]
    assert not fit.status(t+.2)['active']


def test_stationary_world_target_does_not_follow_camera_motion_or_small_jitter():
    fit=BallVelocity();rng=np.random.default_rng(42)
    for i in range(100):
        p=np.array([0.,0.,.5])+rng.uniform(-.001,.001,3)
        fit.update(10+i/30,p,[.02*math.sin(i),.01*math.cos(i),0.],ORIGINS,AXES)
        assert fit.output(10+i/30)==[0.,0.]


def test_reversal_stop_dropouts_and_jumps_do_not_reuse_old_velocity():
    fit=BallVelocity()
    for i in range(6):observe(fit,10+i/30,[i*.01,0.,.5])
    assert fit.output(10+5/30)[1]>0
    for i,x in enumerate((.04,.03,.02),6):observe(fit,10+i/30,[x,0.,.5])
    assert fit.output(10+8/30)[1]<=0
    for i in range(9,13):observe(fit,10+i/30,[.02,0.,.5])
    assert fit.output(10+12/30)==[0.,0.]
    observe(fit,11.,[.3,0.,.5])  # capture gap requires warmup
    assert fit.output(11.)==[0.,0.]
    observe(fit,11.04,[-.3,0.,.5])  # sphere-fit jump
    assert fit.output(11.04)==[0.,0.]
    fit.invalidate('target unavailable')
    assert fit.output(11.05)==[0.,0.]


@pytest.mark.parametrize('stamp,point',[(math.nan,[0.,0.,.5]),(10.,[math.nan,0.,.5]),(10.,[0.,.5])])
def test_invalid_input_clears_estimate(stamp,point):
    fit=BallVelocity()
    for i in range(6):observe(fit,9+i/30,[i*.01,0.,.5])
    observe(fit,stamp,point)
    assert fit.output(10.)==[0.,0.]


def test_duplicate_out_of_order_disabled_and_singular_geometry():
    fit=BallVelocity()
    for i in range(6):observe(fit,10+i/30,[i*.01,0.,.5])
    before=fit.output(10+5/30)
    observe(fit,10.1,[-.4,0.,.5])
    assert fit.output(10+5/30)==before
    fit.update(10.2,[.06,0.,.5],CAMERA,ORIGINS,[[0.,1.,0.]]*2)
    assert fit.output(10.2)==[0.,0.]
    off=BallVelocity(enabled=False)
    for i in range(6):observe(off,10+i/30,[i*.01,0.,.5])
    assert off.output(10.2)==[0.,0.] and not off.status(10.2)['active']


def test_transverse_mapping_includes_camera_translation_and_ignores_radial_motion():
    center=np.array([.2,-.1,.6]);camera=np.array([.06,.04,.13])
    origins=[np.array([-.05,0.,.04]),np.array([.02,-.04,.08])]
    axes=[np.array(a) for a in AXES]
    expected=np.array([.31,-.24])
    velocity=sum(np.cross(axis,center-origin)*rate for axis,origin,rate in zip(axes,origins,expected))
    # Independent finite difference of target coordinates in the moving camera.
    h=1e-6;omega=sum(axis*rate for axis,rate in zip(axes,expected))
    camera_velocity=sum(np.cross(axis,camera-origin)*rate for axis,origin,rate in zip(axes,origins,expected))
    local0=center-camera
    local1=(center+velocity*h-camera-camera_velocity*h)-np.cross(omega,local0)*h
    assert np.linalg.norm(local1/np.linalg.norm(local1)-local0/np.linalg.norm(local0))<1e-10
    assert np.allclose(joint_velocity(center,camera,origins,axes,velocity),expected,atol=.001)
    assert np.linalg.norm(joint_velocity(center,camera,origins,axes,center-camera))<1e-10


def test_feedforward_and_combined_commands_keep_existing_speed_acceleration_workspace_limits():
    fit=BallVelocity()
    for i in range(6):observe(fit,10+i/30,[i*.04,0.,.3])
    assert max(abs(v) for v in fit.output(10+5/30))<=.6
    for dt in (.01,.025,.05):
        q=[0.]*6;v=[0.]*6;lo=[0.,0.,0.,-.25,-.35,0.];hi=[0.,0.,0.,.25,.35,0.]
        for i in range(800):
            direction=1 if i<250 else -1 if i<500 else 0
            target=q[:]
            for j in (3,4):target[j]+=1.1*direction*dt
            target=add_velocity_feedforward(q,target,[.6*direction]*2,dt,1.2,[-2.]*6,[2.]*6)
            old=v[:];q,v=tracking_step(q,v,target,dt,1.2,2.4,lo,hi)
            assert max(abs(x) for x in v)<=1.2+1e-9
            assert max(abs(a-b)/dt for a,b in zip(v,old))<=2.4+1e-8
            assert all(a-1e-9<=x<=b+1e-9 for x,a,b in zip(q,lo,hi))
            assert q[:3]+q[5:]==[0.]*4
        assert max(abs(x) for x in v)<1e-9
