import math
import numpy as np
import pytest

from cyberdog.arm_tracking import ArmTracking,distribute,image_jacobian
from cyberdog.tracking import tracking_step


# Redundant camera yaw/pitch axes, with a nonzero camera lever arm.
POINT=np.array([.03,-.02,.5]);CAMERA=np.array([.02,.01,.05])
ORIGINS=[[0.,0.,-.1],[0.,-.1,0.],[0.,-.05,.05],[0.,0.,.1],[0.,0.,.15]]
AXES=[[0.,1.,0.],[1.,0.,0.],[-1.,0.,0.],[1.,0.,0.],[0.,1.,0.]]


def geometry():return image_jacobian(POINT,CAMERA,np.eye(3),ORIGINS,AXES,[1.25,1.67])


def test_image_jacobian_matches_camera_ray_finite_difference_all_five_axes():
    j,p=geometry();h=1e-7;local=POINT-CAMERA
    image=lambda v:np.array([1.25*v[0]/v[2],1.67*v[1]/v[2]])
    for i,(axis,origin) in enumerate(zip(AXES,ORIGINS)):
        axis=np.array(axis);origin=np.array(origin)
        moved_camera=CAMERA+np.cross(axis,CAMERA-origin)*h
        moved_local=POINT-moved_camera-np.cross(axis,POINT-moved_camera)*h
        assert np.allclose((image(moved_local)-image(local))/h,j[:,i],atol=1e-5)
    assert np.allclose((image(local+np.array([.1,-.2,0.])*h)-image(local))/h,p@np.array([.1,-.2,0.]),atol=1e-5)


@pytest.mark.parametrize('error',[[.4,0.],[0.,.4],[.3,-.4]])
def test_error_is_shared_across_arm_and_wrist_and_decreases_image_error(error):
    j,_=geometry();desired=-2.4*np.array(error)
    v=distribute(j,desired,[0.]*6,[0.]*6,[-1.]*6,[1.]*6,[.45]*3+[1.2]*3)
    assert np.dot(j@v,error)<0
    assert max(abs(x) for x in v[:3])>.02
    assert max(abs(x) for x in v[3:])>.02
    assert max(abs(x) for x in v[:3])<=.45
    assert np.linalg.norm(j@v-desired)<.02


def test_joint_at_boundary_hands_over_to_others_instead_of_driving_past_limit():
    j,_=geometry();q=[-.6,0.,0.,0.,0.,0.]
    lo=q[:];hi=[.6,.3,.35,1.,1.,0.];lo[3:5]=[-1.,-1.]
    desired=np.array([.5,.4]);v=distribute(j,desired,q,q,lo,hi,[.45]*3+[1.2]*3)
    assert all(x>=-1e-9 for x in v[:3])
    assert np.linalg.norm(j@v-desired)<.01


def test_stale_invalid_disabled_geometry_falls_back_and_centred_target_holds():
    arm=ArmTracking(True);args=([.3,.2],[0.]*6,[0.]*6,[-1.]*6,[1.]*6,[.45]*3+[1.2]*3,.025,2.4,[0.]*3,.85,.6)
    assert arm.target(10.,*args) is None
    arm.update(10.,POINT,CAMERA,np.eye(3),ORIGINS,AXES,[1.25,1.67])
    assert arm.target(10.1,*args) is not None
    assert arm.rates[5]==0.
    assert arm.target(10.151,*args) is None
    assert not arm.status(10.151)['active']
    arm.update(11.,POINT,CAMERA,np.eye(3),ORIGINS,AXES,[1.25,1.67])
    centred=([0.,0.],*args[1:])
    assert arm.target(11.1,*centred)==[0.]*6
    assert arm.rates==[0.]*6
    arm.update(11.1,POINT,CAMERA,np.zeros((3,3)),ORIGINS,AXES,[1.25,1.67])
    assert arm.target(11.1,*args) is None
    assert ArmTracking(False).target(10.,*args) is None


def test_arm_and_wrist_use_distinct_speed_acceleration_and_braking_limits():
    speed=[.45]*3+[1.2]*3;acc=[1.]*3+[2.4]*3
    lo=[-.6,0.,0.,-1.5,-1.5,0.];hi=[.6,.3,.35,1.5,1.5,0.]
    for dt in (.01,.025,.05):
        q=[0.]*6;v=[0.]*6
        for i in range(1000):
            direction=1 if i<350 else -1 if i<700 else 0
            target=[x+direction*2*dt for x in q];target[5]=0.;old=v[:]
            q,v=tracking_step(q,v,target,dt,speed,acc,lo,hi)
            assert all(abs(x)<=limit+1e-9 for x,limit in zip(v,speed))
            assert all(abs(a-b)/dt<=limit+1e-8 for a,b,limit in zip(v,old,acc))
            assert all(a-1e-9<=x<=b+1e-9 for x,a,b in zip(q,lo,hi))
            assert q[5]==0
        assert max(abs(x) for x in v)<1e-9


@pytest.mark.parametrize('speeds,accelerations',[([.5]*5,[1.]*6),([.5]*6,[1.]*5),([math.nan]*6,[1.]*6),([.5]*6,[0.]*6)])
def test_invalid_per_axis_limits_are_rejected(speeds,accelerations):
    with pytest.raises(ValueError):tracking_step([0.]*6,[0.]*6,[0.]*6,.025,speeds,accelerations,[-1.]*6,[1.]*6)
