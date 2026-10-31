import math
import numpy as np
import pytest
from cyberdog.ball_grasp_geometry import WorldStillGate,locate_ball,preview_waypoints,rotation


def test_three_seconds_require_continuous_world_stationarity_and_fresh_samples():
    g=WorldStillGate()
    for i in range(30):assert not g.update([.2,0,.1],i*.1)
    assert g.update([.2,0,.1],3.)
    assert not g.update([.2,0,.1],3.)
    assert not g.update([.23,0,.1],3.1)
    assert not g.update([.23,0,.1],4.)
    g.reset()
    for i in range(50):assert not g.update([.2+i*.003,0,.1],i*.1)


def test_rgbd_sphere_center_and_invalid_depth():
    h,w=160,200;f=200.;cx=w/2;cy=h/2
    y,x=np.mgrid[:h,:w];rays=np.stack(((x-cx)/f,(y-cy)/f,np.ones((h,w))),axis=-1)
    center=np.array([0.,0.,.4]);radius=.04
    a=np.sum(rays*rays,axis=-1);b=-2*np.sum(rays*center,axis=-1);c=center@center-radius**2
    disc=b*b-4*a*c;depth=np.where(disc>=0,(-b-np.sqrt(np.maximum(0,disc)))/(2*a),0)
    ball=dict(x=0.,y=0.,width=.4,height=.5)
    result=locate_ball(depth,[f,0,cx,0,f,cy,0,0,1],ball,[.1,.2,.3],[0,0,0,1])
    assert result['center']==pytest.approx([.1,.2,.7],abs=1e-5)
    assert result['radius']==pytest.approx(.04,abs=1e-5)
    with pytest.raises(ValueError):locate_ball(np.zeros_like(depth),[f,0,cx,0,f,cy,0,0,1],ball,[0,0,0],[0,0,0,1])
    with pytest.raises(ValueError):locate_ball(np.full_like(depth,.4),[f,0,cx,0,f,cy,0,0,1],ball,[0,0,0],[0,0,0,1])



def test_grasp_path_has_only_four_stages():
    assert [p['stage'] for p in preview_waypoints([.3,0.,.1],.025)]==['approach','pregrasp','grasp','lift']


def test_cartesian_path_may_leave_zero_reset_but_never_exceed_joint_limits():
    from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
    from cyberdog.ball_sequence import timed_points,JOINTS
    q=[-1.5707,0.,0.,0.,0.,0.]
    path=JointTrajectory(joint_names=JOINTS)
    for seconds,values in [(0,q),(1,[-1.5707,.06,.08,0.,0.,0.])]:
        p=JointTrajectoryPoint(positions=values,velocities=[0.]*6,accelerations=[0.]*6)
        p.time_from_start.sec=seconds;path.points.append(p)
    assert len(timed_points(path,q))==2
    path.points[-1].positions[2]=-.001
    with pytest.raises(ValueError,match='limit'):timed_points(path,q)


def test_near_ball_approach_advances_instead_of_retreating_at_real_capture_pose():
    tcp=np.array([.001961545,.308584315,.457449696])
    center=np.array([-.020091203,.388914240,.515565247])
    q=[-.423597981,.432695641,.558523343,.566913492]
    rot=rotation(q);axis=rot[:,2]
    # Recorded regression: the old fixed stand-off requests -7.48 mm on tool Z.
    assert np.dot(center-.1*axis-tcp,axis)<-.007
    waypoints=preview_waypoints(center,.0151057,orientation=q,tcp_position=tcp)
    previous=tcp
    for waypoint in waypoints[:3]:
        point=np.array(waypoint['position'])
        assert np.dot(point-previous,axis)>0
        assert np.linalg.norm(center-point)<np.linalg.norm(center-previous)
        assert waypoint['orientation']==q
        previous=point
    assert previous==pytest.approx(center)
    assert np.array(waypoints[3]['position'])-center==pytest.approx([0,0,.12])


@pytest.mark.parametrize('forward',[.05,.08,.10,.15,.3,.6])
def test_all_close_and_distant_approach_segments_advance(forward):
    tcp=np.array([.2,.1,.3]);center=tcp+[.01,-.02,forward]
    waypoints=preview_waypoints(center,.02,orientation=[0,0,0,1],tcp_position=tcp)
    z=[tcp[2]]+[p['position'][2] for p in waypoints[:3]]
    assert np.all(np.diff(z)>0)


@pytest.mark.parametrize('center',[[0,0,-.2],[0,0,.02],[.2,0,.1]])
def test_ball_behind_too_close_or_far_sideways_cannot_make_grasp_path(center):
    with pytest.raises(ValueError):
        preview_waypoints(center,.025,orientation=[0,0,0,1],tcp_position=[0,0,0])


def test_distorted_rgb_box_maps_into_rectified_depth_before_fitting():
    import cv2
    # Off-axis sphere: raw RGB and rectified D2C depth have different pixels.
    h,w=240,320;f=250.;k=np.array([f,0,w/2,0,f,h/2,0,0,1.])
    distortion=np.array([.7,.1,.005,-.003,0.,0.,0.,0.])
    y,x=np.mgrid[:h,:w];rays=np.stack(((x-w/2)/f,(y-h/2)/f,np.ones((h,w))),axis=-1)
    center=np.array([.16,.05,.4]);radius=.03
    a=np.sum(rays*rays,axis=-1);b=-2*np.sum(rays*center,axis=-1);c=center@center-radius**2
    disc=b*b-4*a*c;depth=np.where(disc>=0,(-b-np.sqrt(np.maximum(0,disc)))/(2*a),0)
    # Obtain the raw RGB bounding box by projecting the visible sphere surface.
    surface=rays[disc>=0]*depth[disc>=0,None]
    raw=cv2.projectPoints(surface,np.zeros(3),np.zeros(3),k.reshape(3,3),distortion)[0].reshape(-1,2)
    lo,hi=raw.min(axis=0),raw.max(axis=0)
    ball=dict(x=(lo[0]+hi[0])/w-1,y=(lo[1]+hi[1])/h-1,width=2*(hi[0]-lo[0])/w,height=2*(hi[1]-lo[1])/h)
    result=locate_ball(depth,k,ball,[0,0,0],[0,0,0,1],distortion=distortion,depth_k=k,depth_distortion=np.zeros(8),color_size=(h,w))
    assert result['center']==pytest.approx(center,abs=1e-5)
    assert result['radius']==pytest.approx(radius,abs=1e-5)
