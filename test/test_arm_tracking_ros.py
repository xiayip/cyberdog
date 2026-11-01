import math
import time
from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage
from tf2_ros import TransformException
from ghost_game_orchestrator.face_tracking import make_face_sample

from test_behavior import graph
from test_ball_velocity_ros import tracking
from test_arm_tracking import POINT,CAMERA,ORIGINS,AXES


def prepare(g):
    tracking(g);r=g.robot
    r.reference=[-1.5707,.6,.9,0.,0.,0.];r.mock_pose=r.reference[:]
    r.tracking_anchor=r.reference[:];g.anchor=r.reference[:]
    g.ball_sequence.coordination.update(g.get_clock().now().nanoseconds*1e-9,POINT,CAMERA,np.eye(3),ORIGINS,AXES,[1.25,1.67])


def test_full_behavior_moves_arm_and_wrist_then_brakes_on_loss_and_clears_on_grasp():
    with graph(parameters=['ball_multijoint_tracking:=true']) as ([g],e):
        prepare(g);start=g.robot.reference[:]
        for _ in range(10):
            now=time.monotonic();g.sample=make_face_sample(now,1.4,1.3,.2,.2,2,2,.9)
            g.tracking_tick(now,.025)
        assert any(abs(a-b)>.001 for a,b in zip(start[:3],g.robot.reference[:3]))
        assert any(abs(a-b)>.001 for a,b in zip(start[3:5],g.robot.reference[3:5]))
        assert g.robot.reference[5]==start[5] and not g.robot.fault
        g.sample=None
        for _ in range(25):g.tracking_tick(time.monotonic(),.025)
        assert max(abs(v) for v in g.robot.tracking_velocity)<1e-9
        assert not g.ball_sequence.coordination.status(g.get_clock().now().nanoseconds*1e-9)['active']
        g.set_state('active','grasp_settling','test')
        assert g.ball_sequence.coordination.stamp is None


def test_extra_joint_tf_uses_correct_positive_j2_axis_and_falls_back_independently():
    with graph(parameters=['ball_multijoint_tracking:=true','ball_target_feedforward:=false']) as ([g],e):
        prepare(g);seq=g.ball_sequence;queries=[]
        seq.info=SimpleNamespace(k=[500.,0.,320.,0.,500.,200.,0.,0.,1.],width=640,height=400)
        def lookup(base,frame,stamp):
            queries.append((frame,stamp.nanoseconds))
            t=TransformStamped();t.transform.rotation.w=math.sqrt(.5)
            if frame in ('link1','link5'):t.transform.rotation.y=-math.sqrt(.5)
            else:t.transform.rotation.x=math.sqrt(.5)
            return t
        seq.buffer.lookup_transform=lookup;m=CompressedImage();m.header.stamp=g.get_clock().now().to_msg()
        camera=TransformStamped().transform;camera.rotation.w=1.
        seq.update_velocity({'center':[0.,0.,.5]},m,camera)
        assert seq.coordination.jacobian is not None
        assert np.allclose(seq.coordination.jacobian[:,1],-seq.coordination.jacobian[:,2])
        assert sorted(x[0] for x in queries)==['link1','link2','link3','link4','link5']
        assert len(set(x[1] for x in queries))==1
        def missing(base,frame,stamp):
            if frame=='link2':raise TransformException('missing J2')
            return lookup(base,frame,stamp)
        seq.buffer.lookup_transform=missing;seq.update_velocity({'center':[0.,0.,.5]},m,camera)
        assert seq.coordination.jacobian is None
        start=g.robot.reference[:];g.sample=make_face_sample(time.monotonic(),1.4,1.3,.2,.2,2,2,.9)
        g.tracking_tick(time.monotonic(),.025)
        assert g.robot.reference[:3]==start[:3]  # fallback wrist servo
        assert g.robot.reference[3:5]!=start[3:5]
        assert not g.robot.fault


def test_position_trajectory_braking_tail_uses_per_axis_deceleration():
    with graph(parameters=['ball_multijoint_tracking:=true','ball_tracking_speed:=1.2','ball_tracking_acceleration:=2.4']) as ([g],e):
        prepare(g);published=[];g.robot.position_pub=SimpleNamespace(publish=published.append)
        q=g.robot.reference[:];v=[.4,.3,-.2,.8,-.6,0.]
        g.robot.publish_position_reference(q,v)
        points=published[-1].points;end=points[-1]
        assert list(end.velocities)==[0.]*6
        _,acc=g.robot.tracking_limits()
        for j in range(6):assert abs(end.positions[j]-q[j]-(v[j]*abs(v[j])/(2*acc[j])))<1e-8
        for a,b in zip(points,points[1:]):
            dt=(b.time_from_start.sec-a.time_from_start.sec)+(b.time_from_start.nanosec-a.time_from_start.nanosec)*1e-9
            assert all(abs(x-y)/dt<=limit+1e-6 for x,y,limit in zip(a.velocities,b.velocities,acc))


def test_one_depth_fit_dropout_preserves_only_unexpired_geometry():
    with graph(parameters=['ball_multijoint_tracking:=true']) as ([g],e):
        prepare(g);seq=g.ball_sequence;stamp=seq.coordination.stamp
        seq.reject(ValueError('one depth fit'),transient=True)
        assert seq.coordination.stamp==stamp
        assert not seq.coordination.status(stamp+.151)['active']
        seq.reject(ValueError('invalid calibration'))
        assert seq.coordination.stamp is None
