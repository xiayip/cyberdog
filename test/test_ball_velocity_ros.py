import math
import time

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage
from tf2_ros import TransformException
from ghost_game_orchestrator.face_tracking import make_face_sample

from test_behavior import graph


def tracking(g):
    g.mode='ball';g.state='active';g.phase='tracking';g.handshake_completed=True
    r=g.robot;r.set_tracking_profile(True);r.servo=True
    r.reference=[-1.5707,.6,.9,0.,0.,0.];r.mock_pose=r.reference[:]
    r.tracking_anchor=r.reference[:];g.anchor=r.reference[:]


def test_timestamped_joint_axes_and_centered_moving_ball_generate_feedforward_then_brake():
    with graph() as ([g],e):
        tracking(g);seq=g.ball_sequence;queries=[]
        def lookup(base,frame,stamp):
            queries.append((base,frame,stamp.nanoseconds))
            t=TransformStamped()
            # Rotate each child's local -Z into world X or world Y.
            if frame=='link4':t.transform.rotation.y=-math.sqrt(.5)
            else:t.transform.rotation.x=math.sqrt(.5)
            t.transform.rotation.w=math.sqrt(.5)
            return t
        seq.buffer.lookup_transform=lookup
        camera=TransformStamped().transform;camera.rotation.w=1.
        end=g.get_clock().now().nanoseconds
        for i in range(6):
            m=CompressedImage();stamp=end-(5-i)*30_000_000
            m.header.stamp.sec,m.header.stamp.nanosec=divmod(stamp,1_000_000_000)
            seq.update_velocity({'center':[i*.006,0.,.5]},m,camera)
        assert len(queries)==12 and all(b=='base_link' for b,_,_ in queries)
        assert queries[-2:]==[('base_link','link4',end),('base_link','link5',end)]
        assert np.allclose(seq.velocity.world,[.2,0.,0.])
        assert seq.velocity.output(end*1e-9)[1]>.3
        now=time.monotonic();g.sample=make_face_sample(now,1.,1.,.2,.2,2,2,.9)
        g.tracking_tick(now,.025)
        assert g.robot.reference[4]>0 and g.robot.reference[3]==0
        assert not g.robot.fault
        # Lost vision revokes FF immediately and uses the bounded brake path.
        g.sample=None
        for _ in range(10):g.tracking_tick(time.monotonic(),.025)
        assert seq.velocity.output(end*1e-9)==[0.,0.]
        assert max(abs(v) for v in g.robot.tracking_velocity)<1e-9


def test_missing_joint_tf_does_not_reject_grasp_center_or_touch_still_gate():
    with graph() as ([g],e):
        tracking(g);seq=g.ball_sequence
        seq.target={'center':[0.,0.,.5],'radius':.025};target=seq.target
        seq.gate.update(target['center'],10.);seq.gate.update(target['center'],10.5)
        elapsed=seq.gate.elapsed
        def missing(*a):raise TransformException('no joint TF')
        seq.buffer.lookup_transform=missing
        seq.update_velocity(target,CompressedImage(),TransformStamped().transform)
        assert seq.target is target and seq.gate.elapsed==elapsed
        assert seq.velocity.reason=='waiting for joint TF'
        assert seq.velocity.output(11.)==[0.,0.]


def test_lock_and_pause_clear_velocity_without_predicting_the_grasp_center():
    with graph() as ([g],e):
        tracking(g);seq=g.ball_sequence;now=g.get_clock().now().nanoseconds*1e-9
        for i in range(6):
            seq.velocity.update(now-(5-i)*.03,[i*.006,0.,.5],[0.,0.,0.],[[0.,0.,0.]]*2,[[1.,0.,0.],[0.,1.,0.]])
        assert seq.velocity.status(now)['active']
        target={'center':[.3,0.,.1],'radius':.025}
        seq.target=target;seq.target_at=time.monotonic();seq.stable=True
        seq.begin()
        assert g.phase=='grasp_settling' and seq.frozen==target['center']
        assert seq.velocity.output(now)==[0.,0.]
        g.pause('test')
        assert seq.velocity.output(now)==[0.,0.]
