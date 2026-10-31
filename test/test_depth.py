import time
import pytest
from cyberdog.behavior import Cyberdog as Game
from test_behavior import graph
PARAMS=[]
def depth_frame(stamp,array,frame='camera_color_optical_frame',fmt='16UC1; compressedDepth png'):
    import cv2
    from sensor_msgs.msg import CompressedImage
    msg=CompressedImage(format=fmt,data=bytes(12)+cv2.imencode('.png',array)[1].tobytes())
    msg.header.frame_id=frame;msg.header.stamp.sec=int(stamp);msg.header.stamp.nanosec=round((stamp%1)*1e9)
    return msg


def test_compressed_depth_decodes_losslessly_to_metres_and_rejects_other_formats():
    import numpy as np
    from cyberdog.ball_sequence import decode_depth
    raw=np.arange(48,dtype=np.uint16).reshape(6,8)*37
    assert (decode_depth(depth_frame(10.,raw))==raw*.001).all()
    with pytest.raises(ValueError,match='compressedDepth'):decode_depth(depth_frame(10.,raw,fmt='jpeg'))
    with pytest.raises(ValueError,match='16UC1'):decode_depth(depth_frame(10.,raw,fmt='32FC1; compressedDepth png'))
    with pytest.raises(ValueError,match='invalid'):decode_depth(depth_frame(10.,raw.astype(np.uint8)))
    bad=depth_frame(10.,raw);bad.data=bytes(bad.data[:12])+b'not a png'
    with pytest.raises(ValueError,match='invalid'):decode_depth(bad)


def real_depth_sequence(game,monkeypatch):
    import numpy as np
    from sensor_msgs.msg import CameraInfo
    from geometry_msgs.msg import TransformStamped
    s=game.ball_sequence;game.mode='ball';game.robot.mock=False
    s.info=CameraInfo(width=8,height=6,k=[500.,0.,4.,0.,500.,3.,0.,0.,1.]);s.info.header.frame_id='camera_color_optical_frame'
    s.depth_info=s.info
    tf=TransformStamped();tf.transform.rotation.w=1.
    monkeypatch.setattr(s.buffer,'lookup_transform',lambda *a:tf)
    calls=[]
    def locate(depth,k,ball,translation,rotation,**calibration):
        calls.append(depth.copy());return dict(center=[float(depth[0,0]),0.,0.],radius=.02)
    monkeypatch.setattr('cyberdog.ball_sequence.locate_ball',locate)
    now=game.get_clock().now().nanoseconds*1e-9
    return s,calls,now-.05,lambda stamp:dict(stamp=stamp,valid=True,source='camera',ball={'x':0.})


def test_observation_waits_for_the_depth_frame_of_its_own_stamp(monkeypatch):
    import numpy as np
    with graph([Game],PARAMS) as (nodes,exe):
        game=nodes[0];s,calls,stamp,obs=real_depth_sequence(game,monkeypatch)
        s.receive('depth',depth_frame(stamp-.033,np.full((6,8),100,np.uint16)))
        s.observe(obs(stamp))
        assert s.target is None and not calls and len(s.pending)==1
        s.receive('depth',depth_frame(stamp,np.full((6,8),200,np.uint16)))
        assert len(calls)==1 and (calls[0]==.2).all() and s.target['center'][0]==.2 and not s.pending


def test_observation_without_a_depth_frame_within_50ms_is_dropped(monkeypatch):
    import numpy as np
    with graph([Game],PARAMS) as (nodes,exe):
        game=nodes[0];s,calls,stamp,obs=real_depth_sequence(game,monkeypatch)
        s.receive('depth',depth_frame(stamp-.1,np.full((6,8),100,np.uint16)))
        s.observe(obs(stamp))
        s.receive('depth',depth_frame(stamp+.1,np.full((6,8),200,np.uint16)))
        assert s.target is None and not calls and not s.pending


def test_misaligned_depth_or_lost_ball_resets_the_stillness_gate(monkeypatch):
    import numpy as np
    with graph([Game],PARAMS) as (nodes,exe):
        game=nodes[0];s,calls,stamp,obs=real_depth_sequence(game,monkeypatch)
        s.receive('depth',depth_frame(stamp,np.full((6,8),200,np.uint16),frame='other_frame'))
        s.stable=True;s.observe(obs(stamp))
        assert not s.stable and not calls
        s.stable=True;s.receive('depth',depth_frame(stamp+.033,np.full((6,8),200,np.uint16)))
        s.observe(obs(stamp+.033));s.stable=True
        s.observe(dict(stamp=stamp+.066,valid=True,source='camera',ball=None))
        assert not s.stable and not s.pending



def test_timestamped_tf_is_retried_without_resetting_dwell(monkeypatch):
    import numpy as np
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import ExtrapolationException
    with graph([Game],PARAMS) as (nodes,exe):
        game=nodes[0];s,calls,stamp,obs=real_depth_sequence(game,monkeypatch)
        s.gate.update([.2,0.,0.],stamp-.1)
        s.receive('depth',depth_frame(stamp,np.full((6,8),200,np.uint16)))
        def missing(*args):raise ExtrapolationException('TF has not arrived yet')
        monkeypatch.setattr(s.buffer,'lookup_transform',missing)
        s.observe(obs(stamp))
        assert len(s.pending)==1 and not calls and s.gate.since==stamp-.1
        assert s.status()['last_reason']=='waiting for timestamped TF'
        tf=TransformStamped();tf.transform.rotation.w=1.
        monkeypatch.setattr(s.buffer,'lookup_transform',lambda *args:tf)
        s.tick()
        assert not s.pending and len(calls)==1 and s.gate.since==stamp-.1


def test_tf_wait_cannot_turn_expired_source_into_fresh_target(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from tf2_ros import ExtrapolationException
    with graph([Game],PARAMS) as (nodes,exe):
        game=nodes[0];s,calls,stamp,obs=real_depth_sequence(game,monkeypatch)
        s.receive('depth',depth_frame(stamp,np.full((6,8),200,np.uint16)))
        def missing(*args):raise ExtrapolationException('pending')
        monkeypatch.setattr(s.buffer,'lookup_transform',missing)
        s.observe(obs(stamp));assert len(s.pending)==1
        with monkeypatch.context() as m:
            m.setattr(game,'get_clock',lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int((stamp+.16)*1e9))))
            s.tick()
        assert not s.pending and not calls and not s.target_valid()
        assert s.reasons['RGB-D/TF sync timeout']==1
