import cv2
import numpy as np
from cyberdog.ball_tracker import BallTracker


def fixture(tmp_path):
    rng=np.random.default_rng(52)
    patch=rng.integers(0,255,(72,72,3),dtype=np.uint8)
    patch=cv2.GaussianBlur(patch,(3,3),0)
    p=tmp_path/'ball.png';cv2.imwrite(str(p),patch)
    return BallTracker(str(p)),patch


def frame(patch,x=150,y=120,angle=0.):
    out=np.full((320,480,3),180,dtype=np.uint8)
    matrix=cv2.getRotationMatrix2D((36,36),angle,1.)
    out[y:y+72,x:x+72]=cv2.warpAffine(patch,matrix,(72,72),borderMode=cv2.BORDER_REPLICATE)
    return out


def test_fast_tracking_follows_translation_rotation_and_stops_on_occlusion(tmp_path):
    tracker,patch=fixture(tmp_path)
    results=[]
    for i in range(12):
        result=tracker.detect(frame(patch,150+i*5,120,i*2),10+i/30)
        assert result is not None,i
        assert abs(result['x']-(2*(186+i*5)/480-1))<.03
        results.append(result)
    assert any(x['method']=='flow' for x in results)
    assert tracker.detect(np.full((320,480,3),180,dtype=np.uint8),10.4) is None
    assert tracker.ball is None


def test_ambiguity_detected_and_flow_identity_has_bounded_lifetime(tmp_path):
    tracker,patch=fixture(tmp_path)
    assert tracker.detect(frame(patch))
    duplicate=frame(patch);duplicate[20:92,340:412]=patch
    assert any(tracker.detect(duplicate) is None for _ in range(3))
    tracker.reset();assert tracker.detect(frame(patch))
    tracker.detector.detect=lambda _:None
    results=[tracker.detect(frame(patch,150+i)) for i in range(20)]
    assert results[0] is not None and results[-1] is None


def test_capture_gap_resets_flow_and_requires_selected_appearance(tmp_path):
    tracker,patch=fixture(tmp_path)
    assert tracker.detect(frame(patch),10.)
    tracker.detector.detect=lambda _:None
    assert tracker.detect(frame(patch),10.2) is None
