import cv2
import numpy as np
from cyberdog.ball_template import BallTemplate


def test_selected_texture_survives_translation_and_rejects_loss_and_duplicates(tmp_path):
    rng=np.random.default_rng(17)
    patch=np.full((72,72,3),180,dtype=np.uint8)
    for _ in range(120):
        x,y=rng.integers(10,62,size=2)
        cv2.circle(patch,(int(x),int(y)),3,tuple(int(v) for v in rng.integers(0,255,size=3)),-1)
    p=tmp_path/'target.png';cv2.imwrite(str(p),patch);detector=BallTemplate(str(p))
    frame=np.full((320,480,3),180,dtype=np.uint8);frame[120:192,200:272]=patch
    result=detector.detect(frame)
    assert result is not None and abs(result['x']-(2*236/480-1))<.03
    moved=np.full_like(frame,180);moved[80:152,280:352]=patch
    result=detector.detect(moved)
    assert result is not None and result['x']>.25
    assert detector.detect(np.full_like(frame,180)) is None
    frame[20:92,30:102]=patch
    assert detector.detect(frame) is None


def test_local_refinement_follows_gradual_size_and_rotation_changes(tmp_path):
    rng=np.random.default_rng(29)
    patch=np.full((72,72,3),160,dtype=np.uint8)
    for _ in range(120):
        x,y=rng.integers(10,62,size=2)
        cv2.circle(patch,(int(x),int(y)),3,tuple(int(v) for v in rng.integers(0,255,size=3)),-1)
    p=tmp_path/'target.png';cv2.imwrite(str(p),patch);detector=BallTemplate(str(p))
    for i in range(12):
        size=72+2*i
        changed=cv2.resize(patch,(size,size))
        matrix=cv2.getRotationMatrix2D((size/2,size/2),i*3.,1.)
        changed=cv2.warpAffine(changed,matrix,(size,size),borderMode=cv2.BORDER_REPLICATE)
        frame=np.full((320,480,3),160,dtype=np.uint8)
        x=180+i*3;y=120
        frame[y:y+size,x:x+size]=changed
        result=detector.detect(frame)
        assert result is not None,i
        assert abs(result['x']-(2*(x+size/2)/480-1))<.03


def test_real_open_gripper_is_not_acquired_as_rope_ball():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    frame=cv2.imread(str(root/'test/data/ball-with-open-gripper.jpg'))
    detector=BallTemplate(str(root/'config/rope-ball-template.png'))
    # Full resolution previously selected the gripper with score 0.73.
    for shape in ((848,530),(640,400)):
        detector.last=None
        result=detector.detect(cv2.resize(frame,shape))
        assert result is None or (result['x']<0 and result['y']<0)
    # Production resolution must still acquire the real ball.
    detector.last=None
    result=detector.detect(cv2.resize(frame,(640,400)))
    assert result is not None and -.4<result['x']<-.2 and -.6<result['y']<-.3


def test_reacquisition_of_larger_near_ball(tmp_path):
    rng=np.random.default_rng(42)
    patch=rng.integers(0,255,(72,72,3),dtype=np.uint8)
    patch=cv2.GaussianBlur(patch,(3,3),0)
    p=tmp_path/'near.png';cv2.imwrite(str(p),patch)
    detector=BallTemplate(str(p))
    large=cv2.resize(patch,(130,130))
    frame=np.full((400,640,3),180,dtype=np.uint8);frame[120:250,250:380]=large
    result=detector.detect(frame)
    assert result is not None and abs(result['x']-(2*315/640-1))<.03
    frame[120:250,250:380]=180
    assert detector.detect(frame) is None
