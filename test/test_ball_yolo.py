import math
import pytest
from cyberdog.ball_yolo import select_ball


def test_yolo_keeps_real_confidence_and_image_coordinates():
    ball=select_ball([[100,50,180,130,.81,0]],640,400)
    assert ball['score']==.81
    assert ball['x']==pytest.approx(-.5625)
    assert ball['y']==pytest.approx(-.55)
    assert ball['width']==.25
    assert ball['height']==.4


@pytest.mark.parametrize('row',[
    [100,50,180,130,.69,0], [100,50,180,130,.99,1],
    [100,50,180,130,math.nan,0], [-1,50,180,130,.9,0],
    [100,50,180,130,1.1,0], [100,50,101,51,.9,0],
    [0,0,640,400,.9,0], [100,50,180,60,.9,0],
])
def test_yolo_rejects_unreliable_or_invalid_targets(row):
    assert select_ball([row],640,400) is None


def test_yolo_rejects_ambiguity_and_does_not_reuse_lost_target():
    row=[100,50,180,130,.9,0]
    assert select_ball([row],640,400)
    assert select_ball([row,[300,50,380,130,.8,0]],640,400) is None
    assert select_ball([],640,400) is None
