"""Local YOLOE visual-prompt detector; every result comes from the current frame."""
import math
from pathlib import Path


def select_ball(rows, width, height, min_score=.7):
    """Keep original network confidence; ambiguous targets are not actionable."""
    candidates=[]
    for row in rows:
        if len(row)!=6 or not all(math.isfinite(float(v)) for v in row):continue
        x1,y1,x2,y2,score,cls=map(float,row)
        if cls!=0 or not min_score<=score<=1:continue
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):continue
        bw,bh=x2-x1,y2-y1
        if bw<8 or bh<8 or bw*bh>width*height*.15 or not .5<=bw/bh<=2.:continue
        candidates.append(dict(x=(x1+x2)/width-1,y=(y1+y2)/height-1,
            width=2*bw/width,height=2*bh/height,score=score,method='yoloe'))
    return candidates[0] if len(candidates)==1 else None


class YoloBall:
    def __init__(self, model_path, device='0'):
        path=Path(model_path).expanduser()
        if not path.is_file():raise ValueError(f'YOLOE weights missing: {path}')
        import torch
        from ultralytics import YOLOE
        torch.set_num_threads(2)
        self.model=YOLOE(str(path))
        if len(self.model.names)!=1:raise ValueError('Expected a single rope-ball visual-prompt class')
        self.device=device
        # Warm up once in the perception worker. The caller rejects this stale
        # first frame; no checkpoint or text-encoder download occurs at runtime.
        import numpy as np
        self.model.predict(np.zeros((400,640,3),dtype=np.uint8),device=device,imgsz=640,conf=.7,verbose=False)

    def detect(self,frame):
        result=self.model.predict(frame,device=self.device,imgsz=640,conf=.7,verbose=False)[0]
        h,w=frame.shape[:2]
        return select_ball(result.boxes.data.cpu().tolist(),w,h)
