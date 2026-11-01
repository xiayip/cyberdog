"""Mode-aware latest-frame perception; no motor commands, no cloud requests."""
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from .common import STATE_QOS, finite, param, publish, run


def detect_ball(bgr,hsv_low,hsv_high):
    import cv2
    import numpy as np
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array(hsv_low,dtype=np.uint8),np.array(hsv_high,dtype=np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=bgr.shape[:2]
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
        if area<80 or area>w*h*.12 or perimeter<=0:continue
        circularity=4*math.pi*area/(perimeter*perimeter)
        if circularity<.72:continue
        (x,y),radius=cv2.minEnclosingCircle(contour)
        if radius<=0 or area/(math.pi*radius*radius)<.65:continue
        candidates.append((area,dict(x=2*x/w-1,y=2*y/h-1,width=4*radius/w,height=4*radius/h,score=min(1.,circularity))))
    candidates.sort(key=lambda pair:pair[0],reverse=True)
    if not candidates or (len(candidates)>1 and candidates[1][0]>.6*candidates[0][0]):return None
    return candidates[0][1]


class Senses(Node):
    def __init__(self):
        super().__init__('cyberdog_senses')
        # OpenCV's default pool (one worker per core) spins ~12 cores on 640 px frames and is slower than one thread,
        # starving the camera driver and controllers on this machine.
        import cv2
        cv2.setNumThreads(1)
        self.backend=param(self,'senses_backend','mock')
        if self.backend not in ('mock','camera','external'):raise ValueError('unknown senses backend')
        self.ball_detector=param(self,'ball_detector','hsv')
        if self.ball_detector not in ('hsv','template','yoloe'):raise ValueError('invalid ball detector')
        self.ball_template_path=param(self,'ball_template_path',str(Path(get_package_share_directory('cyberdog'))/'config/rope-ball-template.png'))
        self.ball_yolo_path=param(self,'ball_yolo_path',str(Path(get_package_share_directory('cyberdog'))/'models/cyberdog-rope-yoloe.pt'))
        self.ball_yolo_device=param(self,'ball_yolo_device','0')
        self.ball_min_score=finite(param(self,'ball_min_score',.35),.1,.95,'ball confidence threshold')
        self.ball_yolo=None
        self.ball_template=None
        self.ball_context=None
        self.low=list(param(self,'ball_hsv_low',[5,100,80]))
        self.high=list(param(self,'ball_hsv_high',[30,255,255]))
        if len(self.low)!=3 or len(self.high)!=3 or any(type(a) is not int or type(b) is not int or not 0<=a<=b<=limit for a,b,limit in zip(self.low,self.high,[179,255,255])):raise ValueError('invalid HSV range')
        self.state,self.mode,self.phase,self.round_id='paused','','',0
        self.mock_ball_visible=param(self,'mock_ball_visible',False)
        self.watch_ball=False
        self.mode_started=time.monotonic()
        self.session,self.sequence=uuid.uuid4().hex,0
        self.latest,self.future=None,None
        self.last_image_stamp=0.
        self.context_generation=0
        self.last_submit,self.last_preview=0.,0.
        self.loaded_errors={}
        self.pub=self.create_publisher(String,'/cyberdog/observation',1)
        self.status=self.create_publisher(String,'/cyberdog/senses_status',STATE_QOS)
        self.debug=self.create_publisher(CompressedImage,'/cyberdog/camera_preview/compressed',1)
        self.state_sub=self.create_subscription(String,'/cyberdog/state',self.on_state,STATE_QOS)
        self.image_sub=self.create_subscription(CompressedImage,param(self,'image_topic','/camera/color/image_raw/compressed'),self.on_image,
            QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cyberdog-perception')
        self.timer=self.create_timer(.005 if self.backend=='camera' else .066,self.tick)
        self.last_at,self.last_status=0.,0.

    def on_state(self,msg):
        try:
            data=json.loads(msg.data)
            if data.get('schema_version')!=2:return
            key=(data['mode'],data['round_id'])
            if key!=(self.mode,self.round_id):self.mode_started=time.monotonic()
            watch=data.get('watch_ball') is True
            if (data['state'],*key,watch)!=(self.state,self.mode,self.round_id,self.watch_ball):
                self.context_generation+=1
                self.latest=None
            self.state,self.mode,self.phase,self.round_id=data['state'],data['mode'],data['phase'],data['round_id']
            self.watch_ball=watch
        except (ValueError,TypeError,KeyError):pass

    def on_image(self,msg):
        if self.backend!='camera':return
        if not 0<len(msg.data)<=8*1024*1024:return
        now=self.get_clock().now().nanoseconds*1e-9
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if stamp<=self.last_image_stamp or not 0<=now-stamp<=.15:return
        self.last_image_stamp=stamp
        self.latest=(bytes(msg.data),stamp,time.monotonic())

    def envelope(self,stamp,rid,**values):
        self.sequence+=1
        return dict(schema_version=1,stamp=stamp,session=self.session,sequence=self.sequence,round_id=rid,**values)

    def process(self,data,state,mode,preview=True,context=None,source_stamp=None,watch_ball=False):
        started=time.perf_counter()
        import cv2
        import numpy as np
        frame=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
        if frame is None:raise ValueError('invalid image')
        h,w=frame.shape[:2]
        if w>640:frame=cv2.resize(frame,(640,round(h*640/w)))
        h,w=frame.shape[:2]
        result=dict(valid=True,digit=0,ball=None,face=None)
        if watch_ball or mode=='ball':
            if self.ball_detector=='yoloe':
                if self.ball_yolo is None:
                    from .ball_yolo import YoloBall
                    self.ball_yolo=YoloBall(self.ball_yolo_path,self.ball_yolo_device,self.ball_min_score)
                result['ball']=self.ball_yolo.detect(frame)
            elif self.ball_detector=='template':
                if self.ball_template is None:
                    from .ball_tracker import BallTracker
                    self.ball_template=BallTracker(self.ball_template_path)
                if context!=self.ball_context:
                    self.ball_template.reset()
                    self.ball_context=context
                result['ball']=self.ball_template.detect(frame,source_stamp)
            else:
                result['ball']=detect_ball(frame,self.low,self.high)
            if result['ball']:
                ball=result['ball']
                cv2.circle(frame,(round((ball['x']+1)*w/2),round((ball['y']+1)*h/2)),round(ball['width']*w/4),(0,255,0),2)
        jpeg=None
        if preview:
            cv2.putText(frame,f'{mode or "menu"} digit={result["digit"]}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,70])
            if ok:jpeg=encoded.tobytes()
        result['timing']=dict(processing_ms=(time.perf_counter()-started)*1000)
        return result,jpeg

    def tick(self):
        now=time.monotonic()
        stamp=self.get_clock().now().nanoseconds*1e-9
        if self.backend=='external':return
        if self.backend=='mock':
            # Mock observations are explicitly labelled; no unsolicited menu triggers.
            age=now-self.mode_started
            sample=dict(x=.35*math.sin(age) if age<4. else 0.,y=0.,width=.15,height=.15,score=.99)
            publish(self.pub,self.envelope(stamp,self.round_id,valid=True,digit=0,source='mock',
                    ball=sample if self.mock_ball_visible and (self.mode=='ball' or self.watch_ball) else None))
            return
        if self.future is not None and self.future.done():
            future,self.future=self.future,None
            source_stamp,rid,generation=self.inflight
            try:
                value,jpeg=future.result()
                if rid==self.round_id and generation==self.context_generation and 0<=stamp-source_stamp<=.15:
                    value.setdefault('timing',{})['capture_age_ms']=(stamp-source_stamp)*1000
                    publish(self.pub,self.envelope(source_stamp,rid,source='camera',**value))
                    if jpeg:
                        msg=CompressedImage(format='jpeg',data=jpeg)
                        msg.header.stamp.sec=int(source_stamp)
                        msg.header.stamp.nanosec=round((source_stamp-int(source_stamp))*1e9)%1000000000
                        self.debug.publish(msg)
                if now-self.last_status>1.:
                    self.last_status=now
                    publish(self.status,dict(backend='camera',status='ready'))
            except Exception as exc:
                publish(self.status,dict(backend='camera',status='error',reason=str(exc)[:240]))
                publish(self.pub,self.envelope(stamp,self.round_id,valid=False,digit=0,source='camera'))
        if self.future is None and self.latest and (self.state!='idle' or now-self.last_submit>=.066):
            data,source_stamp,received=self.latest
            self.latest=None
            if now-received<=.15 and 0<=stamp-source_stamp<=.15:
                self.inflight=(source_stamp,self.round_id,self.context_generation)
                preview=now-self.last_preview>=.1 and self.debug.get_subscription_count()>0
                if preview:self.last_preview=now
                self.future=self.pool.submit(self.process,data,self.state,self.mode,preview,self.context_generation,source_stamp,self.watch_ball)
                self.last_submit=now
                self.last_at=now
        if now-self.last_at>1. and now-self.last_status>1.:
            self.last_status=now
            publish(self.status,dict(backend='camera',status='waiting_for_fresh_timestamped_image'))
            publish(self.pub,self.envelope(stamp,self.round_id,valid=False,digit=0,source='camera'))

    def destroy_node(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        return super().destroy_node()


def main():run(Senses)
