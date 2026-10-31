"""Short optical-flow tracks seeded by the selected texture, with bounded trust."""
import math
from .ball_template import BallTemplate


class BallTracker:
    def __init__(self,path):
        self.detector=BallTemplate(path)
        self.reset()

    def reset(self):
        self.gray=None;self.points=None;self.ball=None;self.hist=None;self.flow_frames=0;self.frames=0
        self.stamp=None
        self.verified_stamp=None
        self.texture=0.
        self.detector.last=None

    @staticmethod
    def appearance(frame,ball):
        import cv2
        import numpy as np
        h,w=frame.shape[:2]
        cx,cy=(ball['x']+1)*w/2,(ball['y']+1)*h/2
        rx,ry=ball['width']*w/4,ball['height']*h/4
        mask=np.zeros((h,w),np.uint8)
        cv2.ellipse(mask,(round(cx),round(cy)),(max(2,round(rx*.65)),max(2,round(ry*.65))),0,0,360,255,-1)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],mask,[12,8],[0,180,0,256])
        cv2.normalize(hist,hist,1,0,cv2.NORM_L1)
        texture=float(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[mask>0].std())
        return hist,texture,mask

    def detect(self,frame,stamp=None):
        import cv2
        import numpy as np
        if stamp is not None:
            if self.stamp is not None and (stamp<=self.stamp or stamp-self.stamp>.15):self.reset()
            self.stamp=stamp
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.frames+=1
        found=None
        # A flow-only identity expires after 0.5 s at 30 Hz. Reacquisition must
        # match the original selected appearance, never a learned hand patch.
        fresh_identity=stamp is None or self.verified_stamp is None or 0<=stamp-self.verified_stamp<=.5
        if self.gray is not None and self.points is not None and len(self.points)>=8 and self.flow_frames<15 and fresh_identity:
            forward,st,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,winSize=(21,21),maxLevel=3)
            if forward is not None:
                back,bst,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,forward,None,winSize=(21,21),maxLevel=3)
                if back is not None:
                    good=(st.ravel()>0)&(bst.ravel()>0)&(np.linalg.norm(back-self.points,axis=2).ravel()<1.)
                    if good.sum()>=8 and good.mean()>=.6:
                        old=self.points[good].reshape(-1,2);new=forward[good].reshape(-1,2)
                        affine,inliers=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=2.)
                        if affine is not None and inliers.mean()>=.7:
                            scale=math.hypot(affine[0,0],affine[1,0]);h,w=gray.shape
                            center=np.array([(self.ball['x']+1)*w/2,(self.ball['y']+1)*h/2,1.])
                            x,y=affine@center
                            if .92<=scale<=1.08 and np.linalg.norm(np.array([x,y])-center[:2])<60:
                                candidate=dict(x=2*x/w-1,y=2*y/h-1,width=self.ball['width']*scale,
                                    height=self.ball['height']*scale,score=.75,method='flow')
                                if abs(candidate['x'])<1-candidate['width']/2 and abs(candidate['y'])<1-candidate['height']/2:
                                    hist,texture,_=self.appearance(frame,candidate)
                                    if texture>=max(12.,self.texture*.6) and cv2.compareHist(self.hist,hist,cv2.HISTCMP_BHATTACHARYYA)<.30:
                                        found=candidate
        # Template verification every third frame also corrects optical-flow drift.
        if found is None or self.frames%3==0:
            matched=self.detector.detect(frame)
            if matched is not None:
                # Do not jump to another object while a valid local track exists.
                if found is not None and math.hypot(matched['x']-found['x'],matched['y']-found['y'])>.20:
                    self.reset();return None
                found=matched;found['method']='template'
                self.flow_frames=0
                self.verified_stamp=stamp
            elif self.detector.ambiguous:
                self.reset();return None
        if found is None:
            self.reset();return None
        if found.get('method')=='flow':self.flow_frames+=1
        else:self.hist,self.texture,_=self.appearance(frame,found)
        _,_,mask=self.appearance(frame,found)
        self.points=cv2.goodFeaturesToTrack(gray,maxCorners=60,qualityLevel=.03,minDistance=4,mask=mask)
        self.gray=gray;self.ball=found
        # Keep the template refinement centered on the independently tracked ROI.
        if self.detector.last:
            _,_,tw,th,scale,angle=self.detector.last
            self.detector.last=((found['x']+1)*gray.shape[1]*.75/2,(found['y']+1)*gray.shape[0]*.75/2,tw,th,scale,angle)
        return found
