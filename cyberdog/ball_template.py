"""Selected textured-ball matching with local scale/rotation refinement."""
import math


class BallTemplate:
    def __init__(self,path):
        import cv2
        color=cv2.imread(path)
        image=None if color is None else cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
        if image is None or min(image.shape)<16 or max(image.shape)>200 or image.std()<12:
            raise ValueError('ball template must be a textured 16..200 pixel crop')
        self.image=cv2.resize(image,None,fx=.75,fy=.75)
        self.reference_hist=self.color_hist(cv2.resize(color,None,fx=.75,fy=.75))
        self.last=None
        self.frames=0
        self.ambiguous=False
        self.cache={}
        self.global_keys=[(scale,angle) for scale in (.7,.85,1.,1.15,1.3,1.45,1.6,1.8,2.) for angle in range(0,360,30)]

    @staticmethod
    def color_hist(patch):
        import cv2
        import numpy as np
        h,w=patch.shape[:2]
        mask=np.zeros((h,w),np.uint8)
        cv2.ellipse(mask,(w//2,h//2),(max(2,round(w*.325)),max(2,round(h*.325))),0,0,360,255,-1)
        hist=cv2.calcHist([cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)],[0,1],mask,[12,8],[0,180,0,256])
        return cv2.normalize(hist,hist,1,0,cv2.NORM_L1)

    def appearance_matches(self,color,x,y,w,h):
        import cv2
        x0,y0=max(0,round(x-w/2)),max(0,round(y-h/2))
        patch=color[y0:y0+round(h),x0:x0+round(w)]
        return min(patch.shape[:2])>=8 and cv2.compareHist(
            self.reference_hist,self.color_hist(patch),cv2.HISTCMP_BHATTACHARYYA)<.45

    def template(self,scale,angle):
        import cv2
        key=(round(scale,3),round(angle%360,2))
        if key not in self.cache:
            h,w=self.image.shape
            resized=cv2.resize(self.image,(round(w*scale),round(h*scale)))
            rh,rw=resized.shape
            matrix=cv2.getRotationMatrix2D((rw/2,rh/2),angle,1.)
            if len(self.cache)>512:self.cache.clear()
            self.cache[key]=cv2.warpAffine(resized,matrix,(rw,rh),borderMode=cv2.BORDER_REPLICATE)
        return self.cache[key]

    def detect(self,bgr):
        import cv2
        self.frames+=1
        self.ambiguous=False
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,None,fx=.75,fy=.75);height,width=gray.shape
        color=cv2.resize(bgr,(width,height))
        x0,y0=0,0;region=gray;keys=self.global_keys
        if self.last:
            x,y,w,h,scale,angle=self.last
            radius=max(85,round(max(w,h)*1.6))
            x0,y0=max(0,round(x-radius)),max(0,round(y-radius))
            region=gray[y0:min(height,round(y+radius)),x0:min(width,round(x+radius))]
            keys=[(round(scale+ds,3),(angle+da)%360) for ds in (-.05,-.025,0.,.025,.05)
                  for da in (-15.,-7.5,0.,7.5,15.) if .7<=scale+ds<=2.]
            # Most frames need only translation. Keep periodic scale/rotation
            # refinement and fall back immediately if the quick score weakens.
            if self.frames%3:
                quick=cv2.matchTemplate(region,self.template(scale,angle),cv2.TM_CCOEFF_NORMED)
                if cv2.minMaxLoc(quick)[1]>=.80:keys=[(scale,angle)]
        else:
            # Coarse global localization selects a few transforms; all accepted
            # scores are still recomputed at the original tracking resolution.
            coarse=cv2.resize(gray,None,fx=.5,fy=.5)
            ranked=[]
            for scale,angle in keys:
                small=cv2.resize(self.template(scale,angle),None,fx=.5,fy=.5)
                if small.shape[0]>coarse.shape[0] or small.shape[1]>coarse.shape[1]:continue
                scores=cv2.matchTemplate(coarse,small,cv2.TM_CCOEFF_NORMED)
                _,score,_,loc=cv2.minMaxLoc(scores)
                ranked.append((score,scale,angle,loc))
            ranked.sort(reverse=True)
            keys=[(s,a) for _,s,a,_ in ranked[:4]]
        candidates=[]
        for scale,angle in keys:
            template=self.template(scale,angle);h,w=template.shape
            if h>=region.shape[0] or w>=region.shape[1]:continue
            scores=cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED)
            for _ in range(2):
                _,score,_,(x,y)=cv2.minMaxLoc(scores)
                if math.isfinite(score):candidates.append((score,x0+x+w/2,y0+y+h/2,w,h,scale,angle))
                scores[max(0,y-h):y+h+1,max(0,x-w):x+w+1]=-1.
        # Geometry alone can confuse the opened gripper with the rope ball.
        # Verify every acquisition against the immutable selected color model.
        candidates=[c for c in candidates if c[0]>=.70 and self.appearance_matches(color,*c[1:5])]
        if not candidates:
            self.last=None;return None
        candidates.sort(reverse=True);score,x,y,w,h,scale,angle=candidates[0]
        if score<.70:
            self.last=None;return None
        if any(other[0]>=.70 and (abs(other[1]-x)>w or abs(other[2]-y)>h) for other in candidates[1:]):
            self.ambiguous=True
            self.last=None;return None
        # Even local tracking checks the full image for a competing copy. Never
        # update appearance from live pixels: occluding hands cannot become the model.
        scores=cv2.matchTemplate(gray,self.template(scale,angle),cv2.TM_CCOEFF_NORMED)
        bx,by=round(x-w/2),round(y-h/2)
        scores[max(0,by-h):by+h+1,max(0,bx-w):bx+w+1]=-1.
        _,competing_score,_,(cx,cy)=cv2.minMaxLoc(scores)
        if competing_score>=.70 and self.appearance_matches(color,cx+w/2,cy+h/2,w,h):
            self.ambiguous=True
            self.last=None;return None
        self.last=(x,y,w,h,scale,angle)
        return dict(x=2*x/width-1,y=2*y/height-1,width=2*w/width,height=2*h/height,score=float(score))
