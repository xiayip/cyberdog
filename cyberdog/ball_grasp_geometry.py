"""Calibrated RGB-D ball localization and Cartesian grasp geometry."""
import math
import numpy as np


def rotation(q):
    x,y,z,w=np.asarray(q,dtype=float);norm=x*x+y*y+z*z+w*w
    if not math.isfinite(norm) or abs(norm-1)>1e-3:raise ValueError('invalid camera quaternion')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def pixel_rays(pixels,k,distortion=()):
    """Optical rays for raw or rectified pixels, using their own calibration."""
    matrix=np.asarray(k,dtype=float).reshape(3,3)
    d=np.asarray(distortion,dtype=float)
    if not np.isfinite(matrix).all() or min(matrix[0,0],matrix[1,1])<=0:
        raise ValueError('invalid camera intrinsics')
    if not np.isfinite(d).all() or len(d) not in (0,4,5,8,12,14):
        raise ValueError('invalid camera distortion')
    pixels=np.asarray(pixels,dtype=float).reshape(-1,2)
    if np.any(d):
        import cv2
        # Orbbec labels its eight-coefficient rational Brown model plumb_bob;
        # OpenCV consumes all eight, including the denominator coefficients.
        xy=cv2.undistortPoints(pixels[:,None,:],matrix,d).reshape(-1,2)
    else:xy=(pixels-matrix[:2,2])/np.array([matrix[0,0],matrix[1,1]])
    return np.column_stack((xy,np.ones(len(xy))))


def locate_ball(depth,k,ball,translation,quaternion,*,distortion=(),depth_k=None,
                depth_distortion=(),color_size=None):
    """Fit a sphere to the central visible surface; reject weak/non-spherical fits."""
    h,w=color_size if color_size is not None else depth.shape
    depth_k=k if depth_k is None else depth_k
    u=(ball['x']+1)*w/2;v=(ball['y']+1)*h/2
    rx=ball['width']*w/4*.65;ry=ball['height']*h/4*.65
    if min(rx,ry)<5:raise ValueError('target too small for depth fit')
    ys,xs=np.mgrid[max(0,int(v-ry)):min(h,int(v+ry)+1):2,max(0,int(u-rx)):min(w,int(u+rx)+1):2]
    inside=((xs-u)/rx)**2+((ys-v)/ry)**2<=1
    rays=pixel_rays(np.column_stack((xs[inside],ys[inside])),k,distortion)
    dk=np.asarray(depth_k,dtype=float).reshape(3,3)
    # D2C can share the optical frame and K while RGB is still distorted and
    # depth is rectified. A shared frame_id does not imply shared pixel indices.
    pixel_rays([[0.,0.]],depth_k,depth_distortion)  # Validate both calibrations.
    if np.any(depth_distortion):
        import cv2
        uv=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),dk,np.asarray(depth_distortion,dtype=float))[0].reshape(-1,2)
    else:uv=rays[:,:2]*[dk[0,0],dk[1,1]]+dk[:2,2]
    uv=np.rint(uv[np.isfinite(uv).all(axis=1)]).astype(int)
    uv=np.unique(uv,axis=0)
    uv=uv[(uv[:,0]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]>=0)&(uv[:,1]<depth.shape[0])]
    x,y=uv.T;z=depth[y,x]
    valid=np.isfinite(z)&(z>.08)&(z<1.2)
    if valid.sum()<40:raise ValueError('insufficient valid ball depth')
    z=z[valid];x=x[valid];y=y[valid]
    near=np.abs(z-np.median(z))<.04
    z=z[near];x=x[near];y=y[near]
    p=pixel_rays(np.column_stack((x,y)),depth_k,depth_distortion)*z[:,None]
    if len(p)<40:raise ValueError('inconsistent depth')
    origin=np.mean(p,axis=0);local=p-origin
    solution,_,rank,_=np.linalg.lstsq(np.column_stack((2*local,np.ones(len(p)))),np.sum(local*local,axis=1),rcond=None)
    if rank<4:raise ValueError('flat target cannot define a ball')
    center=solution[:3]+origin;r2=solution[3]+np.dot(solution[:3],solution[:3])
    if r2<=0:raise ValueError('invalid ball fit')
    radius=math.sqrt(r2);residual=np.abs(np.linalg.norm(p-center,axis=1)-radius)
    if not .015<=radius<=.085 or np.quantile(residual,.9)>.006:
        raise ValueError('sphere fit does not reliably describe selected ball')
    if center[2]<np.median(z) or center[2]-np.median(z)>radius:
        raise ValueError('ball center is not behind visible surface')
    world=rotation(quaternion)@center+np.asarray(translation)
    return dict(center=world.tolist(),radius=radius,fit_p90_m=float(np.quantile(residual,.9)),points=len(p))


class WorldStillGate:
    def __init__(self,dwell=3.,tolerance=.01):
        self.dwell=dwell;self.tolerance=tolerance;self.reset()
    def reset(self):self.anchor=None;self.since=None;self.last=None
    def update(self,center,stamp):
        p=np.asarray(center,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all() or not math.isfinite(stamp):
            self.reset();return False
        if self.last is not None and stamp<=self.last:return False
        if self.last is not None and stamp-self.last>.2+1e-6:self.reset()
        if self.anchor is None or np.linalg.norm(p-self.anchor)>self.tolerance:
            self.anchor=p;self.since=stamp
        self.last=stamp
        return stamp-self.since>=self.dwell


def preview_waypoints(center,radius,yaw=0.,orientation=None,tcp_position=None):
    """Preserve tool attitude and approach from in front of the current TCP."""
    if not all(math.isfinite(v) for v in (*center,radius,yaw)):raise ValueError('non-finite geometry')
    if not .015<=radius<=.085:raise ValueError('invalid ball radius')
    q=list(orientation) if orientation is not None else [math.cos(yaw/2),math.sin(yaw/2),0.,0.]
    rot=rotation(q);back=-rot[:,2]
    approach=.10
    if tcp_position is not None:
        tcp=np.asarray(tcp_position,dtype=float)
        if tcp.shape!=(3,) or not np.isfinite(tcp).all():raise ValueError('invalid TCP position')
        relative=rot.T@(np.asarray(center)-tcp)
        forward=float(relative[2]);lateral=float(np.linalg.norm(relative[:2]))
        if forward<=radius+.01:raise ValueError('球心过近或不在夹爪前方，请重新放置毛球')
        if lateral>forward:raise ValueError('球心偏离夹爪前方超过 45 度，请重新居中')
        # A fixed 100 mm stand-off lies BEHIND the starting TCP for a nearby
        # ball. Reserve 40% of the axial distance for positive first progress.
        approach=min(.10,.6*forward)
    result=[]
    for name,dz in [('approach',approach),('pregrasp',approach*.7),('grasp',0),('lift',.12)]:
        result.append(dict(stage=name,position=(np.asarray(center)+(np.array([0.,0.,dz]) if name=='lift' else back*dz)).tolist(),orientation=list(q),
                           gripper='close' if name=='grasp' else ('hold' if name=='lift' else 'open')))
    return result
