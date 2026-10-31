"""Reset/hold, J3-triggered handshake and automatic yarn-ball tracking."""
import json
import time

from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ghost_game_orchestrator.face_tracking import make_face_sample, servo_target
from .common import STATE_QOS, finite, param, publish, run
from .game_core import pose, inspection_poses
from .motion import LOWER, UPPER
from .robot import Robot
from .ball_sequence import BallSequence
from .ball_prediction import BallPrediction
from .touch import TouchDetector, DampingMovementDetector


class Cyberdog(Node):
    def __init__(self):
        super().__init__('cyberdog')
        self.robot=Robot(self)
        self.motion=self.robot
        self.home=pose(param(self,'home_pose',[-1.5707,0.,0.,0.,0.,0.]))
        self.lift=pose(param(self,'handshake_pose',[-1.5707,.6,.8,0.,0.,0.]))
        self.wrist_delta=finite(param(self,'handshake_wrist_delta',.25),-.6,.6,'inspection J4 lift')
        offsets=param(self,'handshake_head_offsets',[.25,-.25])
        if not isinstance(offsets,(list,tuple)) or not 1<=len(offsets)<=4:raise ValueError('inspection requires 1..4 J6 offsets')
        self.head_offsets=[finite(v,-.6,.6,'inspection J6 tilt') for v in offsets]
        self.head_hold=finite(param(self,'handshake_head_hold_sec',.35),0.,2.,'inspection hold')
        self.head_poses=[];self.head_hold_until=0.
        self.damping_movement=DampingMovementDetector(
            finite(param(self,'damping_movement_threshold',.04),.005,.3,'damping movement threshold'),
            finite(param(self,'damping_movement_dwell_sec',.15),.05,1.,'damping movement dwell'),
            finite(param(self,'damping_movement_settle_sec',.5),0.,3.,'damping settling time'))
        self.handshake_grip_position=finite(param(self,'handshake_gripper_position',.7),0.,1.,'handshake gripper position')
        self.handshake_grip_sent=False
        self.speed=finite(param(self,'pose_speed',.24),.01,.4,'pose speed')
        self.grasp_enabled=param(self,'grasp_enabled',True)
        self.ball_still_sec=finite(param(self,'ball_still_sec',3.),.5,10.,'ball dwell')
        self.gain=finite(param(self,'ball_tracking_gain',2.),.1,2.,'tracking gain')
        self.touch=TouchDetector(param(self,'touch_signal','either'),
            finite(param(self,'j3_position_threshold',.03),.01,.3,'J3 position threshold'),
            finite(param(self,'j3_torque_threshold',1.),.05,10.,'J3 torque threshold'),
            finite(param(self,'touch_dwell_sec',.25),.1,2.,'touch dwell'))
        self.state,self.mode,self.phase,self.message='paused','','startup','点击复位开始'
        self.outcome=''
        self.round_id=self.generation=0
        self.sample=None;self.anchor=None
        self.sensor_session=None;self.sensor_sequence=-1
        self.ball_seen=0.;self.ball_retry=0.;self.touch_until=0.
        self.pending=None;self.pending_deadline=0.;self.pending_ready=None
        self.reset_deadline=0.;self.last_tick=time.monotonic()
        self.ball_prediction=BallPrediction(lead=.04)
        self.pub=self.create_publisher(String,'/cyberdog/state',STATE_QOS)
        self.telemetry=self.create_publisher(String,'/cyberdog/tracking_status',1)
        self.create_subscription(String,'/cyberdog/observation',self.observation,1)
        self.create_subscription(String,'/cyberdog/command',lambda m:self.command(m.data),10)
        self.command_services=[self.create_service(Trigger,'/cyberdog/'+name,
            lambda req,res,n=name:self.service(n,res)) for name in ('reset','stop')]
        self.ball_sequence=BallSequence(self)
        self.create_timer(.025,self.tick)
        self.create_timer(.2,self.publish_state)
        self.publish_state()

    def publish_state(self):
        publish(self.pub,dict(schema_version=2,state=self.state,mode=self.mode,phase=self.phase,
            message=self.message,outcome=self.outcome,round_id=self.round_id,
            mock=self.robot.mock,motion_enabled=self.robot.motion.enabled,
            motion_busy=self.robot.busy or self.ball_sequence.busy,
            controller_mode=self.robot.modes.mode,watch_ball=self.state!='paused'))
        publish(self.telemetry,dict(touch=self.touch.status(),ball=self.ball_sequence.status(),
            damping_movement=self.damping_movement.status(),gripper_target=self.robot.grip_target))

    def set_state(self,state,phase,message):
        self.state,self.phase,self.message=state,phase,message
        self.get_logger().info(f'{state}/{phase}: {message}')
        self.publish_state()

    def service(self,name,response):
        response.success,response.message=self.command(name)
        return response

    def command(self,name):
        if name=='stop':self.pause('已暂停并保持');return True,'已暂停'
        if name!='reset':return False,'仅支持 reset / stop'
        if self.state=='resetting':return False,'正在复位'
        self.pause('停止当前动作，准备复位')
        self.mode='';self.outcome='';self.round_id+=1
        self.ball_sequence.reset_attempts()
        self.touch.reset();self.touch_until=0.;self.ball_seen=0.;self.sample=None
        self.reset_deadline=time.monotonic()+10.
        self.set_state('resetting','waiting_stop','等待动作停止后复位')
        return True,'已接受复位'

    def callback(self,next_step):
        generation=self.generation
        def done(ok,reason):
            if generation!=self.generation:return
            if not ok:self.pause(reason)
            else:next_step()
        return done

    def issue(self,operation,next_step):
        ok,reason=operation(self.callback(next_step))
        if not ok:self.pause(reason)
        return ok

    def wait_settled(self,next_step,ready=None):
        self.pending=next_step;self.pending_ready=ready or self.robot.motion.ready
        self.pending_deadline=time.monotonic()+6.
        self.set_state('transition','settling','等待关节稳定')

    def position(self,next_step):
        self.robot.hold()
        self.issue(lambda cb:self.robot.mode('position',cb,tracking=False),lambda:self.position_ready(next_step))

    def position_ready(self,next_step):
        self.robot.hold()
        self.wait_settled(next_step)

    def move(self,targets,next_step):
        self.issue(lambda cb:self.robot.move(targets,cb,self.speed,1.),next_step)

    def reset_move(self):
        self.set_state('transition','resetting','回到 J1=-1.5707，其余关节=0')
        self.move([self.home],self.idle)

    def idle(self):
        self.mode='';self.touch.reset();self.touch_until=0.
        self.set_state('idle','holding','保持复位姿态；掰动 J3 握手，看到毛线球自动追踪')

    def handshake(self):
        self.touch_until=0.;self.generation+=1;self.mode='handshake'
        self.damping_movement.reset();self.handshake_grip_sent=False
        self.set_state('transition','handshake_start','J3 触碰触发：准备抬腕、歪头打量')
        self.position(self.inspect_wrist)

    def inspect_wrist(self):
        if not self.robot.fresh():self.pause('抬腕前关节反馈不可用');return
        try:wrist,self.head_poses=inspection_poses(self.robot.current,self.wrist_delta,self.head_offsets)
        except ValueError as exc:self.pause('打量动作超出关节范围：'+str(exc));return
        self.set_state('transition','wrist_lift','先单独抬起 J4')
        self.move([wrist],self.inspect_head)

    def inspect_head(self):
        if self.head_poses:
            target=self.head_poses.pop(0)
            self.set_state('transition','head_tilt','J6 歪头打量')
            self.move([target],self.inspect_hold)
        else:
            self.set_state('transition','lifting','打量结束，通过 JTC 到握手姿态')
            self.move([self.lift],lambda:self.wait_settled(self.damping,self.robot.modes.would_pass))

    def inspect_hold(self):
        self.head_hold_until=time.monotonic()+self.head_hold
        self.set_state('transition','head_looking','歪头停留，打量一下')

    def damping(self):
        self.issue(lambda cb:self.robot.mode('damping',cb),self.damping_ready)

    def damping_ready(self):
        self.damping_movement.reset();self.handshake_grip_sent=False
        self.set_state('active','handshake',f'阻尼握手中；掰动关节夹爪收至 {self.handshake_grip_position:g}，看到毛线球自动追踪')

    def handshake_tick(self,now):
        if self.robot.modes.mode!='damping' or self.robot.busy or self.handshake_grip_sent:return
        fresh=self.robot.fresh() and (self.robot.mock or now-self.robot.motion.pose_at<=.2)
        if not self.damping_movement.update(now,self.robot.current if fresh else None):return
        self.handshake_grip_sent=True
        position=self.handshake_grip_position
        self.set_state('active','handshake',f'检测到 J{self.damping_movement.joint} 运动，夹爪收至 {position:g}')
        self.issue(lambda cb:self.robot.grip_position(position,cb),
            lambda:self.set_state('active','handshake',f'阻尼握手中；夹爪已到 {position:g}'))

    def enter_ball(self):
        self.generation+=1;self.round_id+=1;self.mode='ball';self.touch_until=0.
        self.ball_sequence.reset_attempts()
        self.set_state('transition','ball_entering','发现毛线球，切换位置追踪')
        generation=self.generation
        def switched(ok,reason):
            if generation!=self.generation:return
            if ok:self.position_ready(self.open_for_tracking)
            elif self.robot.modes.refused:
                self.ball_retry=time.monotonic()+.3
                self.mode='handshake' if self.robot.modes.mode=='damping' else ''
                self.set_state('active' if self.mode else 'idle','handshake' if self.mode else 'holding',
                    '等待手臂稳定后追踪：'+reason)
            else:self.pause(reason)
        self.robot.hold()
        ok,reason=self.robot.mode('position',switched,tracking=False)
        if not ok:switched(False,reason)

    def open_for_tracking(self):
        self.issue(lambda cb:self.robot.grip(False,cb),self.begin_tracking)

    def begin_tracking(self):
        self.issue(lambda cb:self.robot.mode('position',cb),self.tracking_started)

    def tracking_started(self):
        self.robot.set_tracking_profile(True)
        self.anchor=self.robot.current;self.sample=None;self.ball_prediction.reset()
        self.ball_sequence.gate.reset();self.ball_sequence.stable=False
        self.ball_sequence.target=None;self.ball_sequence.pending.clear()
        self.ball_sequence.frozen=None;self.ball_sequence.restart_count()
        retry=f'；准备第 {self.ball_sequence.attempts+1}/{self.ball_sequence.max_attempts} 次尝试' if self.ball_sequence.attempts else ''
        self.set_state('active','tracking','跟踪毛线球；空间位置稳定 3 秒后抓取'+retry)

    def observation(self,msg):
        try:
            d=json.loads(msg.data)
            if not isinstance(d,dict) or d.get('schema_version')!=1:return
            if d.get('source') not in (('camera','mock','external') if self.robot.mock else ('camera',)):return
            stamp=finite(d['stamp'],0.,1e12,'stamp')
            age=self.get_clock().now().nanoseconds*1e-9-stamp
            if not 0<=age<=.15 or d.get('round_id')!=self.round_id:return
            session,sequence=d['session'],d['sequence']
            if not isinstance(session,str) or type(sequence) is not int:return
            if session==self.sensor_session and sequence<=self.sensor_sequence:return
            if session!=self.sensor_session:
                self.ball_prediction.reset();self.ball_sequence.gate.reset()
                self.ball_sequence.stable=False;self.ball_sequence.pending.clear()
            self.sensor_session,self.sensor_sequence=session,sequence
            ball=d.get('ball')
            if d.get('valid') is not True or not isinstance(ball,dict):
                self.sample=None;self.ball_seen=0.;self.ball_sequence.observe(dict(d,ball=None));return
            x=finite(ball['x'],-1.,1.,'x');y=finite(ball['y'],-1.,1.,'y')
            w=finite(ball['width'],.001,2.,'width');h=finite(ball['height'],.001,2.,'height')
            finite(ball['score'],.7,1.,'score')
            self.ball_seen=time.monotonic()-age
            if self.mode!='ball' or self.state!='active':return
            self.ball_sequence.observe(d)
            if not self.ball_prediction.update(stamp,x,y,ball['score']):self.sample=None;return
            self.sample=make_face_sample(time.monotonic()-age,x+1,y+1,w,h,2,2,ball['score'])
        except (ValueError,TypeError,KeyError):
            self.sample=None;self.ball_seen=0.
            self.ball_sequence.reject(ValueError('invalid observation'))

    def tracking_tick(self,now,dt):
        if self.ball_sequence.stable and self.ball_sequence.target_valid():
            self.ball_sequence.begin();return
        if self.sample is None or now-self.sample.received_at>.15:
            self.ball_prediction.reset()
            self.robot.track(self.robot.reference,dt)
            return
        error=self.ball_prediction.predict(self.get_clock().now().nanoseconds*1e-9)
        if error is None:error=(self.sample.error_x,self.sample.error_y)
        target=servo_target(self.robot.reference,self.anchor,LOWER,UPPER,4,3,
            *error,self.gain,self.gain,self.robot.tracking_speed,dt,-1.,-1.,self.robot.tracking_yaw_range,self.robot.tracking_pitch_range,.08)
        self.robot.track(target,dt)

    def pause(self,message):
        self.generation+=1;self.pending=None
        self.head_poses=[];self.head_hold_until=0.;self.damping_movement.reset()
        self.ball_sequence.stop();self.robot.stop()
        self.set_state('paused','stopped',message)

    def tick(self):
        now=time.monotonic();dt=max(.001,min(.05,now-self.last_tick));self.last_tick=now
        self.robot.tick()
        if self.robot.fault and self.state not in ('paused','resetting'):
            self.pause(self.robot.fault);return
        if self.state=='resetting':
            if now>self.reset_deadline:self.pause('复位未开始：动作未停止或关节未稳定');return
            if (not self.robot.busy and not self.ball_sequence.busy and self.robot.stationary()
                    and (self.robot.mock or self.robot.motion.ready()=='' or self.robot.modes.would_pass()=='')):
                self.robot.fault='';self.position(self.reset_move)
            return
        if self.state=='paused':return
        self.ball_sequence.tick()
        if self.pending:
            if now>self.pending_deadline:self.pause('控制器未稳定：'+self.pending_ready())
            elif self.robot.mock or self.pending_ready()=='':
                callback,self.pending=self.pending,None;callback()
        if self.state=='transition' and self.phase=='head_looking' and now>=self.head_hold_until:
            self.inspect_head()
        if self.state=='active' and self.phase=='handshake':self.handshake_tick(now)
        eligible=self.state=='idle' or (self.state=='active' and self.phase=='handshake')
        if eligible and not self.robot.busy and now>=self.ball_retry and now-self.ball_seen<=.15:
            if self.robot.stationary() and self.robot.modes.would_pass()=='':
                self.enter_ball();return
        if self.state=='idle':
            effort=self.robot.motion.fresh_effort() if not self.robot.mock else None
            q=self.robot.current if self.robot.fresh() else None
            if self.touch.update(now,None if q is None else q[2],None if effort is None else effort[2],
                                 self.home[2],self.robot.stationary()):self.touch_until=now+6.
            if now<self.touch_until and not self.robot.busy and self.robot.modes.would_pass()=='':
                self.handshake();return
        if self.state=='active' and self.mode=='ball' and self.robot.servo and not self.ball_sequence.busy:
            self.tracking_tick(now,dt)

    def prepare_shutdown(self):self.pause('shutdown')

    def destroy_node(self):
        self.ball_sequence.destroy();self.robot.destroy()
        return super().destroy_node()


def main():run(Cyberdog)
