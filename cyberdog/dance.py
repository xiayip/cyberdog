"""One-shot, cancellable dance with an asynchronous complete-path scene check."""
import time
from concurrent.futures import ThreadPoolExecutor

from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetStateValidity,GetPlanningScene

from .choreography import DancePlan,collision_samples,SECTIONS,TaughtDeparture
from .common import param,finite
from .game_core import pose
from .motion import JOINTS,bounded_start


class Dance:
    def __init__(self,game):
        self.g=game;self.robot=game.robot
        self.centre=pose(param(game,'dance_pose',[0.,.6,.9,0.,0.,0.]))
        self.bpm=finite(param(game,'dance_bpm',90.),60.,110.,'dance tempo')
        self.amplitude=finite(param(game,'dance_amplitude',1.),.4,1.2,'dance amplitude')
        self.auto_after_place=param(game,'dance_after_place',True)
        if type(self.auto_after_place) is not bool:raise ValueError('dance_after_place must be boolean')
        self.auto_pending=False;self.auto_generation=0;self.auto_deadline=0.
        self.scene=game.create_client(GetStateValidity,'/check_state_validity')
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dance_plan');self.work=None
        self.busy=False;self.future=None;self.token=0;self.points=[];self.samples=[]
        self.index=0;self.meta={};self.started=None;self.deadline=0.;self.grip=None
        self.plan=None;self.preflight=''
        self.departure=None
        self.music_started=False

    def request(self):
        g=self.g;r=self.robot
        if self.busy:return False,'正在跳舞'
        if not (g.handshake_completed and g.ball_sequence.dance_ready and g.state=='paused' and g.phase=='complete'):
            return False,'本轮尚未确认抓到并放好毛线球，不能跳舞'
        if r.busy or g.ball_sequence.busy or g.ball_taken.busy or not r.fresh():return False,'等待当前动作完成后再跳舞'
        if r.fault:return False,r.fault
        self.auto_pending=False;g.ball_sequence.dance_ready=False
        g.generation+=1;g.round_id+=1;g.mode='dance';g.touch_until=0.;g.ball_seen=0.;g.sample=None
        self.busy=True;self.token+=1;self.started=None;self.meta={};self.music_started=False
        g.audio.emit(replace=True)
        g.set_state('transition','dance_start','准备跳舞：大幅点头 → 抬起弹跳 → 抬腕竖起 → 左右摇摆')
        g.position(self.prepare)
        return True,'已接受跳舞，完成后回到复位姿态'

    def prepare(self):
        try:
            r=self.robot
            anchor=bounded_start(r.current if r.mock else r.motion.reference_pose)
            # Leave the table with J1 held still, then turn to the dance centre.
            via=[anchor[0]]+self.centre[1:]
            args=(anchor,self.centre,self.g.home,r.motion.velocity_limits(inspection=True),
                  r.motion.acceleration_limits(inspection=True),self.bpm,self.amplitude,via)
            seq=self.g.ball_sequence
            taught=(seq.place_mode=='joint' and seq.held_at_release and seq.release_completed
                    and TaughtDeparture.scene_matches(seq.table_object)
                    and max(abs(a-b) for a,b in zip(anchor,seq.place_joint_pose))<.01)
            self.departure=TaughtDeparture(taught,anchor,via,seq.table_id)
            mock=r.mock
            def plan():
                result=DancePlan(*args)
                return result,[] if mock else collision_samples(result.points)
            self.deadline=time.monotonic()+20.
            self.g.set_state('transition','dance_checking','正在准备舞蹈动作')
            # Polynomial extrema checking must not starve joint/gripper feedback
            # in the behavior node's single-threaded ROS executor.
            self.work=self.pool.submit(plan)
        except Exception as exc:self.g.pause('舞蹈未开始：'+str(exc))

    def planned(self,work):
        try:
            self.plan,self.samples=work.result();r=self.robot
            self.points=self.plan.points;self.meta=dict(self.plan.metadata)
            if r.mock:self.play();return
            if not self.scene.service_is_ready():raise ValueError('MoveIt 场景服务不可用')
            if self.departure.enabled:
                # Re-read the live scene: the grasp's snapshot can predate a
                # table update, which must invalidate the taught exemption.
                client=self.g.ball_sequence.scene
                if not client.service_is_ready():raise ValueError('桌面场景服务不可用')
                req=GetPlanningScene.Request();req.components.components=PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                self.preflight='table';token=self.token
                self.future=client.call_async(req)
                self.future.add_done_callback(lambda f:self.table_checked(f,token))
            else:self.preflight='feedback'
        except Exception as exc:self.g.pause('舞蹈未开始：'+str(exc))

    def table_checked(self,future,token):
        if token!=self.token or not self.busy:return
        self.future=None
        try:
            tables=[o for o in future.result().scene.world.collision_objects if o.id==self.departure.table]
            self.departure.enabled=bool(len(tables)==1 and TaughtDeparture.scene_matches(tables[0]))
            self.preflight='feedback'
        except Exception as exc:self.g.pause('舞蹈桌面检查失败：'+str(exc))

    def check_next(self):
        if not self.busy:return
        if self.index==len(self.samples):
            self.preflight='ready';return
        req=GetStateValidity.Request();req.group_name='zephyr_arm_tcp';req.robot_state.is_diff=True
        req.robot_state.joint_state.name=JOINTS+[self.robot.gripper_joint]
        req.robot_state.joint_state.position=self.samples[self.index]+[self.grip]
        token=self.token
        try:
            self.future=self.scene.call_async(req)
            self.future.add_done_callback(lambda f:self.checked(f,token))
        except Exception as exc:self.g.pause('舞蹈场景检查失败：'+str(exc))

    def checked(self,future,token):
        if token!=self.token or not self.busy:return
        self.future=None
        try:
            result=future.result()
            if not self.departure.accept(result.valid,result.contacts,self.samples[self.index]):
                raise ValueError('路径与桌面或机械臂发生碰撞')
            self.index+=1;self.check_next()
        except Exception as exc:self.g.pause('舞蹈未开始：'+str(exc))

    def play(self):
        self.deadline=0.;self.preflight='';self.started=time.monotonic()
        self.g.set_state('active','dance_running',f'跳舞中 · {self.meta["actual_bpm"]:.0f} BPM · 完成后回到复位姿态')
        self.g.issue(lambda cb:self.robot.dance(self.plan,cb),self.complete)

    def complete(self):
        if self.music_started:self.g.audio.emit(replace=True)
        self.music_started=False
        self.busy=False;self.started=None
        self.g.handshake_completed=False;self.g.ball_sequence.reset_attempts();self.g.idle()

    def queue_after_place(self):
        g=self.g
        if self.auto_after_place and g.ball_sequence.dance_ready and g.state=='paused' and g.phase=='complete':
            self.auto_pending=True;self.auto_generation=g.generation;self.auto_deadline=time.monotonic()+6.

    def stop(self):
        if self.music_started:self.g.audio.emit(replace=True)
        self.music_started=False
        self.token+=1;self.busy=False;self.started=None;self.deadline=0.;self.preflight=''
        self.auto_pending=False
        future,self.future=self.future,None
        if future is not None and not future.done():future.cancel()
        work,self.work=self.work,None
        if work is not None:work.cancel()

    def tick(self):
        if self.auto_pending:
            g=self.g
            if (g.generation!=self.auto_generation or not g.handshake_completed or not g.ball_sequence.dance_ready
                    or g.state!='paused' or g.phase!='complete'):
                self.auto_pending=False
            elif time.monotonic()>self.auto_deadline:g.pause('放球后未确认手臂就绪，未开始舞蹈')
            elif not self.robot.busy and self.robot.fresh():
                ok,reason=self.request()
                if not ok:g.pause('放球后未开始舞蹈：'+reason)
        if self.busy and self.deadline and time.monotonic()>self.deadline:
            reason='场景检查'
            if self.work is not None:reason='轨迹计算'
            elif self.preflight in ('feedback','ready'):
                reason='等待夹爪反馈' if self.robot.fresh_gripper_position() is None else '等待手臂稳定'
            self.g.pause('舞蹈准备超时：'+reason);return
        if self.busy and self.work is not None and self.work.done():
            work,self.work=self.work,None;self.planned(work)
        if (self.busy and self.started is not None and not self.music_started
                and self.robot.motion.phase in ('executing','mock')):
            # The goal was accepted; never play music during path checks or
            # when submission was rejected. Poll latency is one behavior tick.
            self.music_started=True;self.started=time.monotonic()
            self.g.audio.start_dance(self.meta)
        if not self.busy or self.preflight not in ('feedback','ready'):return
        r=self.robot;grip=r.fresh_gripper_position()
        if not r.fresh() or r.motion.ready() or grip is None:return
        if self.preflight=='feedback':
            if grip>.05:
                self.g.pause('夹爪尚未张开，未开始放球后的舞蹈');return
            self.grip=grip;self.index=0;self.preflight='scene';self.check_next()
        elif abs(grip-self.grip)>.02:self.g.pause('舞蹈检查期间夹爪状态改变')
        else:self.play()

    def destroy(self):
        self.stop();self.pool.shutdown(wait=False,cancel_futures=True)

    def status(self):
        elapsed=0. if self.started is None else time.monotonic()-self.started
        section='正在准备舞蹈'
        if self.started is not None:
            if elapsed<self.meta['core_start']:section='抬起准备'
            elif elapsed>=self.meta['core_end']:section='回到复位姿态'
            else:
                beat=(elapsed-self.meta['core_start'])/self.meta['beat_sec']
                section=next(name for start,name in reversed(SECTIONS) if beat>=start)
        return dict(active=self.busy,unlocked=self.g.ball_sequence.dance_ready,auto_pending=self.auto_pending,
                    auto_after_place=self.auto_after_place,section=section,elapsed_sec=round(elapsed,1),duration_sec=round(self.meta.get('duration',0.),1),
                    actual_bpm=round(self.meta.get('actual_bpm',0.),1))
