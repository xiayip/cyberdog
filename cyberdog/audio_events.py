"""Small, nonblocking behavior cues. Playback belongs to a separate ROS node."""
import time
import uuid

from rclpy.qos import DurabilityPolicy,QoSProfile,ReliabilityPolicy
from std_msgs.msg import String

from .common import finite,param,publish
from .touch import DampingMovementDetector

AUDIO_QOS=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class AudioEvents:
    def __init__(self,game):
        self.g=game
        self.enabled=param(game,'audio_enabled',True)
        self.idle_interval=finite(param(game,'audio_idle_interval_sec',30.),20.,40.,'idle sound interval')
        self.wake_after=finite(param(game,'audio_wake_after_sec',20.),1.,59.,'wake sound delay')
        self.pub=game.create_publisher(String,'/cyberdog/audio_event',AUDIO_QOS)
        self.session=uuid.uuid4().hex;self.sequence=0
        self.phase='startup';self.idle_since=None;self.idle_next=0.;self.last_ball=None
        self.last_cue={};self.movement=DampingMovementDetector();self.handshake_next=0.
        self.emit('A1',replace=True)

    def emit(self,*events,replace=False,music=None):
        if not self.enabled:return
        self.sequence+=1
        try:
            message=dict(schema_version=1,session=self.session,sequence=self.sequence,
                stamp=self.g.get_clock().now().nanoseconds*1e-9,events=list(events),replace=replace)
            if music is not None:message['music']=music
            publish(self.pub,message)
        except Exception as exc:
            self.enabled=False
            self.g.get_logger().warning('音频事件发送失败，停用声音：'+str(exc))

    def start_dance(self,metadata):
        if not self.enabled:return
        self.g.get_logger().info(f'请求舞蹈音乐：{metadata["actual_bpm"]:.1f} BPM，'
            f'{metadata["duration"]:.1f} 秒，音频订阅数 {self.pub.get_subscription_count()}')
        self.emit(replace=True,music=dict(bpm=metadata['actual_bpm'],duration=metadata['duration'],
            core_start=metadata['core_start'],core_end=metadata['core_end']))

    def limited(self,event,now,interval):
        if now-self.last_cue.get(event,-1e9)<interval:return
        self.last_cue[event]=now;self.emit(event)

    def reset(self):
        self.movement.reset();self.last_ball=None;self.idle_since=None
        self.emit('E4',replace=True)

    def transition(self,state,phase,attempt=0,*,now=None):
        now=time.monotonic() if now is None else now
        previous=self.phase;self.phase=phase
        if previous==phase:
            # An explicit stop must also interrupt D3 after exhausted retries,
            # even though the robot was already paused/stopped.
            if state=='paused' and phase=='stopped':self.emit(replace=True)
            return
        if phase=='handshake_start':
            away=now-self.idle_since if self.idle_since is not None else 0.
            cue='B1' if away<self.wake_after else ('W3' if away>300. else 'W2' if away>=60. else 'W1')
            self.emit(cue,replace=True)
        if phase!='holding':self.idle_since=None
        if phase!='tracking':self.last_ball=None
        if phase!='handshake':self.movement.reset()
        if state=='paused' and phase=='stopped':self.emit(replace=True)
        elif phase=='waiting_stop':self.emit('E1')
        elif phase=='resetting':self.emit('E2')
        elif state=='idle' and phase=='holding':
            self.idle_since=now;self.idle_next=now+self.idle_interval
            self.limited('A2',now,3.)
        elif phase in ('wrist_lift','lifting'):self.emit('B3')
        elif phase=='head_tilt' or (phase=='head_looking' and previous!='head_tilt'):self.emit('B2')
        elif phase=='ball_entering':
            self.emit(replace=True);self.limited('C1',now,3.)
        elif phase=='tracking':self.emit('C2')
        elif phase=='grasp_settling':
            # C4 is silent; D4 is a separate one-shot cue for a retry's new lock.
            self.emit(*(['D4'] if attempt>1 else []),replace=True)
        elif phase=='grasp_closing':self.emit('C5',replace=True)
        elif phase=='complete':self.emit('C6',replace=True)
        elif phase=='grasp_retry_wait':self.emit('D1' if attempt==1 else 'D2',replace=True)
        elif phase=='ball_taken_bowing':self.emit('D3',replace=True)

    def seen_ball(self,now):
        if self.phase!='tracking':return
        if self.last_ball is not None and now-self.last_ball>=.5:self.limited('C3',now,4.)
        self.last_ball=now

    def tick(self,now):
        if not self.enabled:return
        if self.g.state=='idle' and self.phase=='holding' and now>=self.idle_next:
            self.idle_next=now+self.idle_interval;self.emit('A3')
        if self.g.state!='active' or self.phase!='handshake':return
        r=self.g.robot
        if r.modes.mode!='damping' or r.busy:return
        if now<self.handshake_next:return
        q=r.current if r.fresh() and (r.mock or now-r.motion.pose_at<=.2) else None
        if self.movement.update(now,q):
            self.emit('B4');self.handshake_next=now+4.;self.movement.reset()
