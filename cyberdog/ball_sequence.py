"""RGB-D ball workflow. Plans never double as permission to execute a partial path."""
import math
import time
from collections import Counter, deque

import numpy as np
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Pose
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetCartesianPath, GetPlanningScene
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, CameraInfo, JointState
from tf2_ros import Buffer, TransformListener, TransformException
from trajectory_msgs.msg import JointTrajectoryPoint

from .ball_grasp_geometry import locate_ball, preview_waypoints, rotation, WorldStillGate
from .common import finite, param
from .controller_modes import observed
from .motion import LOWER, UPPER
from controller_manager_msgs.srv import ListControllers

JOINTS = ['joint'+str(i) for i in range(1, 7)]


class RetryableGraspError(ValueError):
    """Perception/geometry failure that can recover with a new stationary target."""


def timed_points(trajectory, current):
    """Keep every planned sample; only slow time, never replace with joint shortcuts."""
    if list(trajectory.joint_names) != JOINTS or len(trajectory.points) < 2:
        raise ValueError('invalid Cartesian trajectory')
    rows = trajectory.points
    if max(abs(a-b) for a,b in zip(rows[0].positions, current)) > .01:
        raise RetryableGraspError('arm moved since planning')
    for p in rows:
        if any(len(v) != 6 for v in (p.positions,p.velocities,p.accelerations)):
            raise ValueError('planner derivatives missing')
        if not all(math.isfinite(x) for row in (p.positions,p.velocities,p.accelerations) for x in row):
            raise ValueError('non-finite path')
        if not all(min(lo+.05,start) <= q <= max(hi-.05,start)
                   and lo <= q <= hi for q,lo,hi,start in zip(p.positions,LOWER,UPPER,current)):
            raise ValueError('path enters joint limit margin')
    scale = max(2., max(abs(v) for p in rows for v in p.velocities)/.04*1.01,
                math.sqrt(max(abs(a) for p in rows for a in p.accelerations)/.12)*1.01)
    points=[]; previous=-1
    for p in rows:
        t=p.time_from_start.sec+p.time_from_start.nanosec*1e-9
        ns=round((.3+scale*t)*1e9)
        if t<0 or ns<=previous or ns>60_000_000_000:raise ValueError('invalid trajectory duration')
        previous=ns
        point=JointTrajectoryPoint(positions=list(p.positions),velocities=[v/scale for v in p.velocities],
                                   accelerations=[a/scale**2 for a in p.accelerations])
        point.time_from_start.sec,point.time_from_start.nanosec=divmod(ns,1_000_000_000)
        points.append(point)
    if max(abs(v) for v in points[-1].velocities)>.001:raise ValueError('path does not end stopped')
    return points


DEPTH_TOPIC='/camera/depth/image_raw/compressedDepth'


def decode_depth(msg):
    """Metres from a '16UC1; compressedDepth png' frame: a 12-byte ConfigHeader, then a lossless PNG."""
    import cv2
    if not msg.format.startswith('16UC1') or 'compressedDepth' not in msg.format:raise ValueError('expected 16UC1 compressedDepth')
    depth=cv2.imdecode(np.frombuffer(msg.data,np.uint8,offset=12),cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype!=np.uint16 or depth.ndim!=2:raise ValueError('invalid compressedDepth png')
    return depth*.001


class BallSequence:
    def __init__(self, game):
        self.g=game;self.robot=game.robot
        self.enabled=param(game,'ball_full_sequence',False)
        self.nominal=param(game,'allow_nominal_tcp_trial',False)
        self.table_id=param(game,'grasp_table_object','cyberdog_observed_table')
        self.continue_on_loss=param(game,'grasp_continue_on_loss',True)
        self.max_attempts=param(game,'grasp_max_attempts',3)
        if type(self.max_attempts) is not int or not 1<=self.max_attempts<=10:raise ValueError('grasp_max_attempts must be 1..10')
        self.retry_delay=finite(param(game,'grasp_retry_delay_sec',1.),.2,10.,'grasp retry delay')
        self.attempt_timeout=finite(param(game,'grasp_attempt_timeout_sec',120.),10.,300.,'grasp attempt timeout')
        self.attempts=0;self.attempt_deadline=0.;self.retry_at=None;self.retry_deadline=0.;self.retry_reason=''
        self.target_shift=None;self.closing_started=False;self.loss_announced=False
        self.gate=WorldStillGate(game.ball_still_sec)
        self.depth=deque(maxlen=12);self.pending=deque(maxlen=8);self.reasons=Counter();self.fits=0;self.still_sec=0.;self.info=None;self.depth_info=None;self.joints=None
        self.target=None;self.target_at=0.;self.stable=False;self.last_stamp=0.
        self.last_reason='';self.waiting_reason=''
        self.running=False;self.phase='';self.future=None;self.handle=None;self.submitting=False
        self.cancelled=False;self.feedback_at=0.;self.feedback=None;self.index=0;self.waypoints=[]
        self.deadline=0.;self.settled_since=None;self.frozen=None;self.frozen_radius=None;self.geometry={};self.generation=0
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,game)
        self.plan=game.create_client(GetCartesianPath,'/compute_cartesian_path')
        self.scene=game.create_client(GetPlanningScene,'/get_planning_scene')
        self.action=ActionClient(game,FollowJointTrajectory,'/'+self.robot.modes.pos+'/follow_joint_trajectory')
        for key,topic,typ in [('depth',DEPTH_TOPIC,CompressedImage),('info','/camera/color/camera_info',CameraInfo),('depth_info','/camera/depth/camera_info',CameraInfo),('joints','/joint_states',JointState)]:
            game.create_subscription(typ,topic,lambda m,k=key:self.receive(k,m),qos_profile_sensor_data)
        game.create_subscription(JointTrajectoryControllerState,'/'+self.robot.modes.pos+'/controller_state',self.receive_feedback,qos_profile_sensor_data)

    @property
    def busy(self):return self.running or self.submitting or self.handle is not None or self.retry_at is not None

    def reset_attempts(self):
        self.attempts=0;self.retry_reason=''

    def receive(self,key,msg):
        if key=='depth':self.depth.append(msg);self.locate_pending()
        else:setattr(self,key,msg)
    def receive_feedback(self,msg):
        try:
            indices=[msg.joint_names.index(j) for j in JOINTS]
            q=[msg.feedback.positions[i] for i in indices]
            ref=[msg.reference.positions[i] for i in indices]
            if not all(math.isfinite(v) for v in q+ref):return
            if abs(self.g.get_clock().now().nanoseconds*1e-9-self.stamp(msg))>.25:return
            self.feedback=(q,max(abs(a-b) for a,b in zip(q,ref)))
            self.feedback_at=time.monotonic()
        except (ValueError,IndexError):return

    @staticmethod
    def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

    def observe(self,d):
        if not self.enabled or self.g.mode!='ball':return
        try:
            stamp=d['stamp'];now=self.g.get_clock().now().nanoseconds*1e-9
            if stamp<=self.last_stamp:return
            self.last_stamp=stamp
            fresh=d.get('valid') is True and 0<=now-stamp<=.15 and (self.robot.mock or d.get('source')=='camera')
            if not fresh or not d.get('ball'):
                self.pending.clear()
                self.reject(ValueError('target unavailable'),transient=fresh)
                return
            if self.robot.mock:
                self.accept(dict(center=[.3,0.,.1],radius=.025),stamp)
            else:
                if not self.depth or self.info is None or self.depth_info is None:raise ValueError('RGB-D calibration unavailable')
                self.pending.append(d);self.locate_pending()
        except Exception as exc:
            self.reject(exc)

    def reject(self,exc,*,transient=False):
        self.last_reason=str(exc) if isinstance(exc,ValueError) and str(exc) else type(exc).__name__
        self.reasons[self.last_reason]+=1;self.still_sec=0.
        self.stable=False;self.target=None;self.waiting_reason=''
        # Missing data cannot start a new attempt. Once begun, the frozen target
        # belongs to this bounded attempt and survives detector/fit dropouts.
        # Never erase positive evidence of movement (target_shift) here.
        if not transient:self.gate.reset()

    def restart_count(self):
        self.reasons.clear();self.fits=0;self.still_sec=0.;self.last_reason='';self.waiting_reason='';self.geometry={}

    def status(self):
        return dict(accepted=self.fits,still_sec=round(self.still_sec,2),dwell_sec=self.g.ball_still_sec,
            target_valid=bool(self.target_valid()),last_reason=self.waiting_reason or self.last_reason,rejected=dict(self.reasons),grasp_geometry=self.geometry,
            attempt=self.attempts,max_attempts=self.max_attempts,retry_pending=self.retry_at is not None,retry_reason=self.retry_reason,
            using_frozen_target=bool(self.running and self.continue_on_loss and not self.closing_started
                and self.target_shift is None and not self.target_valid()))

    def accept(self,target,stamp):
        age=max(0.,self.g.get_clock().now().nanoseconds*1e-9-stamp)
        self.target=target;self.target_at=time.monotonic()-age
        if self.running and not self.closing_started and self.frozen is not None and age<=.2:
            distance=float(np.linalg.norm(np.asarray(target['center'])-self.frozen))
            if distance>=.01:self.target_shift=distance
        self.last_reason='';self.waiting_reason=''
        self.stable=self.gate.update(target['center'],stamp)
        self.fits+=1;self.still_sec=stamp-self.gate.since if self.gate.since is not None else 0.

    def locate_pending(self):
        """Pair each ball observation with the depth frame of its own stamp; the compressed depth can arrive after it."""
        if not self.enabled or self.g.mode!='ball':self.pending.clear()
        now=self.g.get_clock().now().nanoseconds*1e-9
        while self.pending:
            d=self.pending[0];stamp=d['stamp']
            if not 0<=now-stamp<=.15:
                self.pending.popleft();self.reject(ValueError('RGB-D/TF sync timeout'),transient=True);continue
            if not self.depth or max(self.stamp(x) for x in self.depth)<stamp-.02:
                self.waiting_reason='waiting for depth';return
            m=min(self.depth,key=lambda x:abs(self.stamp(x)-stamp))
            if abs(self.stamp(m)-stamp)>.05:
                self.pending.popleft();self.reject(ValueError('depth frame missing'),transient=True);continue
            try:
                if self.info is None or self.depth_info is None:raise ValueError('RGB-D calibration unavailable')
                if m.header.frame_id!=self.info.header.frame_id or m.header.frame_id!=self.depth_info.header.frame_id:raise ValueError('depth not aligned')
                if any(info.distortion_model not in ('','plumb_bob','rational_polynomial') for info in (self.info,self.depth_info)):
                    raise ValueError('unsupported RGB-D distortion model')
                # TF commonly arrives after RGB-D. Keep this exact-stamp pair
                # queued and retry on the next tick; never use latest TF instead.
                try:
                    tr=self.buffer.lookup_transform('base_link',m.header.frame_id,Time.from_msg(m.header.stamp)).transform
                except TransformException:
                    self.waiting_reason='waiting for timestamped TF';return
                depth=decode_depth(m)
                if depth.shape!=(self.depth_info.height,self.depth_info.width):raise ValueError('depth calibration size mismatch')
                target=locate_ball(depth,self.info.k,d['ball'],[tr.translation.x,tr.translation.y,tr.translation.z],
                    [tr.rotation.x,tr.rotation.y,tr.rotation.z,tr.rotation.w],distortion=self.info.d,
                    depth_k=self.depth_info.k,depth_distortion=self.depth_info.d,color_size=(self.info.height,self.info.width))
                self.pending.popleft();self.accept(target,stamp)
            except Exception as exc:
                self.pending.popleft();self.reject(exc)

    def target_valid(self):
        return self.target is not None and time.monotonic()-self.target_at<=.2 and (
            self.frozen is None or np.linalg.norm(np.array(self.target['center'])-self.frozen)<.01)

    def begin(self):
        if self.busy or not self.stable or not self.target_valid():return
        if self.attempts>=self.max_attempts:
            self.g.pause(f'抓球已尝试 {self.attempts} 次，请复位后重试');return
        if not self.g.grasp_enabled or self.robot.gripper_open_only or (not self.robot.mock and (not self.nominal or not self.robot.gripper_enabled)):
            self.g.pause('完整抓取配置未启用：需明确名义 TCP 试验和夹爪配置');return
        self.robot.hold();self.running=True;self.cancelled=False;self.generation=self.g.generation
        self.attempts+=1;self.attempt_deadline=time.monotonic()+self.attempt_timeout
        self.target_shift=None;self.closing_started=False;self.loss_announced=False
        self.frozen=list(self.target['center']);self.frozen_radius=self.target['radius'];self.index=0;self.settled_since=None
        self.phase='settle';self.deadline=time.monotonic()+5
        self.g.set_state('active','grasp_settling',f'抓球第 {self.attempts}/{self.max_attempts} 次：锁定球心，检查完整路径')

    def fail(self,reason,*,retryable=False):
        if retryable and not self.closing_started and self.attempts<self.max_attempts:
            # Invalidate motion/gripper callbacks before cancellation. Old action
            # handles must reach a terminal result before reacquiring the arm.
            self.g.generation+=1;self.g.pending=None
            self.stop();self.robot.stop()
            self.retry_reason=reason;self.retry_at=time.monotonic()+self.retry_delay
            self.retry_deadline=self.retry_at+10.;self.phase='retry_stop'
            self.g.set_state('active','grasp_retry_wait',f'第 {self.attempts} 次未完成：{reason}；停稳后重新定位，准备第 {self.attempts+1}/{self.max_attempts} 次')
            return
        exhausted=f'（已尝试 {self.attempts}/{self.max_attempts} 次）' if retryable else ''
        self.g.pause('抓球暂停'+exhausted+'：'+reason)

    def check_target(self):
        if self.closing_started:return
        if self.target_shift is not None:
            raise RetryableGraspError(f'球心偏移 {self.target_shift*1000:.1f} mm，重新定位')
        fresh=self.target is not None and time.monotonic()-self.target_at<=.2
        if fresh and self.frozen is not None:
            distance=float(np.linalg.norm(np.asarray(self.target['center'])-self.frozen))
            if distance>=.01:raise RetryableGraspError(f'球心偏移 {distance*1000:.1f} mm，重新定位')
        if self.target_valid():return
        if not self.continue_on_loss:raise RetryableGraspError(self.target_failure())
        if self.frozen is None or self.frozen_radius is None:raise ValueError('缺少本轮锁定的球心')
        if not self.loss_announced:
            self.loss_announced=True
            self.g.get_logger().info('目标暂不可见，继续按本轮锁定球心和完整规划路径抓取：'+(self.last_reason or '观测过期'))

    def retry_tick(self,now):
        if now>self.retry_deadline:
            self.g.pause('抓球重试暂停：未确认上一动作停止或关节稳定');return
        if now<self.retry_at or self.submitting or self.handle is not None or self.robot.busy:return
        if not self.robot.fresh() or not self.robot.stationary():return
        if not self.robot.mock and self.robot.modes.would_pass()!='':return
        self.retry_at=None;self.phase='reacquiring'
        # begin_tracking resets the image predictor, 3 s dwell and tracking
        # anchor to the measured stopped pose; it does not reset attempt count.
        self.g.begin_tracking()

    def target_failure(self):
        if self.target is None:return '目标定位丢失：'+(self.waiting_reason or self.last_reason or '无有效球心')
        age=time.monotonic()-self.target_at
        if age>.2:return f'目标定位过期：{age*1000:.0f} ms'
        distance=np.linalg.norm(np.asarray(self.target['center'])-self.frozen)
        return f'球心偏移 {distance*1000:.1f} mm，超过 10 mm'

    def stop(self):
        self.cancelled=True;self.running=False;self.stable=False;self.gate.reset();self.frozen=None
        self.retry_at=None;self.target_shift=None
        self.pending.clear();self.target=None;self.waypoints=[]
        if self.future is not None:self.future.cancel();self.future=None
        if self.handle is not None:self.handle.cancel_goal_async()
        # A late accepted goal is cancelled by accepted(); busy stays latched until terminal.

    def request_plan(self,waypoints,phase):
        if self.joints is None or not 0<=self.g.get_clock().now().nanoseconds*1e-9-self.stamp(self.joints)<=.2:
            raise ValueError('joint feedback stale')
        if not self.plan.service_is_ready():raise RetryableGraspError('MoveIt unavailable')
        req=GetCartesianPath.Request();req.header.frame_id='base_link';req.start_state.joint_state=self.joints;req.start_state.is_diff=True
        req.group_name='zephyr_arm_tcp';req.link_name='omni_picker_tcp';req.max_step=.003
        req.jump_threshold=2.;req.revolute_jump_threshold=.15;req.avoid_collisions=True
        req.max_velocity_scaling_factor=.05;req.max_acceleration_scaling_factor=.05
        for w in waypoints:
            p=Pose();p.position.x,p.position.y,p.position.z=w['position'];p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w=w['orientation'];req.waypoints.append(p)
        self.future=self.plan.call_async(req);self.phase=phase;self.deadline=time.monotonic()+10

    def next_segment(self):
        self.settled_since=None
        self.phase='segment_settle';self.deadline=time.monotonic()+5

    def accepted(self,future):
        self.submitting=False
        try:
            handle=future.result()
            if not handle.accepted:
                self.robot.sequence_motion_busy=False
                if not self.cancelled:self.fail('trajectory rejected',retryable=True)
                return
            self.handle=handle
            handle.get_result_async().add_done_callback(self.result)
            if self.cancelled or self.g.generation!=self.generation:handle.cancel_goal_async()
        except Exception:self.submitting=True;self.fail('trajectory submission outcome unknown')

    def result(self,future):
        self.handle=None;self.robot.sequence_motion_busy=False
        if self.cancelled or self.g.generation!=self.generation:return
        try:
            result=future.result()
            if result.status!=GoalStatus.STATUS_SUCCEEDED or result.result.error_code!=0:raise ValueError('trajectory failed')
            self.phase='arrival';self.settled_since=None;self.deadline=time.monotonic()+5
        except Exception as exc:self.fail(str(exc))

    def arrived(self):
        if self.cancelled:return
        stage=self.waypoints[self.index]['stage']
        if stage=='grasp':
            # Arrival can run in an action callback before the next timer tick.
            try:
                if self.attempt_deadline and time.monotonic()>self.attempt_deadline:raise RetryableGraspError('本轮抓取超时，重新定位')
                self.check_target()
            except ValueError as exc:
                self.fail(str(exc),retryable=isinstance(exc,RetryableGraspError));return
            self.closing_started=True
            self.phase='closing';self.deadline=time.monotonic()+12
            self.g.set_state('active','grasp_closing','已到抓取点，闭爪')
            self.g.issue(lambda cb:self.robot.grip(True,cb),self.closed)
        elif stage=='lift':self.complete()
        else:self.index+=1;self.next_segment()

    def closed(self):
        if self.cancelled:return
        self.index+=1;self.next_segment()

    def complete(self):
        self.running=False;self.phase='complete';self.frozen=None;self.stable=False;self.gate.reset()
        self.g.outcome='simulated_grasp' if self.robot.mock else 'grasp_attempt_unverified'
        self.g.set_state('paused','complete','抓取动作结束并保持；点击复位重新开始')

    def tick(self):
        self.locate_pending()
        if not self.running:
            if self.retry_at is not None:
                self.retry_tick(time.monotonic());return
            stamp=self.g.get_clock().now().nanoseconds*1e-9
            if self.gate.last is not None and stamp-self.gate.last>.2:
                self.gate.reset();self.still_sec=0.;self.stable=False;self.target=None
            return
        now=time.monotonic()
        try:
            if self.attempt_deadline and now>self.attempt_deadline:raise RetryableGraspError('本轮抓取超时，重新定位')
            if now>self.deadline:
                error=RetryableGraspError if self.phase in ('scene','full_plan','segment_plan') else ValueError
                raise error(self.phase+' timeout')
            self.check_target()
            if self.robot.mock:
                self.mock_tick(now);return
            if not self.robot.fresh() or now-self.feedback_at>.25:raise ValueError('joint/controller feedback stale')
            if self.feedback[1]>.06:raise ValueError('position following error')
            if self.g.count_publishers('/'+self.robot.modes.pos+'/joint_trajectory')>int(self.robot.position_pub is not None):raise ValueError('another position publisher')
            if self.phase in ('settle','segment_settle'):
                if not self.robot.stationary():return
                if self.settled_since is None:self.settled_since=now
                if now-self.settled_since<.5:return
                self.future=self.robot.modes.list.call_async(ListControllers.Request())
                self.phase='guard_all' if self.phase=='settle' else 'guard_segment';return
            if self.future is not None:
                if not self.future.done():return
                result=self.future.result();self.future=None
                if self.phase.startswith('guard_'):
                    reason=self.robot.modes.guard(result.controller)
                    if reason or observed(result.controller,self.robot.motion.prefix)!='position':raise ValueError(reason or 'position controller required')
                    if self.phase=='guard_all':
                        if not self.scene.service_is_ready():raise RetryableGraspError('planning scene unavailable')
                        req=GetPlanningScene.Request();req.components.components=PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                        self.future=self.scene.call_async(req);self.phase='scene';return
                    self.request_plan([self.waypoints[self.index]],'segment_plan');return
                if self.phase=='scene':
                    objects=result.scene.world.collision_objects
                    if not self.table_id or not any(o.id==self.table_id and (o.primitives or o.meshes) for o in objects):raise ValueError('缺少桌面碰撞模型')
                    tcp=self.buffer.lookup_transform('base_link','omni_picker_tcp',Time()).transform
                    q=[tcp.rotation.x,tcp.rotation.y,tcp.rotation.z,tcp.rotation.w]
                    position=[tcp.translation.x,tcp.translation.y,tcp.translation.z]
                    try:self.waypoints=preview_waypoints(self.frozen,self.frozen_radius,orientation=q,tcp_position=position)
                    except ValueError as exc:raise RetryableGraspError(str(exc)) from exc
                    relative=rotation(q).T@(np.asarray(self.frozen)-position)
                    delta=rotation(q).T@(np.asarray(self.waypoints[0]['position'])-position)
                    self.geometry=dict(frame='base_link',tcp=position,ball=self.frozen,radius=self.frozen_radius,
                        ball_in_tcp=relative.tolist(),first_step_in_tcp=delta.tolist(),waypoints=self.waypoints)
                    self.g.get_logger().info(f'抓取几何：球心相对 TCP (mm)={np.round(relative*1000,1).tolist()}；首段位移 (mm)={np.round(delta*1000,1).tolist()}，+Z 为夹爪前方')
                    self.request_plan(self.waypoints,'full_plan');return
                if self.phase in ('full_plan','segment_plan'):
                    if result.error_code.val!=1 or not math.isfinite(result.fraction) or result.fraction<.999:raise RetryableGraspError('抓取路径不完整，未执行剩余路径')
                    points=timed_points(result.solution.joint_trajectory,self.robot.current)
                    if self.phase=='full_plan':
                        self.phase='opening';self.deadline=now+12
                        self.g.issue(lambda cb:self.robot.grip(False,cb),self.next_segment);return
                    if not self.action.server_is_ready():raise ValueError('position action unavailable')
                    goal=FollowJointTrajectory.Goal();goal.trajectory.joint_names=JOINTS;goal.trajectory.points=points
                    self.goal=list(points[-1].positions);self.phase='moving';self.submitting=True;self.robot.sequence_motion_busy=True
                    self.deadline=now+points[-1].time_from_start.sec+points[-1].time_from_start.nanosec*1e-9+5
                    self.g.set_state('active','grasp_'+self.waypoints[self.index]['stage'],'执行抓球阶段：'+self.waypoints[self.index]['stage'])
                    self.action.send_goal_async(goal).add_done_callback(self.accepted);return
            if self.phase=='arrival':
                arrived=self.robot.stationary() and max(abs(a-b) for a,b in zip(self.robot.current,self.goal))<.01
                self.settled_since=(self.settled_since or now) if arrived else None
                if self.settled_since is not None and now-self.settled_since>.3:self.arrived()
        except Exception as exc:self.fail(str(exc),retryable=isinstance(exc,RetryableGraspError))

    def mock_tick(self,now):
        # Mock exercises the same approach/grasp/close/lift ordering without hardware.
        if self.phase=='settle':
            self.waypoints=preview_waypoints(self.frozen,self.frozen_radius)
            self.phase='opening';self.deadline=now+5
            self.g.issue(lambda cb:self.robot.grip(False,cb),self.next_segment)
        elif self.phase=='segment_settle':
            self.phase='mock_moving';self.deadline=now+5
            self.g.set_state('active','grasp_'+self.waypoints[self.index]['stage'],'模拟 '+self.waypoints[self.index]['stage'])
            self.g.move([self.robot.current],self.arrived)

    def destroy(self):self.action.destroy()
