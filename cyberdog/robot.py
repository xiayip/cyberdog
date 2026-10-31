"""Hardware adapter for the game: only confirmed Rebot controller/action interfaces."""
import math
import time
from action_msgs.msg import GoalStatus, GoalStatusArray
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_action_status_default
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from builtin_interfaces.msg import Duration
from .common import param, finite
from .controller_modes import ControllerModes
from .game_core import path_points, pose
from .motion import Motion, LOWER, UPPER, bounded_start
from .tracking import tracking_step


class Robot:
    def __init__(self,node):
        self.node=node
        self.motion=Motion(node,self.move_done)
        self.mock=self.motion.backend=='mock'
        if self.motion.backend not in ('mock','trajectory'):
            raise ValueError('cyberdog requires mock or trajectory motion_backend')
        self.mock_pose=[0.,.6,.8,0.,0.,0.]
        self.modes=ControllerModes(node,self.motion)
        if not self.mock and self.motion.action_name!='/'+self.modes.pos+'/follow_joint_trajectory':
            raise ValueError('trajectory_action must be the position trajectory controller')
        self.command_pub=node.create_publisher(JointTrajectoryPoint,'/'+self.modes.imp+'/commands',1)
        self.position_pub=None
        self.sequence_motion_busy=False
        self.servo=False
        self.reference=None
        self.tracking_anchor=None
        self.tracking_velocity=[0.]*6
        self.ball_velocity_feedforward=param(node,'ball_velocity_feedforward',False)
        self.tracking_feedforward=False
        self.tracking_speed=.2*self.motion.limit_scale
        # The old direct tracker had no acceleration ramp. Use the ordinary
        # motion acceleration envelope as an explicit new tracking bound.
        self.tracking_acceleration=.65*self.motion.limit_scale
        self.ball_tracking_speed=finite(param(node,'ball_tracking_speed',self.tracking_speed),.1,.9,'ball tracking speed')
        self.ball_tracking_acceleration=finite(param(node,'ball_tracking_acceleration',self.tracking_acceleration),.1,1.5,'ball tracking acceleration')
        self.tracking_yaw_range=finite(param(node,'ball_tracking_yaw_range',1.2),.1,3.14,'tracking yaw range')
        self.tracking_pitch_range=finite(param(node,'ball_tracking_pitch_range',1.0),.1,3.14,'tracking pitch range')
        self.callback=None
        self.move_target=None
        self.verifying=None
        self.fault=''
        self.error_since=None
        self.gripper_enabled=param(node,'gripper_enabled',False)
        self.gripper_open_only=param(node,'gripper_open_only',False)
        self.gripper_joint=param(node,'gripper_joint','gripper_command_joint')
        self.gripper_name=param(node,'gripper_action','/rebot_arm_rs_gripper_controller/gripper_cmd')
        self.gripper=ActionClient(node,ParallelGripperCommand,self.gripper_name)
        self.gripper_other_busy=False
        self.gripper_matched_since=None
        self.gripper_status_sub=node.create_subscription(GoalStatusArray,self.gripper_name+'/_action/status',
            self.gripper_status,qos_profile_action_status_default)
        self.grip_busy,self.grip_handle,self.grip_cancel=False,None,False
        self.grip_closed=False
        self.grip_target=0.;self.grip_allow_stall=False
        self.grip_callback=None
        self.grip_deadline=0.
        self.mock_delay=param(node,'mock_step_sec',.3)

    @property
    def busy(self):return self.sequence_motion_busy or self.motion.busy or self.modes.busy or self.grip_busy or self.verifying is not None

    def set_tracking_profile(self,ball):
        self.tracking_feedforward=ball and self.ball_velocity_feedforward
        self.tracking_speed=self.ball_tracking_speed if ball else .2*self.motion.limit_scale
        self.tracking_acceleration=self.ball_tracking_acceleration if ball else .65*self.motion.limit_scale
        self.tracking_velocity=[0.]*6

    @property
    def current(self):return list(self.mock_pose if self.mock else self.motion.pose) if (self.mock or self.motion.pose is not None) else None

    def fresh(self):return self.mock or (self.current is not None and time.monotonic()-self.motion.pose_at<.5)

    def stationary(self):return self.mock or (self.fresh() and self.motion.stationary)

    def mode(self,target,callback,*,tracking=True):
        if self.busy:return False,'robot busy'
        if not self.mock and self.node.count_publishers('/'+self.modes.imp+'/commands')>1:
            return False,'another direct impedance publisher exists'
        if target=='servo' and self.fresh() and (self.mock or self.motion.enabled):
            # Prime the subscriber exactly as ghost_game does before unchaining JTC.
            self.publish_reference(self.current)
        def done(ok,message):
            self.servo=ok and tracking and target in ('servo','position')
            if self.servo:
                self.reference=self.current
                self.tracking_anchor=list(self.reference)
                self.tracking_velocity=[0.]*6
                if target=='position' and self.position_pub is None:
                    self.position_pub=self.node.create_publisher(JointTrajectory,'/'+self.modes.pos+'/joint_trajectory',1)
                    self.modes.owned_position_publisher=True
                    self.motion.own_publishers['/'+self.modes.pos+'/joint_trajectory']=1
                self.publish_reference(self.reference)
            callback(ok,message)
        self.servo=False
        return self.modes.request(target,done)

    def move(self,targets,callback,speed=.12,minimum=1.):
        if self.busy or self.servo:return False,'robot busy or servo active'
        if not self.fresh():return False,'joint feedback unavailable'
        if not math.isfinite(speed) or not 0<speed<=.4*self.motion.limit_scale:
            return False,'path speed exceeds ordinary motion envelope'
        try:
            targets=[pose(p) for p in targets]
            path_points(bounded_start(self.current),targets,speed,minimum,.65*self.motion.limit_scale)
        except ValueError as exc:return False,str(exc)
        self.callback,self.move_target=callback,targets[-1]
        accepted,reason=self.motion.start_path(lambda current:path_points(current,targets,speed,minimum,.65*self.motion.limit_scale))
        if self.mock and accepted:self.motion.deadline=time.monotonic()+self.mock_delay
        if not accepted:self.callback=None
        return accepted,reason

    def move_done(self,ok,message):
        if self.mock and ok:self.mock_pose=list(self.move_target)
        if not self.mock and ok:
            # JointState and action result are separate streams. Wait for measured
            # arrival/settling, rather than treating the first result callback as proof.
            self.verifying=(time.monotonic(),None)
            return
        self.move_complete(ok,message)

    def move_complete(self,ok,message):
        callback,self.callback=self.callback,None
        if callback:callback(ok,message)

    def publish_reference(self,target,velocity=None):
        if self.mock:
            self.mock_pose=list(target)
        elif self.motion.enabled and self.modes.mode=='position':
            self.publish_position_reference(target,velocity)
        elif self.motion.enabled:
            self.command_pub.publish(JointTrajectoryPoint(positions=list(target),velocities=list(velocity) if velocity is not None else [0.]*6))

    def publish_position_reference(self,target,velocity=None):
        # Position JTC rolls these finite trajectories; every update carries its
        # own braking tail so loss of the publisher never leaves a moving endpoint.
        if self.position_pub is None:return
        v=[x if abs(x)>=1e-6 else 0. for x in velocity] if velocity is not None else [0.]*6
        a=self.tracking_acceleration
        def point(t,q,dq):
            ns=round(t*1e9)
            return JointTrajectoryPoint(positions=list(q),velocities=list(dq),
                time_from_start=Duration(sec=ns//1000000000,nanosec=ns%1000000000))
        points=[point(.025,target,v)]
        # Nanosecond-quantized stop times avoid duplicate ROS timestamps and
        # floating-point residual velocity at the terminal point.
        stops=[math.ceil(abs(x)/a*1e9)/1e9 if x else 0. for x in v]
        for stop in sorted(set(t for t in stops if t)):
            q=[];dq=[]
            for x,speed,duration in zip(target,v,stops):
                t=min(stop,duration);decel=speed/duration if duration else 0.
                q.append(x+speed*t-.5*decel*t*t)
                dq.append(speed*(1.-t/duration) if duration else 0.)
            points.append(point(.025+stop,q,dq))
        points[-1].velocities=[0.]*6
        self.position_pub.publish(JointTrajectory(joint_names=['joint'+str(i) for i in range(1,7)],points=points))

    def track(self,target,dt):
        if not self.servo or not self.fresh():return False
        target=pose(target)
        if self.reference is None:self.reference=self.current
        if not math.isfinite(dt) or dt<=0:return False
        offsets=[0.,0.,0.,self.tracking_pitch_range,self.tracking_yaw_range,0.]
        anchor=self.tracking_anchor or self.reference
        # Use the same envelope and joint margins as the image servo. Fixed
        # axes may be at zero reset; only the two tracking axes get a margin.
        lower=[max(min(lo+(.05 if d else 0.),a),a-d) for lo,a,d in zip(LOWER,anchor,offsets)]
        upper=[min(max(hi-(.05 if d else 0.),a),a+d) for hi,a,d in zip(UPPER,anchor,offsets)]
        try:
            self.reference,self.tracking_velocity=tracking_step(
                self.reference,self.tracking_velocity,target,min(dt,.05),
                self.tracking_speed,self.tracking_acceleration,lower,upper)
        except ValueError as exc:
            self.hold();self.fault=str(exc);return False
        self.publish_reference(self.reference,self.tracking_velocity if self.tracking_feedforward or self.modes.mode=='position' else None)
        return True

    def hold(self):
        was_tracking=self.servo
        self.tracking_velocity=[0.]*6
        self.servo=False
        # A completed JTC action already holds its endpoint. Re-anchoring while
        # idle would replace that target and invalidate the evidence for settling.
        if was_tracking and self.fresh() and self.modes.mode in ('servo','position'):
            self.reference=self.current
            self.publish_reference(self.reference)
            if not self.mock:
                self.motion.invalidate_reference()
                for windows in self.modes.settling.values():
                    for window in windows:window.reset()
                self.modes.telemetry={name:(0.,False) for name in self.modes.telemetry}

    def grip(self,closed,callback):
        return self.grip_position(1. if closed else 0.,callback,allow_stall=bool(closed))

    def grip_position(self,position,callback,*,allow_stall=False):
        try:position=finite(position,0.,1.,'gripper position must be 0..1')
        except ValueError as exc:return False,str(exc)
        if position>0 and self.gripper_open_only:return False,'gripper is configured to remain open'
        if self.busy:return False,'robot busy'
        if not self.mock and (not self.motion.enabled or not self.gripper_enabled):return False,'gripper disabled'
        if not self.mock and not self.gripper.server_is_ready():return False,'gripper action unavailable'
        if not self.mock and (self.gripper_other_busy or self.gripper_matched_since is None or time.monotonic()-self.gripper_matched_since<1.):
            return False,'gripper status unmatched or another goal is active'
        self.grip_busy,self.grip_cancel,self.grip_callback=True,False,callback
        self.grip_target=position;self.grip_closed=position>0;self.grip_allow_stall=allow_stall
        self.grip_deadline=time.monotonic()+(self.mock_delay if self.mock else 5. if position>0 else 10.)
        if self.mock:return True,'mock accepted'
        goal=ParallelGripperCommand.Goal()
        goal.command.name=[self.gripper_joint]
        goal.command.position=[position]
        goal.command.velocity=[.15]
        if position==0:goal.command.effort=[.4]
        try:self.gripper.send_goal_async(goal).add_done_callback(self.grip_accepted)
        except Exception:self.fault='gripper submission unknown'
        return True,'accepted'

    def gripper_status(self,msg):
        self.gripper_other_busy=any(s.status in (0,1,2,3) for s in msg.status_list)

    def grip_accepted(self,future):
        try:
            self.grip_handle=future.result()
            if not self.grip_handle.accepted:
                self.grip_finish(False,'gripper rejected');return
            self.grip_handle.get_result_async().add_done_callback(self.grip_result)
            if self.grip_cancel:self.grip_handle.cancel_goal_async()
        except Exception:self.fault='gripper acceptance unknown'

    def grip_result(self,future):
        try:
            wrapped=future.result()
            ok=wrapped.status==GoalStatus.STATUS_SUCCEEDED and not self.grip_cancel
            result=wrapped.result
            ok=ok and (result.reached_goal or (self.grip_allow_stall and result.stalled))
            self.grip_finish(ok,'stalled' if result.stalled else 'reached_goal' if result.reached_goal else 'unverified')
        except Exception:self.fault='gripper result unknown'

    def grip_finish(self,ok,message):
        self.grip_busy,self.grip_handle=False,None
        callback,self.grip_callback=self.grip_callback,None
        if callback:callback(ok,message)

    def stop(self):
        self.hold()
        self.modes.cancel()
        self.motion.stop()
        if self.verifying is not None:
            self.verifying=None
            self.move_complete(False,'cancelled during measured-pose verification')
        self.grip_cancel=True
        if self.grip_busy and self.mock:self.grip_finish(False,'cancelled')
        elif self.grip_handle is not None:self.grip_handle.cancel_goal_async()

    def tick(self):
        self.motion.tick()
        self.modes.tick()
        now=time.monotonic()
        if self.node.count_publishers(self.gripper_name+'/_action/status')==0:
            self.gripper_matched_since=None
        elif self.gripper_matched_since is None:self.gripper_matched_since=now
        if self.verifying is not None:
            started,settled=self.verifying
            # Action success can arrive before feedback/reference have settled.
            # A chained move must meet the same readiness gate as its first move.
            arrived=(self.stationary() and self.motion.ready()==''
                     and max(abs(a-b) for a,b in zip(self.current,self.move_target))<=.08)
            settled=(settled or now) if arrived else None
            self.verifying=(started,settled)
            if settled is not None and now-settled>=.2:
                self.verifying=None
                self.move_complete(True,'measured pose settled')
            elif now-started>5.:
                self.verifying=None
                self.move_complete(False,'post-motion settling failed: '+(
                    self.motion.ready() or 'measured pose did not settle at requested target'))
        if self.grip_busy and now>=self.grip_deadline:
            if self.mock:self.grip_finish(True,'simulated, not physical evidence')
            elif not self.grip_cancel:
                self.grip_cancel=True
                if self.grip_handle is not None:self.grip_handle.cancel_goal_async()
                self.fault='gripper timed out; waiting for terminal result'
        if self.servo and not self.mock:
            if not self.fresh():
                self.hold();self.fault='joint feedback lost during tracking';return
            topic='/'+self.modes.pos+'/joint_trajectory' if self.modes.mode=='position' else '/'+self.modes.imp+'/commands'
            if self.node.count_publishers(topic)>1:
                self.hold();self.fault='another tracking writer detected';return
            error=max(abs(a-b) for a,b in zip(self.reference,self.current))
            if error>.15:
                if self.error_since is None:self.error_since=now
                elif now-self.error_since>.4:
                    self.hold();self.fault='tracking reference diverged from measured pose'
            else:self.error_since=None

    def destroy(self):
        self.gripper.destroy()
        self.motion.destroy()
