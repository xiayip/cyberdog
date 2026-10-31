"""Cancel placement on measured empty closure, then bow from the stopped pose."""
import math
import time

from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState

from .common import finite, param
from .motion import JOINTS, LOWER, bounded_start


class BallTakenReaction:
    def __init__(self,game):
        self.g=game;self.robot=game.robot
        self.threshold=finite(param(game,'grasp_taken_threshold',.92),0.,1.,'ball taken threshold')
        if self.threshold>=self.robot.gripper_close_position:
            raise ValueError('ball taken threshold must be below gripper close position')
        self.delta=finite(param(game,'grasp_taken_bow_delta',.45),.02,.5,'ball taken J4 bow')
        self.speed=finite(param(game,'grasp_taken_bow_speed',min(.6,.4*self.robot.motion.limit_scale)),
            .01,.4*self.robot.motion.limit_scale,'ball taken bow speed')
        self.speed_scale=finite(param(game,'grasp_taken_bow_speed_scale',3.),1.,3.,'ball taken bow speed scale')
        self.validity=game.create_client(GetStateValidity,'/check_state_validity')
        self.phase='';self.future=None;self.deadline=0.;self.position=None
        self.start_pose=None;self.target=None;self.samples=[];self.index=0

    @property
    def busy(self):return bool(self.phase)

    def stop(self):
        self.phase=''
        if self.future is not None:self.future.cancel();self.future=None

    def watch(self):
        seq=self.g.ball_sequence
        if not seq.running or not seq.placement_started or seq.cancelled or self.busy:return False
        position=self.robot.fresh_gripper_position()
        if position is None or position<=self.threshold:return False
        # Invalidate every old placement/release callback before cancelling its
        # owned action. Keep the gripper's existing close command unchanged.
        self.g.pause(f'夹爪实际位置 {position:.3f} > {self.threshold:g}，毛球被拿走，取消放球')
        self.position=position;self.phase='stopping';self.deadline=time.monotonic()+8.
        self.g.outcome='ball_taken'
        self.g.set_state('transition','ball_taken_stopping','毛球被拿走，等待放球轨迹取消并停稳')
        return True

    def tick(self):
        if not self.busy:return
        try:
            if time.monotonic()>self.deadline:raise ValueError('等待取消、停稳或低头完成超时')
            if self.phase=='stopping':
                if self.g.ball_sequence.busy or self.robot.busy:return
                if not self.robot.fresh() or not self.robot.stationary():return
                if not self.robot.mock and self.robot.motion.ready():return
                # Match the held JTC reference so the other five axes retain
                # exactly their current command throughout the inserted motion.
                self.start_pose=bounded_start(self.robot.current if self.robot.mock else self.robot.motion.reference_pose)
                self.target=list(self.start_pose)
                self.target[3]=max(min(self.start_pose[3],LOWER[3]+.05),self.start_pose[3]-self.delta)
                if self.start_pose[3]-self.target[3]<.005:
                    raise ValueError('J4 已接近下限，保持当前位置')
                if self.robot.mock:self.bow();return
                if not self.validity.service_is_ready():raise ValueError('低头路径碰撞检查服务不可用')
                count=max(1,math.ceil((self.start_pose[3]-self.target[3])/.02))
                self.samples=[bounded_start(self.robot.current)]+[
                    [a+(b-a)*i/count for a,b in zip(self.start_pose,self.target)] for i in range(count+1)]
                self.index=0;self.phase='checking';self.deadline=time.monotonic()+10.
                self.g.set_state('transition','ball_taken_checking','已停稳，检查仅 J4 向下的低头路径')
            if self.phase=='checking':
                if self.future is not None:
                    if not self.future.done():return
                    result=self.future.result();self.future=None
                    if not result.valid:raise ValueError('低头路径碰撞检查未通过，保持当前位置')
                    self.index+=1
                if self.index==len(self.samples):self.bow();return
                position=self.robot.fresh_gripper_position()
                if position is None:raise ValueError('夹爪反馈过期，无法检查低头路径')
                req=GetStateValidity.Request();req.group_name='zephyr_arm_tcp'
                req.robot_state=RobotState(is_diff=True,joint_state=JointState(
                    name=JOINTS+[self.robot.gripper_joint],position=self.samples[self.index]+[position]))
                self.future=self.validity.call_async(req)
        except Exception as exc:self.g.pause('毛球被拿走后保持：'+str(exc))

    def bow(self):
        if not self.busy:return
        if not self.robot.fresh() or not self.robot.stationary():
            raise ValueError('低头前关节尚未停稳')
        if not self.robot.mock:
            reference=self.robot.motion.reference_pose
            if self.robot.motion.ready() or reference is None or max(abs(a-b) for a,b in zip(reference,self.start_pose))>.001:
                raise ValueError('低头检查期间保持姿态改变')
        self.phase='bowing';self.deadline=time.monotonic()+12.
        self.g.set_state('transition','ball_taken_bowing',f'毛球被拿走，从当前姿态低头，J4 下转 {self.start_pose[3]-self.target[3]:.2f} rad，{self.speed_scale:g} 倍速')
        if self.busy:self.g.issue(lambda cb:self.robot.bow(self.target,cb,
            speed=self.speed,speed_scale=self.speed_scale),self.finished)

    def finished(self):
        if not self.busy:return
        self.phase=''
        self.g.set_state('paused','ball_taken','毛球被拿走，已低头并保持；点击复位重新开始')
