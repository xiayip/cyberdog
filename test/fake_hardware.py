"""Actual ROS action transport tests; fake server, isolated domain, no hardware."""
import time

import pytest
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from cyberdog.motion import JOINTS


class FakeRobot(Node):
    traj = 'rebot_arm_rs_impedance_trajectory_controller'

    def __init__(self):
        super().__init__('rebot_arm_rs_impedance_trajectory_guard')
        self.group = ReentrantCallbackGroup()
        self.requests, self.bt_requests, self.cancelled = [], [], 0
        self.run_seconds, self.accept_delay = .2, 0.
        self.chain_active, self.emit_telemetry, self.reference_offset = True, True, 0.
        self.list_service = self.create_service(ListControllers, '/controller_manager/list_controllers', self.list)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        self.state_pub = self.create_publisher(JointTrajectoryControllerState,
            '/' + self.traj + '/controller_state', qos_profile_sensor_data)
        self.action = ActionServer(
            self, FollowJointTrajectory,
            '/' + self.traj + '/follow_joint_trajectory',
            self.execute, goal_callback=self.goal, cancel_callback=self.cancel,
            callback_group=self.group)
        self.timer = self.create_timer(.03, self.telemetry, callback_group=self.group)

    def list(self, request, response):
        response.controller = [ControllerState(name=name, state='active' if self.chain_active else 'inactive')
                               for name in ['rebot_arm_rs_mit_controller',
                                            'rebot_arm_rs_impedance_controller',
                                            'rebot_arm_rs_impedance_trajectory_controller']]
        return response

    def telemetry(self):
        if not self.emit_telemetry:
            return
        pose = [0., .6, .8, .1, .1, 0.]
        msg = JointState(name=JOINTS, position=pose, velocity=[0.] * 6)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_pub.publish(msg)
        state = JointTrajectoryControllerState(joint_names=JOINTS)
        state.header.stamp = msg.header.stamp
        state.reference.positions = [x + self.reference_offset for x in pose]
        state.feedback.positions = pose
        state.reference.velocities = state.feedback.velocities = [0.] * 6
        self.state_pub.publish(state)

    def goal(self, request):
        from rclpy.action import GoalResponse
        time.sleep(self.accept_delay)
        return GoalResponse.ACCEPT

    def cancel(self, handle):
        self.cancelled += 1
        return CancelResponse.ACCEPT

    def execute(self, handle):
        self.requests.append(handle.request)
        deadline = time.monotonic() + self.run_seconds
        while time.monotonic() < deadline:
            if handle.is_cancel_requested:
                handle.canceled()
                return FollowJointTrajectory.Result(error_code=0)
            time.sleep(.01)
        handle.succeed()
        return FollowJointTrajectory.Result(error_code=0)

    def destroy_node(self):
        self.action.destroy()
        return super().destroy_node()


def executor():
    return MultiThreadedExecutor(num_threads=4)


"""ROS service/action integration against fake controllers; never a real robot."""
import time
import pytest
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import SwitchController
from rclpy.action import ActionServer
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from rebot_arm_hmi_bridge.protocol import Mode
from cyberdog.controller_modes import BASE,CHAINS,IMP,JTC,POS,observed,switch_plan
from cyberdog.motion import JOINTS


class FakeGameRobot(FakeRobot):
    """Controller manager with the commissioned layout: motions run on the position JTC, idle holds on the impedance chain."""
    traj=POS

    def __init__(self):
        self.controllers={BASE:'active'}
        for names in CHAINS.values():
            for name in names:self.controllers[name]='inactive'
        for name in CHAINS[Mode.POSITION]:self.controllers[name]='active'
        self.position=[0.,.6,.8,.1,.1,0.]
        self.measured_offset=[0.]*6
        self.switch_requests=[]
        self.switch_delay=0.
        self.direct_commands=[]
        self.impedance_goals=[]
        self.state_velocities=None
        self.effort=None
        super().__init__()
        self.imp_state_pub=self.create_publisher(JointTrajectoryControllerState,'/'+JTC+'/controller_state',qos_profile_sensor_data)
        self.imp_action=ActionServer(self,FollowJointTrajectory,'/'+JTC+'/follow_joint_trajectory',self.impedance_goal,callback_group=self.group)
        self.switch_service=self.create_service(SwitchController,'/controller_manager/switch_controller',self.switch_mode,callback_group=self.group)
        self.direct_sub=self.create_subscription(JointTrajectoryPoint,'/'+IMP+'/commands',self.direct,1)

    def impedance_goal(self,handle):
        self.impedance_goals.append(handle.request)
        handle.abort()
        return FollowJointTrajectory.Result(error_code=-1)

    def destroy_node(self):
        self.imp_action.destroy()
        return super().destroy_node()

    def list(self,request,response):
        response.controller=[ControllerState(name=n,state=s) for n,s in self.controllers.items()]
        return response

    def switch_mode(self,request,response):
        self.switch_requests.append(request)
        time.sleep(self.switch_delay)
        for n in request.deactivate_controllers:self.controllers[n]='inactive'
        for n in request.activate_controllers:self.controllers[n]='active'
        response.ok=True
        return response

    def direct(self,msg):
        self.direct_commands.append(list(msg.positions))
        if self.controllers[IMP]=='active' and self.controllers[JTC]=='inactive':self.position=list(msg.positions)

    def telemetry(self):
        if not self.emit_telemetry:return
        measured=[q+offset for q,offset in zip(self.position,self.measured_offset)]
        msg=JointState(name=JOINTS,position=measured,velocity=[0.]*6,effort=list(self.effort or []))
        msg.header.stamp=self.get_clock().now().to_msg()
        self.joint_pub.publish(msg)
        for name,pub in ((JTC,self.imp_state_pub),(POS,self.state_pub)):
            if self.controllers[name]=='active':
                state=JointTrajectoryControllerState(joint_names=JOINTS)
                state.header.stamp=msg.header.stamp
                state.reference.positions=self.position
                state.feedback.positions=measured
                state.reference.velocities=[0.]*6
                # The position JTC exports position-only feedback.
                state.feedback.velocities=[0.]*6 if name==JTC else []
                if name==POS:self.state_velocities=list(state.feedback.velocities)
                pub.publish(state)

    def execute(self,handle):
        result=super().execute(handle)
        if not handle.is_cancel_requested:self.position=list(handle.request.trajectory.points[-1].positions)
        return result

