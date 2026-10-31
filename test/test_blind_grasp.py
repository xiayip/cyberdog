"""Locked-target execution and continuous approach over fake ROS services/actions."""
import time
from pathlib import Path
import yaml

import pytest
from geometry_msgs.msg import TransformStamped
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import GetCartesianPath,GetPlanningScene,GetMotionPlan
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint

from cyberdog.behavior import Cyberdog
from cyberdog.motion import JOINTS
from cyberdog.table_scene import table_box
from fake_hardware import FakeGameRobot,executor
from test_behavior import graph,until,spin
from test_grasp_retry import lock_target
from test_handshake import FakeGripper
from test_hardware_adapter import PARAMS


def test_lock_ignores_late_vision_and_discards_pending_depth_work():
    with graph() as ([g],e):
        seq=lock_target(g);center=seq.frozen[:];radius=seq.frozen_radius
        seq.accept(dict(center=[.8,-.3,.9],radius=.08),g.get_clock().now().nanoseconds*1e-9)
        seq.observe(dict(stamp=g.get_clock().now().nanoseconds*1e-9,valid=False,ball=None))
        seq.pending.append({'invalid':'late queued frame'})
        seq.locate_pending()
        seq.reject(ValueError('sphere fit does not reliably describe selected ball'))
        assert seq.stable  # Late perception errors cannot undo the locked decision.
        seq.target_at-=10.
        seq.check_target()
        assert seq.vision_locked and seq.status()['using_frozen_target']
        assert seq.frozen==center and seq.frozen_radius==radius
        assert seq.target_shift is None and not seq.pending and seq.retry_at is None
        g.command('stop')
        assert not seq.vision_locked and seq.frozen is None


def test_blind_mode_still_requires_locked_geometry_and_fresh_arm_feedback():
    with graph() as ([g],e):
        seq=lock_target(g);seq.frozen=None
        seq.tick();assert g.state=='paused' and '锁定的球心' in g.message
        seq.reset_attempts();seq=lock_target(g)
        g.robot.mock=False;g.robot.fresh=lambda:False
        seq.tick()
        assert g.state=='paused' and 'feedback stale' in g.message
        assert not g.robot.grip_closed and not seq.retry_at


class FakeGraspPlanner(Node):
    """Exercise transport/state ordering; geometry is checked separately in MoveIt."""
    def __init__(self):
        super().__init__('cyberdog_fake_grasp_planner');self.requests=[];self.transfers=[]
        self.create_service(GetPlanningScene,'/get_planning_scene',self.scene)
        self.create_service(GetCartesianPath,'/compute_cartesian_path',self.plan)
        self.create_service(GetMotionPlan,'/plan_kinematic_path',self.transfer)

    def scene(self,request,response):
        obj=table_box([0.,0.,0.],[.1,-.6],[.4,-.1],'cyberdog_observed_table')
        response.scene.world.collision_objects=[obj]
        return response

    def plan(self,request,response):
        self.requests.append(request)
        response.error_code.val=1;response.fraction=1.
        response.solution.joint_trajectory=self.path(request.start_state)
        return response

    def transfer(self,request,response):
        self.transfers.append(request)
        response.motion_plan_response.error_code.val=1
        goal=request.motion_plan_request.goal_constraints[0]
        joints={c.joint_name:c.position for c in goal.joint_constraints}
        target=[joints[j] for j in JOINTS] if joints else None
        response.motion_plan_response.trajectory.joint_trajectory=self.path(request.motion_plan_request.start_state,target)
        return response

    @staticmethod
    def path(state,target=None):
        joints=state.joint_state
        start=[joints.position[joints.name.index(j)] for j in JOINTS]
        path=JointTrajectory(joint_names=JOINTS)
        for i in range(3):
            q=[a+(b-a)*i/2 for a,b in zip(start,target)] if target is not None else start[:]
            if target is None:q[3]+=.02*i
            v=[0.]*6
            if i==1:
                if target is None:v[3]=.04
                else:v=[1.875*(b-a) for a,b in zip(start,target)]
            p=JointTrajectoryPoint(positions=q,velocities=v,accelerations=[0.]*6)
            p.time_from_start.sec=i//2;p.time_from_start.nanosec=(i%2)*500000000
            path.points.append(p)
        return path


@pytest.mark.parametrize('stop',[False,True])
@pytest.mark.parametrize('near',[False,True])
def test_real_ros_one_approach_then_close_and_direct_joint_place_or_manual_cancel(stop,near,monkeypatch):
    params=PARAMS+['gripper_enabled:=true','allow_nominal_tcp_trial:=true']
    config=yaml.safe_load((Path(__file__).parents[1]/'config/cyberdog.yaml').read_text())['cyberdog']['ros__parameters']
    names=['motion_limit_scale','grasp_joint_speed','grasp_joint_acceleration','grasp_transfer_speed',
           'grasp_transfer_acceleration','gripper_close_speed','gripper_open_speed','grasp_place_joint_pose']
    params += [f'{name}:={config[name]}' for name in names]
    with graph((Cyberdog,FakeGameRobot,FakeGripper,FakeGraspPlanner),params,executor) as ([g,r,gripper,planner],e):
        until(e,lambda:g.robot.motion.ready()=='',timeout=6.)
        switched=[]
        assert g.robot.mode('position',lambda ok,reason:switched.append(ok),tracking=False)[0]
        until(e,lambda:bool(switched));assert switched==[True]
        seq=g.ball_sequence
        tr=TransformStamped();tr.transform.translation.x=.22;tr.transform.translation.z=.06;tr.transform.rotation.w=1.
        if near:
            tr.transform.translation.x=.3221;tr.transform.translation.y=-.005;tr.transform.translation.z=.0864
        monkeypatch.setattr(seq.buffer,'lookup_transform',lambda *a:tr)
        stages=[];set_state=g.set_state
        def record_state(state,phase,message):
            stages.append(phase);set_state(state,phase,message)
        g.set_state=record_state
        r.run_seconds=.6
        seq=lock_target(g);center=seq.frozen[:]
        until(e,lambda:len(r.requests)==1 and seq.handle is not None,timeout=8.)
        assert g.phase=='grasp_approach'
        assert [list(req.command.position) for req in gripper.requests]==[[0.]]
        # Later estimates and dropouts cannot change the action's locked target.
        seq.accept(dict(center=[.9,-.2,.7],radius=.08),g.get_clock().now().nanoseconds*1e-9)
        seq.reject(ValueError('target unavailable'),transient=True)
        assert seq.frozen==center and seq.target_shift is None
        if stop:
            g.command('stop');until(e,lambda:not seq.busy,timeout=5.);spin(e,.3)
            assert g.state=='paused' and r.cancelled>=1 and len(r.requests)==1
            assert [list(req.command.position) for req in gripper.requests]==[[0.]]
            assert 'grasp_closing' not in stages and 'grasp_lift' not in stages
        else:
            until(e,lambda:g.phase=='complete',timeout=15.)
            assert len(r.requests)==2  # Continuous approach, then direct joint placement.
            assert not planner.transfers
            assert [len(req.waypoints) for req in planner.requests]==([1,1] if near else [4,4])
            assert 'lift' not in [w['stage'] for w in seq.waypoints]
            assert all([req.waypoints[-1].position.x,req.waypoints[-1].position.y,
                        req.waypoints[-1].position.z]==pytest.approx(center) for req in planner.requests)
            assert all(req.avoid_collisions for req in planner.requests)
            assert [list(req.command.position) for req in gripper.requests]==[[0.],[.95],[0.]]
            assert [list(req.command.velocity) for req in gripper.requests]==[[.3],[.6],[.3]]
            assert all(req.max_velocity_scaling_factor==.5 and req.max_acceleration_scaling_factor==.5
                       for req in planner.requests)
            assert (seq.speed,seq.acceleration,seq.transfer_speed,seq.transfer_acceleration)==tuple(
                config[name] for name in ('grasp_joint_speed','grasp_joint_acceleration',
                                         'grasp_transfer_speed','grasp_transfer_acceleration'))
            assert not g.robot.grip_closed
            assert list(r.requests[-1].trajectory.points[-1].positions)==pytest.approx(seq.place_joint_pose)
            assert r.position==pytest.approx(seq.place_joint_pose)
            assert r.requests[0].trajectory.points[1].velocities[3]>0.
            assert [s for s in stages if s.startswith('grasp_')]==[
                'grasp_settling','grasp_approach','grasp_closing',
                'grasp_place','grasp_releasing']
            assert seq.attempts==1 and seq.retry_at is None
