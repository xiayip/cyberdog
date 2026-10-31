"""Speed bounds, measured-table geometry and post-close failure/cancel ordering."""
import copy
import math
import time

import pytest
from control_msgs.action import ParallelGripperCommand,FollowJointTrajectory
from rclpy.action import GoalResponse
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint

from cyberdog.ball_grasp_geometry import rotation
from cyberdog.ball_sequence import JOINTS,timed_points
from cyberdog.behavior import Cyberdog
from cyberdog.placement import placement_waypoints,held_ball,joint_placement_points,DEFAULT_PLACE_JOINTS
from cyberdog.table_scene import table_box
from fake_hardware import FakeGameRobot,executor
from test_behavior import graph,until,spin
from test_blind_grasp import FakeGraspPlanner
from test_grasp_retry import lock_target
from test_handshake import FakeGripper
from test_hardware_adapter import PARAMS


def test_joint_place_keeps_exact_six_angles_and_bounds_jtc_interpolation():
    start=[-1.5707,.6,.8,.4,0.,.25]
    points=joint_placement_points(start,DEFAULT_PLACE_JOINTS,.65,1.3)
    first,last=points
    assert list(first.positions)==start
    assert list(last.positions)==[.2,.635,.199,-.599,-.285,.273]
    dt=last.time_from_start.sec-first.time_from_start.sec+(last.time_from_start.nanosec-first.time_from_start.nanosec)*1e-9
    for a,b in zip(start,last.positions):
        assert 1.875*abs(a-b)/dt<=.65+1e-8
        assert (10/math.sqrt(3))*abs(a-b)/dt**2<=1.3+1e-8
    old=joint_placement_points(start,DEFAULT_PLACE_JOINTS,.5,1.)[-1]
    assert last.time_from_start.sec+last.time_from_start.nanosec*1e-9 < (old.time_from_start.sec+old.time_from_start.nanosec*1e-9)*.8
    assert list(last.velocities)==list(last.accelerations)==[0.]*6
    for target in ([0.]*5,[0.,-.001,0.,0.,0.,0.],[0.,0.,0.,2.,0.,0.],[float('nan')]*6):
        with pytest.raises(ValueError):joint_placement_points(start,target,.5,1.)


@pytest.mark.parametrize('event',['goal_rejected','trajectory_failed','stop_place','arrival_error','release_failure','close_failure'])
def test_joint_place_failure_stop_or_missing_arrival_never_releases_early(event,monkeypatch):
    class Robot(FakeGameRobot):
        def goal(self,request):
            if event=='goal_rejected' and list(request.trajectory.points[-1].positions)==DEFAULT_PLACE_JOINTS:
                return GoalResponse.REJECT
            return super().goal(request)
        def execute(self,handle):
            if event=='trajectory_failed' and list(handle.request.trajectory.points[-1].positions)==DEFAULT_PLACE_JOINTS:
                self.requests.append(handle.request);handle.abort()
                return FollowJointTrajectory.Result(error_code=-1)
            return super().execute(handle)
    class Gripper(FakeGripper):
        def execute(self,handle):
            if (event=='release_failure' and len(self.requests)==2) or (event=='close_failure' and len(self.requests)==1):
                self.requests.append(handle.request);handle.abort()
                return ParallelGripperCommand.Result(reached_goal=False)
            return super().execute(handle)
    params=PARAMS+['gripper_enabled:=true','allow_nominal_tcp_trial:=true']
    with graph((Cyberdog,Robot,Gripper,FakeGraspPlanner),params,executor) as ([g,r,gripper,planner],e):
        until(e,lambda:g.robot.motion.ready()=='',timeout=6.)
        switched=[];assert g.robot.mode('position',lambda ok,reason:switched.append(ok),tracking=False)[0]
        until(e,lambda:bool(switched));assert switched==[True]
        tr=TransformStamped();tr.transform.translation.x=.22;tr.transform.translation.z=.06;tr.transform.rotation.w=1.
        monkeypatch.setattr(g.ball_sequence.buffer,'lookup_transform',lambda *a:tr)
        r.run_seconds=.4;seq=lock_target(g)
        if event in ('stop_place','arrival_error'):
            until(e,lambda:g.phase=='grasp_place' and seq.handle is not None,timeout=15.)
            if event=='stop_place':g.command('stop')
            else:r.measured_offset[0]=.02
        until(e,lambda:g.state=='paused' and not seq.busy,timeout=18.);spin(e,.15)
        assert g.phase!='complete' and seq.retry_at is None and seq.attempts==1
        assert len(r.requests)==(1 if event in ('goal_rejected','close_failure') else 2)
        assert not planner.transfers
        expected=[[0.],[.95],[0.]] if event=='release_failure' else [[0.],[.95]]
        assert [list(req.command.position) for req in gripper.requests]==expected
        if event not in ('release_failure','close_failure'):assert g.robot.grip_closed
        if event=='stop_place':assert r.cancelled>=1


def test_placement_inside_measured_table_and_radius_clearance():
    table=table_box([.0235082073,-.0093218657,.0030443231],[.055582,-.656897],[.398558,-.005605],'table')
    small=placement_waypoints(table,[.25,-.3],.02)
    large=placement_waypoints(table,[.25,-.3],.08)
    for ways,radius in ((small,.02),(large,.08)):
        turn,place=ways
        top=table.primitive_poses[0].position.z+.05
        assert place['position'][2]-radius==pytest.approx(top+.01)
        assert turn['position'][2]-place['position'][2]==pytest.approx(.16)
        assert place['position'][:2]==[.25,-.3] and turn['orientation']==place['orientation']
        assert rotation(place['orientation'])[2,2]==pytest.approx(-math.cos(.7))
        assert [p['gripper'] for p in ways]==['hold','open']
    for xy in ([0.,0.],[.39,-.3],[float('nan'),0.]):
        with pytest.raises(ValueError):placement_waypoints(table,xy,.025)
    wrong=copy.deepcopy(table);wrong.header.frame_id='camera'
    with pytest.raises(ValueError):placement_waypoints(wrong,[.25,-.3],.025)
    wrong=copy.deepcopy(table);wrong.primitive_poses[0].orientation.x=.1
    with pytest.raises(ValueError):placement_waypoints(wrong,[.25,-.3],.025)
    attached=held_ball(.025)
    assert attached.link_name=='omni_picker_tcp'
    assert list(attached.object.primitives[0].dimensions)==[.025]
    assert not any(link in attached.touch_links for link in ('base_link','link3','cyberdog_observed_table'))


def test_faster_retiming_preserves_every_sample_and_derivative_limits():
    path=JointTrajectory(joint_names=JOINTS)
    for i in range(3):
        p=JointTrajectoryPoint(positions=[0.,.6,.8,.08*i,0.,0.],velocities=[0.,0.,0.,.16 if i==1 else 0.,0.,0.],
                               accelerations=[0.,0.,0.,.5 if i<2 else 0.,0.,0.])
        p.time_from_start.sec=i;path.points.append(p)
    start=path.points[0].positions
    slow=timed_points(path,start,.04,.12);fast=timed_points(path,start,.12,.3)
    duration=lambda p:p.time_from_start.sec+p.time_from_start.nanosec*1e-9
    assert duration(fast[-1])<duration(slow[-1])*.6
    assert [list(p.positions) for p in fast]==[list(p.positions) for p in path.points]
    assert max(abs(v) for p in fast for v in p.velocities)<=.12
    assert max(abs(a) for p in fast for a in p.accelerations)<=.3
    assert fast[1].velocities[3]>0 and list(fast[-1].velocities)==[0.]*6
    with pytest.raises(ValueError):timed_points(path,start,float('nan'),.3)


@pytest.mark.parametrize('old_speed,old_acc,speed,acc',[(.12,.3,.24,.6),(.36,.6,.5,1.)])
def test_whole_flow_retiming_is_faster_without_dropping_planned_samples(old_speed,old_acc,speed,acc):
    path=JointTrajectory(joint_names=JOINTS)
    for i in range(3):
        p=JointTrajectoryPoint(positions=[0.,.6,.8,.2*i,0.,0.],
                               velocities=[0.,0.,0.,.8 if i==1 else 0.,0.,0.],
                               accelerations=[0.,0.,0.,2. if i<2 else 0.,0.,0.])
        p.time_from_start.sec=i;path.points.append(p)
    slow=timed_points(path,path.points[0].positions,old_speed,old_acc)
    fast=timed_points(path,path.points[0].positions,speed,acc)
    seconds=lambda p:p.time_from_start.sec+p.time_from_start.nanosec*1e-9
    assert seconds(fast[-1])<seconds(slow[-1])*.75
    assert [list(p.positions) for p in fast]==[list(p.positions) for p in path.points]
    assert max(abs(v) for p in fast for v in p.velocities)<=speed
    assert max(abs(a) for p in fast for a in p.accelerations)<=acc
    assert list(fast[-1].velocities)==[0.]*6


@pytest.mark.parametrize('event',['transfer_failure','precheck_failure','descent_failure','release_failure','stop_transfer','stop_place'])
def test_real_ros_place_failure_or_stop_never_releases_in_air_or_completes_early(event,monkeypatch):
    class Planner(FakeGraspPlanner):
        def transfer(self,request,response):
            result=super().transfer(request,response)
            if event=='transfer_failure':result.motion_plan_response.error_code.val=-1
            return result
        def plan(self,request,response):
            result=super().plan(request,response)
            if ((event=='precheck_failure' and len(self.requests)==4)
                    or (event=='descent_failure' and len(self.requests)==5)):
                result.fraction=.8
            return result
    class Gripper(FakeGripper):
        def execute(self,handle):
            if event=='release_failure' and len(self.requests)==2:
                self.requests.append(handle.request);handle.abort()
                return ParallelGripperCommand.Result(reached_goal=False)
            return super().execute(handle)
    with graph((Cyberdog,FakeGameRobot,Gripper,Planner),PARAMS+['gripper_enabled:=true','allow_nominal_tcp_trial:=true','grasp_place_mode:=cartesian'],executor) as ([g,r,gripper,planner],e):
        until(e,lambda:g.robot.motion.ready()=='',timeout=6.)
        switched=[];assert g.robot.mode('position',lambda ok,reason:switched.append(ok),tracking=False)[0]
        until(e,lambda:bool(switched));assert switched==[True]
        tr=TransformStamped();tr.transform.translation.x=.22;tr.transform.translation.z=.06;tr.transform.rotation.w=1.
        monkeypatch.setattr(g.ball_sequence.buffer,'lookup_transform',lambda *a:tr)
        r.run_seconds=.4;seq=lock_target(g)
        if event.startswith('stop_'):
            stage='grasp_'+event[5:]
            until(e,lambda:g.phase==stage and seq.handle is not None,timeout=15.)
            g.command('stop')
        until(e,lambda:g.state=='paused' and not seq.busy,timeout=15.);spin(e,.15)
        assert g.phase!='complete' and seq.retry_at is None and seq.attempts==1
        assert len(r.requests)==dict(transfer_failure=2,precheck_failure=2,descent_failure=3,
                                    release_failure=4,stop_transfer=3,stop_place=4)[event]
        expected=[[0.],[.95],[0.]] if event=='release_failure' else [[0.],[.95]]
        assert [list(req.command.position) for req in gripper.requests]==expected
        if event!='release_failure':assert g.robot.grip_closed
        if event.startswith('stop_'):assert r.cancelled>=1


def test_expired_placement_and_late_arrival_cannot_open_gripper():
    with graph() as ([g],e):
        seq=lock_target(g);seq.closing_started=True;g.robot.grip_closed=True
        seq.waypoints=[dict(stage='place')];seq.index=0
        seq.attempt_deadline=time.monotonic()-1.
        g.robot.grip=lambda *a:pytest.fail('must not open after timeout')
        seq.arrived()
        assert g.state=='paused' and g.robot.grip_closed and seq.retry_at is None
        seq.arrived()  # Late action callback after the pause.
        assert g.phase!='complete'


def test_placement_can_be_disabled_and_still_finishes_holding_ball():
    with graph(parameters=['grasp_place_on_table:=false']) as ([g],e):
        seq=lock_target(g)
        until(e,lambda:g.phase=='complete')
        assert g.robot.grip_closed and g.outcome=='simulated_grasp'
        assert not seq.place_waypoints
