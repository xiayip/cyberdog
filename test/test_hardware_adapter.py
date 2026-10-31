import time
import math
from pathlib import Path
import yaml
import pytest
from cyberdog.behavior import Cyberdog
from cyberdog.game_core import path_points
from cyberdog.controller_modes import BASE, POS, observed
from fake_hardware import FakeGameRobot, executor
from test_behavior import graph, until, spin

PARAMS=['motion_backend:=trajectory','motion_enabled:=true','required_guard_node:=\'\'',
        'touch_signal:=either']


def test_ros_reset_j3_pull_lift_then_verified_damping():
    config=yaml.safe_load((Path(__file__).parents[1]/'config/cyberdog.yaml').read_text())['cyberdog']['ros__parameters']
    pace=['motion_limit_scale','pose_speed','handshake_pose_speed','handshake_lift_time_scale',
          'handshake_wrist_delta','handshake_head_offsets','handshake_head_hold_sec',
          'handshake_inspection_speed','handshake_inspection_acceleration','handshake_inspection_min_sec']
    parameters=PARAMS+[f'{name}:={config[name]}' for name in pace]
    with graph((Cyberdog,FakeGameRobot),parameters,executor) as ([g,r],e):
        assert g.command('reset')[0]
        until(e,lambda:g.state=='idle',timeout=8.)
        assert r.position==g.home and g.robot.modes.mode=='position'
        def peak_motion(request):
            first,last=request.trajectory.points
            duration=(last.time_from_start.sec-first.time_from_start.sec
                +(last.time_from_start.nanosec-first.time_from_start.nanosec)*1e-9)
            distance=max(abs(a-b) for a,b in zip(first.positions,last.positions))
            # Analytic extrema of the JTC zero-velocity/acceleration quintic.
            return 1.875*distance/duration,(10/math.sqrt(3))*distance/duration**2
        assert .24<peak_motion(r.requests[-1])[0]<=.36+1e-8
        initial=len(r.requests)
        # Alter only a different joint: no handshake.
        r.position[4]=.05;spin(e,.5)
        assert g.state=='idle' and len(r.requests)==initial
        r.position[4]=0.;r.position[2]=.06
        until(e,lambda:g.phase=='handshake',timeout=14.)
        poses=[list(req.trajectory.points[-1].positions) for req in r.requests[initial:]]
        assert len(poses)==3
        assert poses[0][:3]==[g.home[0],0.,.06] and poses[0][3:]==[.40,0.,0.]
        assert poses[1][:5]==poses[0][:5] and poses[1][5]==.25
        assert g.handshake_speed==.6 and g.head_hold==.08 and g.inspection_speed==.8
        assert g.inspection_minimum==.65
        assert g.lift_time_scale==.5
        for index,request in enumerate(r.requests[initial:]):
            velocity,acceleration=peak_motion(request)
            assert .24<velocity<=(.8 if index<2 else 1.2)+1e-8
            assert acceleration<=(2.4 if index<2 else 4*.65*g.robot.motion.limit_scale)+1e-8
            first,last=request.trajectory.points
            old_acc=[.975]*6
            if index<2:old_acc[3]=old_acc[5]=1.5
            old=path_points(list(first.positions),[list(last.positions)],.6 if index<2 else .48,
                            1.,old_acc,time_scale=1. if index<2 else .5)
            assert last.time_from_start.sec+last.time_from_start.nanosec*1e-9<old[-1][0]*.85
            if index==2:
                baseline=path_points(poses[1],[g.lift],.6,1.,[.65*g.robot.motion.limit_scale]*6)
                for point,(at,target) in zip(request.trajectory.points,baseline):
                    assert point.time_from_start.sec+point.time_from_start.nanosec*1e-9==pytest.approx(at*.5,abs=1e-9)
                    assert list(point.positions)==target
        assert not g.robot.motion.path_inspection and g.robot.motion.path_time_scale==1.
        assert not r.impedance_goals and not r.direct_commands
        assert r.position==g.lift and r.position[3]==0.
        assert observed(r.list(None,type('R',(),{})()).controller)=='damping'
        assert r.switch_requests[-1].activate_controllers==['rebot_arm_rs_damping_controller']
        assert all(BASE not in q.deactivate_controllers for q in r.switch_requests)
        assert list(r.requests[-1].trajectory.points[-1].positions)==g.lift
        g.command('reset');until(e,lambda:g.state=='idle',timeout=8.)
        assert r.position==g.home and g.robot.modes.mode=='position'
        assert .24<peak_motion(r.requests[-1])[0]<=.36+1e-8


def test_ros_j3_torque_trigger_and_loss_of_feedback_blocks_actions():
    with graph((Cyberdog,FakeGameRobot),PARAMS,executor) as ([g,r],e):
        r.effort=[0.,0.,5.,0.,0.,0.]
        g.command('reset');until(e,lambda:g.state=='idle',timeout=8.)
        until(e,lambda:g.touch.baseline is not None)
        r.effort[2]=6.2
        until(e,lambda:g.phase=='handshake',timeout=14.)
        assert r.position==g.lift
        r.emit_telemetry=False;spin(e,.6)
        before=len(r.requests);g.command('reset');spin(e,.5)
        assert g.state=='resetting' and len(r.requests)==before
        g.command('stop')


def test_reset_does_not_publish_a_reanchor_or_replace_its_zero_target():
    from trajectory_msgs.msg import JointTrajectory
    from rclpy.node import Node
    with graph((Cyberdog,FakeGameRobot,lambda:Node('hold_probe')),PARAMS,executor) as ([g,r,probe],e):
        messages=[]
        probe.create_subscription(JointTrajectory,'/'+POS+'/joint_trajectory',messages.append,10)
        g.command('reset');until(e,lambda:g.state=='idle',timeout=8.)
        spin(e,.3)
        assert messages==[]
        assert list(r.requests[-1].trajectory.points[-1].positions)==g.home
        g.robot.hold();g.command('stop');spin(e,.2)
        assert messages==[]


def test_ready_rejects_new_unsettled_feedback_before_next_timer_tick():
    from control_msgs.msg import JointTrajectoryControllerState
    from cyberdog.motion import JOINTS
    with graph((Cyberdog,FakeGameRobot),PARAMS,executor) as ([g,r],e):
        until(e,lambda:g.robot.motion.ready()=='',timeout=5.)
        m=g.robot.motion
        state=JointTrajectoryControllerState(joint_names=JOINTS)
        state.header.stamp=g.get_clock().now().to_msg()
        state.reference.positions=list(r.position);state.reference.positions[3]+=.04
        state.reference.velocities=[0.]*6
        state.feedback.positions=list(r.position)
        # Subscriber runs between readiness checks; no motion.tick() in between.
        m.controller_state(state)
        assert m.reference_pose is None
        assert m.ready()!=''
        assert m.quiet_since is None
        assert not m.start_path(lambda _:[])[0]
        assert not r.requests
