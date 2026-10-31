import time
from cyberdog.behavior import Cyberdog
from cyberdog.controller_modes import BASE, POS, observed
from fake_hardware import FakeGameRobot, executor
from test_behavior import graph, until, spin

PARAMS=['motion_backend:=trajectory','motion_enabled:=true','required_guard_node:=\'\'',
        'touch_signal:=either']


def test_ros_reset_j3_pull_lift_then_verified_damping():
    with graph((Cyberdog,FakeGameRobot),PARAMS,executor) as ([g,r],e):
        assert g.command('reset')[0]
        until(e,lambda:g.state=='idle',timeout=8.)
        assert r.position==g.home and g.robot.modes.mode=='position'
        initial=len(r.requests)
        # Alter only a different joint: no handshake.
        r.position[4]=.05;spin(e,.5)
        assert g.state=='idle' and len(r.requests)==initial
        r.position[4]=0.;r.position[2]=.06
        until(e,lambda:g.phase=='handshake',timeout=14.)
        poses=[list(req.trajectory.points[-1].positions) for req in r.requests[initial:]]
        assert len(poses)==4
        assert poses[0][:3]==[g.home[0],0.,.06] and poses[0][3:]==[.25,0.,0.]
        assert poses[1][:5]==poses[0][:5] and poses[1][5]==.25
        assert poses[2][:5]==poses[0][:5] and poses[2][5]==-.25
        assert not r.impedance_goals and not r.direct_commands
        assert r.position==g.lift and r.position[3]==0.
        assert observed(r.list(None,type('R',(),{})()).controller)=='damping'
        assert r.switch_requests[-1].activate_controllers==['rebot_arm_rs_damping_controller']
        assert all(BASE not in q.deactivate_controllers for q in r.switch_requests)
        assert list(r.requests[-1].trajectory.points[-1].positions)==g.lift
        g.command('reset');until(e,lambda:g.state=='idle',timeout=8.)
        assert r.position==g.home and g.robot.modes.mode=='position'


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
