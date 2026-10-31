"""One-click Cyberdog on the Zephyr arm.

Stages: arm bringup (Cyberdog impedance overrides merged into a copy of
controllers.yaml), wrist camera and move_group in parallel -> measured table box
-> startup_check (controllers, guard, camera frames) -> Cyberdog game + panel.
A bringup, camera or move_group that is already running is reused. If a core
process exits, the whole launch shuts down. The game starts paused; the arm
only moves after "复位".

  ros2 launch cyberdog bringup.launch.py [panel:=false] [preview:=true] [use_mock_hardware:=true]
"""
import os
import subprocess
import time

import yaml
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from cyberdog.controller_overrides import merge

CAN = 'can0'
# Exit of any of these (outside shutdown) stops the launch; spawners exit by design.
CORE = {'ros2_control_node', 'impedance_trajectory_guard', 'robot_state_publisher',
        'component_container', 'component_container_mt', 'move_group', 'behavior', 'senses'}


def share(package, *path):
    return os.path.join(get_package_share_directory(package), *path)


def include(package, launch_file, **arguments):
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(share(package, 'launch', launch_file)),
                                    launch_arguments=arguments.items())


def running(pattern):
    return subprocess.run(['pgrep', '-f', pattern], stdout=subprocess.DEVNULL).returncode == 0


def can_needs_bringup():
    info = subprocess.run(['ip', '-details', 'link', 'show', CAN], capture_output=True, text=True)
    if info.returncode:
        raise RuntimeError(f'{CAN} not found')
    if 'state UP' not in info.stdout:
        return True  # bringup runs sudo ip link ... at 1 Mbps
    # Shared bus: never reconfigure an interface that is already up.
    if 'bitrate 1000000 ' not in info.stdout:
        raise RuntimeError(f'{CAN} is up but not at 1 Mbps; fix it by hand')
    return False


def merged_controllers():
    with open(share('zephyr_arm_bringup', 'config', 'controllers.yaml')) as f:
        base = yaml.safe_load(f)
    for name in ('zephyr-hold.yaml', 'zephyr-tracking-limits.yaml', 'zephyr-goal-tolerance.yaml'):
        with open(share('cyberdog', 'config', name)) as f:
            merge(base, yaml.safe_load(f))
    os.makedirs('/tmp/cyberdog-up', exist_ok=True)
    path = f'/tmp/cyberdog-up/controllers-{time.strftime("%Y%m%d-%H%M%S")}.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(base, f, sort_keys=False)
    return path


def stop_on_core_exit(event, context):
    if context.is_shutdown or not CORE & {os.path.basename(arg) for arg in event.cmd[:2]}:
        return None
    reason = f'{event.action.name} exited with code {event.returncode}; stopping Cyberdog'
    get_logger('cyberdog_up').error(reason)
    return EmitEvent(event=Shutdown(reason=reason))


def after_success(stage, *actions):
    def on_exit(event, context):
        if context.is_shutdown:
            return None
        if event.returncode == 0:
            return list(actions)
        return EmitEvent(event=Shutdown(reason=f'{stage} failed; stopping Cyberdog'))
    return on_exit


def stages(context):
    arg = lambda name: LaunchConfiguration(name).perform(context).lower() == 'true'
    mock, panel, log = arg('use_mock_hardware'), arg('panel'), get_logger('cyberdog_up')
    # Cyberdog must be the only motion controller.
    if running('lib/cyberdog/behavior|lib/armie/game'):
        raise RuntimeError('an Cyberdog game is already running; stop it first')
    if running('lib/ghost_game_orchestrator/ghost_game_node'):
        raise RuntimeError('ghost game is running; stop it first')
    actions = []
    display = os.environ.setdefault('DISPLAY', ':1')
    actions.append(SetEnvironmentVariable('DISPLAY', display))
    if panel and subprocess.run(['timeout', '5', 'python3', '-c', 'import tkinter; tkinter.Tk().destroy()'],
                                stderr=subprocess.DEVNULL).returncode:
        log.warning(f'cannot open display {display}; no button panel (use the /cyberdog/* services)')
        panel = False

    if running('zephyr_arm_bringup bringup.launch.py|lib/controller_manager/ros2_control_node'):
        log.warning('arm bringup already running: reused and left running; '
                    'its impedance stiffness/limits are whatever it was started with')
    else:
        actions.append(include(
            'zephyr_arm_bringup', 'bringup.launch.py', can_interface=CAN,
            bring_up_can=str(not mock and can_needs_bringup()).lower(),
            use_mock_hardware=str(mock).lower(), activate_trajectory_controller='true',  # position chain holds the pose; Cyberdog switches from it
            show_rviz='false', controllers_file=merged_controllers()))
        if not mock:
            actions.append(RegisterEventHandler(OnShutdown(on_shutdown=LogInfo(
                msg='bringup stopping: the CAN driver flushes the last hold command and sends no '
                    'disable frame, so the motors keep holding until power-off'))))
    if running('__node:=camera_container'):
        log.warning('wrist camera already running: reused and left running')
    else:
        actions.append(include('zephyr_arm_bringup', 'gemini_305.launch.py', enable_camera='true'))
    if running('lib/moveit_ros_move_group/move_group'):
        log.warning('move_group already running: reused and left running')
    else:
        actions.append(include('zephyr_arm_moveit_config', 'move_group.launch.py'))

    table = Node(package='cyberdog', executable='table_scene', output='screen',
                 parameters=[share('cyberdog', 'config', 'zephyr-table.yaml'), {'timeout_sec': 120.}])
    check = Node(package='cyberdog', executable='startup_check', output='screen')
    game = [include('cyberdog', 'cyberdog.launch.py', config=share('cyberdog', 'config', 'cyberdog.yaml'),
                    motion_backend='trajectory', motion_enabled='true', senses_backend='camera',
                    panel=str(panel).lower(),ball_detector=LaunchConfiguration('ball_detector').perform(context),
                    audio=LaunchConfiguration('audio').perform(context),
                    senses_python=LaunchConfiguration('senses_python').perform(context)),
            LogInfo(msg='Cyberdog started paused: the arm moves only after "复位". Ctrl+C stops everything.')]
    if arg('preview'):
        game += [Node(package='image_transport', executable='republish', output='screen',
                      parameters=[{'in_transport': 'compressed', 'out_transport': 'raw'}],
                      remappings=[('in/compressed', '/cyberdog/camera_preview/compressed'),
                                  ('out', '/cyberdog/ball_preview')]),
                 Node(package='rqt_image_view', executable='rqt_image_view', arguments=['/cyberdog/ball_preview'])]
    return actions + [
        RegisterEventHandler(OnProcessExit(on_exit=stop_on_core_exit)),
        RegisterEventHandler(OnProcessExit(target_action=table,
                                           on_exit=after_success('table collision box', check))),
        RegisterEventHandler(OnProcessExit(target_action=check, on_exit=after_success('startup check', *game))),
        table,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ball_detector', default_value='yoloe'),
        DeclareLaunchArgument('senses_python', default_value=os.path.join(os.path.dirname(os.path.dirname(get_package_prefix('cyberdog'))),
            '.cyberdog-yolo/bin/python')),
        DeclareLaunchArgument('panel', default_value='true', description='Tk button panel on $DISPLAY (default :1)'),
        DeclareLaunchArgument('audio', default_value='true', description='Stage sound effects in a separate node'),
        DeclareLaunchArgument('preview', default_value='false', description='rqt_image_view of the annotated camera'),
        DeclareLaunchArgument('use_mock_hardware', default_value='false',
                              description='ros2_control mock instead of CAN (rehearsal; the camera is still real)'),
        OpaqueFunction(function=stages),
    ])
