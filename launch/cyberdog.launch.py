"""Core behavior, ball perception and reset/stop panel. Defaults to mock behavior."""
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config=LaunchConfiguration('config')
    runtime=str(Path(get_package_prefix('cyberdog')).parents[1]/'.cyberdog-yolo/bin/python') 
    return LaunchDescription([
        DeclareLaunchArgument('config',default_value=os.path.join(get_package_share_directory('cyberdog'),'config','cyberdog.yaml')),
        DeclareLaunchArgument('motion_backend',default_value='mock'),
        DeclareLaunchArgument('motion_enabled',default_value='false'),
        DeclareLaunchArgument('ball_detector',default_value='yoloe'),
        DeclareLaunchArgument('senses_backend',default_value='mock'),
        DeclareLaunchArgument('mock_ball_visible',default_value='false'),
        DeclareLaunchArgument('panel',default_value='true'),
        DeclareLaunchArgument('grasp_planning',default_value='false'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([FindPackageShare('zephyr_arm_moveit_config'),
            '/launch/move_group.launch.py']),condition=IfCondition(LaunchConfiguration('grasp_planning'))),
        DeclareLaunchArgument('senses_python',default_value=runtime if os.path.isfile(runtime) else ''),
        Node(package='cyberdog',executable='behavior',name='cyberdog',output='screen',parameters=[config,{
            'motion_backend':LaunchConfiguration('motion_backend'),
            'motion_enabled':ParameterValue(LaunchConfiguration('motion_enabled'),value_type=bool)}]),
        Node(package='cyberdog',executable='senses',name='cyberdog_senses',output='screen',
             prefix=LaunchConfiguration('senses_python'), additional_env={'MPLCONFIGDIR':'/tmp/cyberdog-mpl'},
             parameters=[config,{'ball_detector':LaunchConfiguration('ball_detector'),'senses_backend':LaunchConfiguration('senses_backend'),
                 'mock_ball_visible':ParameterValue(LaunchConfiguration('mock_ball_visible'),value_type=bool)}]),
        # `docker exec` shells have no DISPLAY; the container's desktop is :1.
        Node(package='cyberdog',executable='panel',name='cyberdog_panel',output='screen',
             additional_env={'DISPLAY':os.environ.get('DISPLAY',':1')},condition=IfCondition(LaunchConfiguration('panel'))),
    ])
