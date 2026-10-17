"""Stage 3 - Nav2 (map server + AMCL + planner + controller + behaviours) on the saved map.

Uses TurtleBot3's tuned waffle parameters with three changes, applied with RewrittenYaml
so we don't have to copy and maintain a 400-line params file:
  * AMCL starts at the spawn pose, so no manual "2D Pose Estimate" click is needed
  * robot_radius 0.20 m in both costmaps: TB3's default (0.1) is smaller than the waffle,
    which is ~0.31 m wide, and lets the planner cut corners into walls and boxes

Normally started by bringup.launch.py after the robot has spawned.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg = get_package_share_directory('nlp_robot')
    nav2 = get_package_share_directory('nav2_bringup')
    tb3_params = os.path.join(get_package_share_directory('turtlebot3_navigation2'),
                              'param', 'waffle.yaml')
    x, y = LaunchConfiguration('x_pose'), LaunchConfiguration('y_pose')

    params = RewrittenYaml(
        source_file=LaunchConfiguration('params_file'),
        param_rewrites={
            'amcl.ros__parameters.set_initial_pose': 'True',
            'amcl.ros__parameters.initial_pose.x': x,
            'amcl.ros__parameters.initial_pose.y': y,
            'amcl.ros__parameters.initial_pose.yaw': '0.0',
            'local_costmap.local_costmap.ros__parameters.robot_radius': '0.20',
            'global_costmap.global_costmap.ros__parameters.robot_radius': '0.20',
        },
        convert_types=True)

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=os.path.join(pkg, 'maps', 'nlp_room.yaml')),
        DeclareLaunchArgument('params_file', default_value=tb3_params),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('rviz', default_value='true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2, 'launch', 'bringup_launch.py')),
            launch_arguments={'map': LaunchConfiguration('map'),
                              'use_sim_time': 'true',
                              'params_file': params,
                              'autostart': 'true'}.items()),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
             arguments=['-d', os.path.join(pkg, 'rviz', 'nlp_robot.rviz')],
             parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
