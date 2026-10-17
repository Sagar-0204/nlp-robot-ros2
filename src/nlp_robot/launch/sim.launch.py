"""Stage 2 - simulation: Gazebo Harmonic + the nlp_room world + a TurtleBot3 waffle.

    ros2 launch nlp_robot sim.launch.py              # with the Gazebo window
    ros2 launch nlp_robot sim.launch.py gui:=false   # headless (lighter on the GPU)

What starts:
  * gz sim server (physics + sensors) and, optionally, the GUI
  * robot_state_publisher - publishes the robot's fixed TF frames from its URDF
  * spawn                 - inserts the waffle model into the running world
  * ros_gz_bridge         - Gazebo <-> ROS topics: /cmd_vel, /odom, /scan, /tf, /clock ...
  * ros_gz_image          - the camera image (/camera/image_raw)
  * ground truth          - where the boxes REALLY are (from Gazebo), drawn in RViz with the
                            detector's error. Display/evaluation only: the robot never uses it.

Later stages include this file and add Nav2 and the language layer on top, started only
after the spawn has succeeded (that's what `on_spawned` is for).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, LogInfo, RegisterEventHandler,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description(on_spawned=None):
    """on_spawned: optional list of launch actions to start once the robot is in Gazebo."""
    pkg = get_package_share_directory('nlp_robot')
    tb3_gz = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    world = os.path.join(pkg, 'worlds', 'nlp_room.sdf')
    robot_sdf = os.path.join(tb3_gz, 'models', 'turtlebot3_waffle', 'model.sdf')
    bridge_yaml = os.path.join(tb3_gz, 'params', 'turtlebot3_waffle_bridge.yaml')
    gt_yaml = os.path.join(pkg, 'config', 'ground_truth_bridge.yaml')
    L = LaunchConfiguration

    def include(path, args, **kw):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                        launch_arguments=args.items(), **kw)

    gz_server = include(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py'),
                        {'gz_args': f'-r -s -v2 {world}', 'on_exit_shutdown': 'true'})
    gz_gui = include(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py'),
                     {'gz_args': '-g -v2', 'on_exit_shutdown': 'true'},
                     condition=IfCondition(L('gui')))
    robot_state_publisher = include(
        os.path.join(tb3_gz, 'launch', 'robot_state_publisher.launch.py'),
        {'use_sim_time': 'true'})
    spawn = Node(
        package='ros_gz_sim', executable='create', name='spawn_waffle', output='screen',
        arguments=['-world', 'nlp_room', '-name', 'waffle', '-file', robot_sdf,
                   '-x', L('x_pose'), '-y', L('y_pose'), '-z', '0.01'])
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                  arguments=['--ros-args', '-p', f'config_file:={bridge_yaml}'])
    image_bridge = Node(package='ros_gz_image', executable='image_bridge', output='screen',
                        arguments=['/camera/image_raw'])
    gt_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                     name='ground_truth_bridge', output='screen',
                     arguments=['--ros-args', '-p', f'config_file:={gt_yaml}'])
    ground_truth = Node(package='nlp_robot', executable='ground_truth', name='ground_truth',
                        output='screen', parameters=[{'use_sim_time': True}])

    def after_spawn(event, _context):
        if event.returncode != 0:
            return [LogInfo(msg='[nlp_robot] Spawning the robot FAILED. '
                                'Run tools/check_setup.sh, then tools/clean.sh, and relaunch.')]
        return [LogInfo(msg='[nlp_robot] Robot spawned in Gazebo.')] + list(on_spawned or [])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo GUI window'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('gz_partition', default_value='nlp_robot',
                              description='Gazebo transport partition (isolates other sims)'),

        # TurtleBot3 launch files read this at include time.
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),
        # Keep Gazebo transport on this machine and in its own partition. Without this, a
        # network change or a leftover Gazebo from another project can leave the GUI blank
        # and the spawner stuck at "Requesting list of world names".
        SetEnvironmentVariable('GZ_IP', '127.0.0.1'),
        SetEnvironmentVariable('GZ_PARTITION', L('gz_partition')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3_gz, 'models')),

        LogInfo(msg='[nlp_robot] Starting Gazebo (partition: nlp_robot)...'),
        gz_server, gz_gui, robot_state_publisher, bridge, image_bridge, gt_bridge, ground_truth,
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=after_spawn)),
    ])
