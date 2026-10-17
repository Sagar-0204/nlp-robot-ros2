"""Everything built so far, in one command.

    ros2 launch nlp_robot bringup.launch.py               # Gazebo + robot + Nav2 + RViz
    ros2 launch nlp_robot bringup.launch.py gui:=false    # without the Gazebo window
    ros2 launch nlp_robot bringup.launch.py nav:=false    # simulation only (= sim.launch.py)
    ros2 launch nlp_robot bringup.launch.py llm_backend:=rules           # no LLM
    ros2 launch nlp_robot bringup.launch.py llm_model:=qwen2.5:3b-instruct
    ros2 launch nlp_robot bringup.launch.py stack:=false  # robot + Nav2 only, no commander

Start-up is event driven: Nav2 starts only after the robot has actually been spawned in
Gazebo, and the language layer (commander) a few seconds after Nav2. If the spawn fails
you get one clear message instead of a wall of TF errors.
"""
import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _load_sim(pkg):
    path = os.path.join(pkg, 'launch', 'sim.launch.py')
    spec = importlib.util.spec_from_file_location('nlp_robot_sim_launch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_launch_description():
    pkg = get_package_share_directory('nlp_robot')
    L = LaunchConfiguration

    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'nav.launch.py')),
        launch_arguments={'rviz': L('rviz'), 'x_pose': L('x_pose'),
                          'y_pose': L('y_pose')}.items(),
        condition=IfCondition(L('nav')))

    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'nlp_stack.launch.py')),
        launch_arguments={'llm_backend': L('llm_backend'), 'llm_model': L('llm_model')}.items(),
        condition=IfCondition(L('stack')))

    ld = _load_sim(pkg).generate_launch_description(
        on_spawned=[nav, TimerAction(period=4.0, actions=[stack])])
    ld.add_action(DeclareLaunchArgument('nav', default_value='true', description='Start Nav2'))
    ld.add_action(DeclareLaunchArgument('rviz', default_value='true', description='Start RViz'))
    ld.add_action(DeclareLaunchArgument('stack', default_value='true',
                                        description='Start the language layer (commander)'))
    ld.add_action(DeclareLaunchArgument('llm_backend', default_value='ollama',
                                        description='ollama | gemini | anthropic | openai | rules'))
    ld.add_action(DeclareLaunchArgument('llm_model', default_value='',
                                        description='empty = backend default (qwen2.5:7b-instruct)'))
    return ld
