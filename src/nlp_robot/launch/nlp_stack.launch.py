"""The language + perception layer on its own: the commander and the colour detector.

Normally started by bringup.launch.py once Nav2 is up. Use it directly on top of any robot
that already runs Nav2:
    ros2 launch nlp_robot nlp_stack.launch.py use_sim_time:=false llm_backend:=ollama
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('nlp_robot')
    cfg = os.path.join(pkg, 'config', 'nlp_robot.yaml')
    sim = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    def as_str(name):
        # value_type=str keeps an empty model name a string instead of null
        return ParameterValue(LaunchConfiguration(name), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('llm_backend', default_value='ollama',
                              description='ollama | anthropic | openai | rules'),
        DeclareLaunchArgument('llm_model', default_value='',
                              description='e.g. qwen2.5:7b-instruct; empty = backend default'),
        DeclareLaunchArgument('places_file',
                              default_value=os.path.join(pkg, 'config', 'places.yaml')),

        Node(package='nlp_robot', executable='color_detector', name='color_detector',
             output='screen', parameters=[cfg, sim]),
        Node(package='nlp_robot', executable='commander', name='nlp_commander', output='screen',
             parameters=[cfg, sim, {'places_file': as_str('places_file'),
                                    'llm.backend': as_str('llm_backend'),
                                    'llm.model': as_str('llm_model')}]),
    ])
