from glob import glob

from setuptools import setup

package_name = 'nlp_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Filled in later stages; empty globs are fine.
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vidya Sagar',
    maintainer_email='tirandasvidyasagar@gmail.com',
    description='Natural-language commands for a ROS 2 robot via an LLM and Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'check_interfaces = nlp_robot.check_interfaces:main',
            'goto = nlp_robot.goto:main',
            'commander = nlp_robot.commander_node:main',
            'nlp_cli = nlp_robot.nlp_cli:main',
            'color_detector = nlp_robot.color_detector_node:main',
            'ground_truth = nlp_robot.ground_truth_node:main',
        ],
    },
)
