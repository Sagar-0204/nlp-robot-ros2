"""goto: send the robot to a named place through Nav2. Stage 3 test tool.

    ros2 run nlp_robot goto kitchen
    ros2 run nlp_robot goto "the dock"        # aliases work
    ros2 run nlp_robot goto --list

This is a minimal Nav2 *action client*: send a NavigateToPose goal, print feedback while
the robot drives, wait for the result, and cancel the goal on Ctrl+C. The stage-5
commander does exactly this for every step of an LLM-generated plan.
"""
import math
import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from .places import find_place, load_places

STATUS = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_ABORTED: 'ABORTED',
          GoalStatus.STATUS_CANCELED: 'CANCELED'}


def make_goal(node, x, y, yaw_deg):
    ps = PoseStamped()
    ps.header.frame_id = 'map'
    ps.header.stamp = node.get_clock().now().to_msg()
    ps.pose.position.x, ps.pose.position.y = float(x), float(y)
    yaw = math.radians(yaw_deg)
    ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return NavigateToPose.Goal(pose=ps)


def main(args=None):
    argv = rclpy.utilities.remove_ros_args(sys.argv if args is None else args)[1:]
    places = load_places(os.path.join(get_package_share_directory('nlp_robot'),
                                      'config', 'places.yaml'))
    if not argv or argv[0] in ('--list', '-l'):
        print('Places:')
        for name, info in places.items():
            x, y, yaw = info['pose']
            aka = f'  (also: {", ".join(info["aliases"])})' if info['aliases'] else ''
            print(f'  {name:13s} ({x:5.1f}, {y:5.1f}, {yaw:6.1f} deg){aka}')
        print('\nUsage: ros2 run nlp_robot goto <place>')
        return

    name = find_place(' '.join(argv), places)
    if name is None:
        sys.exit(f'Unknown place "{" ".join(argv)}". Known: {", ".join(places)}')

    # Let Ctrl+C reach us as KeyboardInterrupt (instead of rclpy shutting down first), so we
    # can still send the cancel request to Nav2 and the robot actually stops.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node('goto')
    client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
    try:
        node.get_logger().info('waiting for Nav2 (navigate_to_pose)...')
        if not client.wait_for_server(timeout_sec=15.0):
            sys.exit('Nav2 is not running - start it with: ros2 launch nlp_robot bringup.launch.py')

        x, y, yaw = places[name]['pose']
        last = [None]

        def on_feedback(msg):
            d = round(msg.feedback.distance_remaining, 1)
            if last[0] is None and d == 0.0:
                return          # Nav2 hasn't planned a path yet; nothing useful to show
            if d != last[0]:
                last[0] = d
                print(f'  {name}: {d:.1f} m to go', flush=True)

        print(f'Going to {name} ({x}, {y}, {yaw} deg)...')
        send = client.send_goal_async(make_goal(node, x, y, yaw), feedback_callback=on_feedback)
        rclpy.spin_until_future_complete(node, send)
        handle = send.result()
        if not handle.accepted:
            sys.exit('Nav2 rejected the goal')
        result = handle.get_result_async()
        try:
            rclpy.spin_until_future_complete(node, result)
        except KeyboardInterrupt:
            print('\nCancelling...')
            cancel = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel, timeout_sec=3.0)
            return
        status = result.result().status
        print(f'Result: {STATUS.get(status, status)}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
