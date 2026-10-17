"""ground_truth: draws where the boxes REALLY are (from Gazebo) in RViz, and how far the
detector's estimate is from the truth.

This is for visualisation and evaluation only. The commander never reads it; the robot
still has to find boxes with its own camera + LiDAR.

Subscribes: nlp_robot/ground_truth/<box>  (geometry_msgs/PoseStamped, bridged from Gazebo)
            detected_objects            (nlp_robot_interfaces/DetectedObjectArray)
Publishes:  nlp_robot/ground_truth/markers (visualization_msgs/MarkerArray)
"""
import math

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from nlp_robot_interfaces.msg import DetectedObjectArray

RGB = {'red': (1.0, 0.15, 0.15), 'green': (0.2, 0.85, 0.2), 'blue': (0.25, 0.45, 1.0),
       'yellow': (1.0, 0.9, 0.1)}


def cube_edges(cx, cy, cz, s):
    h = s / 2.0
    c = [(cx + dx, cy + dy, cz + dz) for dx in (-h, h) for dy in (-h, h) for dz in (-h, h)]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    pts = []
    for a, b in edges:
        pts += [Point(x=c[a][0], y=c[a][1], z=c[a][2]), Point(x=c[b][0], y=c[b][1], z=c[b][2])]
    return pts


class GroundTruth(Node):
    def __init__(self):
        super().__init__('ground_truth')
        self.declare_parameter('boxes', ['red_box', 'blue_box', 'green_box', 'yellow_box'])
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('box_size', 0.30)
        self.boxes = list(self.get_parameter('boxes').value)
        self.frame = self.get_parameter('global_frame').value
        self.size = float(self.get_parameter('box_size').value)
        self.truth, self.detected, self.reported = {}, {}, {}
        for name in self.boxes:
            self.create_subscription(PoseStamped, f'nlp_robot/ground_truth/{name}',
                                     lambda m, n=name: self._on_truth(n, m), 10)
        self.create_subscription(DetectedObjectArray, 'detected_objects', self._on_detected, 10)
        self.pub = self.create_publisher(MarkerArray, 'nlp_robot/ground_truth/markers', 10)
        self.create_timer(0.5, self._publish)
        self.create_timer(15.0, self._warn_if_silent)

    def _on_truth(self, name, msg):
        p = msg.pose.position
        self.truth[name] = (p.x, p.y, p.z)

    def _on_detected(self, msg):
        self.detected = {o.label: (o.pose.pose.position.x, o.pose.pose.position.y)
                         for o in msg.objects}

    def _warn_if_silent(self):
        if not self.truth:
            self.get_logger().warn('no ground-truth poses from Gazebo yet; check the '
                                   'ground-truth bridge (ros2 topic list | grep ground_truth)')

    def _publish(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, name in enumerate(self.boxes):
            if name not in self.truth:
                continue
            x, y, z = self.truth[name]
            r, g, b = RGB.get(name.split('_')[0], (1.0, 1.0, 1.0))

            box = Marker()
            box.header.frame_id, box.header.stamp = self.frame, stamp
            box.ns, box.id, box.type = 'ground_truth', 2 * i, Marker.LINE_LIST
            box.pose.orientation.w = 1.0
            box.scale.x = 0.025
            box.color.r, box.color.g, box.color.b, box.color.a = r, g, b, 1.0
            box.points = cube_edges(x, y, z, self.size)
            box.lifetime = Duration(seconds=2.0).to_msg()

            label = Marker()
            label.header = box.header
            label.ns, label.id, label.type = 'ground_truth', 2 * i + 1, Marker.TEXT_VIEW_FACING
            label.pose.position.x, label.pose.position.y = x, y
            label.pose.position.z = z + self.size
            label.pose.orientation.w = 1.0
            label.scale.z = 0.16
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            if name in self.detected:
                dx, dy = self.detected[name]
                err = math.hypot(dx - x, dy - y)
                label.text = f'{name}  err {err:.2f} m'
                if self.reported.get(name) is None or abs(self.reported[name] - err) > 0.05:
                    self.reported[name] = err
                    self.get_logger().info(
                        f'{name}: true ({x:.2f}, {y:.2f}), detected ({dx:.2f}, {dy:.2f}), '
                        f'error {err:.2f} m')
            else:
                label.text = f'{name} (not seen yet)'
            label.lifetime = box.lifetime
            arr.markers += [box, label]
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruth()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
