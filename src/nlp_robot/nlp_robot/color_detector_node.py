"""color_detector: finds coloured boxes and publishes their map positions. Stage 7.

Subscribes: camera/image_raw (sensor_msgs/Image), camera/camera_info, scan
Publishes:  detected_objects (nlp_robot_interfaces/DetectedObjectArray, 2 Hz)
            detected_objects/markers (visualization_msgs/MarkerArray)
            color_detector/debug_image (annotated, if publish_debug)
Labels are "<colour>_box", which is exactly what the language layer targets.
"""
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from nlp_robot_interfaces.msg import DetectedObject, DetectedObjectArray

from .color_detection import (DEFAULT_HSV, ObjectTracker, detect_blobs, point_on_ray, push_back,
                              range_from_height, range_from_scan)

MARKER_RGB = {'red': (1.0, 0.1, 0.1), 'green': (0.1, 0.8, 0.1), 'blue': (0.1, 0.3, 1.0),
              'yellow': (1.0, 0.9, 0.1)}


def rotate(q, v):
    """Rotate 3-vector v by quaternion q=(x, y, z, w)."""
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def tf_parts(t):
    tr, ro = t.transform.translation, t.transform.rotation
    return np.array([tr.x, tr.y, tr.z]), (ro.x, ro.y, ro.z, ro.w)


def image_to_bgr(msg):
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
    if ch is None:
        raise ValueError(f'unsupported encoding {msg.encoding}')
    buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    conv = {'rgb8': cv2.COLOR_RGB2BGR, 'rgba8': cv2.COLOR_RGBA2BGR, 'bgra8': cv2.COLOR_BGRA2BGR}
    return cv2.cvtColor(img, conv[msg.encoding]) if msg.encoding in conv else img.copy()


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        p = self.declare_parameter
        p('colors', ['red', 'green', 'blue', 'yellow'])
        p('global_frame', 'map')
        p('process_rate_hz', 5.0)
        p('process_width', 480)              # downscale the 1920 px sim image for speed
        p('min_area_px', 120)                # at process_width
        p('horizontal_fov', 1.02974)         # TurtleBot3 waffle camera; used if no CameraInfo
        p('object_depth_m', 0.30)
        p('object_height_m', 0.30)           # used for the camera-only range estimate
        p('camera_range_fallback', True)     # estimate range from the camera beyond LiDAR reach
        p('max_camera_range_m', 9.0)
        p('min_hits', 3)
        p('max_range_m', 6.0)
        p('max_scan_age_s', 0.3)
        p('publish_debug', True)
        p('camera_frame', 'camera_rgb_frame')   # used if the image's frame_id is not in TF
        p('laser_frame', 'base_scan')           # used if the scan's frame_id is not in TF
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.colors = list(g('colors'))
        self.hsv = {c: DEFAULT_HSV[c] for c in self.colors if c in DEFAULT_HSV}
        unknown = set(self.colors) - set(DEFAULT_HSV)
        if unknown:
            self.get_logger().warn(f'no HSV range for {sorted(unknown)}; ignoring them')
        self.global_frame = g('global_frame')
        self.width, self.min_area = int(g('process_width')), int(g('min_area_px'))
        self.hfov, self.depth = float(g('horizontal_fov')), float(g('object_depth_m'))
        self.obj_height = float(g('object_height_m'))
        self.cam_fallback = bool(g('camera_range_fallback'))
        self.max_cam_range = float(g('max_camera_range_m'))
        self.max_range, self.max_scan_age = float(g('max_range_m')), float(g('max_scan_age_s'))
        self.publish_debug = bool(g('publish_debug'))
        self.camera_frame, self.laser_frame = g('camera_frame'), g('laser_frame')

        self.tracker = ObjectTracker(min_hits=int(g('min_hits')))
        self.image = self.scan = self.cam_info = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.create_subscription(Image, 'camera/image_raw', self._on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, 'camera/camera_info', self._on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self._on_scan, qos_profile_sensor_data)
        self.pub = self.create_publisher(DetectedObjectArray, 'detected_objects', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'detected_objects/markers', 10)
        self.debug_pub = self.create_publisher(Image, 'color_detector/debug_image', 2)
        self.create_timer(1.0 / float(g('process_rate_hz')), self._process)
        self.create_timer(0.5, self._publish)
        self._warned = set()
        self.get_logger().info(f'detecting {list(self.hsv)} boxes')

    def _on_image(self, m):
        self.image = m

    def _on_info(self, m):
        self.cam_info = m

    def _on_scan(self, m):
        self.scan = m

    def _warn_once(self, key, text):
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warn(text)

    # ------------------------------------------------------------------ main loop
    def _process(self):
        img_msg, scan = self.image, self.scan
        if img_msg is None:
            return
        self.image = None   # process each frame at most once
        try:
            bgr = image_to_bgr(img_msg)
        except ValueError as e:
            self._warn_once('enc', str(e))
            return
        scale = self.width / bgr.shape[1]
        small = cv2.resize(bgr, (self.width, int(bgr.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        blobs = detect_blobs(small, self.hsv, self.min_area)
        status = self._localise(img_msg, scan, blobs, scale, small.shape[1], small.shape[0])
        self._publish_debug(img_msg, small, blobs, status)

    def _frame(self, frame, fallback):
        """Use the message's frame if TF knows it, otherwise the configured fallback."""
        if frame and self.tf_buffer.can_transform(frame, frame, Time()):
            return frame
        if frame and frame != fallback:
            self._warn_once(f'frame:{frame}', f'frame "{frame}" is not in TF; using "{fallback}"')
        return fallback

    def _localise(self, img_msg, scan, blobs, scale, small_w, small_h):
        """Fill blob['range'] and update the tracker. Returns a short status string."""
        if scan is None:
            return 'no /scan'
        img_t = Time.from_msg(img_msg.header.stamp)
        scan_t = Time.from_msg(scan.header.stamp)
        if abs((img_t - scan_t).nanoseconds) * 1e-9 > self.max_scan_age:
            return 'scan/image out of sync'   # robot may have turned in between
        if not blobs:
            return 'ok'
        cam_frame = self._frame(img_msg.header.frame_id, self.camera_frame)
        laser_frame = self._frame(scan.header.frame_id, self.laser_frame)
        try:
            cam_in_laser = self.tf_buffer.lookup_transform(laser_frame, cam_frame, Time())
            laser_in_map = self.tf_buffer.lookup_transform(self.global_frame, laser_frame, scan_t,
                                                           timeout=Duration(seconds=0.05))
        except TransformException as e:
            self._warn_once('tf', f'waiting for TF ({e}); is Nav2/AMCL running?')
            return 'waiting for TF'

        # Intrinsics at the processed resolution
        if self.cam_info is not None and self.cam_info.k[0] > 0:
            fx = self.cam_info.k[0] * scale
            fy = (self.cam_info.k[4] or self.cam_info.k[0]) * scale
            cx = self.cam_info.k[2] * scale
        else:
            fx = fy = (self.width / 2.0) / math.tan(self.hfov / 2.0)   # square pixels
            cx = self.width / 2.0
        image_size = (small_w, small_h)
        optical = 'optical' in cam_frame
        c_t, c_q = tf_parts(cam_in_laser)
        cam_xy = (c_t[0], c_t[1])

        def ray_angle(u):
            d = ((u - cx) / fx, 0.0, 1.0) if optical else (1.0, (cx - u) / fx, 0.0)
            dl = rotate(c_q, d)
            return math.atan2(dl[1], dl[0])

        m_t, m_q = tf_parts(laser_in_map)
        now = self.get_clock().now()
        for colour, b in blobs.items():
            face = range_from_scan(scan.ranges, scan.angle_min, scan.angle_increment, cam_xy,
                                   ray_angle(b['u_left']), ray_angle(b['u_right']),
                                   range_min=scan.range_min,
                                   range_max=min(scan.range_max, self.max_range))
            b['source'] = 'lidar'
            if face is None and self.cam_fallback:
                # Beyond LiDAR reach (3.5 m on the TB3): use the box's pixel height instead.
                dist = range_from_height(b['bbox'], image_size, fx, fy, cx, self.obj_height)
                if dist is not None and dist <= self.max_cam_range:
                    face = point_on_ray(cam_xy, ray_angle(b['u']), dist)
                    b['source'] = 'camera'
            if face is None:
                continue
            centre_l = push_back(face, cam_xy, self.depth / 2.0)
            p_map = rotate(m_q, (centre_l[0], centre_l[1], 0.0)) + m_t
            self.tracker.update(f'{colour}_box', (float(p_map[0]), float(p_map[1])), now,
                                b['source'])
            b['range'] = math.dist(face, cam_xy)
        return 'ok'

    def _publish_debug(self, img_msg, small, blobs, status):
        if not self.publish_debug or self.debug_pub.get_subscription_count() == 0:
            return
        for colour, b in blobs.items():
            x, y, w, h = b['bbox']
            cv2.rectangle(small, (x, y), (x + w, y + h), (255, 255, 255), 2)
            if 'range' in b:
                cam_only = ' cam' if b.get('source') == 'camera' else ''
                txt = f'{colour} {b["range"]:.2f}m{cam_only}'
            else:
                txt = f'{colour} (no range)'
            cv2.putText(small, txt, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        if status != 'ok':
            cv2.putText(small, status, (6, small.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
        out = Image()
        out.header = img_msg.header
        out.height, out.width = small.shape[:2]
        out.encoding, out.step = 'bgr8', small.shape[1] * 3
        out.data = small.tobytes()
        self.debug_pub.publish(out)

    # ------------------------------------------------------------------ outputs
    def _publish(self):
        now = self.get_clock().now()
        arr = DetectedObjectArray()
        arr.header.frame_id, arr.header.stamp = self.global_frame, now.to_msg()
        markers = MarkerArray()
        for i, (label, tr) in enumerate(sorted(self.tracker.confirmed().items())):
            o = DetectedObject()
            o.label, o.color = label, label.split('_')[0]
            o.pose = PoseStamped()
            o.pose.header = arr.header
            o.pose.pose.position.x, o.pose.pose.position.y = tr['xy']
            o.pose.pose.position.z = self.depth / 2.0
            o.pose.pose.orientation.w = 1.0
            o.observations = int(tr['hits'])
            o.range_source = tr.get('source', '')
            o.last_seen = tr['last'].to_msg()
            arr.objects.append(o)

            m = Marker()
            m.header = arr.header
            m.ns, m.id, m.type = 'detected', i, Marker.CUBE
            m.pose = o.pose.pose
            m.scale.x = m.scale.y = m.scale.z = self.depth
            r, g, b = MARKER_RGB.get(o.color, (1.0, 1.0, 1.0))
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.6
            m.lifetime = Duration(seconds=1.5).to_msg()
            markers.markers.append(m)
        self.pub.publish(arr)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
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
