"""Colour-box perception core (no ROS imports, unit-tested).

Camera gives *bearing*, LiDAR gives *range*:
  1. HSV threshold the image -> one blob per colour -> its left/right pixel columns.
  2. Turn those columns into two rays and express them in the LiDAR frame.
  3. Take the LiDAR returns that fall between the rays; the closest cluster is the
     box's front face. Push the point back by half the box depth -> box centre.
  4. A tracker fuses repeated sightings (EMA) and only "confirms" an object after
     a few consistent hits, so a single noisy frame never sends the robot anywhere.

The TurtleBot3 LiDAR only reaches 3.5 m, so a box further away has no range from step 3.
For those, `range_from_height` estimates the distance from the camera alone: the box
has a known height, so its height in pixels gives its distance (pinhole model). The
height is used rather than the width because a cube's visible width changes as it
turns, but its height does not. This estimate is coarser than the LiDAR estimate.
Once the robot is within LiDAR range, the tracker replaces it with the LiDAR estimate.
"""
import math

import cv2
import numpy as np

# HSV ranges for OpenCV (H: 0-179, S/V: 0-255). Red wraps around 0, hence two ranges.
DEFAULT_HSV = {
    'red': [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (179, 255, 255))],
    'yellow': [((20, 120, 90), (34, 255, 255))],
    'green': [((40, 100, 50), (85, 255, 255))],
    'blue': [((100, 120, 50), (130, 255, 255))],
}

_KERNEL = np.ones((5, 5), np.uint8)


def detect_blobs(bgr, hsv_ranges=None, min_area=150):
    """Return {colour: dict(u=cx, u_left, u_right, v, area, bbox)} for the largest blob of each colour."""
    hsv_ranges = hsv_ranges or DEFAULT_HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = {}
    for colour, ranges in hsv_ranges.items():
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out[colour] = dict(u=x + w / 2.0, u_left=float(x), u_right=float(x + w),
                           v=y + h / 2.0, area=float(area), bbox=(x, y, w, h))
    return out


def range_from_scan(ranges, angle_min, angle_inc, cam_xy, ang_a, ang_b,
                    range_min=0.05, range_max=12.0, shrink=0.2, face_band=0.25):
    """Find the LiDAR point cluster between two rays from the camera.

    cam_xy: camera origin in the LiDAR frame. ang_a/ang_b: ray angles (rad) in the
    LiDAR frame, measured from the camera origin. Returns (x, y) of the nearest
    face in the LiDAR frame, or None.
    """
    r = np.asarray(ranges, dtype=float)
    th = angle_min + angle_inc * np.arange(r.size)
    ok = np.isfinite(r) & (r > range_min) & (r < range_max)
    if not ok.any():
        return None
    px, py = r * np.cos(th), r * np.sin(th)
    vx, vy = px - cam_xy[0], py - cam_xy[1]
    ang = np.arctan2(vy, vx)
    lo, hi = min(ang_a, ang_b), max(ang_a, ang_b)
    centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * (1.0 - shrink)
    diff = np.arctan2(np.sin(ang - centre), np.cos(ang - centre))
    sel = ok & (np.abs(diff) <= max(half, math.radians(0.6)))
    if not sel.any():
        return None
    dist = np.hypot(vx[sel], vy[sel])
    near = dist <= dist.min() + face_band
    return float(px[sel][near].mean()), float(py[sel][near].mean())


def range_from_height(bbox, image_size, fx, fy, cx, object_height, min_px=6, edge_px=2):
    """Camera-only distance (m) from the camera to the near face of a box of known height.

    bbox: (x, y, w, h) in pixels; image_size: (width, height). fx/fy/cx: intrinsics at
    the same resolution. Returns the distance along the horizontal ray through the blob
    centre, or None if the blob is too small or cut off by the image border (a clipped
    blob looks smaller than it is, so the box would seem further away than it is).
    """
    x, y, w, h = bbox
    width, height = image_size
    if h < min_px or fy <= 0:
        return None
    if x <= edge_px or y <= edge_px or x + w >= width - edge_px or y + h >= height - edge_px:
        return None
    depth = fy * object_height / h                  # along the optical axis
    off = (x + w / 2.0 - cx) / fx                   # tan(horizontal bearing)
    return depth * math.sqrt(1.0 + off * off)


def point_on_ray(origin_xy, angle, dist):
    return origin_xy[0] + dist * math.cos(angle), origin_xy[1] + dist * math.sin(angle)


def push_back(point_xy, from_xy, depth):
    """Move a front-face point `depth` metres further along the viewing ray."""
    dx, dy = point_xy[0] - from_xy[0], point_xy[1] - from_xy[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return point_xy
    return point_xy[0] + dx / d * depth, point_xy[1] + dy / d * depth


class ObjectTracker:
    """Per-label position fusion with confirmation and re-acquisition."""

    def __init__(self, min_hits=3, alpha=0.3, gate_m=0.8, jump_hits=3):
        self.min_hits, self.alpha, self.gate, self.jump_hits = min_hits, alpha, gate_m, jump_hits
        self.tracks = {}   # label -> dict(xy, hits, last, jump_xy, jumps, source)

    def update(self, label, xy, t, source=''):
        """source: how this sighting's range was measured ("lidar" or "camera")."""
        tr = self.tracks.get(label)
        if tr is None:
            self.tracks[label] = dict(xy=tuple(xy), hits=1, last=t, jump_xy=None, jumps=0,
                                      source=source)
            return
        d = math.dist(tr['xy'], xy)
        if d <= self.gate:
            a = self.alpha
            tr['xy'] = ((1 - a) * tr['xy'][0] + a * xy[0], (1 - a) * tr['xy'][1] + a * xy[1])
            tr['hits'] += 1
            tr['jumps'], tr['jump_xy'] = 0, None
            tr['source'] = source
        else:
            # Far from the current estimate: the box was moved, or this is an outlier.
            if tr['jump_xy'] is not None and math.dist(tr['jump_xy'], xy) <= self.gate:
                tr['jumps'] += 1
            else:
                tr['jump_xy'], tr['jumps'] = tuple(xy), 1
            if tr['jumps'] >= self.jump_hits or tr['hits'] < self.min_hits:
                self.tracks[label] = dict(xy=tuple(xy), hits=1 if tr['hits'] < self.min_hits
                                          else self.min_hits, last=t, jump_xy=None, jumps=0,
                                          source=source)
                return
        tr['last'] = t

    def confirmed(self):
        return {k: v for k, v in self.tracks.items() if v['hits'] >= self.min_hits}

    def forget(self, label):
        self.tracks.pop(label, None)
