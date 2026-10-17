import math
import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_robot.color_detection import (ObjectTracker, detect_blobs, point_on_ray,  # noqa: E402
                                      push_back, range_from_height, range_from_scan)


def synthetic_image():
    img = np.full((270, 480, 3), (128, 128, 128), np.uint8)   # grey walls/floor (BGR)
    img[100:160, 300:350] = (20, 20, 230)     # red
    img[110:150, 50:90] = (230, 40, 10)       # blue
    img[200:205, 200:205] = (20, 200, 20)     # tiny green speck -> below min_area
    return img


def test_detect_blobs():
    blobs = detect_blobs(synthetic_image(), min_area=150)
    assert set(blobs) == {'red', 'blue'}
    r = blobs['red']
    assert r['u_left'] == 300 and r['u_right'] == 350 and abs(r['u'] - 325) < 1


def test_grey_is_not_a_colour():
    assert detect_blobs(np.full((100, 100, 3), 200, np.uint8)) == {}


def test_range_from_scan_with_camera_offset():
    # Wall-less scan with a 0.3 m wide box face at x=2.0, y in [-0.15, 0.15] (LiDAR frame)
    n = 360
    angle_min, inc = -math.pi, 2 * math.pi / n
    ranges = np.full(n, np.inf)
    for i in range(n):
        th = angle_min + i * inc
        if abs(th) < math.pi / 2:
            y = 2.0 * math.tan(th)
            if abs(y) <= 0.15:
                ranges[i] = 2.0 / math.cos(th)
    cam = (0.13, -0.05)                      # camera ahead of and right of the LiDAR
    a_left = math.atan2(0.15 - cam[1], 2.0 - cam[0])
    a_right = math.atan2(-0.15 - cam[1], 2.0 - cam[0])
    p = range_from_scan(ranges, angle_min, inc, cam, a_left, a_right)
    assert p is not None and abs(p[0] - 2.0) < 0.05 and abs(p[1]) < 0.08
    centre = push_back(p, cam, 0.15)
    assert abs(centre[0] - 2.15) < 0.06


def test_range_from_scan_nothing_there():
    assert range_from_scan(np.full(360, np.inf), -math.pi, math.pi / 180, (0, 0), 0.1, -0.1) is None


def test_tracker_confirmation_and_move():
    t = ObjectTracker(min_hits=3, gate_m=0.8, jump_hits=3)
    for k in range(2):
        t.update('red_box', (1.0, 1.0), k)
    assert 'red_box' not in t.confirmed()
    t.update('red_box', (1.02, 0.98), 2)
    assert 'red_box' in t.confirmed()
    t.update('red_box', (5.0, 5.0), 3)                  # single outlier: ignored
    assert math.dist(t.confirmed()['red_box']['xy'], (1, 1)) < 0.1
    for k in range(3):                                  # consistent new position: re-acquire
        t.update('red_box', (3.0, 0.0), 4 + k)
    assert math.dist(t.tracks['red_box']['xy'], (3, 0)) < 0.1


# ---------------------------------------------------------------- camera-only range
FX = FY = 400.0
CX = 240.0
IMG = (480, 270)


def test_range_from_height_centred():
    # A 0.30 m tall box 5 m away is 400 * 0.30 / 5 = 24 px tall
    assert math.isclose(range_from_height((228, 120, 24, 24), IMG, FX, FY, CX, 0.30), 5.0)


def test_range_from_height_off_axis():
    # Same depth, centre 100 px right of the axis -> longer slant range along the ray
    d = range_from_height((328, 120, 24, 24), IMG, FX, FY, CX, 0.30)
    assert math.isclose(d, 5.0 * math.hypot(1.0, 100 / FX))


@pytest.mark.parametrize('bbox', [
    (0, 120, 24, 24),       # touches the left edge
    (228, 0, 24, 24),       # touches the top edge (could be clipped)
    (228, 250, 24, 20),     # touches the bottom edge
    (228, 120, 4, 4),       # too small to trust
])
def test_range_from_height_rejects_unreliable_blobs(bbox):
    assert range_from_height(bbox, IMG, FX, FY, CX, 0.30) is None


def test_camera_estimate_is_corrected_by_lidar():
    # Far away: camera-only estimate, 0.4 m off. Close up: LiDAR estimates take over.
    t = ObjectTracker(min_hits=3, gate_m=0.8, jump_hits=3)
    for k in range(3):
        t.update('red_box', point_on_ray((0, 0), 0.0, 5.4), k)
    assert 'red_box' in t.confirmed()
    for k in range(5):
        t.update('red_box', (5.0, 0.0), 3 + k)
    assert math.dist(t.confirmed()['red_box']['xy'], (5.0, 0.0)) < 0.1


def test_tracker_reports_range_source():
    t = ObjectTracker(min_hits=3)
    for k in range(3):
        t.update('red_box', (5.0, 0.0), k, 'camera')
    assert t.confirmed()['red_box']['source'] == 'camera'
    t.update('red_box', (5.05, 0.0), 3, 'lidar')
    assert t.confirmed()['red_box']['source'] == 'lidar'
    t.update('red_box', (9.0, 9.0), 4, 'camera')        # ignored outlier keeps the source
    assert t.confirmed()['red_box']['source'] == 'lidar'
