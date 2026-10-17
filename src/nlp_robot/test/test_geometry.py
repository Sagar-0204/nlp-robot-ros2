"""Stage 5 tests: the small geometry helpers the commander relies on."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_robot import geometry as G  # noqa: E402


def test_standoff_goal():
    x, y, yaw = G.standoff_goal((0, 0), (2, 0), 0.5)
    assert (round(x, 6), round(y, 6), round(yaw, 6)) == (1.5, 0, 0)
    x, y, yaw = G.standoff_goal((0, 0), (0.2, 0.2), 0.5)  # already close: turn in place
    assert (x, y) == (0, 0) and math.isclose(yaw, math.pi / 4)


def test_quat_roundtrip():
    for yaw in (-3.0, -1.0, 0.0, 0.7, 3.1):
        assert math.isclose(G.quat_to_yaw(*G.yaw_to_quat(yaw)), yaw, abs_tol=1e-9)


def test_wrap():
    assert math.isclose(G.wrap(3 * math.pi / 2), -math.pi / 2)
