"""Small 2-D geometry helpers (pure Python, unit-tested)."""
import math


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_to_quat(yaw):
    """(x, y, z, w) quaternion for a rotation about z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def standoff_goal(robot_xy, obj_xy, standoff):
    """Goal pose `standoff` metres short of the object on the line from the robot,
    facing the object. If the robot is already closer than the standoff, it just
    turns to face the object where it stands."""
    rx, ry = robot_xy
    ox, oy = obj_xy
    dx, dy = ox - rx, oy - ry
    d = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx) if d > 1e-6 else 0.0
    if d <= standoff:
        return rx, ry, yaw
    k = (d - standoff) / d
    return rx + k * dx, ry + k * dy, yaw


def bearing_in_frame(u, image_width, fx, cx=None):
    """Horizontal bearing (rad, +left) of pixel column u for a forward-looking camera."""
    if cx is None:
        cx = image_width / 2.0
    return math.atan2(cx - u, fx)


def fx_from_hfov(image_width, hfov):
    return (image_width / 2.0) / math.tan(hfov / 2.0)
