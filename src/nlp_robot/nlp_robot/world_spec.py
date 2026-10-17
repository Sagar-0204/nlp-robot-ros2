"""Single source of truth for the demo room.

tools/generate_world.py turns this into the Gazebo world (worlds/nlp_room.sdf)
and the matching Nav2 map (maps/nlp_room.pgm + .yaml), so the simulation and
the map can never drift apart. Coordinates are metres in the map frame; the
robot spawns at the origin facing +x.
"""

ROOM = {'xmin': -4.0, 'xmax': 4.0, 'ymin': -3.0, 'ymax': 3.0}
WALL_THICKNESS = 0.10
WALL_HEIGHT = 0.50

# Interior walls as axis-aligned segments: (x0, y0, x1, y1)
INTERIOR_WALLS = [
    (1.5, -3.0, 1.5, -0.9),   # partitions the south-east "storage" bay
    (-1.5, 3.0, -1.5, 0.9),   # partitions the north-west "reading nook"
]

BOX_SIZE = 0.30  # cube edge (m); taller than the LiDAR plane so it shows in /scan

# Coloured boxes: name -> (x, y, rgb). They are NOT drawn into the static map,
# so you can move them in Gazebo and the robot will re-detect them.
BOXES = {
    'red_box': (3.4, 2.4, (0.9, 0.05, 0.05)),
    'blue_box': (-3.5, 2.5, (0.05, 0.15, 0.9)),
    'green_box': (3.5, -2.5, (0.05, 0.75, 0.1)),
    'yellow_box': (-3.4, -2.4, (0.95, 0.85, 0.05)),
}

MAP_RESOLUTION = 0.05
MAP_MARGIN = 0.5


def all_wall_segments():
    r = ROOM
    outer = [
        (r['xmin'], r['ymin'], r['xmax'], r['ymin']),
        (r['xmin'], r['ymax'], r['xmax'], r['ymax']),
        (r['xmin'], r['ymin'], r['xmin'], r['ymax']),
        (r['xmax'], r['ymin'], r['xmax'], r['ymax']),
    ]
    return outer + list(INTERIOR_WALLS)


def distance_to_walls(x, y):
    """Smallest distance from (x, y) to any wall surface."""
    best = float('inf')
    half = WALL_THICKNESS / 2.0
    for x0, y0, x1, y1 in all_wall_segments():
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / L2))
        px, py = x0 + t * dx, y0 + t * dy
        d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5 - half
        best = min(best, d)
    return best


def inside_room(x, y):
    r = ROOM
    return r['xmin'] < x < r['xmax'] and r['ymin'] < y < r['ymax']
