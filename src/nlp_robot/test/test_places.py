"""Stage 3 tests: places.yaml is valid, aliases resolve, and every place is somewhere the
robot can actually stand (inside the room, clear of walls and boxes)."""
import math
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from nlp_robot import world_spec as W  # noqa: E402
from nlp_robot.places import find_place, load_places, normalise  # noqa: E402

ROBOT_RADIUS = 0.20   # the value nav.launch.py gives Nav2's costmaps


@pytest.fixture(scope='module')
def places():
    return load_places(os.path.join(PKG, 'config', 'places.yaml'))


@pytest.mark.parametrize('text,expected', [
    ('kitchen', 'kitchen'), ('The Kitchen', 'kitchen'), ('the dock', 'home'),
    ('Storage Bay', 'storage'), ('nook', 'reading_nook'), ('reading nook', 'reading_nook'),
    ('garage', None), ('', None),
])
def test_find_place(places, text, expected):
    assert find_place(text, places) == expected


def test_normalise():
    assert normalise('  The Storage-Bay! ') == 'storage_bay'


def test_places_are_reachable(places):
    for name, info in places.items():
        x, y, _ = info['pose']
        assert W.inside_room(x, y), name
        assert W.distance_to_walls(x, y) > ROBOT_RADIUS + 0.25, f'{name} too close to a wall'
        for box, (bx, by, _) in W.BOXES.items():
            assert math.hypot(bx - x, by - y) > 0.7, f'{name} too close to {box}'


def test_aliases_are_unique(places):
    seen = {}
    for name, info in places.items():
        for key in [name] + [normalise(a) for a in info['aliases']]:
            assert key not in seen, f'"{key}" used by both {seen[key]} and {name}'
            seen[key] = name


def test_bad_file_is_rejected(tmp_path):
    p = tmp_path / 'bad.yaml'
    p.write_text('places:\n  x:\n    pose: [1, 2]\n')
    with pytest.raises(ValueError, match='pose must be'):
        load_places(str(p))
