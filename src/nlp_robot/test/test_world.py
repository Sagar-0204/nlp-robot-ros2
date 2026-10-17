"""Stage 2 tests: the world spec is sane and the generated files match it.
Run with:  cd src/nlp_robot && python3 -m pytest test/"""
import os
import sys

import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from nlp_robot import world_spec as W  # noqa: E402


def test_boxes_inside_room_with_approach_room():
    for box, (bx, by, _) in W.BOXES.items():
        assert W.inside_room(bx, by), box
        assert W.distance_to_walls(bx, by) > W.BOX_SIZE / 2 + 0.2, box


def test_spawn_point_is_free():
    assert W.inside_room(0.0, 0.0) and W.distance_to_walls(0.0, 0.0) > 0.5
    for box, (bx, by, _) in W.BOXES.items():
        assert (bx ** 2 + by ** 2) ** 0.5 > 1.0, box


def test_map_yaml_matches_spec():
    with open(os.path.join(PKG, 'maps', 'nlp_room.yaml')) as f:
        m = yaml.safe_load(f)
    assert m['image'] == 'nlp_room.pgm'
    assert m['resolution'] == W.MAP_RESOLUTION
    assert m['origin'][:2] == [W.ROOM['xmin'] - W.MAP_MARGIN, W.ROOM['ymin'] - W.MAP_MARGIN]


def test_map_image_size():
    with open(os.path.join(PKG, 'maps', 'nlp_room.pgm'), 'rb') as f:
        header = f.read(80).split(b'\n')
    w, h = map(int, header[2].split())
    r, m, res = W.ROOM, W.MAP_MARGIN, W.MAP_RESOLUTION
    assert (w, h) == (round((r['xmax'] - r['xmin'] + 2 * m) / res),
                      round((r['ymax'] - r['ymin'] + 2 * m) / res))


def test_world_has_every_box_and_wall():
    with open(os.path.join(PKG, 'worlds', 'nlp_room.sdf')) as f:
        sdf = f.read()
    assert '<world name="nlp_room">' in sdf
    for box in W.BOXES:
        assert f'<model name="{box}">' in sdf
    assert sdf.count('<link name="wall_') == len(W.all_wall_segments())


def test_generated_files_are_up_to_date():
    """Fails if world_spec.py was edited without re-running tools/generate_world.py."""
    sys.path.insert(0, os.path.join(PKG, 'tools'))
    import generate_world as G
    with open(os.path.join(PKG, 'worlds', 'nlp_room.sdf')) as f:
        assert f.read() == G.make_sdf(), 'run: python3 tools/generate_world.py'
    with open(os.path.join(PKG, 'maps', 'nlp_room.pgm'), 'rb') as f:
        assert f.read() == G.make_map()[0], 'run: python3 tools/generate_world.py'


def test_ground_truth_bridge_matches_world():
    """Stage 7: every box publishes its true pose and the bridge forwards exactly those."""
    with open(os.path.join(PKG, 'config', 'ground_truth_bridge.yaml')) as f:
        entries = yaml.safe_load(f)
    assert {e['gz_topic_name'] for e in entries} == {f'/model/{b}/pose' for b in W.BOXES}
    assert all(e['ros_topic_name'].startswith('nlp_robot/ground_truth/') for e in entries)
    with open(os.path.join(PKG, 'worlds', 'nlp_room.sdf')) as f:
        sdf = f.read()
    for b in W.BOXES:
        assert f'<topic>/model/{b}/pose</topic>' in sdf
