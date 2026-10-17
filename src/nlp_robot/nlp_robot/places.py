"""Named places ("kitchen", "the dock", ...) loaded from config/places.yaml.

Pure Python (no ROS imports) so it can be unit-tested anywhere. Used by the `goto` tool
now, and by the language layer and commander in later stages.
"""
import re

import yaml

_ARTICLES = re.compile(r'^(the|a|an|my)\s+')


def normalise(text):
    """'The Storage-Bay!' -> 'storage_bay'"""
    t = str(text).strip().lower()
    t = re.sub(r'[^\w\s-]', '', t)
    t = _ARTICLES.sub('', t)
    return re.sub(r'[\s-]+', '_', t).strip('_')


def load_places(path):
    """Return {name: {'pose': [x, y, yaw_deg], 'aliases': [...], 'description': str,
    'lookout': bool}}. Raises ValueError with a clear message if the file is malformed."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    places = data.get('places')
    if not isinstance(places, dict) or not places:
        raise ValueError(f'{path}: expected a non-empty "places:" mapping')
    for name, info in places.items():
        pose = info.get('pose', [])
        if len(pose) != 3 or not all(isinstance(v, (int, float)) for v in pose):
            raise ValueError(f'{path}: place "{name}": pose must be [x, y, yaw_deg]')
        info.setdefault('aliases', [])
        info.setdefault('description', '')
        info.setdefault('lookout', False)
    return places


def find_place(text, places):
    """Resolve a name or alias to the canonical place name, or None."""
    t = normalise(text)
    for name, info in places.items():
        if t == name or t in (normalise(a) for a in info.get('aliases', [])):
            return name
    return None
