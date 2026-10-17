"""The contract between the LLM and the robot.

The LLM may only emit plans made of the primitives below. Everything it emits
is validated here before a single motor command is sent: unknown actions,
unknown targets, out-of-range numbers, over-long plans and turns/moves in the wrong
direction are rejected with a message that is fed back to the LLM for one repair attempt.

Pure Python (no ROS imports) so it can be unit-tested anywhere.
"""
import re
from dataclasses import dataclass, field

MAX_STEPS = 12

# action -> (description shown to the LLM, required fields)
ACTIONS = {
    'navigate_to': ('Drive to a named place or a detected object, e.g. target="kitchen" or '
                    'target="red_box". If the object has not been seen yet the robot searches for '
                    'it first.', ['target']),
    'find': ('Search for an object (spin, then visit lookout places) without driving up to it.',
             ['target']),
    'go_to_pose': ('Drive to explicit map coordinates x, y in metres, optional yaw_deg.',
                   ['x', 'y']),
    'return_to_start': ('Drive back to where the robot was when the user\'s previous movement '
                        'command began ("come back", "go back", "return").', []),
    'rotate': ('Turn in place by angle_deg degrees; positive = left/counter-clockwise, '
               'negative = right/clockwise.', ['angle_deg']),
    'move': ('Drive straight by distance_m metres; positive = forward, negative = backward.',
             ['distance_m']),
    'wait': ('Pause for seconds.', ['seconds']),
    'say': ('Report something to the user, e.g. what was found. Field: text.', ['text']),
}

LIMITS = {
    'angle_deg': (-720.0, 720.0),
    'distance_m': (-2.0, 3.0),
    'seconds': (0.1, 60.0),
}

_STEP_PROPS = {
    'action': {'type': 'string', 'enum': list(ACTIONS)},
    'target': {'type': 'string'},
    'x': {'type': 'number'},
    'y': {'type': 'number'},
    'yaw_deg': {'type': 'number'},
    'angle_deg': {'type': 'number'},
    'distance_m': {'type': 'number'},
    'seconds': {'type': 'number'},
    'text': {'type': 'string'},
}

# JSON schema handed to the LLM (Ollama `format`, Anthropic tool input_schema,
# OpenAI response_format). Kept flat on purpose: small local models follow a
# flat step object far more reliably than a oneOf per action.
PLAN_JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'reply': {'type': 'string',
                  'description': 'One short sentence to the user: what you will do, or a '
                                 'clarifying question if the command cannot be done.'},
        'steps': {'type': 'array', 'maxItems': MAX_STEPS,
                  'items': {'type': 'object', 'properties': _STEP_PROPS,
                            'required': ['action']}},
    },
    'required': ['reply', 'steps'],
}

_ARTICLES = re.compile(r'^(the|a|an|that|this|my)\s+')

# Words that point back at the previous target in the same plan
# ("find the yellow box and go to it", "go to the office and wait there").
PRONOUNS = {'it', 'there', 'that', 'this', 'them', 'that_one', 'this_one', 'same', 'same_place',
            'same_box', 'one'}


@dataclass
class WorldContext:
    """What the robot currently knows; used both for prompting and validation."""
    places: dict = field(default_factory=dict)       # name -> {'pose': [x,y,yaw], 'aliases': [...]}
    objects: dict = field(default_factory=dict)      # label -> (x, y)
    colors: list = field(default_factory=lambda: ['red', 'green', 'blue', 'yellow'])
    bounds: tuple = (-4.0, 4.0, -3.0, 3.0)           # xmin, xmax, ymin, ymax
    robot_pose: tuple = None                         # (x, y, yaw_deg) or None


def _clean(text):
    t = str(text).strip().lower()
    t = re.sub(r'[^\w\s-]', '', t)
    t = _ARTICLES.sub('', t)
    return re.sub(r'[\s-]+', '_', t).strip('_')


def resolve_target(text, ctx):
    """Map free text to ('place', name) or ('object', label). Returns (kind, name) or (None, why)."""
    t = _clean(text)
    if not t:
        return None, 'empty target'
    for name, info in ctx.places.items():
        if t == name or t in (_clean(a) for a in info.get('aliases', [])):
            return 'place', name
    if t in ctx.objects:
        return 'object', t
    tokens = t.split('_')
    colours = [c for c in ctx.colors if c in tokens]
    if len(colours) == 1:
        return 'object', f'{colours[0]}_box'
    if len(tokens) == 1 and tokens[0] in ('box', 'cube', 'object', 'block') and len(ctx.objects) == 1:
        return 'object', next(iter(ctx.objects))
    options = sorted(ctx.places) + [f'{c}_box' for c in ctx.colors]
    return None, f'unknown target "{text}". Valid targets: {", ".join(options)}'


def _num(step, key, errors, i):
    try:
        return float(step[key])
    except (KeyError, TypeError, ValueError):
        errors.append(f'step {i}: "{key}" must be a number')
        return None


# Direction words. Small models often get the sign convention wrong ("turn right 45" ->
# +45, which turns LEFT), so the sign is checked against the user's own words.
_LEFT = re.compile(r'\b(left|counter[- ]?clockwise|anti[- ]?clockwise)\b')
_RIGHT = re.compile(r'\b(right|clockwise)\b')
_FORWARD = re.compile(r'\b(forwards?|ahead)\b')
# Not plain "back": "come back"/"go back" mean return_to_start, not a backward move.
_BACKWARD = re.compile(r'\b(back(?:ing)?\s+up|backwards?|reverse'
                       r'|back\s+(?:by\s+)?(?:\d|a\s+(?:little|bit)|half|one|two|three))')


def check_directions(command, steps):
    """Errors for rotate/move steps whose sign contradicts the command's own direction
    words. Only applies when the command uses one direction, so "turn left then right"
    is left alone."""
    text = str(command).lower()
    left, right = bool(_LEFT.search(text)), bool(_RIGHT.search(_LEFT.sub(' ', text)))
    fwd, back = bool(_FORWARD.search(text)), bool(_BACKWARD.search(text))
    errors = []
    for s in steps:
        if s['action'] == 'rotate' and left != right:
            ang = s['angle_deg']
            if right and ang > 0:
                errors.append(f'rotate angle_deg={ang:g} turns LEFT (positive = counter-clockwise),'
                              f' but the command says right: use angle_deg={-ang:g}')
            if left and ang < 0:
                errors.append(f'rotate angle_deg={ang:g} turns RIGHT (negative = clockwise),'
                              f' but the command says left: use angle_deg={-ang:g}')
        if s['action'] == 'move' and fwd != back:
            d = s['distance_m']
            if back and d > 0:
                errors.append(f'move distance_m={d:g} drives FORWARD, but the command says '
                              f'backward: use distance_m={-d:g}')
            if fwd and d < 0:
                errors.append(f'move distance_m={d:g} drives BACKWARD, but the command says '
                              f'forward: use distance_m={-d:g}')
    return errors


_DIST = re.compile(r'(\d+(?:\.\d+)?)\s*(cm|centimet(?:er|re)s?|m|met(?:er|re)s?)\b')


def requested_distances(command):
    """Explicit distances in the command, in metres: "drive 50 metres" -> [50.0]."""
    out = []
    for num, unit in _DIST.findall(str(command).lower()):
        out.append(float(num) / (100.0 if unit.startswith('c') else 1.0))
    return out


def check_distances(command, steps):
    """A plan must not quietly shorten a distance the robot can't do in one step."""
    too_far = [d for d in requested_distances(command) if d > LIMITS['distance_m'][1]]
    if too_far and any(s['action'] == 'move' for s in steps):
        return [f'the user asked to move {too_far[0]:g} m, more than the {LIMITS["distance_m"][1]:g} m '
                f'limit for one move. Do not shorten it: return empty steps and explain the limit.']
    return []


_RETURN_TO = re.compile(r'\b(?:return|go back|come back|head back|get back)\s+to\b')
_RETURN_ALONE = re.compile(r'\b(?:return|go back|come back|head back|get back)\b(?!\s+to\b)')


def check_return_to(command, steps):
    """"return to the kitchen" means navigate_to kitchen, not return_to_start + navigate_to."""
    text = str(command).lower()
    if _RETURN_TO.search(text) and not _RETURN_ALONE.search(text):
        if any(s['action'] == 'return_to_start' for s in steps):
            return ['the command says "return to <place>": that is navigate_to <place> only. '
                    'Remove return_to_start.']
    return []


def validate_plan(raw, ctx, command=None):
    """Validate an LLM plan. Returns (reply, steps, errors). steps are normalised dicts.
    If `command` is given, the plan is also checked against the user's own words: turn and
    move directions, distances beyond the limit, and "return to <place>"."""
    errors = []
    if not isinstance(raw, dict):
        return '', [], ['plan must be a JSON object with "reply" and "steps"']
    reply = str(raw.get('reply', '')).strip()
    raw_steps = raw.get('steps', [])
    if not isinstance(raw_steps, list):
        return reply, [], ['"steps" must be a list']
    if len(raw_steps) > MAX_STEPS:
        errors.append(f'plan has {len(raw_steps)} steps; the maximum is {MAX_STEPS}')

    steps = []
    last_target = None   # (kind, name) of the most recent navigate_to/find in this plan
    for i, s in enumerate(raw_steps[:MAX_STEPS]):
        if not isinstance(s, dict):
            errors.append(f'step {i}: must be an object')
            continue
        a = str(s.get('action', '')).strip().lower()
        if a not in ACTIONS:
            errors.append(f'step {i}: unknown action "{a}". Allowed: {", ".join(ACTIONS)}')
            continue
        out = {'action': a}
        if a in ('navigate_to', 'find'):
            text = s.get('target', '')
            if _clean(text) in PRONOUNS:
                if last_target is None:
                    errors.append(f'step {i}: "{text}" does not refer to an earlier target in '
                                  f'this plan; name the place or box explicitly')
                    continue
                kind, name = last_target
            else:
                kind, name = resolve_target(text, ctx)
            if kind is None:
                errors.append(f'step {i}: {name}')
                continue
            last_target = (kind, name)
            if a == 'find' and kind != 'object':
                errors.append(f'step {i}: find needs an object such as red_box, not a place')
                continue
            out.update(target=name, target_kind=kind)
        elif a == 'go_to_pose':
            x, y = _num(s, 'x', errors, i), _num(s, 'y', errors, i)
            if x is None or y is None:
                continue
            xmin, xmax, ymin, ymax = ctx.bounds
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                errors.append(f'step {i}: ({x:.2f}, {y:.2f}) is outside the map '
                              f'x[{xmin}, {xmax}] y[{ymin}, {ymax}]')
                continue
            yaw = s.get('yaw_deg')
            out.update(x=x, y=y, yaw_deg=None if yaw is None else float(yaw))
        elif a in ('rotate', 'move', 'wait'):
            key = {'rotate': 'angle_deg', 'move': 'distance_m', 'wait': 'seconds'}[a]
            v = _num(s, key, errors, i)
            if v is None:
                continue
            lo, hi = LIMITS[key]
            if not (lo <= v <= hi) or (a != 'wait' and v == 0):
                # Worded so the repair round doesn't just clamp the number: in v2,
                # "drive 50 metres forward" came back as move 2.0 after this error.
                errors.append(f'step {i}: {key}={v} is outside what the robot can do in one step '
                              f'[{lo}, {hi}] (and must be non-zero). If the user asked for more '
                              f'than that, do NOT shorten it: return empty steps and say why.')
                continue
            out[key] = v
        elif a == 'say':
            text = str(s.get('text', '')).strip()
            if not text:
                errors.append(f'step {i}: say needs text')
                continue
            out['text'] = text[:300]
        steps.append(out)
    if command is not None:
        errors += check_directions(command, steps)
        errors += check_distances(command, steps)
        errors += check_return_to(command, steps)
    return reply, steps, errors


def describe_step(step):
    a = step['action']
    if a == 'navigate_to':
        return f'navigate to {step["target"]}'
    if a == 'find':
        return f'search for {step["target"]}'
    if a == 'go_to_pose':
        yaw = '' if step.get('yaw_deg') is None else f', yaw {step["yaw_deg"]:.0f} deg'
        return f'go to ({step["x"]:.2f}, {step["y"]:.2f}{yaw})'
    if a == 'return_to_start':
        return 'return to start'
    if a == 'rotate':
        return f'rotate {step["angle_deg"]:+.0f} deg'
    if a == 'move':
        return f'move {step["distance_m"]:+.2f} m'
    if a == 'wait':
        return f'wait {step["seconds"]:.1f} s'
    if a == 'say':
        return f'say "{step["text"]}"'
    return a


def strip_internal(steps):
    """Steps as JSON-friendly dicts without internal keys (for plan_json)."""
    return [{k: v for k, v in s.items() if k != 'target_kind' and v is not None} for s in steps]


def _name(target):
    return target.replace('_', ' ')


def _ref(target):
    """'home' reads without an article; everything else gets 'the'."""
    return 'home' if target == 'home' else f'the {_name(target)}'


def _phrase(step):
    a = step['action']
    if a == 'navigate_to':
        return f'go to {_ref(step["target"])}'
    if a == 'find':
        return f'look for {_ref(step["target"])}'
    if a == 'go_to_pose':
        return f'drive to ({step["x"]:.1f}, {step["y"]:.1f})'
    if a == 'return_to_start':
        return 'come back'
    if a == 'rotate':
        ang = step['angle_deg']
        return f'turn {"left" if ang > 0 else "right"} {abs(ang):.0f} degrees'
    if a == 'move':
        d = step['distance_m']
        return f'move {"forward" if d > 0 else "back"} {abs(d):.2f} m'
    if a == 'wait':
        return f'wait {step["seconds"]:g} s'
    if a == 'say':
        return 'report back'
    return a


def summarize_plan(steps):
    """One readable sentence for a validated plan, e.g. "I'll go to the red box, then come back."."""
    if not steps:
        return 'Nothing to do.'
    return "I'll " + ', then '.join(_phrase(s) for s in steps) + '.'


def completion_text(step):
    """What the robot says after a step succeeds."""
    a = step['action']
    if a == 'navigate_to':
        return f'Reached {_ref(step["target"])}.'
    if a == 'go_to_pose':
        return f'Reached ({step["x"]:.1f}, {step["y"]:.1f}).'
    if a == 'return_to_start':
        return 'Back where I started.'
    if a == 'rotate':
        return f'Turned {abs(step["angle_deg"]):.0f} degrees {"left" if step["angle_deg"] > 0 else "right"}.'
    if a == 'move':
        return f'Moved {abs(step["distance_m"]):.2f} m {"forward" if step["distance_m"] > 0 else "back"}.'
    if a == 'wait':
        return f'Waited {step["seconds"]:g} s.'
    return ''
