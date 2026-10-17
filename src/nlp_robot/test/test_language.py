"""Stage 4 tests: schema validation, target resolution, direction checks, the parser's
repair/fallback loop and the rule backend. Pure Python - no ROS, no LLM, no network."""
import json
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from nlp_robot.command_parser import CommandParser, build_user_message  # noqa: E402
from nlp_robot.llm_backends import BackendError, LLMBackend, RuleBackend  # noqa: E402
from nlp_robot.places import load_places  # noqa: E402
from nlp_robot.plan_schema import (PLAN_JSON_SCHEMA, WorldContext, check_directions,  # noqa: E402
                                   completion_text, resolve_target, summarize_plan,
                                   validate_plan)


@pytest.fixture
def ctx():
    return WorldContext(places=load_places(os.path.join(PKG, 'config', 'places.yaml')),
                        objects={'red_box': (3.4, 2.4)}, robot_pose=(0.0, 0.0, 0.0))


class ScriptedBackend(LLMBackend):
    """Stands in for an LLM: returns canned plans in order and records the prompts."""
    name = 'scripted'

    def __init__(self, plans):
        self.plans, self.prompts = list(plans), []

    def complete(self, system, user, schema):
        self.prompts.append(user)
        return self.plans.pop(0)


class DeadBackend(LLMBackend):
    name = 'dead'

    def complete(self, system, user, schema):
        raise BackendError('connection refused')


def plan(*steps, reply='ok'):
    return {'reply': reply, 'steps': list(steps)}


# ---------------------------------------------------------------- targets
@pytest.mark.parametrize('text,expected', [
    ('red box', ('object', 'red_box')),
    ('the Red Box', ('object', 'red_box')),
    ('blue cube', ('object', 'blue_box')),      # not seen yet, but a valid colour
    ('kitchen', ('place', 'kitchen')),
    ('the dock', ('place', 'home')),             # alias
    ('Storage Bay', ('place', 'storage')),
    ('the box', ('object', 'red_box')),          # only one object known
])
def test_resolve_target(ctx, text, expected):
    assert resolve_target(text, ctx) == expected


@pytest.mark.parametrize('text', ['purple elephant', 'purple box', 'garage'])
def test_resolve_unknown_target(ctx, text):
    kind, why = resolve_target(text, ctx)
    assert kind is None and 'Valid targets' in why


# ---------------------------------------------------------------- validation
def test_valid_plan(ctx):
    _, steps, errors = validate_plan(plan({'action': 'navigate_to', 'target': 'red box'},
                                          {'action': 'return_to_start'}), ctx)
    assert errors == []
    assert steps == [{'action': 'navigate_to', 'target': 'red_box', 'target_kind': 'object'},
                     {'action': 'return_to_start'}]


@pytest.mark.parametrize('step,fragment', [
    ({'action': 'fly', 'target': 'kitchen'}, 'unknown action'),
    ({'action': 'move', 'distance_m': 50}, 'outside what the robot can do'),
    ({'action': 'rotate', 'angle_deg': 0}, 'outside what the robot can do'),
    ({'action': 'move'}, 'must be a number'),
    ({'action': 'go_to_pose', 'x': 10, 'y': 0}, 'outside the map'),
    ({'action': 'find', 'target': 'kitchen'}, 'find needs an object'),
    ({'action': 'navigate_to', 'target': 'mars'}, 'unknown target'),
])
def test_invalid_steps(ctx, step, fragment):
    _, _, errors = validate_plan(plan(step), ctx)
    assert errors and fragment in errors[0]


def test_too_many_steps(ctx):
    _, _, errors = validate_plan(plan(*[{'action': 'wait', 'seconds': 1}] * 20), ctx)
    assert any('maximum' in e for e in errors)


def test_non_dict_plan(ctx):
    assert validate_plan(['nope'], ctx)[2]


def test_schema_is_json_serialisable():
    json.dumps(PLAN_JSON_SCHEMA)


# ---------------------------------------------------------------- pronouns
@pytest.mark.parametrize('word', ['it', 'there', 'that one', 'the same', 'IT'])
def test_pronoun_refers_to_previous_target(ctx, word):
    _, steps, errors = validate_plan(plan({'action': 'find', 'target': 'yellow box'},
                                          {'action': 'navigate_to', 'target': word}), ctx)
    assert errors == []
    assert steps[1] == {'action': 'navigate_to', 'target': 'yellow_box', 'target_kind': 'object'}


def test_pronoun_without_antecedent(ctx):
    _, _, errors = validate_plan(plan({'action': 'navigate_to', 'target': 'it'}), ctx)
    assert errors and 'does not refer' in errors[0]


# ---------------------------------------------------------------- direction check
@pytest.mark.parametrize('command,steps,bad', [
    ('turn right by 45 degrees', [{'action': 'rotate', 'angle_deg': 45}], True),
    ('turn right by 45 degrees', [{'action': 'rotate', 'angle_deg': -45}], False),
    ('turn left', [{'action': 'rotate', 'angle_deg': -90}], True),
    ('turn counter-clockwise 30', [{'action': 'rotate', 'angle_deg': 30}], False),
    ('rotate clockwise 30', [{'action': 'rotate', 'angle_deg': 30}], True),
    ('turn left then turn right', [{'action': 'rotate', 'angle_deg': 90},
                                   {'action': 'rotate', 'angle_deg': -90}], False),
    ('turn around', [{'action': 'rotate', 'angle_deg': 180}], False),
    ('back up half a metre', [{'action': 'move', 'distance_m': 0.5}], True),
    ('back up half a metre', [{'action': 'move', 'distance_m': -0.5}], False),
    ('go back 30 cm', [{'action': 'move', 'distance_m': 0.3}], True),
    ('move forward one metre', [{'action': 'move', 'distance_m': -1.0}], True),
    ('reverse 20 cm', [{'action': 'move', 'distance_m': -0.2}], False),
    # "come back" is return_to_start, not a backward move: a forward move stays valid
    ('drive forward 1 m then come back', [{'action': 'move', 'distance_m': 1.0},
                                          {'action': 'return_to_start'}], False),
])
def test_check_directions(command, steps, bad):
    assert bool(check_directions(command, steps)) is bad


@pytest.mark.parametrize('command,expected', [
    ('drive 50 metres forward', [50.0]), ('back up 20 cm', [0.2]),
    ('move 1.5m then 30 centimetres', [1.5, 0.3]), ('go to the office', [])])
def test_requested_distances(command, expected):
    from nlp_robot.plan_schema import requested_distances
    assert requested_distances(command) == pytest.approx(expected)


def test_too_far_is_not_quietly_shortened(ctx):
    """Replays the real v2 failure: 50 m -> rejected -> the model clamped it to 2.0 m."""
    _, _, errors = validate_plan(plan({'action': 'move', 'distance_m': 2.0}), ctx,
                                 'drive 50 metres forward')
    assert errors and 'Do not shorten' in errors[0]
    _, steps, errors = validate_plan(plan(reply='I can only move 3 m.'), ctx,
                                     'drive 50 metres forward')
    assert errors == [] and steps == []


def test_range_error_says_not_to_clamp(ctx):
    _, _, errors = validate_plan(plan({'action': 'move', 'distance_m': 50}), ctx)
    assert 'do NOT shorten' in errors[0]


@pytest.mark.parametrize('command,steps,bad', [
    # the real v2 failure
    ('go to the green box then return to the kitchen',
     [{'action': 'navigate_to', 'target': 'green_box'}, {'action': 'return_to_start'},
      {'action': 'navigate_to', 'target': 'kitchen'}], True),
    ('go to the green box then return to the kitchen',
     [{'action': 'navigate_to', 'target': 'green_box'},
      {'action': 'navigate_to', 'target': 'kitchen'}], False),
    ('go to the office then come back', [{'action': 'navigate_to', 'target': 'office'},
                                         {'action': 'return_to_start'}], False),
])
def test_check_return_to(ctx, command, steps, bad):
    _, _, errors = validate_plan(plan(*steps), ctx, command)
    assert bool(errors) is bad


def test_wrong_turn_is_repaired(ctx):
    """Replays the real v1 failure: qwen2.5:7b answered +45 for "turn right by 45 degrees".
    The validator rejects it, the errors go back to the model, the fix is accepted."""
    be = ScriptedBackend([plan({'action': 'rotate', 'angle_deg': 45}),
                          plan({'action': 'rotate', 'angle_deg': -45})])
    r = CommandParser(be, lambda: ctx).parse('turn right by 45 degrees')
    assert r.ok and r.attempts == 2 and r.steps[0]['angle_deg'] == -45
    assert 'turns LEFT' in be.prompts[1]


def test_reply_always_matches_the_plan(ctx):
    """Replays the 3B eval: plan repaired to -45, but the model's reply still said left."""
    be = ScriptedBackend([plan({'action': 'rotate', 'angle_deg': 45}, reply='Turning left 45.'),
                          plan({'action': 'rotate', 'angle_deg': -45}, reply='Turning left 45.')])
    r = CommandParser(be, lambda: ctx).parse('turn right by 45 degrees')
    assert r.reply == "I'll turn right 45 degrees." and r.llm_reply == 'Turning left 45.'


def test_wrong_turn_twice_moves_nothing(ctx):
    be = ScriptedBackend([plan({'action': 'rotate', 'angle_deg': 45})] * 2)
    r = CommandParser(be, lambda: ctx).parse('turn right by 45 degrees')
    assert not r.ok and r.steps == [] and r.errors


# ---------------------------------------------------------------- parser loop
def test_repair_loop(ctx):
    be = ScriptedBackend([plan({'action': 'navigate_to', 'target': 'the moon'}),
                          plan({'action': 'navigate_to', 'target': 'kitchen'})])
    r = CommandParser(be, lambda: ctx).parse('go to the moon kitchen')
    assert r.ok and r.attempts == 2
    assert 'rejected because' in be.prompts[1]


def test_clarification_is_not_executed(ctx):
    be = ScriptedBackend([plan(reply='Which box do you mean?')])
    r = CommandParser(be, lambda: ctx).parse('go to the box over there')
    assert not r.ok and r.errors == [] and r.reply == 'Which box do you mean?'


def test_fallback_backend(ctx):
    r = CommandParser(DeadBackend(), lambda: ctx, fallback=RuleBackend()).parse('go home')
    assert r.ok and r.backend == 'rules' and 'connection refused' in r.fallback_reason


def test_no_backend(ctx):
    r = CommandParser(DeadBackend(), lambda: ctx).parse('go home')
    assert not r.ok and 'connection refused' in r.reply


def test_prime_does_not_touch_history(ctx):
    be = ScriptedBackend([plan({'action': 'wait', 'seconds': 1})])
    p = CommandParser(be, lambda: ctx)
    assert p.prime() == '' and len(be.prompts) == 1 and len(p.history) == 0
    assert CommandParser(DeadBackend(), lambda: ctx).prime() == 'connection refused'


def test_history_in_prompt(ctx):
    be = ScriptedBackend([plan({'action': 'navigate_to', 'target': 'office'})] * 2)
    p = CommandParser(be, lambda: ctx)
    p.parse('go to the office')
    p.parse('do that again')
    assert 'RECENT COMMANDS' in be.prompts[1] and 'go to the office' in be.prompts[1]


def test_prompt_contains_grounding(ctx):
    msg = build_user_message('hi', ctx, [])
    assert 'kitchen' in msg and 'red_box at (3.4, 2.4)' in msg and msg.endswith('COMMAND: "hi"')


# ---------------------------------------------------------------- rule backend
@pytest.mark.parametrize('cmd,actions', [
    ('go to the red box, then come back', ['navigate_to', 'return_to_start']),
    ('turn left 45 degrees and move forward 50 cm', ['rotate', 'move']),
    ('find the green box then go home', ['find', 'navigate_to']),
    ('turn around', ['rotate']),
    ('wait 3 seconds then go back', ['wait', 'return_to_start']),
    ('find the yellow box and go to it', ['find', 'navigate_to']),
    ('turn right 45 degrees and back up 20 cm', ['rotate', 'move']),
])
def test_rule_backend(ctx, cmd, actions):
    r = CommandParser(RuleBackend(), lambda: ctx).parse(cmd)
    assert r.ok, r.errors
    assert [s['action'] for s in r.steps] == actions


def test_rule_backend_values(ctx):
    r = CommandParser(RuleBackend(), lambda: ctx).parse('turn right 30 then move back 0.4 m')
    assert r.steps[0]['angle_deg'] == -30 and r.steps[1]['distance_m'] == -0.4


def test_rule_backend_back_up(ctx):
    r = CommandParser(RuleBackend(), lambda: ctx).parse('back up 20 cm')
    assert r.ok and r.steps[0]['distance_m'] == pytest.approx(-0.2)


def test_rule_backend_unknown(ctx):
    r = CommandParser(RuleBackend(), lambda: ctx).parse('make me a sandwich')
    assert not r.ok and 'could not understand' in r.reply


# ---------------------------------------------------------------- user-facing text
def test_summary_and_completion_text(ctx):
    r = CommandParser(RuleBackend(), lambda: ctx).parse('go to the red box, then come back')
    assert r.reply == "I'll go to the red box, then come back."
    assert summarize_plan([]) == 'Nothing to do.'
    assert completion_text({'action': 'navigate_to', 'target': 'office'}) == 'Reached the office.'
    assert completion_text({'action': 'rotate', 'angle_deg': -90}) == 'Turned 90 degrees right.'
