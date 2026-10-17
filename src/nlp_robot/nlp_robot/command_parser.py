"""Natural language -> validated plan.

Pipeline: build a grounded prompt (places, seen objects, robot pose, recent
history) -> ask the backend for JSON -> validate against plan_schema -> if
invalid, send the errors back once for a repair -> if the backend itself
fails (no network, Ollama down), try the fallback backend.

Pure Python; the ROS node only calls `CommandParser.parse()`.
"""
import json
import time
from collections import deque
from dataclasses import dataclass, field

from .llm_backends import BackendError
from .plan_schema import ACTIONS, MAX_STEPS, PLAN_JSON_SCHEMA, summarize_plan, validate_plan

SYSTEM_PROMPT = """You are the command interpreter for a mobile robot (a TurtleBot3) in a room.
Turn the user's instruction into a short plan using ONLY these actions:
{actions}

Rules:
- Output JSON only, matching the schema: {{"reply": str, "steps": [{{"action": ..., ...}}]}}.
- Use only the fields an action needs. Angles are degrees, distances metres, times seconds.
- Targets must be one of the known places or a coloured box written as <colour>_box
  (detectable colours: {colors}). You may target a box that has not been seen yet;
  the robot will search for it.
- Write the full target in every step: "find the yellow box and go to it" ->
  find yellow_box, then navigate_to yellow_box (not "it").
- "come back", "go back", "return" with NO place -> return_to_start.
  "return to <place>" / "go back to <place>" -> navigate_to <place> (not return_to_start).
- "go home" -> navigate_to home.
- Only search for objects the user names. "look in the nook" is just navigate_to
  reading_nook; do not add a find for a box nobody mentioned.
- "find / look for / search for X" alone is ONLY find X (search, don't drive up to it).
  Add navigate_to X only if the user also asks to go to it.
- Signs: rotate left = POSITIVE angle_deg, right = NEGATIVE. move forward = POSITIVE
  distance_m, backward/back up/reverse = NEGATIVE.
- Never invent places or coordinates. If the request is impossible (flying, picking things
  up, making things), unsafe, or ambiguous, return an EMPTY steps list and say why or ask a
  short clarifying question in "reply". Never substitute a different action for an
  impossible one.
- At most {max_steps} steps. Keep "reply" to one sentence.

Examples:
"go to the red box, then come back" ->
{{"reply": "Going to the red box, then returning.", "steps": [{{"action": "navigate_to", "target": "red_box"}}, {{"action": "return_to_start"}}]}}
"turn left and drive half a metre" ->
{{"reply": "Turning left 90 degrees and driving 0.5 m.", "steps": [{{"action": "rotate", "angle_deg": 90}}, {{"action": "move", "distance_m": 0.5}}]}}
"turn right 30 degrees and back up 20 cm" ->
{{"reply": "Turning right 30 degrees and backing up 0.2 m.", "steps": [{{"action": "rotate", "angle_deg": -30}}, {{"action": "move", "distance_m": -0.2}}]}}
"is there a green box in the storage bay?" ->
{{"reply": "I'll go to storage and look for the green box.", "steps": [{{"action": "navigate_to", "target": "storage"}}, {{"action": "find", "target": "green_box"}}]}}
"find the blue box" ->
{{"reply": "Searching for the blue box.", "steps": [{{"action": "find", "target": "blue_box"}}]}}
"drive 10 metres forward" ->
{{"reply": "I can move at most 3 m in one go - try a named place instead.", "steps": []}}
"make me a coffee" ->
{{"reply": "I can only drive, turn, search and report - I can't make coffee.", "steps": []}}
"""


@dataclass
class ParseResult:
    ok: bool
    reply: str
    steps: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    backend: str = ''
    latency_s: float = 0.0
    attempts: int = 0
    fallback_reason: str = ''   # set when the primary backend failed and the fallback answered
    llm_reply: str = ''         # the model's own wording, kept for logs/evaluation only


def build_user_message(command, ctx, history):
    lines = ['KNOWN PLACES:']
    for name, info in ctx.places.items():
        x, y, yaw = info['pose']
        al = info.get('aliases') or []
        al_txt = f' (also: {", ".join(al)})' if al else ''
        desc = info.get('description', '')
        lines.append(f'- {name}{al_txt}: ({x:.1f}, {y:.1f}) {desc}')
    lines.append('OBJECTS SEEN SO FAR:')
    if ctx.objects:
        for label, (x, y) in ctx.objects.items():
            lines.append(f'- {label} at ({x:.1f}, {y:.1f})')
    else:
        lines.append('- none yet')
    if ctx.robot_pose:
        x, y, yaw = ctx.robot_pose
        lines.append(f'ROBOT POSE: ({x:.1f}, {y:.1f}), heading {yaw:.0f} deg')
    xmin, xmax, ymin, ymax = ctx.bounds
    lines.append(f'MAP BOUNDS: x {xmin}..{xmax}, y {ymin}..{ymax}')
    if history:
        lines.append('RECENT COMMANDS (oldest first):')
        for h_cmd, h_plan in history:
            lines.append(f'- "{h_cmd}" -> {json.dumps(h_plan)}')
    lines.append(f'COMMAND: "{command}"')
    return '\n'.join(lines)


class CommandParser:
    def __init__(self, backend, ctx_provider, fallback=None, history_len=3, logger=None):
        """ctx_provider: zero-arg callable returning a fresh WorldContext."""
        self.backend = backend
        self.fallback = fallback
        self.ctx_provider = ctx_provider
        self.history = deque(maxlen=history_len)
        self.log = logger or (lambda msg: None)
        actions = '\n'.join(f'- {a}: {desc}' for a, (desc, _) in ACTIONS.items())
        self._actions_txt = actions

    def system_prompt(self, ctx):
        return SYSTEM_PROMPT.format(actions=self._actions_txt, colors=', '.join(ctx.colors),
                                    max_steps=MAX_STEPS)

    def prime(self):
        """Send one throwaway request so the backend loads the model and caches the system
        prompt. Does not touch the command history. Returns an error string or ''."""
        ctx = self.ctx_provider()
        try:
            self.backend.complete(self.system_prompt(ctx),
                                  build_user_message('wait 1 second', ctx, []), PLAN_JSON_SCHEMA)
        except BackendError as e:
            return str(e)
        return ''

    def parse(self, command):
        ctx = self.ctx_provider()
        backends = [self.backend] + ([self.fallback] if self.fallback else [])
        last_err = ''
        for be in backends:
            t0 = time.monotonic()
            try:
                result = self._parse_with(be, command, ctx)
            except BackendError as e:
                last_err = str(e)
                self.log(f'backend {be.describe()} failed: {e}')
                continue
            result.latency_s = time.monotonic() - t0
            result.backend = be.describe()
            if be is not self.backend:
                result.fallback_reason = last_err
            if result.ok:
                self.history.append((command, [{k: v for k, v in s.items()
                                                if k != 'target_kind'} for s in result.steps]))
            return result
        return ParseResult(False, f'No language backend available: {last_err}',
                           errors=[last_err])

    def _parse_with(self, be, command, ctx):
        system = self.system_prompt(ctx)
        user = build_user_message(command, ctx, list(self.history))
        raw = be.complete(system, user, PLAN_JSON_SCHEMA)
        reply, steps, errors = validate_plan(raw, ctx, command)
        attempts = 1
        if errors and be.name != 'rules':
            self.log(f'plan rejected ({"; ".join(errors)}); asking for a repair')
            repair = (user + '\n\nYour previous plan was:\n' + json.dumps(raw) +
                      '\nIt was rejected because:\n- ' + '\n- '.join(errors) +
                      '\nReturn a corrected plan, or empty steps with a clarifying reply.')
            raw = be.complete(system, repair, PLAN_JSON_SCHEMA)
            reply, steps, errors = validate_plan(raw, ctx, command)
            attempts = 2
        if errors:
            return ParseResult(False, f'I could not turn that into a valid plan: {errors[0]}',
                               steps=[], errors=errors, attempts=attempts)
        if not steps:
            # A valid "I need clarification" answer: not executed, but not a crash either.
            return ParseResult(False, reply or 'Nothing to do.', attempts=attempts)
        # What the robot SAYS it will do is generated from the validated steps, never taken
        # from the LLM: in the 3B eval the plan was repaired to rotate -45 but the model's
        # reply still said "Turning left 45 degrees". The LLM's own words are only used
        # when nothing will move (refusals and clarifying questions, above).
        return ParseResult(True, summarize_plan(steps), steps=steps, attempts=attempts,
                           llm_reply=reply)
