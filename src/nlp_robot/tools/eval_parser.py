#!/usr/bin/env python3
"""Benchmark a language backend on a fixed set of commands - no ROS or robot needed.

    python3 tools/eval_parser.py --backend rules
    python3 tools/eval_parser.py --backend ollama --model qwen2.5:7b-instruct
    python3 tools/eval_parser.py --backend ollama --model qwen2.5:3b-instruct
    ANTHROPIC_API_KEY=... python3 tools/eval_parser.py --backend anthropic
    GEMINI_API_KEY=...    python3 tools/eval_parser.py --backend gemini --delay 6   # free tier

    # compare against an earlier run (prints FIXED / BROKEN per command)
    python3 tools/eval_parser.py --backend ollama --label v2 \\
        --compare ../../results/eval_ollama_qwen2.5-7b-instruct_v1.csv

Reports plan accuracy, correct refusals, repair rate and latency, and writes a CSV to
<workspace>/results/ that you can drop straight into a report. Plans are compared on
action + target; numbers are checked loosely (same sign, within 10 %). Run it with the
simulation CLOSED, otherwise Gazebo competes for the CPU/GPU and the timings are wrong.
"""
import argparse
import csv
import os
import statistics
import sys
import time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, PKG)

from nlp_robot.command_parser import CommandParser  # noqa: E402
from nlp_robot.llm_backends import make_backend  # noqa: E402
from nlp_robot.places import load_places  # noqa: E402
from nlp_robot.plan_schema import WorldContext  # noqa: E402

N, F, R, M, W, G = 'navigate_to', 'find', 'return_to_start', 'move', 'wait', 'go_to_pose'
ROT = 'rotate'

# (command, expected steps). Each expected step: (action, target-or-number-or-None).
# Expected [] = the model should refuse or ask for clarification (nothing may move).
CASES = [
    # paraphrases of places
    ('go to the red box, then come back', [(N, 'red_box'), (R, None)]),
    ('drive over to the kitchen', [(N, 'kitchen')]),
    ('head to the dock', [(N, 'home')]),
    ('could you please go to the storage bay?', [(N, 'storage')]),
    ('take a look in the nook', [(N, 'reading_nook')]),
    # objects
    ('find the blue box', [(F, 'blue_box')]),
    ('where is the yellow cube? go there', [(N, 'yellow_box')]),
    ('find the green box and go to it', [(N, 'green_box')]),
    # turning and moving: signs matter
    ('turn left', [(ROT, 90)]),
    ('turn right by 45 degrees', [(ROT, -45)]),
    ('do a full turn', [(ROT, 360)]),
    ('turn around and go back 30 cm', [(ROT, 180), (M, -0.3)]),
    ('move forward one metre', [(M, 1.0)]),
    ('back up a little, half a meter', [(M, -0.5)]),
    # multi-step
    ('wait five seconds then go home', [(W, 5), (N, 'home')]),
    ('go to the office, wait 2 seconds, then come back', [(N, 'office'), (W, 2), (R, None)]),
    ('visit the red box and then the green box', [(N, 'red_box'), (N, 'green_box')]),
    ('go to the green box then return to the kitchen', [(N, 'green_box'), (N, 'kitchen')]),
    ('check whether there is a blue box in the reading nook',
     [(N, 'reading_nook'), (F, 'blue_box')]),
    ('go to coordinates x 1.5, y -0.5', [(G, None)]),
    ('come back', [(R, None)]),
    # must refuse / ask
    ('bring me a coffee', []),
    ('go to the purple box', []),
    ('fly to the ceiling', []),
    ('drive 50 metres forward', []),
    ('go to the garage', []),
]


BACKEND_ERROR = 'No language backend available'
FIELDS = ['command', 'expected', 'correct', 'attempts', 'latency_s', 'plan', 'reply', 'llm_reply',
          'rate_limit_retries']


def _collapse(steps):
    """find X immediately followed by navigate_to X behaves exactly like navigate_to X
    (navigate_to searches for an unseen object anyway), so grade them the same."""
    out = []
    for s in steps:
        if (out and s[0] == N and out[-1][0] == F and out[-1][1] == s[1]):
            out[-1] = s
        else:
            out.append(s)
    return out


def _as_tuples(steps):
    return [(s['action'], s.get('target', s.get('angle_deg', s.get('distance_m',
                                                                   s.get('seconds')))))
            for s in steps]


def matches(steps, expected):
    got, exp = _collapse(_as_tuples(steps)), _collapse(list(expected))
    if len(got) != len(exp):
        return False
    for (ga, gv), (ea, ev) in zip(got, exp):
        if ga != ea:
            return False
        if isinstance(ev, str) and gv != ev:
            return False
        if isinstance(ev, (int, float)):
            if gv is None or gv * ev <= 0 or abs(gv - ev) > 0.1 * abs(ev) + 1e-6:
                return False
    return True


def fmt_plan(steps):
    return '; '.join(f'{a}:{"" if v is None else v}' for a, v in _as_tuples(steps))


def compare(rows, path):
    with open(path) as f:
        old = {r['command']: int(r['correct']) for r in csv.DictReader(f)}
    print(f'\nCompared with {os.path.basename(path)}:')
    fixed = [r['command'] for r in rows if old.get(r['command']) == 0 and r['correct']]
    broken = [r['command'] for r in rows if old.get(r['command']) == 1 and not r['correct']]
    for c in fixed:
        print(f'  FIXED   {c}')
    for c in broken:
        print(f'  BROKEN  {c}')
    common = [r for r in rows if r['command'] in old]
    print(f'  score on the {len(common)} shared commands: '
          f'{sum(old[r["command"]] for r in common)} -> {sum(r["correct"] for r in common)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', default='rules', help='ollama | gemini | anthropic | openai | rules')
    ap.add_argument('--model', default='')
    ap.add_argument('--base-url', default='')
    ap.add_argument('--label', default='', help='tag for this run, e.g. v1, v2, prompt-b')
    ap.add_argument('--out', default='', help='CSV path (default: <workspace>/results/...)')
    ap.add_argument('--compare', default='', help='earlier CSV to compare against')
    ap.add_argument('--resume', default='',
                    help='earlier CSV of the SAME backend/model: keep its results and rerun only '
                         'the commands that hit a backend error (timeout, quota)')
    ap.add_argument('--delay', type=float, default=0.0,
                    help='seconds to pause between commands (cloud free tiers allow only a few '
                         'requests per minute); not counted in the latency')
    a = ap.parse_args()

    places = load_places(os.path.join(PKG, 'config', 'places.yaml'))
    ctx = WorldContext(places=places, objects={'red_box': (3.4, 2.4)}, robot_pose=(0, 0, 0))
    be = make_backend(a.backend, model=a.model, base_url=a.base_url)
    ok, why = be.check()
    if not ok:
        sys.exit(f'Cannot evaluate {be.describe()}: {why}')
    secs, where = be.warmup()
    if secs > 0.5 or where:
        print(f'model ready in {secs:.1f} s' + (f' ({where})' if where else ''))

    previous = {}
    if a.resume:
        with open(a.resume) as f:
            previous = {r['command']: r for r in csv.DictReader(f)}

    rows, lat, sent, errors = [], [], 0, 0
    for cmd, expected in CASES:
        old = previous.get(cmd)
        if old and not old['reply'].startswith(BACKEND_ERROR):
            row = {k: old.get(k, '') for k in FIELDS}
            row.update(correct=int(old['correct']), attempts=int(old['attempts']),
                       latency_s=float(old['latency_s']),
                       rate_limit_retries=int(old.get('rate_limit_retries') or 0))
            rows.append(row)
            lat.append(row['latency_s'])
            print(f"KEPT  {'PASS' if row['correct'] else 'FAIL'}  {cmd!r:58} -> {row['plan']}")
            continue
        if a.delay and sent:
            time.sleep(a.delay)
        sent += 1
        parser = CommandParser(be, lambda: ctx)   # fresh history per case
        r = parser.parse(cmd)
        if not r.backend:          # the backend errored: that is a failure, not a refusal
            good = False
            errors += 1            # ...and its 0 s "latency" must not count in the timings
        else:
            good = matches(r.steps, expected) if expected else not r.steps
            lat.append(r.latency_s)
        plan = fmt_plan(r.steps)
        rows.append(dict(command=cmd, expected=fmt_plan([{'action': x, 'target': v}
                                                          if isinstance(v, str) else
                                                          {'action': x, 'angle_deg': v}
                                                          for x, v in expected]),
                         correct=int(good), attempts=r.attempts,
                         latency_s=round(r.latency_s, 3), plan=plan, reply=r.reply,
                         llm_reply=r.llm_reply,
                         rate_limit_retries=getattr(be, 'retries_used', 0)))
        print(f"{'PASS' if good else 'FAIL'}  {r.latency_s:6.2f}s  {cmd!r:58} -> "
              f"{plan or '(nothing) ' + r.reply}", flush=True)

    pos = [r for r, (_, e) in zip(rows, CASES) if e]
    neg = [r for r, (_, e) in zip(rows, CASES) if not e]
    print('\n' + '=' * 72)
    print(f'backend           : {be.describe()}' + (f'   [{a.label}]' if a.label else ''))
    print(f'overall accuracy  : {sum(r["correct"] for r in rows)}/{len(rows)}')
    print(f'executable plans  : {sum(r["correct"] for r in pos)}/{len(pos)}')
    print(f'correct refusals  : {sum(r["correct"] for r in neg)}/{len(neg)}')
    print(f'needed a repair   : {sum(r["attempts"] > 1 for r in rows)}')
    if lat:
        print(f'latency median/max: {statistics.median(lat):.2f}s / {max(lat):.2f}s')
    errors += sum(1 for r in rows if r['reply'].startswith(BACKEND_ERROR)) - errors
    if errors:
        print(f'backend errors    : {errors} command(s) never got an answer (timeout / quota) - '
              f'counted as FAIL. Rerun them later with --resume <this csv>')
    retried = sum(1 for r in rows if r['rate_limit_retries'])
    if retried:
        print(f'rate limited      : {retried} command(s) had to wait and retry - their latency '
              f'includes the wait; rerun with a larger --delay for clean timings')

    tag = (getattr(be, 'model', '') or a.model or 'default').replace(':', '-').replace('/', '-')
    out = a.out or os.path.join(WS, 'results',
                                f'eval_{a.backend}_{tag}' + (f'_{a.label}' if a.label else '')
                                + '.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}')
    if a.compare:
        compare(rows, a.compare)


if __name__ == '__main__':
    main()
