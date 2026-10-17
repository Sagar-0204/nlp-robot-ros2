#!/usr/bin/env python3
"""Type sentences, see the validated plan. Stage 4 playground - no ROS or robot needed.

    python3 tools/plan_shell.py                              # rules (no LLM)
    python3 tools/plan_shell.py --backend ollama             # qwen2.5:7b-instruct
    python3 tools/plan_shell.py --backend ollama --model qwen2.5:3b-instruct
    python3 tools/plan_shell.py --backend anthropic          # needs ANTHROPIC_API_KEY
    python3 tools/plan_shell.py --show-prompt                # also print what the LLM sees

Commands inside the shell:  :prompt  toggles prompt display,  :history  shows the
recent-command memory,  :seen red_box 3.4 2.4  pretends the detector saw a box,  :q  quits.
"""
import argparse
import json
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from nlp_robot.command_parser import CommandParser, build_user_message  # noqa: E402
from nlp_robot.llm_backends import BackendError, make_backend  # noqa: E402
from nlp_robot.places import load_places  # noqa: E402
from nlp_robot.plan_schema import WorldContext, strip_internal  # noqa: E402

DIM, BOLD, GREEN, RED, CYAN, RESET = '\033[2m', '\033[1m', '\033[32m', '\033[31m', '\033[36m', '\033[0m'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', default='rules', help='ollama | anthropic | openai | rules')
    ap.add_argument('--model', default='')
    ap.add_argument('--base-url', default='')
    ap.add_argument('--fallback', default='rules', help='used if the main backend fails; "" = none')
    ap.add_argument('--show-prompt', action='store_true')
    a = ap.parse_args()

    places = load_places(os.path.join(PKG, 'config', 'places.yaml'))
    ctx = WorldContext(places=places, objects={}, robot_pose=(0.0, 0.0, 0.0))
    be = make_backend(a.backend, model=a.model, base_url=a.base_url)
    fb = make_backend(a.fallback) if a.fallback and a.fallback != a.backend else None
    ok, why = be.check()
    if not ok:
        print(f'{RED}{be.describe()} is not usable: {why}{RESET}')
        if not fb:
            sys.exit(1)
        print(f'{DIM}continuing with the {fb.describe()} fallback{RESET}')
    else:
        try:
            secs, where = be.warmup()
            if secs > 0.5 or where:
                print(f'{DIM}{be.describe()} loaded in {secs:.1f} s'
                      + (f' ({where})' if where else '') + RESET)
        except BackendError as e:
            print(f'{RED}{e}{RESET}')

    parser = CommandParser(be, lambda: ctx, fallback=fb)
    show = a.show_prompt
    print(f'{BOLD}Type a command{RESET} {DIM}(:prompt, :history, :seen <label> <x> <y>, :q){RESET}')
    while True:
        try:
            line = input(f'\n{BOLD}you> {RESET}').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (':q', ':quit', 'quit', 'exit'):
            break
        if line == ':prompt':
            show = not show
            print(f'{DIM}prompt display {"on" if show else "off"}{RESET}')
            continue
        if line == ':history':
            for cmd, plan in parser.history:
                print(f'  {cmd!r} -> {json.dumps(plan)}')
            continue
        if line.startswith(':seen'):
            try:
                _, label, x, y = line.split()
                ctx.objects[label] = (float(x), float(y))
                print(f'{DIM}objects now: {ctx.objects}{RESET}')
            except ValueError:
                print(':seen <label> <x> <y>   e.g.  :seen red_box 3.4 2.4')
            continue

        if show:
            print(f'{DIM}--- system prompt ---\n{parser.system_prompt(ctx)}\n'
                  f'--- user message ---\n{build_user_message(line, ctx, list(parser.history))}'
                  f'\n---{RESET}')
        r = parser.parse(line)
        colour = GREEN if r.ok else (CYAN if not r.errors else RED)
        print(f'{colour}robot> {r.reply}{RESET}')
        if r.steps:
            print(json.dumps(strip_internal(r.steps), indent=2))
        if r.errors:
            for e in r.errors:
                print(f'{RED}  ! {e}{RESET}')
        extra = f', fell back: {r.fallback_reason}' if r.fallback_reason else ''
        print(f'{DIM}[{r.backend or "no backend"} | {r.latency_s:.2f} s | '
              f'{r.attempts} attempt(s){extra}]{RESET}')


if __name__ == '__main__':
    main()
