#!/usr/bin/env python3
"""Summarise every evaluation run in <workspace>/results/ - stage 8.

    python3 tools/summarize_results.py            # table + figures
    python3 tools/summarize_results.py --no-plots # table only (no matplotlib needed)

Prints one markdown table (paste it into the report) and, if matplotlib is installed
(sudo apt install python3-matplotlib), writes results/figures/*.png:
  accuracy_by_version.png   - rules -> LLM v1 -> v2 -> v3 on the same commands
  speed_vs_accuracy.png     - model size / hardware trade-off
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(os.path.dirname(PKG))
RESULTS = os.path.join(WS, 'results')
sys.path.insert(0, os.path.join(PKG, 'tools'))

from eval_parser import CASES  # noqa: E402

REFUSALS = {cmd for cmd, expected in CASES if not expected}

# Hardware each run used (not stored in the CSV). Add a line when you add a run.
WHERE = {
    ('qwen2.5:7b-instruct', 'v3-gpu'): '53 % GPU / 47 % CPU',   # 5.2 GB model, 4 GB GPU
    ('qwen2.5:3b-instruct', 'v3-gpu'): '100 % GPU',
}

BLUE, INK, MUTED, GRID, SURFACE = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load(path):
    name = os.path.basename(path)[len('eval_'):-len('.csv')]
    m = re.match(r'(?P<backend>[^_]+)_(?P<model>.+?)(?:_(?P<label>v[\w-]+))?$', name)
    backend, model, label = m['backend'], m['model'], m['label'] or ''
    with open(path) as f:
        rows = list(csv.DictReader(f))
    pos = [r for r in rows if r['command'] not in REFUSALS]
    neg = [r for r in rows if r['command'] in REFUSALS]
    lat = [float(r['latency_s']) for r in rows]
    model = model.replace('qwen2.5-', 'qwen2.5:') if backend == 'ollama' else model
    return dict(
        file=os.path.basename(path), backend=backend, model=model, label=label,
        name=('rules (regex baseline)' if backend == 'rules' else f'{model} {label}'.strip()),
        where=('-' if backend == 'rules' else WHERE.get((model, label), 'CPU')),
        n=len(rows), correct=sum(int(r['correct']) for r in rows),
        plans=(sum(int(r['correct']) for r in pos), len(pos)),
        refusals=(sum(int(r['correct']) for r in neg), len(neg)),
        repairs=sum(int(r['attempts']) > 1 for r in rows),
        median=statistics.median(lat), maxlat=max(lat))


def order(run):
    return (run['backend'] != 'rules', 'gpu' in run['label'], run['label'], run['model'])


def table(runs):
    out = ['| Run | Runs on | Accuracy | Plans carried out | Correct refusals | Repairs '
           '| Median latency |',
           '|---|---|---|---|---|---|---|']
    for r in runs:
        out.append(f"| {r['name']} | {r['where']} | {r['correct']}/{r['n']} "
                   f"({100 * r['correct'] / r['n']:.0f} %) | {r['plans'][0]}/{r['plans'][1]} "
                   f"| {r['refusals'][0]}/{r['refusals'][1]} | {r['repairs']} "
                   f"| {r['median']:.2f} s |")
    return '\n'.join(out)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _titles(fig, title, subtitle):
    fig.text(0.02, 0.955, title, color=INK, fontsize=13, fontweight='bold', va='top')
    fig.text(0.02, 0.895, subtitle, color=MUTED, fontsize=9.5, va='top')


def plots(runs, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'text.color': INK, 'axes.labelcolor': MUTED,
                         'figure.facecolor': SURFACE})

    # 1. Accuracy by version: the 7B on CPU, prompt/validator v1 -> v3, against the baseline
    seq = [r for r in runs if r['backend'] == 'rules'] + sorted(
        [r for r in runs if r['model'].startswith('qwen2.5:7b') and 'gpu' not in r['label']],
        key=lambda r: r['label'])
    labels = ['Rules\n(regex)' if r['backend'] == 'rules' else f"LLM {r['label']}" for r in seq]
    vals = [100 * r['correct'] / r['n'] for r in seq]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _style(ax)
    bars = ax.bar(labels, vals, width=0.5, color=BLUE)
    for b, v, r in zip(bars, vals, seq):
        b.set_visible(False)        # redraw with rounded data-end
        ax.add_patch(FancyBboxPatch((b.get_x(), 0), b.get_width(), v,
                                    boxstyle='round,pad=0,rounding_size=0.04',
                                    mutation_aspect=40, color=BLUE, linewidth=0))
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{r['correct']}/{r['n']}",
                ha='center', va='bottom', color=INK, fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f'{t} %' for t in (0, 25, 50, 75, 100)])
    _titles(fig, 'Commands turned into the correct plan',
            'qwen2.5:7b-instruct on CPU, same 25-26 commands; v1 → v3 change only the prompt '
            'and validator')
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(outdir, 'accuracy_by_version.png'), dpi=160)
    plt.close(fig)

    # 2. Speed vs accuracy: same v3 prompt, different model / hardware
    v3 = [r for r in runs if r['label'].startswith('v3')]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    for r in v3:
        acc = 100 * r['correct'] / r['n']
        ax.scatter(r['median'], acc, s=110, color=BLUE, edgecolor=SURFACE, linewidth=2,
                   zorder=3)
        size = '7B' if '7b' in r['model'] else '3B'
        hw = 'GPU' if 'gpu' in r['label'] else 'CPU'
        if size == '7B' and hw == 'GPU':
            hw = '53 % GPU'
        # labels go below the two 100 % points (one each side) so they never collide
        dx, dy, ha = {('7B', '53 % GPU'): (-10, -34, 'right'),
                      ('7B', 'CPU'): (10, -34, 'left')}.get((size, hw), (12, -4, 'left'))
        ax.annotate(f"{size}, {hw}\n{r['correct']}/{r['n']} · {r['median']:.1f} s",
                    (r['median'], acc), textcoords='offset points', xytext=(dx, dy), ha=ha,
                    color=INK, fontsize=10)
    ax.set_xscale('log')
    ax.set_xlim(0.3, 12)
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(['0.5 s', '1 s', '2 s', '5 s', '10 s'])
    ax.set_ylim(60, 105)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_yticklabels([f'{t} %' for t in (60, 70, 80, 90, 100)])
    ax.set_xlabel('Median time to plan one command (log scale)')
    _titles(fig, 'Faster model, more wrong plans',
            'Same v3 prompt and validator; qwen2.5 7B vs 3B, RTX 3050 Laptop GPU (4 GB)')
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(outdir, 'speed_vs_accuracy.png'), dpi=160)
    plt.close(fig)
    return ['accuracy_by_version.png', 'speed_vs_accuracy.png']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-plots', action='store_true')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RESULTS, 'eval_*.csv')))
    if not files:
        sys.exit(f'no results in {RESULTS} - run tools/eval_parser.py first')
    runs = sorted((load(f) for f in files), key=order)
    print(table(runs))
    if not a.no_plots:
        try:
            made = plots(runs, os.path.join(RESULTS, 'figures'))
            print('\nfigures: ' + ', '.join(f'results/figures/{m}' for m in made))
        except ImportError:
            print('\n(matplotlib not installed - sudo apt install python3-matplotlib for figures)')


if __name__ == '__main__':
    main()
