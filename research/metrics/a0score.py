"""The candidate metrics against the user's picks in the A0 rounds (PLAN.md, the amendments to stage A).

For each item the user answered validly, every metric scores the options against the original at the viewing the
user had (the grid at x2: 19.4 pixels per degree, 4 device pixels a Spectrum pixel); a metric agrees when its
lowest-scored option is the one the user picked. Reads the rounds' manifests and keys (data/vote/), the votes saved
from the page's db (data/vote/db/votes/), and the files as published.

    python research/metrics/a0score.py [--ppd 19.4] [--metrics a,b]   prints the table, writes a0-human.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VOTE = os.path.join(HERE, 'data', 'vote')
ROUNDS = ('a0h-1', 'a0h-2', 'a0h-3')
CHECK = dict(dark='dark ramp', **{'band-white': 'banding', 'band-yellow': 'banding', 'grey50': '50% grey',
             'clash-grey': 'cell-scale noise', 'clash-salmon': 'cell-scale noise', 'edge': 'seam beside an edge',
             'stair': 'staircase'}, **{f'seam-{g}': 'seam on a gradient' for g in ('sky', 'sunset', 'skin', 'field')})


def load(rel):
    return np.asarray(Image.open(os.path.join(VOTE, 'files', rel)).convert('RGB'), np.float32) / 255


def items():
    """[(round, item, check, original, {label: image}, picked label)] for the valid answers, repeats included."""
    out = []
    for r in ROUNDS:
        manifest = json.load(open(os.path.join(VOTE, f'{r}.manifest.json')))
        key = json.load(open(os.path.join(VOTE, f'{r}.key.json')))
        for it in manifest['items']:
            path = os.path.join(VOTE, 'db', 'votes', f'{r}__{it["id"]}.json')
            if not os.path.exists(path):
                continue
            v = json.load(open(path))
            v = v.get('data', v)
            if abs(v['dpr'] * v['zoom'] - round(v['dpr'] * v['zoom'])) > 1e-6:
                continue                    # uneven pixels: an answer the page should have refused (LOG, a0h-1)
            labels = key[it['id']]['options']
            base = key[it['id']]['repeat_of'] or it['id']
            out.append((r, it['id'], CHECK[base], load(it['original']),
                        {l: load(p) for l, p in zip(labels, it['options'])}, labels[v['pick'] - 1] if v['pick'] else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppd', type=float, default=19.4)
    ap.add_argument('--metrics', default=','.join(M.METRICS))
    args = ap.parse_args()
    names = args.metrics.split(',')
    rows = []
    for r, iid, check, ref, options, pick in items():
        scores = {n: {l: M.score(n, ref, img, args.ppd)[0] for l, img in options.items()} for n in names}
        rows.append((r, iid, check, pick, {n: min(s, key=s.get) for n, s in scores.items()}, scores))
        print(iid, pick, {n: rows[-1][4][n] for n in names}, flush=True)
    lines = [f'# The metrics against the user\'s picks in the A0 rounds\n',
             f'At {args.ppd} pixels per degree (the grid at x2), device factor {M.device_factor(args.ppd)}. '
             'A cell is the option the metric scores lowest; **bold** where it is the user\'s pick.\n',
             '| item | check | user | ' + ' | '.join(names) + ' |', '|' + '---|' * (3 + len(names))]
    for r, iid, check, pick, best, _ in rows:
        cells = [f'**{best[n]}**' if best[n] == pick else best[n] for n in names]
        lines.append(f'| {r}/{iid} | {check} | {pick} | ' + ' | '.join(cells) + ' |')
    agree = {n: sum(best[n] == pick for *_, pick, best, _ in rows) for n in names}
    lines.append('| **agree** | | | ' + ' | '.join(f'{agree[n]}/{len(rows)}' for n in names) + ' |')
    lines.append('\nScores (lower = nearer), per item and option:\n')
    for r, iid, check, pick, best, scores in rows:
        lines.append(f'- {r}/{iid} ({pick}): ' + '; '.join(
            f'{n} ' + ', '.join(f'{l} {v:.4g}' for l, v in s.items()) for n, s in scores.items()))
    open(os.path.join(HERE, 'a0-human.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:len(rows) + 5]))


if __name__ == '__main__':
    main()
