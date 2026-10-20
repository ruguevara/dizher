"""B3 rounds: per picture and palette, many free conversions, and the four on which the candidate metrics disagree.

The variants: Floyd-Steinberg and Jarvis in each mixing space (linear, srgb, lab), Knoll with the Bayer and the
blue-noise maps in each space at error multipliers 1 and 0.5 (n = 64): 18 per picture and palette. The options shown:
the best variant by each of the selectors in turn (tone at sigma 1, 2 and 3 Spectrum pixels, the converter's eye
baseline, ColorVideoVDP, SSIMULACRA2), skipping one already taken, until there are four.

    python research/metrics/b3build.py ROUND TITLE SEED PICTURE:PALETTE...   (pictures as tests/images file names)
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from free import free, prepare  # noqa: E402
from rounds import make_pick_round  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SELECTORS = [('tone1', {}), ('tone2', {}), ('tone3', {}), ('eye', {}), ('cvvdp', {}), ('ssimulacra2', {})]
PPD = 19.4                                   # the grid at x2


def variants(src, palette):
    out = {}
    for space in ('linear', 'srgb', 'lab'):
        for method in ('fs', 'jjn'):
            out[f'{method}-{space}'] = free(src, palette, method, space)[1]
        for method in ('knoll-bayer', 'knoll-blue'):
            for mult in (1.0, 0.5):
                out[f'{method}-{space}-m{mult:g}'] = free(src, palette, method, space, n=64, mult=mult)[1]
    return out


def score(name, ref, test):
    if name == 'tone3':
        return M.tone(ref, test, PPD, None, 3.0)[0]
    return M.score(name, ref, test, PPD)[0]


def build(round_id, title, seed, specs):
    d = os.path.join(HERE, 'data', round_id)
    os.makedirs(d, exist_ok=True)
    items, chosen = [], {}
    for spec in specs:
        picture, palette = spec.split(':')
        stem = os.path.splitext(picture)[0]
        src = prepare(os.path.join(ROOT, 'tests', 'images', picture))
        Image.fromarray(src).save(os.path.join(d, f'{stem}.png'))
        ref = src.astype(np.float32) / 255
        vs = variants(src, palette)
        scores = {n: {k: score(n, ref, v.astype(np.float32) / 255) for k, v in vs.items()} for n, _ in SELECTORS}
        picks = []
        for n, _ in SELECTORS:
            for k in sorted(scores[n], key=scores[n].get):
                if k not in picks:
                    picks.append(k)
                    break
            if len(picks) == 4:
                break
        opts = {}
        for k in picks:
            path = os.path.join(d, f'{stem}_{palette}_{k}.png')
            Image.fromarray(vs[k]).save(path)
            opts[k] = path
        iid = f'{stem}-{palette}'
        items.append((iid, os.path.join(d, f'{stem}.png'), opts))
        chosen[iid] = {'options': picks, 'scores': {n: {k: scores[n][k] for k in picks} for n in scores},
                       'best': {n: min(scores[n], key=scores[n].get) for n in scores}}
        print(iid, picks, flush=True)
    json.dump(chosen, open(os.path.join(HERE, 'data', 'vote', f'{round_id}.selection.json'), 'w'), indent=1)
    return make_pick_round(round_id, title, items, repeats=0.15, seed=int(seed))


if __name__ == '__main__':
    m = build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
    print([i['id'] for i in m['items']])
