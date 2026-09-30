"""Every candidate metric with both sides of a pair and the picture first seen through an eye model.

    python research/pairs/eyes.py [EYE...]         after build.py, variants.py, flat.py; writes data/eyes.csv
    python research/pairs/eyes.py --deep EYE...    LPIPS and DISTS only, data/eyes-deep.csv; an EYE also as g1.5 (a
                                                   Gaussian of sigma 1.5 px) or a0.95s1 (alpha-stable, alpha, scale)

Eyes (sRGB in, sRGB out): none; the project's default eye (a Gaussian of sigma 1 px on linear RGB); the random-portrait
kernel exp(-r^0.95); S-CIELAB at the user's viewing (26 Spectrum pixels per degree) and at half that (13: twice as
far, or squinting). The pairs are fit.py's: B and C on the painted segments, the user's decisive votes (cal), the flat
counterexamples (F), and the null pairs (N) for the noise. Per metric and eye: the share of pairs where the metric
prefers the side the label prefers (a tie counts half), and the noise, median |A - N| over median |A - B|."""
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, window   # noqa: E402
from metrics import all_metrics, deep, encode, linear, seen_srgb   # noqa: E402
from dizher.converter.eye import eye_blur   # noqa: E402

HERE = Path(__file__).resolve().parent


def alpha_stable(scale, alpha):
    return lambda x: encode(eye_blur(linear(x).astype(np.float32), scale, alpha))


EYES = {
    'none': lambda x: x,
    'project': alpha_stable(1.4, 2.0),
    'portrait': alpha_stable(1.0, 0.95),
    'scielab26': lambda x: seen_srgb(x, 26),
    'scielab13': lambda x: seen_srgb(x, 13),
}
SOURCES = ('B', 'C', 'cal', 'F')


def eye(name):
    """An eye by name: one of EYES, g<sigma> or a<alpha>s<scale>."""
    if name in EYES:
        return EYES[name]
    if name.startswith('g'):
        return alpha_stable(float(name[1:]) * np.sqrt(2), 2.0)
    alpha, scale = name[1:].split('s')
    return alpha_stable(float(scale), float(alpha))


def measure(X, T, mask, only):
    return {k: deep(X, T, mask, k) for k in only} if only else all_metrics(X, T, mask)


def pairs(name):
    """(source, id, A, other, mask, y) of one image, y 1 when A is the better side (N: the same colouring)."""
    f = lambda z, k: z[k].astype(np.float32) / 255
    out, built = [], (DATA / f'{name}.npz').exists()   # david and burning-hand: not painted, votes only
    z = np.load(DATA / f'{name}.npz') if built else None
    A = f(z, 'A') if built else None
    meta = json.loads((DATA / f'{name}.json').read_text()) if built else []
    for m in meta:
        s = m['segment']
        mask = window(z['diff'] & (z['segments'] == s))
        out += [('B', m['id'], A, f(z, f'B{s}'), mask, 1), ('C', m['id'], A, f(z, f'C{s}'), mask, 1)]
        if m['null']:
            out.append(('N', m['id'], A, f(z, 'N'), mask, 1))
    fl = np.load(DATA / 'flat' / f'{name}.npz') if built else None
    for m in json.loads((DATA / 'flat' / f'{name}.json').read_text()) if built else []:
        out.append(('F', m['id'], A, f(fl, f'F{m["segment"]}'), window(z['segments'] == m['segment']), 1))
    key = json.loads((HERE / 'rounds' / 'cal1' / 'key.json').read_text())
    votes = json.loads((HERE / 'rounds' / 'cal1' / 'votes.json').read_text())
    v = np.load(DATA / 'variants' / f'{name}.npz') if (DATA / 'variants' / f'{name}.npz').exists() else None
    for sheet, k in key.items():
        vote = votes[sheet]['verdict']
        if vote == 0 or k['id'].split('/')[0] != name:
            continue
        y = int((vote == 1) == (k['painting'] == 2))
        if k['kind'] == 'pilot':
            s = int(k['id'].split('/')[1])
            cells, other, a = z['diff'] & (z['segments'] == s), f(z, f'B{s}'), A
        else:
            _, variant, s = k['id'].split('/')
            if f'cells_{variant}_{s}' not in v:
                continue
            cells, other, a = v[f'cells_{variant}_{s}'], f(v, f'B_{variant}_{s}'), f(v, 'A')
        if cells.sum() == k['cells']:   # else the rebuild no longer makes the pair the user saw
            out.append(('cal', sheet, a, other, window(cells), y))
    return out


def score_image(job):
    name, eyes, only = job
    path = DATA / f'{name}.npz'
    T = np.load(path if path.exists() else DATA / 'variants' / path.name)['target'].astype(np.float32) / 255
    rows = []
    for name_ in eyes:
        see = eye(name_)
        t = see(T)
        for source, pid, A, other, mask, y in pairs(name):
            a = measure(see(A), t, mask, only)
            o = measure(see(other), t, mask, only)
            rows.append(dict(eye=name_, source=source, id=pid, image=name, y=y, **{k: o[k] - a[k] for k in a}))
    return rows


def main(eyes, only=()):
    names = sorted(f.stem for f in DATA.glob('*.npz')) + ['david', 'burning-hand']   # the last two: votes only
    with Pool(9) as pool:
        rows = [r for rs in pool.map(score_image, [(n, eyes, only) for n in names]) for r in rs]
    with open(DATA / ('eyes-deep.csv' if only else 'eyes.csv'), 'w', newline='') as out:
        w = csv.DictWriter(out, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows, eyes)


def report(rows, eyes):
    metrics = [k for k in rows[0] if k not in ('eye', 'source', 'id', 'image', 'y')]
    print('share agreeing with the label (tie half), mean of B, C, cal; then F; noise')
    print(f"{'metric':22}" + ''.join(f' {e:>10}' for e in eyes) + '   |' + ''.join(f' {e[:6]:>6}' for e in eyes)
          + '   |' + ''.join(f' {e[:6]:>6}' for e in eyes))
    for k in metrics:
        agree, flat, noise = [], [], []
        for e in eyes:
            acc = {}
            for s in SOURCES:
                r = [x for x in rows if x['eye'] == e and x['source'] == s]
                d, y = np.array([x[k] for x in r]), np.array([x['y'] for x in r])
                acc[s] = np.where(d == 0, 0.5, (d > 0) == (y == 1)).mean()
            agree.append(np.mean([acc[s] for s in ('B', 'C', 'cal')]))
            flat.append(acc['F'])
            nb = np.median([abs(x[k]) for x in rows if x['eye'] == e and x['source'] == 'B'])
            nn = [abs(x[k]) for x in rows if x['eye'] == e and x['source'] == 'N']
            noise.append(np.median(nn) / max(nb, 1e-12))
        print(f"{k:22}" + ''.join(f' {v:10.2f}' for v in agree) + '   |' + ''.join(f' {v:6.2f}' for v in flat)
              + '   |' + ''.join(f' {v:6.2f}' for v in noise))


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['--deep']:
        main(args[1:], ('lpips', 'dists'))
    else:
        main(args or list(EYES))
