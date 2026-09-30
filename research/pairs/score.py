"""Score the candidate metrics on the local pairs build.py made.

    python research/pairs/score.py            writes data/metrics.csv and prints the table

For each metric, over the pairs (lower is better for every metric):
  user>alg   the share of segments where it rates the painting (A) better than Select pairs' cells there (B);
             the project's own energy loses nearly all of them, as B is its optimum
  user>next  the same against the cells' next best pair by the energy (C), assumed worse than the painting
  noise      median |A - N| over the null pairs (the same colouring from another halftone start) over the median
             |A - B|: near 0, the metric sees the change, not the dither
The ranges are 95% bootstrap intervals over images.
"""
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, window   # noqa: E402
from metrics import all_metrics   # noqa: E402


def score_image(path: Path) -> list:
    z = np.load(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    f = lambda k: z[k].astype(np.float32) / 255
    T, A = f('target'), f('A')
    rows = []
    for m in meta:
        s = m['segment']
        mask = window(z['diff'] & (z['segments'] == s))
        a = all_metrics(A, T, mask)
        b, c = all_metrics(f(f'B{s}'), T, mask), all_metrics(f(f'C{s}'), T, mask)
        a['energy'], b['energy'], c['energy'] = float(z['energy_A']), float(z[f'energy_B{s}']), float(z[f'energy_C{s}'])
        n = all_metrics(f('N'), T, mask) if m['null'] else None
        if n is not None:
            n['energy'] = a['energy']
        for variant, vals in (('A', a), ('B', b), ('C', c), ('N', n)):
            if vals is not None:
                rows.append(dict(id=m['id'], image=m['image'], cells=m['cells'], variant=variant, **vals))
    return rows


def boot(images, wins, n=2000, seed=0):
    """95% interval of the pooled share over images resampled with replacement."""
    rng, names = np.random.default_rng(seed), sorted(set(images))
    by = {k: wins[images == k] for k in names}
    shares = []
    for _ in range(n):
        pick = rng.choice(names, len(names))
        v = np.concatenate([by[k] for k in pick])
        shares.append(v.mean())
    return np.percentile(shares, [2.5, 97.5])


def main():
    files = sorted(DATA.glob('*.npz'))
    with Pool(4) as pool:
        rows = [r for rs in pool.map(score_image, files) for r in rs]
    keys = [k for k in rows[0] if k not in ('id', 'image', 'cells', 'variant')]
    with open(DATA / 'metrics.csv', 'w', newline='') as out:
        w = csv.DictWriter(out, ['id', 'image', 'cells', 'variant'] + keys)
        w.writeheader()
        w.writerows(rows)
    table = {}
    for r in rows:
        table.setdefault(r['id'], {})[r['variant']] = r
    ids = sorted(table)
    images = np.array([table[i]['A']['image'] for i in ids])
    print(f"{len(ids)} segments on {len(set(images))} images, {sum('N' in table[i] for i in ids)} null pairs\n")
    print(f"{'metric':24} {'user>alg':>8} {'95%':>11} {'user>next':>9} {'95%':>11} {'noise':>6}")
    for k in keys:
        a = np.array([table[i]['A'][k] for i in ids])
        b = np.array([table[i]['B'][k] for i in ids])
        c = np.array([table[i]['C'][k] for i in ids])
        wb, wc = (a < b).astype(float), (a < c).astype(float)
        nulls = [abs(table[i]['A'][k] - table[i]['N'][k]) for i in ids if 'N' in table[i]]
        noise = np.median(nulls) / max(np.median(np.abs(a - b)), 1e-12) if nulls else np.nan
        lb, hb = boot(images, wb)
        lc, hc = boot(images, wc)
        print(f"{k:24} {wb.mean():8.2f} {lb:5.2f}-{hb:4.2f} {wc.mean():9.2f} {lc:5.2f}-{hc:4.2f} {noise:6.2f}")


if __name__ == '__main__':
    main()
