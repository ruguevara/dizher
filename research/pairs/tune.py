"""Session C: the converter's settings tuned by the judges, one at a time, on the training pictures only.

    python research/pairs/tune.py dbs [N]     can the search skip DBS? N one-setting changes from the base (20), each
                                              picture converted with DBS and without, the three judges' ranks compared

The base is the Exact mixture preset (new projects' Metric and Select pairs values) over each project's own Tune,
Target, Halftoner, Eye and Optimise; no cell is painted. Held out of all tuning: three painted pictures and the two
unpainted ones (HELD), for the user's blind check."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Project   # noqa: E402
from metrics import JUDGE, JUDGE_FAST, score   # noqa: E402

from dizher.converter.energy import METHODS, NEWEST   # noqa: E402

HELD = ('andy', 'sunset', 'golden-axe', 'david', 'burning-hand')
TRAIN = ('RC1', 'anubis', 'autumn', 'diver-sunset', 'jojo', 'rocket-rackoon')
JUDGES = {'v4': JUDGE, 'v3': {'lpips_eye': 16.41, 'seam_L_1': 1.459}, 'fast': JUDGE_FAST}
PRESET = METHODS[NEWEST].preset
BASE = {'metric': dict(method=NEWEST, chroma=PRESET['chroma'], flare=PRESET['flare']),
        'select': {k: PRESET[k] for k in ('coherence', 'edge', 'luma_noise', 'chroma_noise')}}
KNOBS = {   # (node, param): the slider's range
    ('metric', 'chroma'): (0.0, 4.0), ('metric', 'flare'): (0.0, 1.0),
    ('select', 'coherence'): (0.0, 8.0), ('select', 'edge'): (0.02, 0.4),
    ('select', 'luma_noise'): (0.0, 0.5), ('select', 'chroma_noise'): (0.0, 0.5),
    ('eye', 'luma_scale'): (0.3, 1.9), ('eye', 'chroma_scale'): (0.3, 1.9)}


def setting(change: dict) -> dict:
    """The base with {(node, param): value} changed, as Project.convert's node params."""
    out = {nid: dict(p) for nid, p in BASE.items()}
    for (nid, k), v in change.items():
        out.setdefault(nid, {})[k] = v
    return out


def judged(X, T) -> dict:
    return {name: round(score(j, X, T), 4) for name, j in JUDGES.items()}


def dbs_job(job):
    name, changes = job
    p, rows = Project(name), []
    T = p.target()
    for change in changes:
        s = setting(change)
        t = time.time()
        on = p.convert(**s)
        t_on = time.time() - t
        off = p.convert(**s, optimise=dict(enabled=False))
        rows.append(dict(change={f'{n}.{k}': v for (n, k), v in change.items()}, seconds=round(t_on, 1),
                         on=judged(on, T), off=judged(off, T)))
    return name, rows


def dbs(n=20, seed=0):
    rng = np.random.default_rng(seed)
    knobs = list(KNOBS)
    changes = [{}] + [{k: round(float(rng.uniform(*KNOBS[k])), 3)} for k in
                      (knobs[i] for i in rng.integers(len(knobs), size=n))]
    out = DATA / 'tune'
    out.mkdir(parents=True, exist_ok=True)
    with Pool(len(TRAIN)) as pool:
        results = dict(pool.map(dbs_job, [(name, changes) for name in TRAIN]))
    (out / 'dbs.json').write_text(json.dumps(results, indent=1))
    print('Spearman, judge scores with DBS against without, over the settings, per picture:')
    print(f'{"":16}' + ''.join(f'{j:>7}' for j in JUDGES))
    for name, rows in results.items():
        print(f'{name:16}' + ''.join(f'{spearmanr([r["on"][j] for r in rows], [r["off"][j] for r in rows])[0]:7.2f}'
                                     for j in JUDGES))
    secs = [r['seconds'] for rows in results.values() for r in rows]
    print(f'one conversion with DBS: median {np.median(secs):.1f} s')


if __name__ == '__main__':
    {'dbs': lambda: dbs(*map(int, sys.argv[2:]))}[sys.argv[1]]()
