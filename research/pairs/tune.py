"""Session C: the converter's settings tuned by the judges, one at a time, on the training pictures only.

    python research/pairs/tune.py dbs [N]     can the search skip DBS? N one-setting changes from the base (20), each
                                              picture converted with DBS and without, the three judges' ranks compared
    python research/pairs/tune.py sweep       each setting alone over its slider (STEPS values), and the base at
                                              ORIGINS halftone origins for the judges' noise; renders and scores in
                                              data/tune/sweep/
    python research/pairs/tune.py report      per setting and value: on how many pictures all three judges put it
                                              better than the base by more than the noise
    python research/pairs/tune.py fine [--held] NODE.PARAM V...   those values and the base, each at every origin;
                                              per value the judges' mean against the base's in standard errors;
                                              --held on the held-out pictures instead
    python research/pairs/tune.py sheets NODE.PARAM V   round c1: the held-out base against V, for the agents
                                              (agents.write_round, every origin) and blind for the user (data/c1/user/,
                                              origin 0, raw 2x, key in rounds/c1/user-key.json)

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

HERE = Path(__file__).resolve().parent

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

STEPS, ORIGINS = 6, ((0, 0), (131, 57), (263, 311), (389, 173))   # origins: offsets from the project's own


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


def sweep_runs():
    return [({}, o) for o in ORIGINS] + [({k: round(float(v), 3)}, ORIGINS[0]) for k in KNOBS
                                         for v in np.linspace(*KNOBS[k], STEPS)]


def sweep_job(job):
    name, runs, folder = job
    p, rows, renders = Project(name), [], {}
    T, h = p.target(), p.graph['halftoner'].params
    for i, (change, (dx, dy)) in enumerate(runs):
        X = p.convert(**setting(change), halftoner=dict(noise_x=(h.noise_x + dx) % 512, noise_y=(h.noise_y + dy) % 512))
        renders[f'r{i}'] = np.round(X * 255).astype(np.uint8)
        rows.append(dict(run=f'r{i}', change={f'{n}.{k}': v for (n, k), v in change.items()}, origin=[dx, dy],
                         **judged(X, T)))
    out = DATA / 'tune' / folder
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'{name}.npz', target=np.round(T * 255).astype(np.uint8), **renders)
    (out / f'{name}.json').write_text(json.dumps(rows, indent=1))
    return name


def sweep():
    with Pool(len(TRAIN)) as pool:
        for name in pool.imap_unordered(sweep_job, [(name, sweep_runs(), 'sweep') for name in TRAIN]):
            print(name, 'done', flush=True)
    report()


def fine(*args: str):
    names = HELD if args[0] == '--held' else TRAIN
    knob, *values = args[1:] if args[0] == '--held' else args
    nid, k = knob.split('.')
    runs = [({(nid, k): float(v)} if v != 'base' else {}, o) for v in ['base', *values] for o in ORIGINS]
    folder = f'fine-{knob}' + ('-held' if names == HELD else '')
    with Pool(len(names)) as pool:
        for name in pool.imap_unordered(sweep_job, [(name, runs, folder) for name in names]):
            print(name, 'done', flush=True)
    print(f'{knob}: per picture, each judge\'s mean against the base\'s in standard errors (< 0 better); '
          'all three below -2: better, above +2: worse')
    total = {}
    for name in names:
        rows = json.loads((DATA / 'tune' / folder / f'{name}.json').read_text())
        by = {}
        for r in rows:
            by.setdefault(next(iter(r['change'].values()), 'base'), []).append(r)
        b = by.pop('base')
        line = []
        for v, rs in by.items():
            z = {j: (np.mean([r[j] for r in rs]) - np.mean([r[j] for r in b]))
                 / np.sqrt((np.var([r[j] for r in rs], ddof=1) + np.var([r[j] for r in b], ddof=1)) / len(ORIGINS))
                 for j in JUDGES}
            mark = '+' if all(x < -2 for x in z.values()) else '-' if all(x > 2 for x in z.values()) else ' '
            total.setdefault(v, []).append(mark)
            line.append(f'{v:5} ' + '/'.join(f'{x:+.0f}' for x in z.values()) + mark)
        print(f'  {name:15} ' + '  '.join(line))
    for v, marks in total.items():
        print(f'  {v}: better {marks.count("+")}, worse {marks.count("-")} of {len(names)}')


def report():
    """A value wins on a picture when every judge scores it below the base's mean by more than twice the spread of
    the base over the origins (lower is better); loses when every judge scores it above by as much."""
    runs = {name: json.loads((DATA / 'tune' / 'sweep' / f'{name}.json').read_text()) for name in TRAIN}
    table = {}
    for name, rows in runs.items():
        base = [r for r in rows if not r['change']]
        mean = {j: np.mean([r[j] for r in base]) for j in JUDGES}
        noise = {j: 2 * np.std([r[j] for r in base], ddof=1) for j in JUDGES}
        for r in rows:
            if r['change']:
                (k, v), = r['change'].items()
                d = {j: (r[j] - mean[j]) / noise[j] for j in JUDGES}   # in noise units, < 0 better
                cell = table.setdefault((k, v), dict(win=0, lose=0, d=[]))
                cell['win'] += all(x < -1 for x in d.values())
                cell['lose'] += all(x > 1 for x in d.values())
                cell['d'].append(d)
    print(f'per value: pictures where all three judges agree it is better / worse than the base (of {len(TRAIN)}), '
          'and each judge\'s median shift in noise units (< 0 better)')
    for (k, v), c in table.items():
        med = ' '.join(f'{j} {np.median([d[j] for d in c["d"]]):+5.1f}' for j in JUDGES)
        print(f'  {k:20} {v:6.3f}   better {c["win"]}  worse {c["lose"]}   {med}')


def sheets(knob: str, value: str, seed=0):
    import cv2
    from agents import write_round
    from views import up
    rng = np.random.default_rng(seed)
    pairs, user = [], {}
    out = DATA / 'c1' / 'user'
    out.mkdir(parents=True, exist_ok=True)
    for name in HELD:
        rows = json.loads((DATA / 'tune' / f'fine-{knob}-held' / f'{name}.json').read_text())
        z = np.load(DATA / 'tune' / f'fine-{knob}-held' / f'{name}.npz')
        base = {tuple(r['origin']): z[r['run']] for r in rows if not r['change']}
        cand = {tuple(r['origin']): z[r['run']] for r in rows if r['change'] == {knob: float(value)}}
        for i, o in enumerate(ORIGINS):
            pairs.append((f'{name}/o{i}', dict(image=name, knob=knob, value=float(value)), z['target'], cand[o], base[o]))
        side = int(rng.integers(1, 3))
        X1, X2 = (cand[ORIGINS[0]], base[ORIGINS[0]])[::1 if side == 1 else -1]
        gap = np.full((384, 8, 3), 90, np.uint8)
        sheet = np.concatenate([up(z['target'], 2), gap, up(X1, 2), gap, up(X2, 2)], 1)
        cv2.imwrite(str(out / f'{name}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        user[name] = dict(knob=knob, value=float(value), x=side)
    write_round('c1', pairs, seed)
    (HERE / 'rounds' / 'c1' / 'user-key.json').write_text(json.dumps(user, indent=1))


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
    {'dbs': lambda: dbs(*map(int, sys.argv[2:])), 'sweep': sweep, 'report': report,
     'fine': lambda: fine(*sys.argv[2:]), 'sheets': lambda: sheets(*sys.argv[2:])}[sys.argv[1]]()
