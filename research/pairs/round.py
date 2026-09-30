"""A calibration round: blind sheets for the user and the judges, the same sheets for both.

    python research/pairs/round.py OUT

Three parts, one sheet per pair (sides at random), under shuffled names:
  - pilot:  painted segments where the pilot judges went against the painting or split (rounds/pilot1), the user's
            own choice put back to them blind;
  - kept:   cells the user kept from Select pairs against a smoother or looser coherence's (variants.py), where the
            structure terms (seam, neighbour) and S-CIELAB disagree about which is better, half each way;
  - repeat: a few of the above again with the sides swapped, for the user's self-agreement.
key.json maps each sheet to its pair and to the side the painting (or the kept cells) is on; swapped/ has every
sheet with the sides the other way, for the judges."""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA   # noqa: E402
from views import sheet   # noqa: E402

HERE = Path(__file__).resolve().parent
KEPT, PER_IMAGE, REPEATS = 20, 3, 5
STRUCT, FID = ('seam_excess', 'neighbour_excess'), 'scielab_dE'


def pilot_pairs() -> list:
    """Pilot segments whose two verdicts did not both pick the painting."""
    folder = HERE / 'rounds' / 'pilot1'
    key = json.loads((folder / 'key.json').read_text())
    res = {r['file'].removesuffix('.png'): r for r in json.loads((folder / 'results-all.json').read_text())}
    by = {}
    for name, k in key.items():
        if k['kind'] == 'user':
            v = res[name]['verdict']
            by.setdefault(k['id'], []).append('tie' if v == '=' else 'user' if int(v) == k['user'] else 'other')
    return [pid for pid, p in sorted(by.items()) if p != ['user', 'user']]


def kept_pairs() -> list:
    """Variant pairs where structure and fidelity disagree most, both ways, a few per image."""
    rows = []
    for f in sorted((DATA / 'variants').glob('*.json')):
        rows += json.loads(f.read_text())
    d = lambda r, k: r['B'][k] - r['A'][k]   # > 0: the kept cells are better by k
    sd = {k: np.std([d(r, k) for r in rows]) or 1 for k in STRUCT + (FID,)}
    for r in rows:
        r['struct'] = sum(d(r, k) / sd[k] for k in STRUCT) / len(STRUCT)
        r['fid'] = d(r, FID) / sd[FID]
    conflict = [r for r in rows if np.sign(r['struct']) != np.sign(r['fid'])]
    conflict.sort(key=lambda r: -min(abs(r['struct']), abs(r['fid'])))
    picked, per = [], Counter()
    for want in (1, -1):   # structure prefers the kept cells, then the variant's
        n = 0
        for r in conflict:
            if n < KEPT // 2 and np.sign(r['struct']) == want and per[r['image']] < PER_IMAGE and r not in picked:
                picked.append(r)
                per[r['image']] += 1
                n += 1
    return picked


def main(out: Path, seed=1):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    jobs = [('pilot', pid, None) for pid in pilot_pairs()] + [('kept', r['id'], r) for r in kept_pairs()]
    jobs += [(kind, pid, r, 'repeat') for kind, pid, r in
             [jobs[i] for i in rng.choice(len(jobs), REPEATS, replace=False)]]
    names = [f'c{i:03d}' for i in rng.permutation(len(jobs)) + 1]
    key, cache, first_side = {}, {}, {}
    for job in jobs:
        kind, pid, r = job[:3]
        repeat = len(job) > 3
        if kind == 'pilot':
            image, s = pid.split('/')
            z = cache.setdefault(image, np.load(DATA / f'{image}.npz'))
            T, A, B = z['target'], z['A'], z[f'B{s}']
            cells = z['diff'] & (z['segments'] == int(s))
        else:
            image, variant, s = pid.split('/')
            z = cache.setdefault(('v', image), np.load(DATA / 'variants' / f'{image}.npz'))
            T, A, B = z['target'], z['A'], z[f'B_{variant}_{s}']
            cells = z[f'cells_{variant}_{s}']
        side = 3 - first_side[pid] if repeat else int(rng.integers(1, 3))
        first_side.setdefault(pid, side)
        X1, X2 = (A, B) if side == 1 else (B, A)
        name = names.pop()
        cv2.imwrite(str(out / f'{name}.png'), cv2.cvtColor(sheet(T, X1, X2, cells), cv2.COLOR_RGB2BGR))
        (out / 'swapped').mkdir(exist_ok=True)   # the judges see each sheet both ways
        cv2.imwrite(str(out / 'swapped' / f'{name}.png'), cv2.cvtColor(sheet(T, X2, X1, cells), cv2.COLOR_RGB2BGR))
        key[name] = dict(id=pid, kind=kind, repeat=repeat, painting=side, cells=int(cells.sum()),
                         at=np.argwhere(cells).tolist(),   # the cells, so a rebuild can be checked against them
                         **({'struct': round(r['struct'], 2), 'fid': round(r['fid'], 2)} if r else {}))
    (out / 'key.json').write_text(json.dumps(dict(sorted(key.items())), indent=1))
    print(f'{len(key)} sheets: ' + ', '.join(f'{k} {v}' for k, v in Counter(
        ('repeat' if k['repeat'] else k['kind']) for k in key.values()).items()))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
