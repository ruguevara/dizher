"""Before more galleries: the Metric method, Exact mixture against Halftoned, each at its own preset (ops.apply_preset)
over the project's other settings, through its whole pipeline without DBS, as the galleries run; one blind sheet per
training picture. If one is never worse, the galleries fix it; if the pictures split, a gallery per method.

    python research/pairs/methods.py user     the renders and the sheets, data/methods/user/, key
                                              rounds/methods/user-key.json; vote:
                                              python research/pairs/dp.py vote rounds/methods data/methods/user
    python research/pairs/methods.py tally    per picture the method the user prefers
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Project, project_graph   # noqa: E402
from tune import TRAIN   # noqa: E402
from views import GAP, up   # noqa: E402
from dizher import ops   # noqa: E402
from dizher.converter.energy import METHODS   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'methods', HERE / 'rounds' / 'methods'


def renders(name: str) -> dict:
    """uint8 the target and the picture at each method's preset."""
    p = Project(name)
    out = {'target': p.target()}
    for m in METHODS:
        p.graph = ops.apply_preset(project_graph(name), m)
        out[m] = p.convert(optimise=dict(enabled=False))
    return {k: np.round(v * 255).astype(np.uint8) for k, v in out.items()}


def user(seed=0):
    """Per picture one sheet: the picture, then the two methods in a random order, raw 3x as the app shows them; the
    sheets shuffled."""
    rng = np.random.default_rng(seed)
    with Pool(len(TRAIN)) as pool:
        z = dict(zip(TRAIN, pool.map(renders, TRAIN)))
    out = OUT / 'user'
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    for i, j in enumerate(rng.permutation(len(TRAIN))):
        name = TRAIN[j]
        a, b = rng.permutation(list(METHODS))
        gap = np.full((576, GAP, 3), 90, np.uint8)
        sheet = np.concatenate([up(z[name]['target'], 3), gap, up(z[name][a], 3), gap, up(z[name][b], 3)], 1)
        cv2.imwrite(str(out / f'u{i + 1:02d}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        key[f'u{i + 1:02d}'] = dict(id=name, left=str(a), right=str(b),
                                    same=bool((z[name][a] == z[name][b]).all()))
    ROUND.mkdir(parents=True, exist_ok=True)
    (ROUND / 'user-key.json').write_text(json.dumps(key, indent=1))
    print(f'{len(key)} sheets in {out}')


def tally():
    key = json.loads((ROUND / 'user-key.json').read_text())
    votes = json.loads((ROUND / 'user-verdicts.json').read_text())
    for u, k in sorted(key.items(), key=lambda uk: uk[1]['id']):
        v = votes.get(u, '?')
        print(f"{k['id']:15} {k['left'] if v == '1' else k['right'] if v == '2' else {'=': 'both fine', 'x': 'both bad'}.get(v, v)}")


if __name__ == '__main__':
    {'user': user, 'tally': tally}[sys.argv[1]]()
