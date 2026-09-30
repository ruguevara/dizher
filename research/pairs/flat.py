"""Flat counterexamples: the painting with a whole segment on its one most common painted pair, a colouring with no
seams and no cell-to-cell change inside the segment, assumed worse than the painting. A judge of smoothness alone
would prefer it; fit.py reports how often the judge does, without training on it.

    python research/pairs/flat.py [NAME...]      the projects build.py made; writes data/flat/NAME.json"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Project, window   # noqa: E402
from metrics import all_metrics   # noqa: E402

OUT = DATA / 'flat'


def build(name: str):
    t = time.time()
    p = Project(name)
    z = np.load(DATA / f'{name}.npz')
    T, A = z['target'].astype(np.float32) / 255, z['A'].astype(np.float32) / 255
    painted, seg = z['painted'], z['segments']
    meta = []
    for m in json.loads((DATA / f'{name}.json').read_text()):
        cells = seg == m['segment']
        F = painted.copy()
        F[cells] = np.bincount(painted[cells]).argmax()
        if (F == painted).all():
            continue
        mask = window(cells)
        meta.append(dict(id=m['id'], image=name, cells=int(cells.sum()),
                         A=all_metrics(A, T, mask), B=all_metrics(p.render(F), T, mask)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{name}.json').write_text(json.dumps(meta, indent=1))
    return name, len(meta), time.time() - t


if __name__ == '__main__':
    names = sys.argv[1:] or sorted(f.stem for f in DATA.glob('*.npz'))
    with Pool(min(len(names), 9)) as pool:
        for name, n, s in pool.imap_unordered(build, names):
            print(f'{name}: {n} flat pairs, {s:.0f} s', flush=True)
