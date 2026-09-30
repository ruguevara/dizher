"""Local pairs from the hand-painted test projects, every image rendered as the project exports it.

    python research/pairs/build.py [NAME...]      all painted projects by default; writes data/NAME.npz

Per project: the painting (A) and Select pairs alone (Hide). The cells whose shown colours differ are grouped by the
picture's segments (common.segments), one pair per segment: A against A with that segment's changed cells given back
to Select pairs (B, the painting is better there). Two controls per segment: A against the cells' next best pair by
the project's energy, neither the painted nor the selected one (C, assumed worse than the painting); and, for a few
segments, A against A rendered from another halftone origin (N, the same colouring: a metric should call it even).
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from common import DATA, Project, painted_projects, segments, shown   # noqa: E402

NULLS = 4                                   # segments per project that get a null pair
NULL_ORIGIN = {'noise_x': 97, 'noise_y': 53}   # another start for the halftone and DBS


def colours(pairs: np.ndarray, rgb: np.ndarray) -> list:
    """Per label, the frozenset of its two colours as RGB bytes (the two blacks alike)."""
    return [frozenset(tuple(rgb[i]) for i in p) for p in pairs]


def build(name: str):
    t = time.time()
    p = Project(name)
    painted, bare = p.labels(), p.selection.best_attr_indexes.copy()
    A, H = p.render(painted), p.render(bare)
    diff = shown(A) != shown(H)
    seg = segments(p.target())
    sets = colours(p.pairs, p.selection.palette.as_ubyte())
    unary = p.selection.energy.unary()                        # (P, R, C) each pair's own cost at each cell
    out = dict(target=p.target(), A=A, H=H, painted=painted, bare=bare, segments=seg, diff=diff,
               energy_A=p.energy(painted), energy_H=p.energy(bare))
    meta, rng = [], np.random.default_rng(0)
    touched = [s for s in range(seg.max() + 1) if (diff & (seg == s)).any()]
    nulls = set(rng.choice(touched, size=min(NULLS, len(touched)), replace=False).tolist())
    if nulls:
        out['N'] = p.render(painted, halftoner=NULL_ORIGIN)
    for s in touched:
        cells = diff & (seg == s)
        B = painted.copy()
        B[cells] = bare[cells]
        C = painted.copy()
        for r, c in zip(*np.nonzero(cells)):
            skip = {sets[painted[r, c]], sets[bare[r, c]]}
            costs = np.where([x in skip for x in sets], np.inf, unary[:, r, c])
            C[r, c] = int(costs.argmin())
        key = f'{name}/{s}'
        out[f'B{s}'], out[f'C{s}'] = p.render(B), p.render(C)
        out[f'Blabels{s}'], out[f'Clabels{s}'] = B, C
        out[f'energy_B{s}'], out[f'energy_C{s}'] = p.energy(B), p.energy(C)
        meta.append(dict(id=key, image=name, segment=int(s), cells=int(cells.sum()), null=s in nulls))
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / f'{name}.npz', **{k: (np.round(v * 255).astype(np.uint8)
                                                    if isinstance(v, np.ndarray) and v.dtype.kind == 'f' else v)
                                                for k, v in out.items()})
    (DATA / f'{name}.json').write_text(json.dumps(meta, indent=1))
    return name, len(meta), time.time() - t


if __name__ == '__main__':
    names = sys.argv[1:] or painted_projects()
    with Pool(4) as pool:
        for name, n, s in pool.imap_unordered(build, names):
            print(f'{name}: {n} segments, {s:.0f} s', flush=True)
