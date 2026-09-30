"""Pairs where the user kept Select pairs' cells: the painting against the same cells from Select pairs run with
another coherence, smoother (coherence x3) or looser (0), rendered through the project's pipeline.

    python research/pairs/variants.py [NAME...]      every project by default; writes data/variants/NAME.npz, .json

Per project and variant, the up to 4 segments with the most kept cells the variant colours differently (by the colours
of the pairs) become a pair each: A, the painting, against A with those cells from the variant. Each pair carries the
metrics of both sides over its window, so the calibration round can pick the ones where structure and fidelity pull
apart."""
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from mokit.graph import evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import colours   # noqa: E402
from common import DATA, IMAGES, Project, segments, shown, window   # noqa: E402
from metrics import all_metrics   # noqa: E402

OUT = DATA / 'variants'
PER = 4        # segments per project and variant
MIN_CELLS = 2


def variants(p: Project) -> dict:
    coherence = p.graph['select'].params.coherence
    out = {}
    for name, value in (('smooth', max(3 * coherence, coherence + 4)), ('loose', 0.0)):
        g = p.graph.with_params('select', replace(p.graph['select'].params, coherence=value))
        out[name] = evaluate(g, 'select', p.memo).best_attr_indexes.copy()
    return out


def build(name: str):
    t = time.time()
    p = Project(name)
    painted = p.labels()
    old = DATA / f'{name}.npz'
    if old.exists():   # build.py's renders of the painting and of Select pairs alone
        z = np.load(old)
        A, H = z['A'].astype(np.float32) / 255, z['H'].astype(np.float32) / 255
    else:
        A = p.render(painted)
        H = A if not p.graph['overpaint'].params.overrides else p.render(p.selection.best_attr_indexes)
    kept = shown(A) == shown(H)
    seg = segments(p.target())
    sets = colours(p.pairs, p.selection.palette.as_ubyte())
    T = p.target()
    out, meta = dict(target=T, A=A, segments=seg), []
    for vname, V in variants(p).items():
        differs = np.array([[sets[V[r, c]] != sets[painted[r, c]] for c in range(V.shape[1])]
                            for r in range(V.shape[0])]) & kept
        counts = sorted(((int((differs & (seg == s)).sum()), s) for s in range(seg.max() + 1)), reverse=True)
        for n, s in [x for x in counts if x[0] >= MIN_CELLS][:PER]:
            cells = differs & (seg == s)
            B = painted.copy()
            B[cells] = V[cells]
            X = p.render(B)
            mask = window(cells)
            key = f'{name}/{vname}/{s}'
            out[f'B_{vname}_{s}'], out[f'cells_{vname}_{s}'] = X, cells
            meta.append(dict(id=key, image=name, variant=vname, segment=int(s), cells=n,
                             A=all_metrics(A, T, mask), B=all_metrics(X, T, mask)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'{name}.npz', **{k: (np.round(v * 255).astype(np.uint8)
                                                   if isinstance(v, np.ndarray) and v.dtype.kind == 'f' else v)
                                               for k, v in out.items()})
    (OUT / f'{name}.json').write_text(json.dumps(meta, indent=1))
    return name, len(meta), time.time() - t


if __name__ == '__main__':
    names = sys.argv[1:] or sorted(p.parent.name for p in IMAGES.glob('*/project.json'))
    with Pool(min(len(names), 9)) as pool:
        for name, n, s in pool.imap_unordered(build, names):
            print(f'{name}: {n} pairs, {s:.0f} s', flush=True)
