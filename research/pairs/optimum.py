"""The fast judge's optimum: is what the judge likes best a good colouring, or does it cheat (flat, wrong colours,
noise)? Started from Select pairs, each cell in turn takes the pair the judge rates best on the cells around it, until
a pass changes nothing.

    python research/pairs/optimum.py ROUND [--full [--mps]] [NAME...]   the fast judge's optimum, or the full one's
                                        (LPIPS: --mps on Apple silicon, else ~80 min a pass a picture); the projects
                                        build.py made; writes data/optimum/rROUND/NAME.*
    python research/pairs/optimum.py ROUND --pairs      counterexamples from that round's optima, for fit.py

The optima are worse than the paintings (the user, round 1: in 8 of 9 pictures, the other slightly better). --pairs
makes them local labels like build.py's B: per picture segment where the optimum shows other colours than the
painting, the painting against itself with that segment from the optimum (O), the painting better, assumed.

The search renders without DBS (Converter.render_labels: each pair's own halftone): the judge decides a painted segment
the same way on it as on the full render in 0.90 of the segments (B) and 0.94 (C), rank correlation 0.96. Every
distinct colour pair is a candidate: the painted pair is the energy's 9th best in andy's median cell. The result is
rendered through the project's pipeline and scored there by both judges, against Select pairs and the painting."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import colours   # noqa: E402
from common import DATA, Project, segments, shown, window   # noqa: E402
from metrics import JUDGE, JUDGE_FAST, all_metrics, judge_fast_score, judge_score, score_batch, seen_change   # noqa: E402

ROOT = DATA / 'optimum'
REACH, PASSES = 2, 4   # cells around a cell the judge sees; passes at most


def candidates(p: Project, sets: list) -> np.ndarray:
    """(R, C, K) per cell one label for each distinct colour pair, the energy's cheapest of those that show it;
    -1 pads."""
    unary = p.selection.energy.unary()                           # (P, R, C)
    groups = {}
    for label, s in enumerate(sets):
        groups.setdefault(s, []).append(label)
    reps = np.stack([np.array(g)[unary[g].argmin(0)] for g in groups.values()], -1)   # (R, C, K)
    return reps


def search(p: Project, start: np.ndarray, judge: dict, seed=0):
    sel, T, (h, w) = p.selection, p.target(), p.selection.cell
    realized = sel.realized.astype(np.float32)
    labels = start.copy()
    X = sel.render_labels(labels).astype(np.float32)
    R, C = labels.shape
    cand = candidates(p, colours(p.pairs, sel.palette.as_ubyte()))
    rng, changed = np.random.default_rng(seed), []
    for _ in range(PASSES):
        n = 0
        for i in rng.permutation(R * C):
            r, c = divmod(int(i), C)
            r0, r1, c0, c1 = max(r - REACH - 1, 0), min(r + REACH + 2, R), max(c - REACH - 1, 0), min(c + REACH + 2, C)
            crop = (slice(r0 * h, r1 * h), slice(c0 * w, c1 * w))
            Xc, Tc = X[crop].copy(), T[crop]
            mask = np.zeros(Xc.shape[:2], bool)
            mask[max(r - REACH - r0, 0) * h:(min(r + REACH + 1, R) - r0) * h,
                 max(c - REACH - c0, 0) * w:(min(c + REACH + 1, C) - c0) * w] = True
            cell = (slice((r - r0) * h, (r - r0 + 1) * h), slice((c - c0) * w, (c - c0 + 1) * w))
            px = (slice(r * h, (r + 1) * h), slice(c * w, (c + 1) * w))
            Xs = []
            for label in cand[r, c]:
                Xc[cell] = realized[label][px]
                Xs.append(Xc.copy())
            best_label = cand[r, c][int(np.argmin(score_batch(judge, Xs, Tc, mask)))]
            if best_label != labels[r, c] and not np.array_equal(realized[best_label][px], X[px]):
                labels[r, c] = best_label
                X[px] = realized[best_label][px]
                n += 1
        changed.append(n)
        if not n:
            break
    return labels, changed


def run(job):
    name, out, full = job
    t = time.time()
    p = Project(name)
    z = np.load(DATA / f'{name}.npz')
    T, painted, bare = p.target(), z['painted'], z['bare']
    judge = JUDGE if full else JUDGE_FAST
    labels, changed = search(p, bare, judge)
    O = p.render(labels)
    f = lambda k: z[k].astype(np.float32) / 255
    sets = colours(p.pairs, p.selection.palette.as_ubyte())
    same = lambda a, b: np.array([[sets[a[r, c]] == sets[b[r, c]] for c in range(a.shape[1])] for r in range(a.shape[0])])
    moved, was_painted = ~same(labels, bare), ~same(painted, bare)
    report = dict(image=name, judge=judge, passes=changed, seconds=round(time.time() - t),
                  cells_moved=int(moved.sum()), painted_cells=int(was_painted.sum()),
                  moved_to_painting=int((moved & was_painted & same(labels, painted)).sum()),
                  moved_where_painted=int((moved & was_painted).sum()),
                  shown_changed=int((shown(O) != shown(f('H'))).sum()),
                  fast={k: round(judge_fast_score(X, T), 4) for k, X in (('select', f('H')), ('painting', f('A')),
                                                                          ('optimum', O))},
                  full={k: round(judge_score(X, T), 4) for k, X in (('select', f('H')), ('painting', f('A')),
                                                                     ('optimum', O))})
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'{name}.npz', labels=labels, O=np.round(O * 255).astype(np.uint8))
    (out / f'{name}.json').write_text(json.dumps(report, indent=1))
    return report


def counterexamples(job):
    """The painting against itself with one segment from the optimum, per segment where they show other colours."""
    name, out, _ = job
    p = Project(name)
    z = np.load(DATA / f'{name}.npz')
    T, A, painted = p.target(), z['A'].astype(np.float32) / 255, z['painted']
    best = np.load(out / f'{name}.npz')['labels']
    sets = colours(p.pairs, p.selection.palette.as_ubyte())
    differs = np.array([[sets[best[r, c]] != sets[painted[r, c]] for c in range(best.shape[1])]
                        for r in range(best.shape[0])])
    seg, meta, renders = segments(T), [], {}
    for s in range(seg.max() + 1):
        cells = differs & (seg == s)
        if not cells.any():
            continue
        L = painted.copy()
        L[cells] = best[cells]
        X, mask = p.render(L), window(cells)
        renders[f'O{s}'] = np.round(X * 255).astype(np.uint8)
        changed = np.repeat(np.repeat(cells, 8, 0), 8, 1)
        meta.append(dict(id=f'{name}/{out.name}/{s}', image=name, segment=s, cells=int(cells.sum()),
                         change=seen_change(X, A, changed), A=all_metrics(A, T, mask), B=all_metrics(X, T, mask)))
    np.savez_compressed(out / f'pairs-{name}.npz', **renders)
    (out / f'pairs-{name}.json').write_text(json.dumps(meta, indent=1))
    return name, len(meta)


def use_mps():
    import metrics
    metrics.DEVICE = 'mps'   # the network terms on the GPU: ~20x a CPU thread for a cell's candidates


if __name__ == '__main__':
    out, args = ROOT / f'r{sys.argv[1]}', sys.argv[2:]
    pairs, full, mps = '--pairs' in args, '--full' in args, '--mps' in args
    names = [a for a in args if not a.startswith('--')] or sorted(f.stem for f in DATA.glob('*.npz'))
    with Pool(min(len(names), 3 if mps else 9), initializer=use_mps if mps else None) as pool:
        for rep in pool.imap_unordered(counterexamples if pairs else run, [(n, out, full) for n in names]):
            print(json.dumps(rep), flush=True)
