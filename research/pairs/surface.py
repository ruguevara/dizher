"""After session E: pairs chosen per surface, not per cell, as an artist paints (this face red/yellow, that sky
blue/white): per surface (common.segments) the K pairs that cover its cells at least cost; every other pair costs
too much there; then Select pairs' own DP (eye-model seams, coherence) as usual, and the labels rendered by the base
pipeline. Against the patchwork of the base, whose every cell picks its own nearest mixture, so the cells of a smooth
surface fall on both sides of the border between two families (session E, the user's "both bad").

    python research/pairs/surface.py make     per picture the base (Exact mixture preset) and each of VARIANTS, in
                                              data/surface/NAME.npz, a sheet to look at data/surface/NAME.png
    python research/pairs/surface.py user R V...   round R: blind sheets for the user, the base against each variant
                                              V, in data/surface/R/user/, key rounds/surface/R/user-key.json; vote:
                                              python research/pairs/dp.py vote rounds/surface/R data/surface/R/user
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Project, segments   # noqa: E402
from tune import setting   # noqa: E402
from views import GAP, up   # noqa: E402
from dizher.converter.energy import GROUPS, OFFSETS, optimise   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'surface', HERE / 'rounds' / 'surface'
PICTURES = ('RC1', 'anubis', 'autumn', 'diver-sunset', 'jojo', 'rocket-rackoon', 'vangog')   # not the held-out ones
VARIANTS = {   # (surfaces, pairs per surface, the cost of leaving them: None prohibitive, else this quantile of
    # the cells' gaps between their cheapest pair and their surface's); round surface: 1 pair hard is better than the
    # base on jojo, vangog, diver-sunset, worse on RC1, anubis, rocket-rackoon (a surface of several colours gets a
    # grey compromise, the raccoon black/white), so the soft ones: a near tie goes to the surface's pair, a real
    # difference keeps its own; they only blend the base and the hard one (jojo's faces stay a patchwork at 0.9),
    # so the last: only a surface's cells within T dE of its mean colour are bound to its pair, the rest are free
    # (surfaces, pairs, quantile, T)
    '32 surfaces, 1 pair': (32, 1, None, None), '64 surfaces, 1 pair': (64, 1, None, None),
    '32 surfaces, 2 pairs': (32, 2, None, None), '32 surfaces, 1 pair, soft 0.5': (32, 1, 0.5, None),
    '32 surfaces, 1 pair, soft 0.75': (32, 1, 0.75, None), '32 surfaces, 1 pair, soft 0.9': (32, 1, 0.9, None),
    '32 surfaces, 1 pair, within 10 dE': (32, 1, None, 10.0), '32 surfaces, 1 pair, within 20 dE': (32, 1, None, 20.0),
    '64 surfaces, 1 pair, within 20 dE': (64, 1, None, 20.0)}


def cell_lab(target: np.ndarray, cell=(8, 8)) -> np.ndarray:
    """(R, C, 3) each cell's mean CIELAB."""
    h, w = cell
    R, C = target.shape[0] // h, target.shape[1] // w
    return cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2Lab).reshape(R, h, C, w, 3).mean((1, 3))


def allowed(D: np.ndarray, seg: np.ndarray, k: int, lab=None, near=None) -> np.ndarray:
    """(P, R, C) bool: per surface the k pairs whose summed own cost over its cells, each cell taking the cheaper of
    them, is least (exhaustive over the sets of k). With near: only the cells within near dE of the surface's mean
    colour (lab: each cell's) choose and are bound; the rest may take any pair."""
    out = np.zeros(D.shape, bool)
    for s in np.unique(seg):
        cells = seg == s
        if near is not None:
            far = cells & (np.linalg.norm(lab - lab[cells].mean(0), axis=-1) >= near)
            out[:, far] = True
            cells &= ~far
            if not cells.any():
                continue
        cost = D[:, cells]                                   # (P, n)
        if k == 1:
            best = [int(cost.sum(1).argmin())]
        else:
            both = np.minimum(cost[:, None], cost[None]).sum(-1)   # (P, P)
            best = list(np.unravel_index(both.argmin(), both.shape))
        for p in best:
            out[p, cells] = True
    return out


def leaving(D: np.ndarray, mask: np.ndarray, q) -> np.ndarray:
    """(P, R, C) the extra cost of a pair outside mask: prohibitive (q None), or the q quantile over the cells of the
    gap between a cell's cheapest pair and its cheapest allowed one."""
    if q is None:
        return np.where(mask, 0, 100 * np.abs(D).max())
    gap = np.where(mask, D, np.inf).min(0) - D.min(0)
    return np.where(mask, 0, np.quantile(gap, q))


def select(p: Project, extra=None) -> np.ndarray:
    """Select pairs' DP at the project's settings, extra (P, R, C) added to the pairs' own costs."""
    c = p.selection
    e, w = c.energy, c.energy.weights
    D = e.unary() if extra is None else e.unary() + extra
    S = {off: sum(w[g] * e.S[g][off] for g in GROUPS) for off in OFFSETS}
    Lh, Lv = e.seam_smoothness()
    return optimise(D, S, c.pair_dissimilarity, Lh, Lv, c.coherence)


def make_job(name: str):
    p = Project(name, **setting({}))
    base = p.selection.best_attr_indexes.copy()
    assert (select(p) == base).all()
    D = p.selection.energy.unary()
    out, changed = dict(target=p.target(), base=p.render(base)), []
    for i, (k, n, q, near) in enumerate(VARIANTS.values()):
        L = select(p, leaving(D, allowed(D, segments(p.target(), k=k), n, cell_lab(p.target()), near), q))
        out[f'v{i}'] = p.render(L)
        changed.append(int((L != base).sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    out = {key: np.round(v * 255).astype(np.uint8) for key, v in out.items()}
    np.savez_compressed(OUT / f'{name}.npz', **out)
    gap = np.full((384, GAP, 3), 90, np.uint8)
    row = [x for key in ('target', 'base', *(f'v{i}' for i in range(len(VARIANTS)))) for x in (up(out[key], 2), gap)]
    cv2.imwrite(str(OUT / f'{name}.png'), cv2.cvtColor(np.concatenate(row[:-1], 1), cv2.COLOR_RGB2BGR))
    return name, changed


def make():
    with Pool(len(PICTURES)) as pool:
        for name, changed in pool.imap_unordered(make_job, PICTURES):
            print(f'{name}: cells changed ' + ', '.join(f'{v}: {n}' for v, n in zip(VARIANTS, changed)), flush=True)


def user(tag: str, *vs: int, seed=0):
    """Per picture and variant one sheet: the picture, then base and the variant in a random order, raw 3x as the app
    shows them; all shuffled."""
    rng = np.random.default_rng(seed)
    out = OUT / tag / 'user'
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    items = [(name, v) for name in PICTURES for v in vs]
    for i, j in enumerate(rng.permutation(len(items))):
        name, v = items[j]
        z = np.load(OUT / f'{name}.npz')
        side = int(rng.integers(1, 3))   # the variant's side
        a, b = (z[f'v{v}'], z['base'])[::1 if side == 1 else -1]
        gap = np.full((576, GAP, 3), 90, np.uint8)
        sheet = np.concatenate([up(z['target'], 3), gap, up(a, 3), gap, up(b, 3)], 1)
        cv2.imwrite(str(out / f'u{i + 1:02d}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        key[f'u{i + 1:02d}'] = dict(id=str(name), variant=list(VARIANTS)[v], x=side)
    (ROUND / tag).mkdir(parents=True, exist_ok=True)
    (ROUND / tag / 'user-key.json').write_text(json.dumps(key, indent=1))
    print(f'{len(key)} sheets in {out}')


if __name__ == '__main__':
    {'make': make, 'user': lambda: user(sys.argv[2], *map(int, sys.argv[3:]))}[sys.argv[1]]()
