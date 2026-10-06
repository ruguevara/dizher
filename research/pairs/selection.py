"""PLAN step 5d: the seam score (seamfit.SEAM, README 17) in the selection, in place of the coherence term (V, the
pairs' attribute distance, times the target's edge weight). For each border of two cells and each two pairs, from what
the selection holds before it chooses, each pair's halftone (Converter.realized, .bitmaps) in each cell:
  - E_L: the lightness change of the border's pixels when the pair switches, the dots the same (seams.switch): each
    side's border column, its pair's ink and paper at its own dots against the other pair's at the same dots;
  - M_ab: the colour difference of the two cells' mean colours (linear light, CIELAB), beyond the target's;
  - the solidity step |exp(-sd_a / SOLID) - exp(-sd_b / SOLID)|, sd a cell's pixels' L*;
  - Y: the two cells' mean L* / 100;
the score 0 for the same pair, else SEAM. It goes into the DP's pairwise table (energy.optimise's S, the borders of 4
neighbours) times coherence x scale, the coherence term off. The scale: the old term's sum over the current
selections' pair changes over the score's there, all training pictures (make prints it), so a coherence costs about
what it did.

    python research/pairs/selection.py make [held] [fixed]   fixed: the score at WEIGHT x scale, not coherence x
                                                scale (data/seams/selection/[held-]fixed/). Per training picture at its gallery best (DBS on), or per
                                                held-out one at its project's settings (DBS on): the current selection
                                                and the score's, data/seams/selection[/held]/NAME.npz and a sheet
                                                NAME.png (the target, current, the score's); the cells changed; how
                                                the score at the halftone holds after DBS (Spearman over the changed
                                                seams). The scale always from the training pictures
    python research/pairs/selection.py strength   per training picture the current selection and the score's at
                                                STRENGTHS times the scale, labelled side by side on one page
                                                (data/seams/selection/strength/index.html, square pixels, whole zoom)
    python research/pairs/selection.py user [held] [fixed]   blind sheets: the picture, then both in a random order, raw at
                                                1x (shown enlarged by whole screen pixels), shuffled; held: each again
                                                at the end, the sides swapped. data/seams/selection[/held]/user/, the
                                                key in rounds/seams/selection[/held]/user-key.json; vote:
                                                python research/pairs/dp.py vote rounds/seams/selection[/held] data/seams/selection[/held]/user
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seamfit   # noqa: E402
import seams   # noqa: E402
from common import Project   # noqa: E402
from metrics import SRGB2XYZ, linear, xyz2lab   # noqa: E402
from views import GAP, up   # noqa: E402
from dizher.converter.energy import GROUPS, OFFSETS, SEAM_COST, optimise, surface_binding   # noqa: E402

OUT, ROUND = seams.OUT / 'selection', seams.ROUND / 'selection'
STRENGTHS = (1, 2, 4, 8, 16)   # the score's weight against the scale that matched the old term's cost (the user: low)
WEIGHT = 16.0   # fixed: the score's weight in coherence x strength units, the picture's coherence aside; the user's
                # picks on the strength page, coherence x strength: 16, 26, 23, 14, 8 (and jojo none), geometric mean


def per_cell(c) -> dict:
    """Each pair's halftone in each cell, (P, R, C, ...): its border columns' and rows' ink counts (n), its mean
    colour's L* and ab, its pixels' solidity; and the pairs' paper and ink L* (P, 2)."""
    P, H, W = c.bitmaps.shape
    R, C = H // 8, W // 8
    bits = c.bitmaps.reshape(P, R, 8, C, 8)
    mean = xyz2lab(linear(c.realized).reshape(P, R, 8, C, 8, 3).mean((2, 4)) @ SRGB2XYZ.T)
    L = seams.lab(c.realized)[..., 0].reshape(P, R, 8, C, 8)
    return dict(left=bits[:, :, :, :, 0].sum(2), right=bits[:, :, :, :, 7].sum(2),
                top=bits[:, :, 0, :, :].sum(-1), bottom=bits[:, :, 7, :, :].sum(-1),
                L=mean[..., 0], ab=mean[..., 1:], solid=np.exp(-L.std((2, 4)) / seams.SOLID),
                ends=seams.lab(c.color_pairs)[..., 0])


def table(c, cells: dict, offset) -> np.ndarray:
    """(R', C', P, P) the score of the border between cell (r, c) at pair p and its neighbour at offset at pair q."""
    dr, dc = offset
    near, far = ('bottom', 'top') if dr else ('right', 'left')
    a = (slice(None), slice(0, cells['L'].shape[1] - dr), slice(0, cells['L'].shape[2] - dc))
    b = (slice(None), slice(dr, None), slice(dc, None))
    pick = lambda k, s: np.moveaxis(cells[k][s], 0, 2)                     # (R', C', P[, ...])
    ink = pick(near, a)[..., :, None] + pick(far, b)[..., None, :]          # ink pixels of the 16 at the border
    dLi = np.abs(cells['ends'][:, None, 1] - cells['ends'][None, :, 1])     # (P, P)
    dLp = np.abs(cells['ends'][:, None, 0] - cells['ends'][None, :, 0])
    E = (ink * dLi + (16 - ink) * dLp) / 16
    abT = xyz2lab(linear(c.image_rgb).reshape(cells['L'].shape[1], 8, cells['L'].shape[2], 8, 3).mean((1, 3))
                  @ SRGB2XYZ.T)[..., 1:]
    stepT = np.linalg.norm(abT[a[1:]] - abT[b[1:]], axis=-1)[..., None, None]
    M = np.maximum(np.linalg.norm(pick('ab', a)[..., :, None, :] - pick('ab', b)[..., None, :, :], axis=-1) - stepT, 0)
    Y = (pick('L', a)[..., :, None] + pick('L', b)[..., None, :]) / 200
    S = np.abs(pick('solid', a)[..., :, None] - pick('solid', b)[..., None, :])
    w = seamfit.SEAM
    out = w['M cell means ab'] * M + Y * (w['E switch L x Y cell lightness L'] * E +
                                          w['S solid step L x Y cell lightness L'] * S)
    out[..., np.arange(out.shape[-1]), np.arange(out.shape[-1])] = 0
    return out.astype(np.float32)


def select(c, scale=None, weight=None) -> np.ndarray:
    """Select pairs' DP at the converter's settings: with scale None the current coherence term (it must give the
    converter's own labels); else the seam score times weight x scale in its place (weight: the converter's coherence
    by default)."""
    e, w = c.energy, c.energy.weights
    D = e.unary()
    if c.surface > 0:
        D = D + surface_binding(D, c.image_rgb, c.cell, c.surface)
    S = {off: sum(w[g] * e.S[g][off] for g in GROUPS) for off in OFFSETS}
    Lh, Lv = e.seam_smoothness()
    if scale is None:
        return optimise(D, S, c.pair_dissimilarity, Lh, Lv, c.coherence)
    cells = per_cell(c)
    for off in ((0, 1), (1, 0)):
        S[off] = S[off] + (c.coherence if weight is None else weight) * scale * table(c, cells, off)
    return optimise(D, S, c.pair_dissimilarity, Lh, Lv, 0.0)


def old_and_new(c, labels):
    """(old, new) on the borders where the labels change pair: the coherence term's cost and the score."""
    cells, V = per_cell(c), c.pair_dissimilarity
    Lh, Lv = c.energy.seam_smoothness()
    old, new = [], []
    for off, L in (((1, 0), Lh), ((0, 1), Lv)):
        dr, dc = off
        p, q = labels[:labels.shape[0] - dr, :labels.shape[1] - dc], labels[dr:, dc:]
        T = table(c, cells, off)
        i, j = np.nonzero(p != q)
        old.append(SEAM_COST * V[p[i, j], q[i, j]] * L[i, j])
        new.append(T[i, j, p[i, j], q[i, j]])
    return np.concatenate(old), np.concatenate(new)


def settings(name: str) -> dict:
    """A training picture's gallery best; a held-out one's own project settings; DBS on."""
    from tune import TRAIN
    return seams.gallery_best(name) if name in TRAIN else {'optimise': {'enabled': True}}


def where(held: bool, fixed=False):
    """(data, rounds) folders: the training pictures' round at the root, held out in held/, at WEIGHT in -fixed."""
    sub = 'held' * held + '-fixed' * fixed
    sub = sub.lstrip('-')
    return (OUT / sub, ROUND / sub) if sub else (OUT, ROUND)


def scale_job(name: str):
    c = Project(name, **settings(name)).selection
    return old_and_new(c, c.best_attr_indexes)


def make_job(job):
    name, scale, out_dir, weight = job
    p = Project(name, **settings(name))
    c = p.selection
    base = c.best_attr_indexes.copy()
    assert (select(c) == base).all(), name
    new = select(c, scale, weight)
    out = dict(target=p.target(), base=p.render(base), new=p.render(new))
    # the score at the halftone against the score measured on the render after DBS, on its changed seams
    X, T = out['new'], out['target']
    m = {**seams.measures(X, T), 'E switch': seams.switch(X, c.color_pairs[new])}
    after = seamfit.score({f'{fam} {g}': v for fam, ks in m.items() for g, v in ks.items()})
    cells = per_cell(c)
    before, changed = [], []
    for off in ((1, 0), (0, 1)):
        dr, dc = off
        a, b = new[:new.shape[0] - dr, :new.shape[1] - dc], new[dr:, dc:]
        before.append(table(c, cells, off)[np.indices(a.shape)[0], np.indices(a.shape)[1], a, b].ravel())
        changed.append((a != b).ravel())
    before, changed = np.concatenate(before), np.concatenate(changed)
    rho = spearmanr(before[changed], after[changed])[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {k: np.round(v * 255).astype(np.uint8) for k, v in out.items()}
    np.savez_compressed(out_dir / f'{name}.npz', **out, base_labels=base, new_labels=new)
    gap = np.full((384, GAP, 3), 90, np.uint8)
    sheet = np.concatenate([up(out['target'], 2), gap, up(out['base'], 2), gap, up(out['new'], 2)], 1)
    cv2.imwrite(str(out_dir / f'{name}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return name, int((new != base).sum()), int((base[1:] != base[:-1]).sum() + (base[:, 1:] != base[:, :-1]).sum()), \
        int((new[1:] != new[:-1]).sum() + (new[:, 1:] != new[:, :-1]).sum()), rho


def make(held=False, fixed=False):
    from tune import HELD, TRAIN
    with Pool(len(TRAIN)) as pool:
        pairs = pool.map(scale_job, TRAIN)
        old, new = (np.concatenate(x) for x in zip(*pairs))
        scale = float(old.sum() / new.sum())
        print(f'SCALE {scale:.4f}: the coherence term over the score on the current selections\' pair changes',
              flush=True)
        names, out_dir = (HELD if held else TRAIN), where(held, fixed)[0]
        jobs = [(t, scale, out_dir, WEIGHT if fixed else None) for t in names]
        for name, n, b, s, rho in pool.imap_unordered(make_job, jobs):
            print(f'{name:15} cells changed {n:3}; pair changes between neighbours {b} -> {s}; the score at the '
                  f'halftone against after DBS, Spearman {rho:+.2f}', flush=True)


def strength_job(job):
    """One picture at its settings: the current selection and the score's at each of STRENGTHS, written as 1x PNGs;
    the cells changed and the score's sum over the changed seams of each, by the score's table."""
    name, scale = job
    p = Project(name, **settings(name))
    c = p.selection
    out = OUT / 'strength'
    out.mkdir(parents=True, exist_ok=True)
    png = lambda X, f: cv2.imwrite(str(out / f), cv2.cvtColor(np.round(X * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    png(p.target(), f'{name}-source.png')
    base = c.best_attr_indexes.copy()
    png(p.render(base), f'{name}-current.png')
    row = [('current', 0, float(old_and_new(c, base)[1].sum()))]
    for k in STRENGTHS:
        labels = select(c, scale * k)
        png(p.render(labels), f'{name}-x{k}.png')
        row.append((f'x{k}', int((labels != base).sum()), float(old_and_new(c, labels)[1].sum())))
    return name, row


def strength():
    from tune import TRAIN
    with Pool(len(TRAIN)) as pool:
        old, new = (np.concatenate(x) for x in zip(*pool.map(scale_job, TRAIN)))
        scale = float(old.sum() / new.sum())
        rows = dict(pool.map(strength_job, [(t, scale) for t in TRAIN]))
    for name, row in rows.items():
        print(f'{name:15} ' + '  '.join(f'{v}: {n} cells, seams {s:.0f}' for v, n, s in row))
    cols = ['source', 'current'] + [f'x{k}' for k in STRENGTHS]
    cells = ''.join(f'<div><b>{name}</b><div class="row">' + ''.join(
        f'<figure><img src="{name}-{v}.png"><figcaption>{v}</figcaption></figure>' for v in cols) + '</div></div>'
        for name in TRAIN)
    (OUT / 'strength' / 'index.html').write_text(f"""<!doctype html><meta charset="utf-8"><title>seam score strength</title>
<style>body{{background:#222;color:#ddd;font:14px system-ui;margin:8px}}.row{{display:flex;gap:6px;flex-wrap:wrap;
margin-bottom:14px}}figure{{margin:0}}img{{display:block;image-rendering:pixelated}}figcaption{{padding:2px 0}}</style>
<p>The current selection and the seam score at x1-x16 of its first scale; +/- zoom (whole screen pixels).</p>{cells}
<script>let z = +(localStorage.strengthZoom || 2);
const set = () => {{ for (const i of document.images) i.style.width = 256 * Math.round(z * devicePixelRatio) / devicePixelRatio + 'px'; }};
addEventListener('keydown', e => {{ if ('+=-'.includes(e.key)) {{ z = Math.max(1, z + (e.key === '-' ? -1 : 1)); localStorage.strengthZoom = z; set(); }} }});
set();</script>""")
    print(OUT / 'strength' / 'index.html')


def user(held=False, fixed=False, seed=0):
    """Per picture one sheet: the picture, then the current selection and the score's in a random order, raw at 1x
    (the vote page enlarges it by whole screen pixels, square; baked in at 3x a window narrower than it smoothed it).
    held: then each again, the sides swapped, in another order."""
    from tune import HELD, TRAIN
    names, (data, rounds) = (HELD if held else TRAIN), where(held, fixed)
    rng = np.random.default_rng(seed)
    out = data / 'user'
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    order = [(j, int(rng.integers(1, 3))) for j in rng.permutation(len(names))]   # the score's side
    if held:
        order += [(j, 3 - side) for j, side in (order[k] for k in rng.permutation(len(order)))]
    for i, (j, side) in enumerate(order):
        z = np.load(data / f'{names[j]}.npz')
        a, b = (z['new'], z['base'])[::1 if side == 1 else -1]
        gap = np.full((192, GAP // 2, 3), 90, np.uint8)   # 1x: the vote page enlarges it by whole screen pixels
        sheet = np.concatenate([z['target'], gap, a, gap, b], 1)
        cv2.imwrite(str(out / f'u{i + 1:02d}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        key[f'u{i + 1:02d}'] = dict(id=names[j], variant='seam score', x=side)
    rounds.mkdir(parents=True, exist_ok=True)
    (rounds / 'user-key.json').write_text(json.dumps(key, indent=1))
    print(f'{len(key)} sheets in {out}')


if __name__ == '__main__':
    if sys.argv[1] == 'strength':
        strength()
    else:
        {'make': make, 'user': user}[sys.argv[1]]('held' in sys.argv[2:], 'fixed' in sys.argv[2:])
