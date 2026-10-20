"""Stage A0's clash checks (PLAN.md, amendment 2): seams that do not follow the original, against what avoids them.

Clash is a conspicuous seam where the pair changes and the original has no such structure (PLAN.md, Terms). Three
kinds of stimuli, all 256x192 ZX renders with blue-noise dots (a0.BLUE), each pixel the exact mixture of its cell's
pair nearest the original in linear light:

- seam: a smooth vertical gradient. "seam" takes one pair above and another below a row of cells, the pairs and the
  row of least error: one straight seam the gradient does not have; "one pair" takes the single pair nearest
  over the whole surface: no seam, a larger colour error. Four gradients, so the error of "one pair" covers a range.
- edge: a vertical edge between two colours 3 pixels right of a cell border. The pair changes at the border to the
  left, at the one to the right, or the column of cells holding the edge takes a pair of the two colours' inks and
  draws the edge where it is.
- stair: a diagonal edge. Whole cells, each in the pair of its larger side, against the cells the edge crosses in a
  pair of the two sides' inks, drawing the edge.

    python research/metrics/a0clash.py   writes the stimuli to data/a0clash/ and prints the colour errors
"""
import itertools
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a0  # noqa: E402
import metrics as M  # noqa: E402
from skimage.color import deltaE_ciede2000  # noqa: E402

H, W = a0.H, a0.W
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'a0clash')
LIN = M.srgb_to_linear(a0.ZX.astype(np.float64))
# Pairs as the Spectrum allows them: two colours of one brightness (black is in both).
PAIRS = [(a, b) for bright in (0, 8) for a, b in itertools.combinations(range(bright, bright + 8), 2)]


def de(x, y):
    return deltaE_ciede2000(a0.lab(x), a0.lab(y))


def exact(target, pair):
    """(H, W) ink coverage of the pair nearest the target in linear light, and the mixture in sRGB."""
    p, i = LIN[pair[0]], LIN[pair[1]]
    lt = M.srgb_to_linear(target.astype(np.float64))
    d = i - p
    t = np.clip(((lt - p) @ d) / max(d @ d, 1e-12), 0, 1)
    return t.astype(np.float32), M.linear_to_srgb(p + t[..., None] * d).astype(np.float32)


def render(target, labels):
    """labels (H/8, W/8) pair indices -> the dithered render and the exact mixture."""
    out, mixture = np.zeros_like(target), np.zeros_like(target)
    for k in np.unique(labels):
        pair = PAIRS[k]
        t, m = exact(target, pair)
        cells = np.kron(labels == k, np.ones((8, 8), bool))
        out[cells] = a0.dither(a0.ZX[pair[0]], a0.ZX[pair[1]], t)[cells]
        mixture[cells] = m[cells]
    return out, mixture


def cell_errors(target):
    """(P, H/8, W/8) mean CIEDE2000 of each pair's exact mixture over each cell."""
    return np.stack([de(exact(target, pair)[1], target).reshape(H // 8, 8, W // 8, 8).mean((1, 3)) for pair in PAIRS])


def vertical(stops):
    """A vertical gradient through sRGB stops [(row fraction, (r, g, b)), ...], interpolated in linear light."""
    f = np.linspace(0, 1, H)
    xs = [s for s, _ in stops]
    lin = np.stack([np.interp(f, xs, [M.srgb_to_linear(np.array(c, float))[ch] for _, c in stops]) for ch in range(3)], -1)
    return np.broadcast_to(M.linear_to_srgb(lin)[:, None], (H, W, 3)).astype(np.float32).copy()


GRADIENTS = {
    'sky': [(0, (0.10, 0.20, 0.55)), (0.6, (0.45, 0.70, 0.90)), (1, (0.85, 0.92, 0.95))],
    'sunset': [(0, (0.45, 0.08, 0.10)), (0.5, (0.90, 0.45, 0.15)), (1, (0.98, 0.85, 0.35))],
    'skin': [(0, (0.55, 0.32, 0.25)), (1, (0.95, 0.78, 0.68))],
    'field': [(0, (0.10, 0.25, 0.08)), (1, (0.65, 0.80, 0.30))],
}


def one_seam(rows):
    """(rows,) labels with one pair change: the pair above, the pair below and the row of the change of least total
    error (rows: (P, R) mean CIEDE2000 of each pair over each row of cells)."""
    above, below = np.cumsum(rows, 1), np.cumsum(rows[:, ::-1], 1)[:, ::-1]
    best = None
    for r in range(1, rows.shape[1]):
        a, b = above[:, r - 1].argmin(), below[:, r].argmin()
        cost = above[a, r - 1] + below[b, r]
        if a != b and (best is None or cost < best[0]):
            best = (cost, a, b, r)
    _, a, b, r = best
    return np.where(np.arange(rows.shape[1]) < r, a, b)


def seam_stimuli():
    out = {}
    for name, stops in GRADIENTS.items():
        target = vertical(stops)
        err = cell_errors(target)
        rows = err.mean(2)                                                # (P, rows of cells)
        seam = np.repeat(one_seam(rows)[:, None], W // 8, 1)
        one = np.full_like(seam, err.mean((1, 2)).argmin())
        for label, labels in (('seam', seam), ('one pair', one)):
            r, m = render(target, labels)
            e = de(m, target)
            out[f'{name} {label}'] = (target, r, dict(mean_de=float(e.mean()), max_de=float(e.max()),
                                                     switches=int((np.diff(labels[:, 0]) != 0).sum())))
    return out


def two_colour(mask, left, right):
    return np.where(mask[..., None], np.asarray(left, np.float32), np.asarray(right, np.float32))


def edge_stimuli():
    """Pale blue sky against yellow sand, the edge at x = 131 (3 px right of the border at 128)."""
    left, right = np.array((0.55, 0.75, 0.95), np.float32), np.array((0.90, 0.80, 0.30), np.float32)
    target = two_colour(a0.XX < 131, left, right)
    err_l, err_r = cell_errors(np.broadcast_to(left, (H, W, 3))), cell_errors(np.broadcast_to(right, (H, W, 3)))
    pl, pr = int(err_l.mean((1, 2)).argmin()), int(err_r.mean((1, 2)).argmin())
    out = {}
    for label, col in (('switch left', 16), ('switch right', 17)):
        labels = np.where(np.arange(W // 8)[None] < col, pl, pr).repeat(H // 8, 0)
        r, m = render(target, labels)
        out[f'edge {label}'] = (target, r, dict(mean_de=float(de(m, target).mean())))
    # the cell column 16 (x 128-135) in the pair of the two sides' inks, the edge drawn at 131
    inks = (PAIRS[pl][1], PAIRS[pr][1])
    if inks[0] // 8 != inks[1] // 8:                                       # one brightness: the brighter one's
        inks = tuple(i | 8 for i in inks)
    k = PAIRS.index(tuple(sorted(inks))) if tuple(sorted(inks)) in PAIRS else None
    labels = np.where(np.arange(W // 8)[None] < 16, pl, pr).repeat(H // 8, 0)
    r, m = render(target, labels)
    if k is not None:
        col = np.zeros((H, W), bool)
        col[:, 128:136] = True
        drawn = two_colour(a0.XX < 131, a0.ZX[inks[0]], a0.ZX[inks[1]])
        r = np.where(col[..., None], drawn, r)
        out['edge drawn'] = (target, r, dict(inks=[int(i) for i in inks]))
    return out


def stair_stimuli():
    """Dark red rock against a pale sky, the edge a line of slope 0.6 across the picture."""
    sky, rock = np.array((0.70, 0.85, 0.95), np.float32), np.array((0.45, 0.12, 0.10), np.float32)
    above = a0.YY < 40 + 0.6 * a0.XX
    target = two_colour(above, sky, rock)
    es, er = cell_errors(np.broadcast_to(sky, (H, W, 3))), cell_errors(np.broadcast_to(rock, (H, W, 3)))
    ps, pr = int(es.mean((1, 2)).argmin()), int(er.mean((1, 2)).argmin())
    share = above.reshape(H // 8, 8, W // 8, 8).mean((1, 3))
    labels = np.where(share >= 0.5, ps, pr)
    whole, _ = render(np.where((np.kron(share >= 0.5, np.ones((8, 8))) > 0)[..., None], sky, rock), labels)
    out = {'stair whole cells': (target, whole, {})}
    inks = tuple(sorted((PAIRS[ps][1] | 8, PAIRS[pr][1] | 8)))
    if inks in PAIRS:
        crossed = np.kron((share > 0) & (share < 1), np.ones((8, 8), bool))
        drawn = two_colour(above, a0.ZX[PAIRS[ps][1] | 8], a0.ZX[PAIRS[pr][1] | 8])
        full, _ = render(target, labels)
        out['stair drawn'] = (target, np.where(crossed[..., None], drawn, full), dict(inks=list(inks)))
    return out


def all_stimuli():
    return {**seam_stimuli(), **edge_stimuli(), **stair_stimuli()}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for k, (ref, test, info) in all_stimuli().items():
        name = k.replace(' ', '-')
        Image.fromarray((np.clip(test, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, f'{name}.png'))
        Image.fromarray((np.clip(ref, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, f'{name}-ref.png'))
        print(k, info)
