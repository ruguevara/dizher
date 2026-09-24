"""Palette snap of the Tune column (ops.py): flat surfaces moved, whole, onto colours the screen paints cleanly.

A colour a few CIELAB units off a palette colour dithers as sparse stray dots, the most visible halftone texture,
where on the palette colour it is a solid cell and on the half-and-half mix of a pair a checkerboard. Pulling
each pixel on its own (a colour curve, a per-pixel mean shift) breaks a surface that lies between two targets
into ragged patches and leaves half-pulled fringes; so this labels regions first, as colour harmonization assigns
hue sectors with a graph cut (Cohen-Or et al. 2006):

    E(l) = sum_p U_p(l_p) + smoothness * sum_{p~q} w_pq [l_p != l_q],   l_p in {keep, target 1..K}

U_p(keep) = 1; U_p(target i) = ((d_i + extra_i)^2 + var_p) / radius^2 with d_i the CIELAB distance of the pixel's
base to the target and var_p the squared change of the base across a cell (CELL times its gradient, averaged in
the window; steps sharper than radius / 2 over two pixels are edges and left to w_pq), so a pixel joins a target only when its surface is near it and flat: a ramp that would be cut into a
plateau stays, and texture on the surface does not count, it rides along; w_pq = exp(-|x_p - x_q|^2 / 2 radius^2) makes a label change free across an edge of the image. Each
pixel considers its CANDIDATES nearest targets and keep. Labels by exact dynamic programming on every other row
(rows of one parity do not interact given the rest), then the other rows, then the same on columns, until
nothing changes: the line coordinate descent of converter/energy.py.

The labelled pixels' base moves `strength` of the way to their target and the detail on it goes along (Durand and
Dorsey 2002); the change map is then smoothed by a guided filter on the image (Rabin, Delon and Gousseau 2011) so
a region's border is a soft edge that follows the image rather than a step where a gradient leaves the radius.
"""
import cv2
import numpy as np

from .converter.colors import lab2rgb, rgb2lab

GAMMA = 2.2         # the converter's encoding: mixes are averaged in linear light, as the eye sees dots
WINDOW = 4          # px, half the box of the base and of the flatness: half a cell, so a cell is one surface
MIX_COST = 0.15     # CIELAB distance a mix target counts as further away per unit of its pair's own distance
CELL = 8            # px, the span over which the base's slope counts as a change: an attribute cell
CANDIDATES = 4      # nearest targets each pixel may take, besides keep
SWEEPS = 6          # row and column passes at most; they settle in 3..4


def targets(palette, mixes: bool = True):
    """The colours a flat surface dithers cleanly with, as (K, 3) CIELAB and (K,) extra distance: every colour an
    allowed pair uses (a solid cell, no extra) and, with mixes, the half-and-half mix of each allowed pair in
    linear light, a checkerboard, counted MIX_COST of its pair's CIELAB distance further away: black and white
    dots are a harsher texture than two close reds."""
    rgb = palette.as_float()
    pairs = np.array(list(palette.iter_idxs_pairs()))
    solid = np.unique(pairs)
    lab = lambda c: rgb2lab(np.asarray(c, np.float32)[None])[0]
    found, extra = [lab(rgb[solid])], [np.zeros(len(solid), np.float32)]
    if mixes:
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        found.append(lab(((rgb[pairs[:, 0]] ** GAMMA + rgb[pairs[:, 1]] ** GAMMA) / 2) ** (1 / GAMMA)))
        extra.append(MIX_COST * np.linalg.norm(lab(rgb[pairs[:, 0]]) - lab(rgb[pairs[:, 1]]), axis=-1))
    found, extra = np.concatenate(found), np.concatenate(extra).astype(np.float32)
    _, first = np.unique(found.round(2), axis=0, return_index=True)   # coinciding colours: keep the one listed first
    return found[np.sort(first)], extra[np.sort(first)]


def _box(x):
    return cv2.boxFilter(x, -1, (2 * WINDOW + 1,) * 2)


def _guided(guide, x, eps):
    """He et al.'s guided filter of each channel of x by a scalar guide, WINDOW box."""
    mg, mx = _box(guide), _box(x)
    cov = _box(guide[..., None] * x) - mg[..., None] * mx
    a = cov / (_box(guide * guide) - mg * mg + eps)[..., None]
    b = mx - a * mg[..., None]
    return _box(a) * guide[..., None] + _box(b)


def _rows(U, ids, Wh, Wv, smoothness, labels):
    """One pass over the rows, every other row at once: each row's labels re-solved exactly (Viterbi) given the
    rows above and below. U, ids: (H, W, M) candidate costs and global ids (-1 keep); Wh: (H, W-1) and
    Wv: (H-1, W) seam weights; labels: (H, W) candidate indexes, updated. Returns how many changed."""
    H, W, M = U.shape
    changed = 0
    for parity in (0, 1):
        r = np.arange(parity, H, 2)
        cur = np.take_along_axis(ids, labels[..., None], -1)[..., 0]      # (H, W) current global ids
        un = U[r].copy()
        up, down = r > 0, r < H - 1
        un[up] += smoothness * Wv[r[up] - 1][..., None] * (ids[r[up]] != cur[r[up] - 1][..., None])
        un[down] += smoothness * Wv[r[down]][..., None] * (ids[r[down]] != cur[r[down] + 1][..., None])
        cost = un[:, 0]
        back = np.zeros((len(r), W, M), dtype=np.int8)
        for c in range(1, W):
            step = smoothness * Wh[r, c - 1][:, None, None] * (ids[r, c - 1][:, :, None] != ids[r, c][:, None, :])
            total = cost[:, :, None] + step                               # (rows, M_prev, M)
            back[:, c] = total.argmin(1)
            cost = un[:, c] + np.take_along_axis(total, back[:, c][:, None, :], 1)[:, 0]
        new = np.empty((len(r), W), dtype=labels.dtype)
        new[:, -1] = cost.argmin(1)
        for c in range(W - 1, 0, -1):
            new[:, c - 1] = np.take_along_axis(back[:, c], new[:, c:c + 1].astype(np.int64), 1)[:, 0]
        changed += int((new != labels[r]).sum())
        labels[r] = new
    return changed


def label(base, var, found, extra, radius, smoothness, image):
    """(H, W) target index per pixel, -1 to keep it (see the module doc)."""
    H, W, _ = base.shape
    b = base.reshape(-1, 3)
    d2 = (b * b).sum(1)[:, None] - 2 * b @ found.T + (found * found).sum(1)[None]
    d = np.sqrt(np.maximum(d2, 0)) + extra[None]
    near = np.argsort(d, axis=1)[:, :CANDIDATES]
    cost = (np.take_along_axis(d, near, 1) ** 2 + var.reshape(-1, 1)) / radius ** 2
    U = np.concatenate([np.ones((H * W, 1), np.float32), cost], 1).reshape(H, W, -1).astype(np.float32)
    ids = np.concatenate([np.full((H * W, 1), -1), near], 1).reshape(H, W, -1)
    seam = lambda a, b: np.exp(-((a - b) ** 2).sum(-1) / (2 * radius ** 2)).astype(np.float32)
    Wh, Wv = seam(image[:, 1:], image[:, :-1]), seam(image[1:], image[:-1])
    labels = U.argmin(-1)
    for _ in range(SWEEPS):
        changed = _rows(U, ids, Wh, Wv, smoothness, labels)
        labels_t = np.ascontiguousarray(labels.T)
        changed += _rows(U.transpose(1, 0, 2), ids.transpose(1, 0, 2), Wv.T, Wh.T, smoothness, labels_t)
        labels = np.ascontiguousarray(labels_t.T)
        if not changed:
            break
    return np.take_along_axis(ids, labels[..., None], -1)[..., 0]


def snap(rgb: np.ndarray, found, strength: float = 0.0, radius: float = 24.0, smoothness: float = 2.0,
         labels_out=None) -> np.ndarray:
    """Float sRGB in 0..1 in and out; strength 0 returns the input. found: targets(). labels_out, a list, gets the
    labelling for inspection."""
    if not strength:
        return rgb
    found, extra = found
    lab = rgb2lab(rgb.astype(np.float32))
    mean = _box(lab)
    var = _box(lab * lab) - mean ** 2
    a = var / (var + radius ** 2)
    base = _box(a) * lab + _box(mean - a * mean)        # self-guided: smooth, keeping steps over the radius
    gy, gx = np.gradient(base, axis=(0, 1))
    slope = (gx ** 2 + gy ** 2).sum(-1)
    slope[slope > (radius / 4) ** 2] = 0                   # a step, an edge between surfaces: the seam weights' job
    ramp = _box(CELL ** 2 * slope)                         # the base's change across a cell on a ramp
    labels = label(base, ramp, found, extra, radius, smoothness, lab)
    if labels_out is not None:
        labels_out.append(labels)
    delta = np.where(labels[..., None] >= 0, found[np.maximum(labels, 0)] - base, 0) * min(strength, 1.0)
    delta = _guided(lab[..., 0] / 100, delta.astype(np.float32), 1e-3)
    return np.clip(lab2rgb(lab + delta), 0, 1).astype(np.float32)
