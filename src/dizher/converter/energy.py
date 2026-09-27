"""Colour-pair selection as one eye-model energy over the whole composite image.

E(labels) = sum_ch w_ch || h_ch * (M (Y - X))_ch ||^2   with M the per-pixel local metric (local_metric: the
Jacobian of CIELAB at the target's colour, so an error costs about its CIELAB difference there), Y the
composite and X the target in linear RGB, h_ch the eye kernel of the channel's group (Luma: L*, Chroma: a* and
b*). A pair's candidate on a block is the exact mixture its per-pixel level asks for (paper + t (ink - paper)),
not one halftone of it: a halftone's noise decided near ties at random, block by block. Because the composite
is a sum of per-block candidates, E splits exactly into a per-block term D[p, b] and pairwise terms
S[p, q, b, b'] = 2 e_p[b]^T K e_q[b'] with K = h (*) h the kernel autocorrelation. The pairwise term is the
visible seam a pair change paints, in the same units as D, with no extra knob.

The dots cost apart from the mixture: each group's unblurred error, the mixture's residual plus the
t (1 - t) contrast^2 the dots add around it, weighted by luma_noise and chroma_noise (hue_contrast).
Chroma contrast counts hue only: blue dots on yellow, which average to a pale colour, cost the most, and
bright yellow dots on black cost nothing, as the hand-painted references prefer them to a dim pair with
fewer dots. For the same reason lightness dot contrast has a weight of only 0.001 by default: enough to break an
exact tie towards the lower-contrast pair (a grey that black mixes with dim or bright white alike), too small to
outweigh any difference of mixture.

Blurred mean-square error is blind to texture: a lone block dithered with green dots among blocks
dithered with yellow dots is plainly a cell even when the mean colours agree. So a coherence term
charges a pair change between 4-neighbours by how different the two pairs look (CIELUV distance of the
papers plus of the inks), scaled down where the original itself has an edge across that seam:
    coherence * SEAM_COST * sum_{b~b'} V[p_b, p_b'] * exp(-|x_b - x_b'|^2 / 2 edge^2)
This is the contrast-sensitive Potts prior of MRF segmentation. It is graded, so the bright
variant of the same colours is nearly free, and a change along a real edge costs nothing.
The eye kernels have radius at most half the smallest cell dimension, so their autocorrelations
couple only the 8 neighbouring blocks. No nonzero interaction is dropped.
Labels by block coordinate descent on whole lines: each row, then each column, is
re-solved exactly by dynamic programming given the rest, so a run of blocks can switch together
(single-block ICM gets trapped by clusters that are wrong in the same way).
"""
from collections import OrderedDict

import numpy as np
import cv2

from .colors import convert_color
from ..progress import report_progress, report_stage

SRGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                     [0.2126, 0.7152, 0.0722],
                     [0.0193, 0.1192, 0.9505]], dtype=np.float32)
WHITE_XYZ = SRGB2XYZ.sum(axis=1)
XYZ2OPP = np.array([[ 0.279,  0.72, -0.107],      # Poirson & Wandell opponent space, as in S-CIELAB
                    [-0.449,  0.29, -0.077],
                    [ 0.086, -0.59,  0.501]], dtype=np.float32)
# The Eye view's blur space (display only; the energy measures in local_metric). Poirson & Wandell's chroma axes
# are not orthogonal to the neutral axis: white maps to O2 = -0.22. Remove the neutral component so every grey has
# zero chroma and the chroma blur leaves greys alone.
LRGB2OPP = XYZ2OPP @ SRGB2XYZ
_white = LRGB2OPP @ np.ones(3, dtype=np.float32)
LRGB2OPP[1:] -= (_white[1:] / _white[0])[:, None] * LRGB2OPP[0]

GROUPS = OrderedDict(Luma=[0], Chroma=[1, 2])   # weight name -> opponent channels
OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1)]      # unordered neighbour pairs, block units
# Default step of the original's block means (CIELAB over _unit, ~90 at flare 0.1: 0.4 is ~36 delta E) that counts
# as a real edge (Converter.edge). The test is per seam, so it cannot tell an edge from a steep smooth gradient: faces
# and sharpened texture step by 10-20 delta E per block and lost their coherence at a lower threshold.
EDGE_SIGMA = 0.4
SEAM_COST = 0.2     # energy of one seam between totally different pairs at coherence 1; fitted with tests/pair_bench.py

LIGHTNESS_REF = 0.18   # mid grey's luminance: its error keeps weight 1

def _f(u):
    """CIELAB's compression of X/Xn, Y/Yn, Z/Zn: a cube root with a linear toe."""
    return np.where(u > (6 / 29) ** 3, np.cbrt(np.maximum(u, 1e-12)), u * (29 / 6) ** 2 / 3 + 4 / 29)

def _slope(u):
    """d _f / du."""
    return np.where(u > (6 / 29) ** 3, np.cbrt(np.maximum(u, 1e-12)) ** -2 / 3, (29 / 6) ** 2 / 3)

def _unit(flare: float) -> float:
    """L* per unit of luminance at mid grey: the scale that keeps a mid grey error at weight 1."""
    return float(116 * _slope(LIGHTNESS_REF + flare))

def local_metric(image_lrgb: np.ndarray, flare: float) -> np.ndarray:
    """(H, W, 3, 3) per pixel: the Jacobian of CIELAB (L*, a*, b*) with respect to linear RGB at the target's colour,
    over _unit. Errors are mapped through it before the eye blur, so within a flat area mixing stays linear while
    each error costs about its CIELAB difference there: ~7x more lightness error in black than in mid grey, and, as
    X, Y and Z each go through their own cube root, chroma compressed along the target's own dominant primary. A
    single lightness gain from Y for all three channels over-charged blue in a dark navy (Z far above Y), so the
    level fit stopped at 7% blue, too dark, and sparse magenta dots (lightness from red, hue off) won. flare, stray
    light on the screen in units of white, is added to the target first and flattens the metric towards plain
    linear light."""
    xyz = ((image_lrgb.astype(np.float32) + flare) @ SRGB2XYZ.T) / WHITE_XYZ
    fx, fy, fz = np.moveaxis(_slope(xyz) / WHITE_XYZ, -1, 0)
    zero = np.zeros_like(fy)
    dlab = np.stack([np.stack([zero, 116 * fy, zero], -1),
                     np.stack([500 * fx, -500 * fy, zero], -1),
                     np.stack([zero, 200 * fy, -200 * fz], -1)], -2)          # d(L*, a*, b*) / d(X, Y, Z)
    return (dlab @ SRGB2XYZ / _unit(flare)).astype(np.float32)

def lab_coordinates(image_lrgb: np.ndarray, flare: float) -> np.ndarray:
    """(..., 3) CIELAB of linear RGB plus flare, over _unit: absolute coordinates on local_metric's scale."""
    f = _f(((image_lrgb.astype(np.float32) + flare) @ SRGB2XYZ.T) / WHITE_XYZ)
    lab = np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
    return (lab / _unit(flare)).astype(np.float32)

def local(metric: np.ndarray, linear_rgb: np.ndarray) -> np.ndarray:
    """Linear RGB (..., H, W, 3), or one colour (3,), through the per-pixel metric (H, W, 3, 3) -> (..., H, W, 3)."""
    linear_rgb = np.asarray(linear_rgb, dtype=np.float32)
    if linear_rgb.ndim == 1:
        return metric @ linear_rgb
    return np.einsum('hwij,...hwj->...hwi', metric, linear_rgb, optimize=True)

def hue_contrast(color_pairs: np.ndarray) -> np.ndarray:
    """(P, 2, 3) gamma-encoded RGB pairs -> (P,) squared contrast of hue alone between a pair's paper and ink dots, in
    CIELAB, 2 (|ab_p| |ab_i| - ab_p . ab_i) / 100^2: none for black, white or grey dots on any colour, most for
    complementary pairs (blue on yellow, green on magenta) whose mixture the eye sees as dots of both. Yellow dots on
    black cost no chroma: the hand-painted references (tests/pair_bench.py) keep bright dots of the target's hue over
    a dim pair with fewer dots."""
    ab = convert_color(color_pairs.astype(np.float32).reshape(1, -1, 3), 'RGB', 'LAB').reshape(-1, 2, 3)[..., 1:] / 100
    hue = 2 * (np.linalg.norm(ab[:, 0], axis=-1) * np.linalg.norm(ab[:, 1], axis=-1) - (ab[:, 0] * ab[:, 1]).sum(-1))
    return np.maximum(hue, 0).astype(np.float32)

def autocorrelation(h: np.ndarray) -> np.ndarray:
    r = h.shape[0] // 2
    return cv2.filter2D(np.pad(h, r), -1, h, borderType=cv2.BORDER_CONSTANT)

def block_kernel_matrix(cpp: np.ndarray, dr: int, dc: int, cell) -> np.ndarray:
    """K[x, y] = cpp(pos_x - pos_y) for pixel x of block (0, 0) and pixel y of block (dr, dc); cell = (h, w)."""
    centre = cpp.shape[0] // 2
    h, w = cell
    xi, xj = np.divmod(np.arange(h * w), w)
    di = xi[:, None] - (xi[None, :] + h * dr) + centre
    dj = xj[:, None] - (xj[None, :] + w * dc) + centre
    inside = (di >= 0) & (di < cpp.shape[0]) & (dj >= 0) & (dj < cpp.shape[1])
    K = np.zeros(di.shape, dtype=np.float32)
    K[inside] = cpp[di[inside], dj[inside]]
    return K

def pair_dissimilarity(color_pairs: np.ndarray) -> np.ndarray:
    """(P, 2, 3) gamma-encoded RGB pairs -> (P, P) in 0..1: CIELUV distance between the papers plus
    between the inks (pairs are sorted dark to bright, so that matching is natural)."""
    luv = convert_color(color_pairs.astype(np.float32).reshape(1, -1, 3), 'RGB', 'LUV').reshape(-1, 2, 3)
    luv /= np.array([100, 180, 180], dtype=np.float32)
    dist = lambda a: np.linalg.norm(a[:, None] - a[None], axis=-1)
    V = dist(luv[:, 0]) + dist(luv[:, 1])
    return (V / V.max()).astype(np.float32)

def _ranges(dr, dc, R, C):
    """Block index ranges of 'me' such that the neighbour at (dr, dc) exists."""
    return slice(max(0, -dr), R - max(0, dr)), slice(max(0, -dc), C - max(0, dc))

class SelectionEnergy:
    def __init__(self, converter, weights) -> None:
        self.converter = converter
        self.weights = OrderedDict(weights)
        assert list(self.weights) == list(GROUPS)
        self.invalidate()

    def invalidate(self) -> None:
        self.D, self.S, self.X = {}, {}, {}
        self.N = {}     # group -> (P, R, C) unblurred squared error per block

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.weights:
                raise ValueError("Can not set weight for unexistent metric {}".format(k))
            self.weights[k] = v

    def calc(self) -> None:
        report_stage('selection energy')
        c = self.converter
        X = local(c.metric, c.image_lrgb)
        pal = np.einsum('hwij,nj->nhwi', c.metric, c.palette.as_float() ** c.gamma)   # (N, H, W, 3) every palette colour
        idx = np.array(list(c.palette.iter_idxs_pairs()))                  # (P, 2) paper, ink
        span = pal[idx[:, 1]] - pal[idx[:, 0]]                             # (P, H, W, 3)
        t = c.levels.astype(np.float32)
        E = pal[idx[:, 0]] + t[..., None] * span - X                       # the exact mixtures' error, (P, H, W, 3)
        P, H, W, _ = E.shape
        h, w = c.cell
        R, C = H // h, W // w
        # t(1 - t) contrast^2 is the mean squared error the dots add around their mixture; lightness contrast in the
        # local metric, hue contrast in CIELAB scaled by the metric's lightness gain at the target
        block = lambda a: a.reshape(P, R, h, C, w).sum(axis=(2, 4))       # (P, H, W) -> (P, R, C) sums
        spread = t * (1 - t)
        gain = c.metric[..., 0, :].sum(-1)                                 # dL* per unit of grey light, over _unit
        dots = dict(Luma=block(spread * span[..., 0] ** 2),
                    Chroma=hue_contrast(c.color_pairs)[:, None, None] * block(spread * gain ** 2))
        E = E.reshape(P, R, h, C, w, 3).transpose(1, 3, 0, 2, 4, 5).reshape(R, C, P, h * w, 3)
        Xs = lab_coordinates(c.image_lrgb, c.flare)                        # absolute, for the seams' edge test
        luma, chroma, _ = c.eye_kernels()
        kernels = dict(Luma=luma, Chroma=chroma)
        for g, channels in GROUPS.items():
            cpp = autocorrelation(kernels[g])
            K0 = block_kernel_matrix(cpp, 0, 0, c.cell)
            A = [np.ascontiguousarray(E[..., k]) for k in channels]         # each (R, C, P, 64)
            self.X[g] = Xs[..., channels].reshape(R, h, C, w, len(channels)).mean(axis=(1, 3))   # (R, C, nch) target block means
            self.N[g] = sum((a ** 2).sum(-1) for a in A).transpose(2, 0, 1) + dots[g]
            self.D[g] = sum(np.einsum('rcpx,xy,rcpy->prc', a, K0, a, optimize=True) for a in A)
            self.S[g] = {}
            for dr, dc in OFFSETS:
                K = block_kernel_matrix(cpp, dr, dc, c.cell)
                rs, cs = _ranges(dr, dc, R, C)
                ns = slice(rs.start + dr, rs.stop + dr), slice(cs.start + dc, cs.stop + dc)
                self.S[g][(dr, dc)] = 2 * sum(
                    np.einsum('rcpy,rcqy->rcpq', np.einsum('rcpx,xy->rcpy', a[rs, cs], K), a[ns])
                    for a in A)

    def unary(self) -> np.ndarray:
        w, c = self.weights, self.converter
        noise = dict(Luma=c.luma_noise, Chroma=c.chroma_noise)
        return sum(w[g] * self.D[g] for g in GROUPS) + sum(w[g] * noise[g] * self.N[g] for g in GROUPS)

    def apply(self) -> np.ndarray:
        """The (R, C) pair labels of least energy."""
        w = self.weights
        D = self.unary()
        S = {off: sum(w[g] * self.S[g][off] for g in GROUPS) for off in OFFSETS}
        Lh, Lv = self.seam_smoothness()
        V = self.converter.pair_dissimilarity
        preview = lambda labels: report_progress(lambda: self.converter.snapshot(labels=labels))
        return optimise(D, S, V, Lh, Lv, self.converter.coherence, on_step=preview)

    def seam_smoothness(self):
        """Weight 0..1 of every seam: 1 where the original is flat across it, ~0 across a real edge.
        Lh: (R-1, C) between (r, c) and (r+1, c); Lv: (R, C-1) between (r, c) and (r, c+1)."""
        w = self.weights
        X = np.concatenate([np.sqrt(w[g]) * self.X[g] for g in GROUPS], axis=-1)
        step = lambda d: np.exp(-(d ** 2).sum(-1) / (2 * self.converter.edge ** 2)).astype(np.float32)
        return step(X[1:] - X[:-1]), step(X[:, 1:] - X[:, :-1])

    def energy(self, labels: np.ndarray) -> float:
        """Total energy of a labelling (for checks)."""
        return float(self.cell_energies(labels).sum())

    def cell_energies(self, labels: np.ndarray) -> np.ndarray:
        """(R, C, 3) a labelling's energy per block: its own term, the eye-model seam terms (negative where the
        neighbours' errors cancel) and the coherence cost; each seam split half and half between its blocks."""
        w = self.weights
        P, R, C = next(iter(self.D.values())).shape
        ri, ci = np.indices((R, C))
        own, seam, coherence = self.unary()[labels, ri, ci], np.zeros((R, C)), np.zeros((R, C))
        for (dr, dc) in OFFSETS:
            rs, cs = _ranges(dr, dc, R, C)
            ns = slice(rs.start + dr, rs.stop + dr), slice(cs.start + dc, cs.stop + dc)
            me, nb = labels[rs, cs], labels[ns]
            i, j = np.indices(me.shape)
            half = sum(w[g] * self.S[g][(dr, dc)][i, j, me, nb] for g in GROUPS) / 2
            seam[rs, cs] += half
            seam[ns] += half
        Lh, Lv = self.seam_smoothness()
        V, k = self.converter.pair_dissimilarity, self.converter.coherence * SEAM_COST / 2
        half = k * Lh * V[labels[1:], labels[:-1]]
        coherence[1:] += half
        coherence[:-1] += half
        half = k * Lv * V[labels[:, 1:], labels[:, :-1]]
        coherence[:, 1:] += half
        coherence[:, :-1] += half
        return np.stack([own, seam, coherence], axis=-1)

    def cell_candidates(self, labels: np.ndarray, r: int, c: int) -> np.ndarray:
        """(P, 3) every pair's cost at block (r, c) with the other labels fixed: its own term, its eye-model seams
        and its coherence with the neighbours, each seam whole, as a line solve trades them. The sum differs from
        the total energy of the labelling with that pair at (r, c) by one constant for all pairs."""
        w, R, C = self.weights, *labels.shape
        own, seam, coherence = self.unary()[:, r, c], 0, 0
        for dr, dc in OFFSETS:
            rs, cs = _ranges(dr, dc, R, C)
            S = lambda i, j, p, q: sum(w[g] * self.S[g][(dr, dc)][i - rs.start, j - cs.start, p, q] for g in GROUPS)
            if rs.start <= r < rs.stop and cs.start <= c < cs.stop:                   # the neighbour at (r+dr, c+dc)
                seam = seam + S(r, c, slice(None), labels[r + dr, c + dc])
            if rs.start <= r - dr < rs.stop and cs.start <= c - dc < cs.stop:         # the one at (r-dr, c-dc)
                seam = seam + S(r - dr, c - dc, labels[r - dr, c - dc], slice(None))
        Lh, Lv = self.seam_smoothness()
        V = self.converter.pair_dissimilarity
        for near, L in (((r - 1, c), Lh[r - 1, c] if r > 0 else 0), ((r + 1, c), Lh[r, c] if r < R - 1 else 0),
                        ((r, c - 1), Lv[r, c - 1] if c > 0 else 0), ((r, c + 1), Lv[r, c] if c < C - 1 else 0)):
            if L:
                coherence = coherence + L * V[:, labels[near]]
        return np.stack(np.broadcast_arrays(own, seam, self.converter.coherence * SEAM_COST * coherence), axis=-1)

def _transpose(D, S, Lh, Lv):
    """The same problem with rows and columns swapped."""
    St = {(1, 0): S[(0, 1)].transpose(1, 0, 2, 3),
          (0, 1): S[(1, 0)].transpose(1, 0, 2, 3),
          (1, 1): S[(1, 1)].transpose(1, 0, 2, 3),
          (1, -1): S[(1, -1)].transpose(1, 0, 3, 2)}   # the pair reverses its roles, so p and q swap
    return D.transpose(0, 2, 1), St, Lv.T, Lh.T

def _row_pass(D, S, V, Lh, Lv, coherence, labels, on_row=None):
    """Re-solve every row exactly (Viterbi over the labels) given the other rows. Returns changes."""
    P, R, C = D.shape
    k = coherence * SEAM_COST
    changed = 0
    for r in range(R):
        # unary: own cost plus everything coupling this row to the rows above and below (fixed)
        U = D[:, r, :].T.copy()                                           # (C, P)
        for (dr, dc), Sd in S.items():
            if dr == 0:
                continue
            rs, cs = _ranges(dr, dc, R, C)
            if rs.start <= r < rs.stop:                                   # me at (r, c), neighbour below
                cols = np.arange(cs.start, cs.stop)
                U[cols] += Sd[r - rs.start, cols - cs.start, :, labels[r + dr, cols + dc]]
            if rs.start <= r - dr < rs.stop:                              # neighbour above at (r-dr, c-dc), me
                cols = np.arange(cs.start + dc, cs.stop + dc)
                U[cols] += Sd[r - dr - rs.start, cols - dc - cs.start, labels[r - dr, cols - dc], :]
        if k > 0:
            if r > 0:
                U += k * Lh[r - 1][:, None] * V[:, labels[r - 1]].T
            if r < R - 1:
                U += k * Lh[r][:, None] * V[:, labels[r + 1]].T
        # transitions between (r, c) and (r, c+1)
        T = S[(0, 1)][r]                                                  # (C-1, P, P)
        if k > 0:
            T = T + k * Lv[r][:, None, None] * V[None]
        Vc = U[0]
        back = np.zeros((C, P), dtype=np.int64)
        for c in range(1, C):
            cand = Vc[:, None] + T[c - 1]                                 # (P_prev, P)
            back[c] = cand.argmin(0)
            Vc = U[c] + cand[back[c], np.arange(P)]
        new = np.zeros(C, dtype=labels.dtype)
        new[-1] = int(Vc.argmin())
        for c in range(C - 1, 0, -1):
            new[c - 1] = back[c, new[c]]
        changed += int((new != labels[r]).sum())
        labels[r] = new
        if on_row:
            on_row()
    return changed

def optimise(D: np.ndarray, S: dict, V: np.ndarray, Lh: np.ndarray, Lv: np.ndarray, coherence: float,
             max_sweeps: int = 10, on_step=None) -> np.ndarray:
    """D: (P, R, C) unary; S[(dr, dc)]: (R', C', P, P) pairwise for block (r, c) with (r+dr, c+dc);
    V: (P, P) pair dissimilarity; Lh, Lv: seam smoothness weights. Returns (R, C) labels."""
    labels = D.argmin(0)
    Dt, St, Lht, Lvt = _transpose(D, S, Lh, Lv)
    changed = None
    for sweep in range(max_sweeps):
        report_stage(f'pair sweep {sweep + 1}' + (f' · {changed} changed' if changed is not None else ''))
        changed = _row_pass(D, S, V, Lh, Lv, coherence, labels, on_step and (lambda: on_step(labels)))
        labels = np.ascontiguousarray(labels.T)
        changed += _row_pass(Dt, St, V, Lht, Lvt, coherence, labels, on_step and (lambda: on_step(labels.T)))
        labels = np.ascontiguousarray(labels.T)
        if changed == 0:
            break
    return labels
