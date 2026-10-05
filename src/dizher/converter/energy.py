"""Colour-pair selection as one eye-model energy over the whole composite image.

E(labels) = sum_ch w_ch || h_ch * (g (Y_ch - X_ch)) ||^2   in a linear opponent space (S-CIELAB's
O1 luminance, O2 red-green, O3 blue-yellow), Y the composite, X the target, h_ch the eye kernel of the
channel's group (Luma: O1, Chroma: O2 and O3), g the per-pixel lightness gain (lightness_gain: linear-light
error weighted as CIELAB sees it at the target). A pair's candidate on a block is what the selection method
(METHODS) makes of its per-pixel level: the exact mixture paper + t (ink - paper), or one halftone of it.
Because the composite is a sum of per-block candidates, E splits exactly
into a per-block term D[p, b] and pairwise terms S[p, q, b, b'] = 2 e_p[b]^T K e_q[b'] with K = h (*) h
the kernel autocorrelation. The pairwise term is the visible seam a pair change paints, in the same units
as D, with no extra knob.

Each group's unblurred error, weighted by luma_noise and chroma_noise, charges what the blur hides. For the
exact mixture it is the mixture's residual plus the t (1 - t) contrast^2 the dots add around it (dot_contrast).
Chroma contrast counts hue only: blue dots on yellow, which average to a pale colour, cost the most, and
bright yellow dots on black cost nothing, as the hand-painted references prefer them to a dim pair with
fewer dots. Luma dot contrast has weight 0 by default for the same reason.

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
XYZ2OPP = np.array([[ 0.279,  0.72, -0.107],      # Poirson & Wandell opponent space, as in S-CIELAB
                    [-0.449,  0.29, -0.077],
                    [ 0.086, -0.59,  0.501]], dtype=np.float32)
LRGB2OPP = XYZ2OPP @ SRGB2XYZ
# Poirson & Wandell's chroma axes are not orthogonal to the neutral axis: white maps to O2 = -0.22, so a
# luminance error leaks into "chroma". Remove the neutral component so every grey has zero chroma.
_white = LRGB2OPP @ np.ones(3, dtype=np.float32)
LRGB2OPP[1:] -= (_white[1:] / _white[0])[:, None] * LRGB2OPP[0]
# Put the three channels on one scale: over the ZX palette red-green spans ~3x less than luminance and
# blue-yellow ~1.2x less, which made chroma error almost free. Each row is scaled so the palette's spread
# is the same in every channel; the Luma/Chroma weights then compare like with like.
_palette_std = np.array([0.2859, 0.0995, 0.2416], dtype=np.float32)
LRGB2OPP = LRGB2OPP * (_palette_std[0] / _palette_std)[:, None]

GROUPS = OrderedDict(Luma=[0], Chroma=[1, 2])   # weight name -> opponent channels
OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1)]      # unordered neighbour pairs, block units
# Default step of the original's block means (weighted opponent units) that counts as a real edge (Converter.edge).
# The test is per seam, so it cannot tell an edge from a steep smooth gradient; at 0.05 a red-to-yellow sky (0.03..0.07
# per block in linear light) read as edges on every row and lost its coherence, real outlines step by 0.2 and more.
EDGE_SIGMA = 0.1
SEAM_COST = 0.1     # energy of one seam between totally different pairs at coherence 1; a block's own cost is ~0.2
SURFACE_CELLS = 24  # cells per surface, about: 32 surfaces on the Spectrum's 768 cells

LIGHTNESS_REF = 0.18   # mid grey's luminance: its error keeps weight 1

def lightness_gain(luminance: np.ndarray, flare: float) -> np.ndarray:
    """(H, W, 1) per-pixel gain of the opponent error: dL*/dY at the target's luminance over mid grey's, the eye's
    sensitivity to an error in linear light, in CIELAB ~7x higher at black than at mid grey and ~3x lower at white.
    Errors are scaled by it before the eye blur: within a flat area mixing stays linear, while a brown patch in a
    black shadow costs about what it looks like instead of its ~4% of luminance. Chroma takes the same gain, as
    CIELAB's a* b* do: on luma alone, dim red dots won dark greys from sparse white ones, their colour error left
    cheap. flare, stray light on the screen in units of white, flattens the gain towards plain linear light."""
    slope = lambda y: np.where(y > (6 / 29) ** 3, np.cbrt(np.maximum(y, 1e-6)) ** -2 / 3, (29 / 6) ** 2 / 3)
    return (slope(luminance + flare) / slope(LIGHTNESS_REF + flare))[..., None].astype(np.float32)

def dot_contrast(color_pairs: np.ndarray, gamma: float) -> np.ndarray:
    """(P, 2, 3) gamma-encoded RGB pairs -> (P, 2) squared contrast between a pair's paper and ink dots, per GROUPS.
    Luma: of the opponent luminance channel. Chroma: of hue alone, in CIELAB, 2 (|ab_p| |ab_i| - ab_p . ab_i) / 100^2:
    none for black, white or grey dots on any colour, most for complementary pairs (blue on yellow, green on magenta)
    whose mixture the eye sees as dots of both. Yellow dots on black cost no chroma: the hand-painted references
    (tests/pair_bench.py) keep bright dots of the target's hue over a dim pair with fewer dots."""
    lin = (color_pairs.astype(np.float32) ** gamma) @ LRGB2OPP.T
    ab = convert_color(color_pairs.astype(np.float32).reshape(1, -1, 3), 'RGB', 'LAB').reshape(-1, 2, 3)[..., 1:] / 100
    hue = 2 * (np.linalg.norm(ab[:, 0], axis=-1) * np.linalg.norm(ab[:, 1], axis=-1) - (ab[:, 0] * ab[:, 1]).sum(-1))
    return np.stack([(lin[:, 1, 0] - lin[:, 0, 0]) ** 2, np.maximum(hue, 0)], axis=-1).astype(np.float32)

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

class Halftoned:
    """Each pair scored on one halftone of its per-pixel mixture, painted by the chosen halftoner: the pairs are chosen
    for the dots they will get. The halftone's noise decides near ties block by block, and the dot contrast of a
    bright pair is charged twice (in the blurred error and in the unblurred one), so dim pairs win. Develop's and
    0.2.4's scoring, kept exactly; a project saved without a method opens with it. No chroma noise and a wider chroma
    blur by the user's picks."""
    preset = dict(chroma=1.0, flare=0.1, coherence=2.0, edge=EDGE_SIGMA, luma_noise=0.0, chroma_noise=0.0,
                  luma_scale=1.4, chroma_scale=1.6)

    def candidates(self, c, X):
        Y = (c.realized ** c.gamma) @ LRGB2OPP.T
        return (Y - X) * c.gain, None


class ExactMixture:
    """Each pair scored on the exact mixture its per-pixel level asks for, and its dots apart from the mixture:
    t (1 - t) times the squared contrast of paper and ink (dot_contrast), in lightness and in hue alone. Blue dots
    on yellow cost the most, black, white or grey dots nothing. Defaults fitted to the hand-painted references of
    tests/pair_bench.py; no chroma noise and a narrower luma blur by the user's picks."""
    preset = dict(chroma=2.0, flare=0.1, coherence=6.0, edge=EDGE_SIGMA, luma_noise=0.0, chroma_noise=0.0,
                  luma_scale=0.9, chroma_scale=1.4)

    def candidates(self, c, X):
        lin = (c.color_pairs.astype(np.float32) ** c.gamma) @ LRGB2OPP.T              # (P, 2, 3) paper, ink
        t = c.levels.astype(np.float32)
        Y = lin[:, None, None, 0] + t[..., None] * (lin[:, None, None, 1] - lin[:, None, None, 0])
        P, H, W = t.shape
        h, w = c.cell
        # t(1 - t) contrast^2 is the mean squared error dots add around their mixture, gained per pixel like E
        spread = (t * (1 - t) * c.gain[..., 0] ** 2).reshape(P, H // h, h, W // w, w).sum(axis=(2, 4))
        dots = dict(zip(GROUPS, dot_contrast(c.color_pairs, c.gamma).T))
        return (Y - X) * c.gain, {g: dots[g][:, None, None] * spread for g in GROUPS}


# How a pair is scored on a block (Metric: method): the rest of pair selection, the halftoner and DBS are shared.
# Each has the Metric, Select pairs and Eye values it was tuned with, set when it is picked (ops.apply_preset).
METHODS = {'Exact mixture': ExactMixture(), 'Halftoned': Halftoned()}
NEWEST = 'Exact mixture'    # new projects
LEGACY = 'Halftoned'        # projects saved before there was a choice

def surfaces(image_rgb: np.ndarray, cell) -> tuple:
    """(R, C) each cell's surface and (R, C, 3) its mean CIELAB: k-means of the cells' mean colour and position, so a
    surface is a patch of one colour, about SURFACE_CELLS cells. Seeded, so a rerun gives the same surfaces."""
    h, w = cell
    R, C = image_rgb.shape[0] // h, image_rgb.shape[1] // w
    lab = cv2.cvtColor(image_rgb.astype(np.float32), cv2.COLOR_RGB2Lab).reshape(R, h, C, w, 3).mean(axis=(1, 3))
    r, c = np.indices((R, C))
    feats = np.concatenate([lab, 2.0 * np.stack([r, c], -1) * 100 / C], -1).reshape(-1, 5).astype(np.float32)
    cv2.setRNGSeed(1)
    _, km, _ = cv2.kmeans(feats, max(1, R * C // SURFACE_CELLS), None,
                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5, cv2.KMEANS_PP_CENTERS)
    return km.reshape(R, C), lab


def surface_binding(D: np.ndarray, image_rgb: np.ndarray, cell, near: float) -> np.ndarray:
    """(P, R, C) extra cost that binds the cells within near dE of their surface's mean colour to the one pair whose
    own cost over them is least: a face or a sky in one pair, where each cell's own nearest mixture would make a
    patchwork of two families. Cells further off (accents, a surface of several colours) stay free."""
    seg, lab = surfaces(image_rgb, cell)
    best = np.full(seg.shape, -1)
    for s in np.unique(seg):
        cells = seg == s
        cells &= np.linalg.norm(lab - lab[cells].mean(axis=0), axis=-1) < near
        if cells.any():
            best[cells] = D[:, cells].sum(axis=1).argmin()
    off = (best >= 0) & (np.arange(len(D))[:, None, None] != best)
    return np.where(off, 100 * np.abs(D).max(), 0).astype(D.dtype)


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
        X = c.image_lrgb.astype(np.float32) @ LRGB2OPP.T
        E, extra = METHODS[c.method].candidates(c, X)                      # (P, H, W, 3), group -> (P, R, C) or None
        P, H, W, _ = E.shape
        h, w = c.cell
        R, C = H // h, W // w
        E = E.reshape(P, R, h, C, w, 3).transpose(1, 3, 0, 2, 4, 5).reshape(R, C, P, h * w, 3)
        luma, chroma, _ = c.eye_kernels()
        kernels = dict(Luma=luma, Chroma=chroma)
        for g, channels in GROUPS.items():
            cpp = autocorrelation(kernels[g])
            K0 = block_kernel_matrix(cpp, 0, 0, c.cell)
            A = [np.ascontiguousarray(E[..., k]) for k in channels]         # each (R, C, P, 64)
            self.X[g] = X[..., channels].reshape(R, h, C, w, len(channels)).mean(axis=(1, 3))   # (R, C, nch) target block means
            self.N[g] = sum((a ** 2).sum(-1) for a in A).transpose(2, 0, 1)
            if extra is not None:
                self.N[g] = self.N[g] + extra[g]
            self.D[g] = sum(np.einsum('rcpx,xy,rcpy->prc', a, K0, a, optimize=True) for a in A)
            self.S[g] = {}
            for dr, dc in OFFSETS:
                K = block_kernel_matrix(cpp, dr, dc, c.cell)
                rs, cs = _ranges(dr, dc, R, C)
                ns = slice(rs.start + dr, rs.stop + dr), slice(cs.start + dc, cs.stop + dc)
                # (R', C', P, P) every pair of a block against every pair of its neighbour: batched matmuls, BLAS;
                # einsum's own loop took ~10x longer and most of a conversion without DBS
                self.S[g][(dr, dc)] = 2 * sum((a[rs, cs] @ K) @ a[ns].swapaxes(-1, -2) for a in A)

    def unary(self) -> np.ndarray:
        w, c = self.weights, self.converter
        noise = dict(Luma=c.luma_noise, Chroma=c.chroma_noise)
        return sum(w[g] * self.D[g] for g in GROUPS) + sum(w[g] * noise[g] * self.N[g] for g in GROUPS)

    def apply(self) -> np.ndarray:
        """The (R, C) pair labels of least energy."""
        w, c = self.weights, self.converter
        D = self.unary()
        if c.surface > 0:
            D = D + surface_binding(D, c.image_rgb, c.cell, c.surface)
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
