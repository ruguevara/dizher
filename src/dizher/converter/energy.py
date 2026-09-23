"""Colour-pair selection as one eye-model energy over the whole composite image.

E(labels) = sum_ch w_ch || h_ch * (Y_ch - X_ch) ||^2   in a linear opponent space (S-CIELAB's
O1 luminance, O2 red-green, O3 blue-yellow), Y the realised composite, X the target, h_ch the
eye kernel of the channel's group (Luma: O1, Chroma: O2 and O3). Because the composite is a sum
of per-block candidates, E splits exactly into a per-block term D[p, b] and pairwise terms
S[p, q, b, b'] = 2 e_p[b]^T K e_q[b'] with K = h (*) h the kernel autocorrelation. The pairwise
term is the visible seam a pair change paints, in the same units as D, with no extra knob.

Blurred mean-square error is blind to texture: a lone block dithered with green dots among blocks
dithered with yellow dots is plainly a cell even when the mean colours agree, and where two pairs
tie the noise of the realisation picks one per block at random. So a coherence term charges a
pair change between 4-neighbours by how different the two pairs look (CIELUV distance of the
papers plus of the inks), scaled down where the original itself has an edge across that seam:
    coherence * SEAM_COST * sum_{b~b'} V[p_b, p_b'] * exp(-|x_b - x_b'|^2 / 2 EDGE_SIGMA^2)
This is the contrast-sensitive Potts prior of MRF segmentation. It is graded, so the bright
variant of the same colours is nearly free, and a change along a real edge costs nothing.

The kernels are pure low-pass, so they call blue dots on yellow (the palette's largest chroma
contrast) invisible once blurred, and then prefer that pair for a salmon target on mean colour
alone; likewise black dots on white for a grey a palette could paint with two close greys. Real
sensitivity does not vanish at the pixel pitch, so each kernel gets a delta component:
h = g + noise * delta, whose extra energy is the unblurred error of the channel group, a per-block
term with no cross-block part (the cross term with g is dropped). The weights are luma_noise and
chroma_noise: dot contrast the eye still sees at the viewing distance.
ponytail: fine interactions truncated to the 8 neighbouring blocks (offset-2 blocks see < 10% of
the kernel peak for 8x8 cells; cells thinner than the kernel radius, like 8x1, would need farther
offsets and a higher-order chain in the optimiser). Labels by block coordinate descent on whole lines: each row, then each column, is
re-solved exactly by dynamic programming given the rest, so a run of blocks can switch together
(single-block ICM gets trapped by clusters that are wrong in the same way).
"""
from collections import OrderedDict

import numpy as np
import cv2

from .colors import convert_color
from .eye import eye_kernel

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
EDGE_SIGMA = 0.05   # step of the original's block means (weighted opponent units) that counts as a real edge
SEAM_COST = 0.1     # energy of one seam between totally different pairs at coherence 1; a block's own cost is ~0.2

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
        c = self.converter
        X = c.image_lrgb.astype(np.float32) @ LRGB2OPP.T
        Y = (c.realized ** c.gamma) @ LRGB2OPP.T
        E = Y - X                                                         # (P, H, W, 3)
        P, H, W, _ = E.shape
        h, w = c.cell
        R, C = H // h, W // w
        E = E.reshape(P, R, h, C, w, 3).transpose(1, 3, 0, 2, 4, 5).reshape(R, C, P, h * w, 3)
        kernels = dict(Luma=eye_kernel(c.luma_scale, c.luma_alpha), Chroma=eye_kernel(c.chroma_scale, c.chroma_alpha))
        for g, channels in GROUPS.items():
            cpp = autocorrelation(kernels[g])
            K0 = block_kernel_matrix(cpp, 0, 0, c.cell)
            A = [np.ascontiguousarray(E[..., k]) for k in channels]         # each (R, C, P, 64)
            self.X[g] = X[..., channels].reshape(R, h, C, w, len(channels)).mean(axis=(1, 3))   # (R, C, nch) target block means
            self.N[g] = sum((a ** 2).sum(-1) for a in A).transpose(2, 0, 1)
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

    def apply(self) -> None:
        w = self.weights
        D = self.unary()
        S = {off: sum(w[g] * self.S[g][off] for g in GROUPS) for off in OFFSETS}
        Lh, Lv = self.seam_smoothness()
        V = self.converter.pair_dissimilarity
        self.converter.set_best_conversion(optimise(D, S, V, Lh, Lv, self.converter.coherence))

    def seam_smoothness(self):
        """Weight 0..1 of every seam: 1 where the original is flat across it, ~0 across a real edge.
        Lh: (R-1, C) between (r, c) and (r+1, c); Lv: (R, C-1) between (r, c) and (r, c+1)."""
        w = self.weights
        X = np.concatenate([np.sqrt(w[g]) * self.X[g] for g in GROUPS], axis=-1)
        step = lambda d: np.exp(-(d ** 2).sum(-1) / (2 * EDGE_SIGMA ** 2)).astype(np.float32)
        return step(X[1:] - X[:-1]), step(X[:, 1:] - X[:, :-1])

    def energy(self, labels: np.ndarray) -> float:
        """Total energy of a labelling (for checks)."""
        w = self.weights
        P, R, C = next(iter(self.D.values())).shape
        ri, ci = np.indices((R, C))
        total = self.unary()[labels, ri, ci].sum()
        for (dr, dc) in OFFSETS:
            rs, cs = _ranges(dr, dc, R, C)
            me, nb = labels[rs, cs], labels[rs.start + dr:rs.stop + dr, cs.start + dc:cs.stop + dc]
            i, j = np.indices(me.shape)
            total += sum(w[g] * self.S[g][(dr, dc)][i, j, me, nb].sum() for g in GROUPS)
        Lh, Lv = self.seam_smoothness()
        V = self.converter.pair_dissimilarity
        total += self.converter.coherence * SEAM_COST * (
            (Lh * V[labels[1:], labels[:-1]]).sum() + (Lv * V[labels[:, 1:], labels[:, :-1]]).sum())
        return float(total)

def _transpose(D, S, Lh, Lv):
    """The same problem with rows and columns swapped."""
    St = {(1, 0): S[(0, 1)].transpose(1, 0, 2, 3),
          (0, 1): S[(1, 0)].transpose(1, 0, 2, 3),
          (1, 1): S[(1, 1)].transpose(1, 0, 2, 3),
          (1, -1): S[(1, -1)].transpose(1, 0, 3, 2)}   # the pair reverses its roles, so p and q swap
    return D.transpose(0, 2, 1), St, Lv.T, Lh.T

def _row_pass(D, S, V, Lh, Lv, coherence, labels):
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
    return changed

def optimise(D: np.ndarray, S: dict, V: np.ndarray, Lh: np.ndarray, Lv: np.ndarray, coherence: float,
             max_sweeps: int = 10) -> np.ndarray:
    """D: (P, R, C) unary; S[(dr, dc)]: (R', C', P, P) pairwise for block (r, c) with (r+dr, c+dc);
    V: (P, P) pair dissimilarity; Lh, Lv: seam smoothness weights. Returns (R, C) labels."""
    labels = D.argmin(0)
    Dt, St, Lht, Lvt = _transpose(D, S, Lh, Lv)
    for _ in range(max_sweeps):
        changed = _row_pass(D, S, V, Lh, Lv, coherence, labels)
        labels = np.ascontiguousarray(labels.T)
        changed += _row_pass(Dt, St, V, Lht, Lvt, coherence, labels)
        labels = np.ascontiguousarray(labels.T)
        if changed == 0:
            break
    return labels
