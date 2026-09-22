"""Spatial coherence of colour-pair selection as a Markov random field.

Minimise  sum_b D[p_b, b] + lam * sum_{b~b'} S[p_b, p_b', b, b']  over the pair label p_b of every
block. D is the per-block cost of each pair. S is the seam cost: the perceptual step across the
shared boundary that the two candidates would create, minus the step the original has there, so a
pair change along a real edge is free and one in a smooth area costs the seam it paints.
ponytail: iterated conditional modes (checkerboard sweeps) from the unary argmin; swap for
alpha-expansion graph cuts if the local minimum is ever the limiting factor.
"""
import numpy as np

from .colors import convert_color

STRIP = 2  # pixels on each side of a boundary that define its colour

def _strip_colour(lrgb: np.ndarray) -> np.ndarray:
    """(..., 3) mean linear RGB of a strip -> (..., 3) perceptual vector: L/100, u/180, v/180."""
    shape = lrgb.shape
    rgb = lrgb.reshape(1, -1, 3).astype(np.float32).clip(0, 1) ** (1 / 2.2)
    luv = convert_color(rgb, 'RGB', 'LUV').reshape(shape) / np.array([100, 180, 180], dtype=np.float32)
    return luv

def seam_costs(expected: np.ndarray, image_lrgb: np.ndarray):
    """expected: (P, H, W, 3) expected linear RGB of every candidate; image_lrgb: (H, W, 3).
    Returns S_h (P, P, R-1, C) for boundaries between block (r, c) above and (r+1, c) below,
    and S_v (P, P, R, C-1) for (r, c) left and (r, c+1) right. Units: per-pixel MSE on a 0..255 scale."""
    P, H, W, _ = expected.shape
    R, C = H // 8, W // 8
    blocks = lambda a: a.reshape(a.shape[:-3] + (R, 8, C, 8, 3))
    e, o = blocks(expected), blocks(image_lrgb)
    # strips: mean over the STRIP rows (cols) nearest each boundary and the 8 pixels along it
    top, bottom = e[..., :STRIP, :, :, :].mean(axis=(-4, -2)), e[..., -STRIP:, :, :, :].mean(axis=(-4, -2))
    left, right = e[..., :STRIP, :].mean(axis=(-4, -2)), e[..., -STRIP:, :].mean(axis=(-4, -2))
    o_top, o_bottom = o[:, :STRIP].mean(axis=(1, 3)), o[:, -STRIP:].mean(axis=(1, 3))
    o_left, o_right = o[..., :STRIP, :].mean(axis=(1, 3)), o[..., -STRIP:, :].mean(axis=(1, 3))

    def costs(above, below, o_above, o_below):
        # step across the boundary that candidate p (one side) and q (other side) would paint, vs the original's
        a, b = _strip_colour(above), _strip_colour(below)
        o_step = _strip_colour(o_above) - _strip_colour(o_below)
        S = np.zeros((P, P) + o_step.shape[:-1], dtype=np.float32)
        for ch in range(3):
            S += (255 * (a[:, None, ..., ch] - b[None, :, ..., ch] - o_step[..., ch])) ** 2
        return S

    S_h = costs(bottom[:, :-1], top[:, 1:], o_bottom[:-1], o_top[1:])          # block r's bottom strip vs r+1's top
    S_v = costs(right[:, :, :-1], left[:, :, 1:], o_right[:, :-1], o_left[:, 1:])
    return S_h, S_v

def icm_labels(D: np.ndarray, S_h: np.ndarray, S_v: np.ndarray, lam: float, max_sweeps: int = 20) -> np.ndarray:
    """D: (P, R, C) unary costs. Returns (R, C) labels."""
    labels = D.argmin(0)
    if lam <= 0:
        return labels
    P, R, C = D.shape
    rows, cols = np.indices((R, C))
    parity = (rows + cols) % 2
    r_h, c_h = np.indices((R - 1, C))
    r_v, c_v = np.indices((R, C - 1))
    for _ in range(max_sweeps):
        changed = 0
        for par in (0, 1):
            cost = D.copy()
            cost[:, :-1, :] += lam * S_h[:, labels[1:, :], r_h, c_h]      # me above, neighbour below
            cost[:, 1:, :] += lam * S_h[labels[:-1, :], :, r_h, c_h].transpose(2, 0, 1)  # neighbour above, me below
            cost[:, :, :-1] += lam * S_v[:, labels[:, 1:], r_v, c_v]
            cost[:, :, 1:] += lam * S_v[labels[:, :-1], :, r_v, c_v].transpose(2, 0, 1)
            best = cost.argmin(0)
            update = (parity == par) & (best != labels)
            labels[update] = best[update]
            changed += int(update.sum())
        if changed == 0:
            break
    return labels
