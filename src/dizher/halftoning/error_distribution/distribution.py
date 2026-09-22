import numpy as np

def ed_dither_duo(luma, paper, ink, positions, weights):
    """Binary error diffusion where every pixel quantises to its own pair of luminances.
    luma, paper, ink : ndarray (rows, cols), linear luminance. paper/ink vary per pixel
    (per character block on the ZX). Error crosses block boundaries in absolute units.
    ponytail: raster ED has no clean form under a per-block range constraint. The running error of
    a wide-range block is a positive sawtooth and can saturate the first row of a narrow-range
    neighbour into a visible line; keeping error inside blocks instead makes every 8th row a 1-D
    diffusion with periodic stripes and worse eye-model error. DBS (halftoning/dbs.py) is the
    principled solution; this stays as the classic look.
    """
    positions = np.asarray(positions)
    down, side = positions[:, 0].max(), np.abs(positions[:, 1]).max()
    kernel = np.zeros((down + 1, 2 * side + 1))
    for (di, dj), w in zip(positions, weights):
        kernel[di, dj + side] = w
    kernel /= kernel.sum()

    rows, cols = luma.shape
    padded = np.zeros((rows + down, cols + 2 * side))
    padded[:rows, side:side + cols] = luma
    threshold = (paper + ink) / 2
    out = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            lo, hi = paper[i, j], ink[i, j]
            value = min(max(padded[i, j + side], lo), hi)  # luminance outside the range is unrepresentable
            is_ink = value >= threshold[i, j]
            out[i, j] = is_ink
            d = value - (hi if is_ink else lo)
            padded[i:i + down + 1, j:j + 2 * side + 1] += d * kernel
    return out

STUCKI = dict(
    positions=[(0, 1), (0, 2), (1, -2), (1, -1),
               (1, 0), (1, 1), (1, 2),
               (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)],
    weights=[8, 4,
             2, 4, 8, 4, 2,
             1, 2, 4, 2, 1],
)

def stucki_duo(luma, paper, ink):
    return ed_dither_duo(luma, paper, ink, **STUCKI)
