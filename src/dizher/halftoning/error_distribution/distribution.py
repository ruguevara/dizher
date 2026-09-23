import numpy as np

def ed_dither_duo(luma, paper, ink, positions, weights):
    """Binary error diffusion of scalar luminance or weighted opponent colour.
    luma, paper, ink : ndarray (rows, cols[, channels]). paper/ink vary per pixel
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

    if luma.ndim == 2:
        luma, paper, ink = [a[..., None] for a in (luma, paper, ink)]
    rows, cols, channels = luma.shape
    kernel = kernel[..., None]
    padded = np.zeros((rows + down, cols + 2 * side, channels))
    padded[:rows, side:side + cols] = luma
    spans = ink - paper
    lengths = (spans * spans).sum(-1)
    out = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            span = spans[i, j]
            length = lengths[i, j]
            level = np.clip(np.dot(padded[i, j + side] - paper[i, j], span) / length, 0, 1) if length else 0
            is_ink = level >= 0.5
            out[i, j] = is_ink
            d = (level - is_ink) * span  # diffuse only the representable part of the colour error
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
