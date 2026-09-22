import numpy as np

def ed_dither_duo(luma, paper, ink, positions, weights):
    """Binary error diffusion where every pixel quantises to its own pair of luminances.
    luma, paper, ink : ndarray (rows, cols), linear luminance. paper/ink vary per pixel
    (per character block on the ZX), so error crosses block boundaries in absolute units.
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
    # ponytail: python loop, ~0.2 s per image; wavefront-vectorise (j + (side+1)*i == t) if it matters
    for i in range(rows):
        for j in range(cols):
            value = min(max(padded[i, j + side], 0.0), 1.0)  # ponytail: clamp to valid luminance so clipped regions do not pile up error
            is_ink = value >= threshold[i, j]
            out[i, j] = is_ink
            d = value - (ink[i, j] if is_ink else paper[i, j])
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
