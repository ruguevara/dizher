from skimage import img_as_float
import numpy as np

def ed_dither(image, n_levels, positions=None, weights=None):
    """Quantize an image, using error-diffusion dithering.
    Parameters
    ----------
    image : ndarray, shape (..., rows, cols)
        Input image or a stack of images. A stack is dithered in one pass over the
        pixels, with the error of all images propagated together as a vector.
    n_levels : int
        Number of quantization levels.
    positions : list of (i, j) offsets
        Position offset to which the quantization error is distributed.
        By default, implement Sierra's "Filter Lite".
    weights : list of ints
        Weights for propagated error.
        By default, implement Sierra's "Filter Lite".
    References
    ----------
    http://www.efg2.com/Lab/Library/ImageProcessing/DHALF.TXT
    """
    image = img_as_float(image, force_copy=True)
    levels = np.linspace(0, 1, n_levels)
    bins = levels[1:] / 2 + levels[:-1] / 2

    if positions is None or weights is None:
        positions = [(0, 1), (1, -1), (1, 0)]
        weights = [2, 1, 1]

    # error kernel as a dense window: rows 0..down, cols -side..side, centred on the current pixel
    positions = np.asarray(positions)
    down, side = positions[:, 0].max(), np.abs(positions[:, 1]).max()
    kernel = np.zeros((down + 1, 2 * side + 1))
    for (di, dj), w in zip(positions, weights):
        kernel[di, dj + side] = w
    kernel /= kernel.sum()

    rows, cols = image.shape[-2:]
    lead = image.shape[:-2]
    padded = np.zeros(lead + (rows + down, cols + 2 * side))
    padded[..., :rows, side:side + cols] = image
    out = np.zeros(lead + (rows, cols))
    # ponytail: python loop over pixels, vectorised over the leading (stack) axis; ~0.5 s for 72x192x256
    for i in range(rows):
        for j in range(cols):
            value = padded[..., i, j + side]
            quantized = levels[np.searchsorted(bins, value, side='right')]
            out[..., i, j] = quantized
            padded[..., i:i + down + 1, j:j + 2 * side + 1] += (value - quantized)[..., np.newaxis, np.newaxis] * kernel
    return out

def floyd_steinberg(image, n_levels):
    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    weights = [7,
               3, 5, 1]
    return ed_dither(image, n_levels, offsets, weights)

def stucki(image, n_levels):
    offsets = [(0, 1), (0, 2), (1, -2), (1, -1),
               (1, 0), (1, 1), (1, 2),
               (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
    weights = [8, 4,
               2, 4, 8, 4, 2,
               1, 2, 4, 2, 1]
    return ed_dither(image, n_levels, offsets, weights)

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
