#!/usr/bin/env python3

from skimage import img_as_float
import numpy as np


def ed_dither(image, n_levels, positions=None, weights=None):
    """Quantize an image, using dithering.
    Parameters
    ----------
    image : ndarray
        Input image.
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
    if not isinstance(bins, np.ndarray):
        bins = np.array([bins])

    if positions is None or weights is None:
        positions = [(0, 1), (1, -1), (1, 0)]
        weights = [2, 1, 1]

    weights = weights / np.sum(weights)
    rows, cols = image.shape

    out = np.zeros_like(image, dtype=float)
    for i in range(rows):
        for j in range(cols):
            # Quantize
            value = image[i, j]
            Ti, = np.digitize([value], bins)
            out[i, j] = levels[Ti]

            # Propagate quantization noise
            d = (value - out[i, j])
            # TODO vectorize this
            for (ii, jj), w in zip(positions, weights):
                ii = i + ii
                jj = j + jj
                if ii < rows and jj < cols:
                    image[ii, jj] += d * w
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
