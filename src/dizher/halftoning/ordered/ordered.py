import numpy as np

from .matrices import MATRICES

def ordered_dither(levels: np.ndarray, matrix: str = 'Bayer 4x4', origin=(0, 0)) -> np.ndarray:
    """Ink where a pixel's level in 0..1 exceeds its threshold from the tiled matrix; origin (y, x) rolls the tile."""
    m, divisor = MATRICES[matrix]
    threshold = np.roll((m + 0.5) / divisor, origin, axis=(0, 1))
    h, w = levels.shape[-2:]
    oh, ow = m.shape
    tiled = np.tile(threshold, (-(-h // oh), -(-w // ow)))[:h, :w]   # ceil division: cover any screen size
    return levels > tiled
