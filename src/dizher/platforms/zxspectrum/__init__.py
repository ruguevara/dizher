"""ZX Spectrum: 15 colours (8 not bright, 8 bright, the two blacks coincide), one (paper, ink) pair
per 8x8 cell, both from the same brightness. The flash bit is never used."""
import numpy as np

from .. import Mode
from ...converter.palette import Palette
from .scr import to_scr


class ZXPalette(Palette):
    # Indexes 0..7 not bright, 8..15 bright, 7/15 white, 0/8 black.
    SUBSETS = {
        'All colours': lambda i1, i2: True,
        'Bright only': lambda i1, i2: i1 >= 8,
        'Not bright only': lambda i1, i2: i1 < 8,
        'Grayscale': lambda i1, i2: {i1, i2} <= {0, 7, 8, 15},  # black, gray (not-bright white), white
        'Mono': lambda i1, i2: (i1, i2) in ((8, 8), (8, 15), (15, 15)),  # black and bright white
    }

    def __init__(self, not_bright_level=205, bright_level=255, subset='All colours'):
        # index bits: 0 blue, 1 red, 2 green; 3 bright
        i = np.arange(16)
        rgb = np.stack([(i >> 1) & 1, (i >> 2) & 1, i & 1], axis=-1)
        level = np.where(i >= 8, bright_level, not_bright_level)[:, None]
        super().__init__(rgb * level, subset)

    def iter_idxs_pairs(self):
        """Both colours of a cell share the bright bit."""
        for i1, i2 in super().iter_idxs_pairs():
            if (i1 >= 8) == (i2 >= 8):
                yield i1, i2


STANDARD = Mode('ZX Spectrum standard', size=(192, 256), cell=(8, 8), palette=ZXPalette(),
                file_type=('ZX Spectrum screen', '*.scr'), encode=to_scr)

