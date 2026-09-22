# -*- coding: utf-8 -*-

from abc import abstractmethod

import numpy as np


class BasePalette:
    @abstractmethod
    def as_ubyte(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    def as_float(self):
        return (self.as_ubyte() / 255).astype(np.float32)

    def __array__(self):
        return self.as_float()

    def __repr__(self):
        return repr(self.__array__())

    def iter_idxs_pairs(self):
        length = len(self)
        for i1 in range(length):
            for i2 in range(i1, length):
                yield i1, i2

    def iter_color_pairs(self):
        pal = self.as_float()
        for i1, i2 in self.iter_idxs_pairs():
            yield pal[i1], pal[i2]

    def color_pairs(self):
        return np.array(list(self.iter_color_pairs()))


class Palette(BasePalette):
    def __init__(self, palette):
        self.palette = palette
        self.pairs_len = len(self.color_pairs())

    def as_ubyte(self):
        return self.palette

    def __len__(self):
        return len(self.palette)

    def __getitem__(self, index):
        return self.palette[index]


class ZXPalette(Palette):
    # Which attribute pairs the selection may use; indexes 0..7 not bright, 8..15 bright, 7/15 white, 0/8 black.
    SUBSETS = {
        'All colours': lambda i1, i2: True,
        'Bright only': lambda i1, i2: i1 >= 8,
        'Not bright only': lambda i1, i2: i1 < 8,
        'Grayscale': lambda i1, i2: {i1, i2} <= {0, 7, 8, 15},  # black, gray (not-bright white), white
        'Mono': lambda i1, i2: (i1, i2) in ((8, 8), (8, 15), (15, 15)),  # black and bright white
    }

    def __init__(self, not_bright_level=205, bright_level=255, subset='All colours'):
        self.subset = subset
        palette = np.empty((2, 8, 3), dtype=np.uint8)
        i = 0
        for g in range(2):
            for r in range(2):
                for b in range(2):
                    palette[0, i, 0] = r * not_bright_level
                    palette[0, i, 1] = g * not_bright_level
                    palette[0, i, 2] = b * not_bright_level
                    palette[1, i, 0] = r * bright_level
                    palette[1, i, 1] = g * bright_level
                    palette[1, i, 2] = b * bright_level
                    i += 1
        super().__init__(palette.reshape(-1, 3))

    def iter_idxs_pairs(self):
        allowed = self.SUBSETS[self.subset]
        for base in (0, 8):
            for i1 in range(base, base + 8):
                for i2 in range(i1, base + 8):
                    if allowed(i1, i2):
                        yield i1, i2

    def invert_bright_index(self, index):
        return (index + (self.pairs_len // 2)) % self.pairs_len
