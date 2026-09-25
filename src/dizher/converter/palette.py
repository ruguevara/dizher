# -*- coding: utf-8 -*-

import copy

import numpy as np


class Palette:
    """Colours a device can show, as (N, 3) gamma-encoded RGB bytes, and the indexes a cell may pair (enabled).
    SUBSETS names sets of indexes; subclasses add theirs, `subsets` puts All colours first."""
    SUBSETS = {}

    def __init__(self, colours):
        self.colours = np.asarray(colours, dtype=np.uint8).reshape(-1, 3)
        self.enabled = frozenset(range(len(self.colours)))

    @property
    def subsets(self) -> dict:
        return {'All colours': frozenset(range(len(self))), **self.SUBSETS}

    def with_colours(self, enabled) -> 'Palette':
        palette = copy.copy(self)
        palette.enabled = frozenset(enabled)
        return palette

    def with_subset(self, subset: str) -> 'Palette':
        return self.with_colours(self.subsets[subset])

    def subset_name(self, enabled) -> str:
        """The subset these indexes make, else Custom."""
        return next((name for name, s in self.subsets.items() if s == frozenset(enabled)), 'Custom')

    def as_ubyte(self):
        return self.colours

    def as_float(self):
        return (self.colours / 255).astype(np.float32)

    def __array__(self):
        return self.as_float()

    def __repr__(self):
        return repr(self.__array__())

    def __len__(self):
        return len(self.colours)

    def __getitem__(self, index):
        return self.colours[index]

    def iter_idxs_pairs(self):
        """(paper, ink) index pairs with paper the darker colour: the error diffuser and the coherence
        metric rely on that order, and palette index order need not follow luminance (C64 index 1 is white)."""
        lum = self.as_float() ** 2.2 @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        for i1 in range(len(self)):
            for i2 in range(i1, len(self)):
                if i1 in self.enabled and i2 in self.enabled:
                    yield (i1, i2) if lum[i1] <= lum[i2] else (i2, i1)

    def iter_color_pairs(self):
        pal = self.as_float()
        for i1, i2 in self.iter_idxs_pairs():
            yield pal[i1], pal[i2]

    def color_pairs(self):
        return np.array(list(self.iter_color_pairs()))
