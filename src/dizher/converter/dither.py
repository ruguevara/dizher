# -*- coding: utf-8 -*-

import numpy as np
from abc import abstractmethod

from ..halftoning.error_distribution import stucki
from ..halftoning.ordered import ordered_dither
from ..halftoning.noise import noise_dither


class Ditherer:
    label = 'You can not get label of an abstract base Ditherer class'

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, color_levels: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class EDStucki(Ditherer):
    label = 'ED Stucki'

    def __call__(self, color_levels: np.ndarray) -> np.ndarray:
        return stucki(color_levels, 2)


class OrderedBayer(Ditherer):
    label = 'Ordered Bayer'

    def __call__(self, color_levels: np.ndarray) -> np.ndarray:
        return ordered_dither(color_levels, 2)


class Stohastic(Ditherer):
    label = 'Stohastic'

    def __call__(self, color_levels: np.ndarray) -> np.ndarray:
        return noise_dither(color_levels)
