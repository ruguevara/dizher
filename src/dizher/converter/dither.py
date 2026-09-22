import numpy as np

from ..halftoning.error_distribution import stucki_duo
from ..halftoning.ordered import ordered_dither
from ..halftoning.noise import noise_dither
from ..halftoning.dbs import dbs_duo

def duo_levels(luma, paper, ink):
    """Fraction of ink in a linear-light mix of paper and ink that matches luma."""
    span = ink - paper
    return np.divide(luma - paper, span, out=np.zeros_like(luma), where=span != 0).clip(0, 1)

class Ditherer:
    """Halftones luma (linear luminance) to a bitmap where each pixel is paper (0) or ink (1).
    paper and ink are per-pixel linear luminances of the attribute colours.
    eye: eye-model parameters (scale = luminance blur in pixels, alpha = kernel shape, see eye.py)."""
    label = 'You can not get label of an abstract base Ditherer class'

    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, luma: np.ndarray, paper: np.ndarray, ink: np.ndarray, **eye) -> np.ndarray:
        raise NotImplementedError()

class ThresholdDitherer(Ditherer):
    def __call__(self, luma, paper, ink, **eye):
        return self.threshold(duo_levels(luma, paper, ink))

    def threshold(self, levels: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class DBS(Ditherer):
    label = 'DBS'

    def __call__(self, luma, paper, ink, scale=1.4, alpha=2.0, structure=0.06, **eye):
        return dbs_duo(luma, paper, ink, init=noise_dither(duo_levels(luma, paper, ink)), scale=scale, alpha=alpha, structure=structure)

class EDStucki(Ditherer):
    label = 'ED Stucki'

    def __call__(self, luma, paper, ink, **eye):
        return stucki_duo(luma, paper, ink)

class OrderedBayer(ThresholdDitherer):
    label = 'Ordered Bayer'

    def threshold(self, levels):
        return ordered_dither(levels, 2) > 0

class Stohastic(ThresholdDitherer):
    label = 'Stohastic'

    def threshold(self, levels):
        return noise_dither(levels)
