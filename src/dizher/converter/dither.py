import numpy as np

from ..halftoning.error_distribution import stucki_duo
from ..halftoning.ordered import ordered_dither
from ..halftoning.noise import noise_dither
from ..halftoning.dbs import dbs_duo

def duo_levels(luma, paper, ink):
    """Project an (H, W) scalar or (H, W, channels) colour image onto its paper/ink segment."""
    span = ink - paper
    if luma.ndim == 3:
        numerator = ((luma - paper) * span).sum(-1)
        denominator = (span * span).sum(-1)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0).clip(0, 1)
    return np.divide(luma - paper, span, out=np.zeros_like(luma), where=span != 0).clip(0, 1)

class Ditherer:
    """Halftones scalar luminance or weighted opponent colour to a paper (0) / ink (1) bitmap.
    paper and ink are per-pixel values in the same space as the target.
    eye: eye-model parameters (scale = luminance blur in pixels, alpha = kernel shape, see eye.py)."""
    label = 'You can not get label of an abstract base Ditherer class'
    controls = ()   # names of the halftone op's params this method uses; the UI shows only those

    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, luma: np.ndarray, paper: np.ndarray, ink: np.ndarray, **eye) -> np.ndarray:
        raise NotImplementedError()

class ThresholdDitherer(Ditherer):
    def __call__(self, luma, paper, ink, origin=(0, 0), **eye):
        return self.threshold(duo_levels(luma, paper, ink), origin)

    def threshold(self, levels: np.ndarray, origin=(0, 0)) -> np.ndarray:
        raise NotImplementedError()

class DBS(Ditherer):
    label = 'DBS'
    controls = ('structure',)

    def __call__(self, luma, paper, ink, scale=1.4, alpha=2.0, structure=0.06, kernels=None, noise=0, on_step=None,
                 origin=(0, 0), **eye):
        return dbs_duo(luma, paper, ink, init=noise_dither(duo_levels(luma, paper, ink), origin=origin),
                       scale=scale, alpha=alpha, structure=structure, kernels=kernels, noise=noise, on_step=on_step)

class EDStucki(Ditherer):
    label = 'ED Stucki'

    def __call__(self, luma, paper, ink, **eye):
        return stucki_duo(luma, paper, ink)

class Ordered(ThresholdDitherer):
    label = 'Ordered'
    controls = ('matrix',)

    def __init__(self, matrix='Bayer 4x4', **kwargs):
        self.matrix = matrix

    def threshold(self, levels, origin=(0, 0)):
        return ordered_dither(levels, self.matrix, origin)

class Stohastic(ThresholdDitherer):
    label = 'Stohastic'

    def threshold(self, levels, origin=(0, 0)):
        return noise_dither(levels, origin=origin)
