import numpy as np

from ..halftoning.error_distribution import ed_dither_duo, ed_dither_levels
from ..halftoning.ordered import ordered_dither
from ..halftoning.noise import noise_dither

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

    def threshold(self, levels: np.ndarray) -> np.ndarray:
        """Levels in 0..1 with any leading axes to a bitmap: the pair candidates (Converter.set_image)."""
        raise NotImplementedError()

class ThresholdDitherer(Ditherer):
    def __init__(self, origin=(0, 0), **kwargs):
        self.origin = origin   # (y, x) roll of the tile: another start for pair selection and DBS, which settle in local optima

    def __call__(self, luma, paper, ink, **eye):
        return self.threshold(duo_levels(luma, paper, ink))

class ErrorDiffusion(Ditherer):
    label = 'Error diffusion'
    controls = ('kernel',)

    def __init__(self, kernel='Stucki', **kwargs):
        self.kernel = kernel

    def __call__(self, luma, paper, ink, **eye):
        return ed_dither_duo(luma, paper, ink, self.kernel)

    def threshold(self, levels):
        return ed_dither_levels(levels, self.kernel)

class Ordered(ThresholdDitherer):
    label = 'Ordered'
    controls = ('matrix',)

    def __init__(self, matrix='Bayer 4x4', **kwargs):
        super().__init__(**kwargs)
        self.matrix = matrix

    def threshold(self, levels):
        return ordered_dither(levels, self.matrix, self.origin)

class Stohastic(ThresholdDitherer):
    label = 'Stohastic'
    controls = ('noise_x', 'noise_y')

    def threshold(self, levels):
        return noise_dither(levels, origin=self.origin)
