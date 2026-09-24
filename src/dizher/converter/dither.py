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

MIX_HANDOVER = 0.3   # width of the hand-over between two levels, in units of the radius: the nearest one wins
MIX_PATTERN = 'Bayer 4x4'   # paints the snapped cells: a checkerboard at 1/2, a regular grid at 1/4 and 3/4

def snap_levels(t, cell, levels, strength, radius):
    """Mix levels t (H, W) pulled onto `levels` (0, 1/2, 1...), the tones a flat cell dithers cleanly: solid, a
    checkerboard, a regular grid. The pull is Converter-side twin of tone.snap in one dimension:
        t -> (t + K sum_i p_i l_i) / (1 + K),  K = k exp(-d^2 / 2 radius^2),  p = softmax(-d_i^2 / 2 (MIX_HANDOVER radius)^2)
    with d the distance to the nearest level and k = strength / (1 - strength), scaled by exp(-(var / radius^2)^2) of
    the cell's levels, so a cell that is not flat keeps its tones. Returns the levels and where they sit on one
    (within half a 4x4 threshold step), for the pattern seed."""
    levels = np.asarray(levels, dtype=np.float32)
    h, w = cell
    H, W = t.shape
    var = t.reshape(H // h, h, W // w, w).var(axis=(1, 3))
    var = np.repeat(np.repeat(var, h, axis=0), w, axis=1)
    s = min(strength, 0.99)
    k = s / (1 - s) * np.exp(-(var / radius ** 2) ** 2)   # steep: k reaches 99, a soft fade would still pull
    d2 = (t[..., None] - levels) ** 2
    nearest = d2.min(-1)
    p = np.exp(-(d2 - nearest[..., None]) / (2 * (MIX_HANDOVER * radius) ** 2))
    K = k * np.exp(-nearest / (2 * radius ** 2))
    snapped = ((t + K * (p @ levels) / p.sum(-1)) / (1 + K)).astype(np.float32)
    on = np.abs(snapped[..., None] - levels).min(-1) < 1 / 32
    return snapped, on

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
