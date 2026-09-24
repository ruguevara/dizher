"""Tone and colour adjustments of the Tune column (ops.py) with the usual semantics: exposure and white balance
as in Lightroom's Basic panel, Levels as in Photoshop. Float sRGB-encoded RGB in 0..1 in and out; an
adjustment at its neutral values returns its input untouched."""
import cv2
import numpy as np
from scipy.special import expit, logit

from .converter.colors import lab2rgb, rgb2lab

GAMMA = 2.2             # the converter's encoding (converter.py): a stop here is a stop there
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
WB_STOPS = 0.5          # channel gain at temperature or tint ±100, in stops
CONTRAST_SLOPE = 8.0    # logistic steepness at contrast ±100: twice the mid-grey slope
VIBRANCE_CHROMA = 60.0  # CIELAB chroma beyond which vibrance leaves a colour alone
TEXTURE_SIGMA = 8.0     # px, the scale of texture: a cell or two at 256x192
TEXTURE_EDGE = 20.0     # L* step the texture base keeps as an edge rather than smoothing it, so edges get no halo
BASE_RADIUS = 24        # px, the box of the guided filter that splits off the base (the broad lighting) for local tone
BASE_EDGE = 15.0        # L* spread within that box which the base keeps as an edge rather than smoothing it
SHADOW_SHIFT = 0.15     # the most shadows or highlights at ±100 move the base, in units of L* 100; keeps it monotone


def light(rgb: np.ndarray, exposure: float = 0.0, temperature: float = 0.0, tint: float = 0.0) -> np.ndarray:
    """Exposure in stops and white balance as per-channel gains, both in linear light. Temperature + warms
    (red up, blue down), tint + goes magenta (green down); the gains keep luminance, so white balance alone
    never brightens. Highlights pushed past white clip, as on a camera."""
    if not (exposure or temperature or tint):
        return rgb
    gains = 2.0 ** (WB_STOPS / 100 * np.array([temperature, -tint, -temperature], dtype=np.float32))
    gains *= 2.0 ** exposure / (gains @ LUMA)
    return (np.clip(rgb ** GAMMA * gains, 0, 1) ** (1 / GAMMA)).astype(np.float32)


def levels(rgb: np.ndarray, in_black: int = 0, in_white: int = 255, gamma: float = 1.0,
           out_black: int = 0, out_white: int = 255) -> np.ndarray:
    """Photoshop Levels over all channels: input black and white points in 0..255, midtone gamma (> 1
    brightens), output range (out_black > out_white inverts)."""
    if (in_black, in_white, gamma, out_black, out_white) == (0, 255, 1.0, 0, 255):
        return rgb
    if in_black >= in_white:
        raise ValueError('input black must be below input white')
    t = np.clip((rgb * 255 - in_black) / (in_white - in_black), 0, 1) ** (1 / gamma)
    return ((out_black + t * (out_white - out_black)) / 255).astype(np.float32)


def histogram(rgb: np.ndarray) -> np.ndarray:
    """256 bins over the values of all three channels: the composite Levels shows and Auto clips."""
    return np.bincount(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8).ravel(), minlength=256)


def auto_levels(rgb: np.ndarray, clip: float = 0.001):
    """Photoshop's Enhance Monochromatic Contrast: the (in_black, in_white) that clip `clip` of the channel
    values at each end, the same for every channel so colours keep their balance."""
    lo, hi = np.quantile(rgb, [clip, 1 - clip]) * 255
    lo, hi = int(np.floor(lo)), int(np.ceil(hi))
    return (lo, hi) if hi - lo >= 2 else (0, 255)


def contrast(rgb: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """S-curve on L* around mid grey, chroma kept: a logistic normalised to fix black and white, its inverse
    for negative amounts (which flattens by the same measure). ±100 doubles or halves the mid-grey slope."""
    if not amount:
        return rgb
    k = CONTRAST_SLOPE * abs(amount) / 100
    lo, hi = expit(-k / 2), expit(k / 2)
    lab = rgb2lab(rgb.astype(np.float32))
    x = np.clip(lab[..., 0] / 100, 0, 1)
    y = (expit(k * (x - 0.5)) - lo) / (hi - lo) if amount > 0 else 0.5 + logit(lo + x * (hi - lo)) / k
    lab[..., 0] = y * 100
    return np.clip(lab2rgb(lab), 0, 1).astype(np.float32)


def color(rgb: np.ndarray, vibrance: float = 0.0, saturation: float = 0.0) -> np.ndarray:
    """CIELAB chroma: saturation scales all of it (-100 greys out, +100 doubles), vibrance mostly the muted
    colours, fading to nothing at VIBRANCE_CHROMA."""
    if not (vibrance or saturation):
        return rgb
    lab = rgb2lab(rgb.astype(np.float32))
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    scale = (1 + saturation / 100) * (1 + vibrance / 100 * np.clip(1 - chroma / VIBRANCE_CHROMA, 0, 1))
    lab[..., 1:] *= scale[..., None]
    return np.clip(lab2rgb(lab), 0, 1).astype(np.float32)



def _midtones(L):
    """1 in mid grey, 0 at black and white: detail gains fade there instead of clipping."""
    return 1 - (L / 50 - 1) ** 2


def _base(L):
    """The broad lighting of L*: self-guided filter (He et al. 2010), smooth within a BASE_RADIUS box but keeping
    steps bigger than BASE_EDGE, so what is taken off it has no halo."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * BASE_RADIUS + 1,) * 2)
    mean = box(L)
    var = box(L * L) - mean ** 2
    a = var / (var + BASE_EDGE ** 2)
    return box(a) * L + box(mean - a * mean)


def local_tone(rgb: np.ndarray, local_contrast: float = 0.0, shadows: float = 0.0, highlights: float = 0.0,
               clarity: float = 0.0) -> np.ndarray:
    """Local tone mapping on L*, chroma kept (Durand and Dorsey 2002): L* split into the base (_base) and the
    detail above it. Local contrast compresses the base towards its mean, taking off low frequencies (+100 leaves
    no broad lighting, -100 doubles it); shadows and highlights lift (+) or darken (-) the dark or bright base,
    black and white fixed; clarity scales the detail (+100 doubles it, midtones most)."""
    if not (local_contrast or shadows or highlights or clarity):
        return rgb
    lab = rgb2lab(rgb.astype(np.float32))
    L = lab[..., 0]
    base = _base(L)
    b = (base.mean() + (1 - local_contrast / 100) * (base - base.mean())) / 100
    bump = lambda x: np.clip(x, 0, 1) ** 2 * np.clip(1 - x, 0, 1) ** 4 * 729 / 16   # 0 at 0 and 1, peak 1 at 1/3
    b = b + SHADOW_SHIFT * (shadows / 100 * bump(b) + highlights / 100 * bump(1 - b))
    lab[..., 0] = np.clip(b * 100 + (1 + clarity / 100 * _midtones(L)) * (L - base), 0, 100)
    return np.clip(lab2rgb(lab), 0, 1).astype(np.float32)


def detail(rgb: np.ndarray, texture: float = 0.0, sharpen: float = 0.0, radius: float = 1.0) -> np.ndarray:
    """Both on L*, chroma kept. Texture: fine local contrast, L* pushed away from (or towards, when negative) an
    edge-preserving blur of TEXTURE_SIGMA px, midtones most, so texture gains contrast while strong edges get no
    halo; +100 doubles it. Sharpen: unsharp mask at the screen's size, amount in % of the detail under a Gaussian
    of `radius` px added back."""
    if not (texture or sharpen):
        return rgb
    lab = rgb2lab(rgb.astype(np.float32))
    L = lab[..., 0]
    if texture:
        L = L + texture / 100 * _midtones(L) * (L - cv2.bilateralFilter(L, -1, TEXTURE_EDGE, TEXTURE_SIGMA))
    if sharpen:
        L = L + sharpen / 100 * (L - cv2.GaussianBlur(L, (0, 0), radius))
    lab[..., 0] = np.clip(L, 0, 100)
    return np.clip(lab2rgb(lab), 0, 1).astype(np.float32)
