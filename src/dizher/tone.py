"""Tone and colour adjustments of the Tune column (ops.py) with the usual semantics: exposure and white balance
as in Lightroom's Basic panel, Levels and Curves as in Photoshop, with eyedroppers that pull a sampled colour onto a
palette colour. Float sRGB-encoded RGB in 0..1 in and out; an adjustment at its neutral values returns its input
untouched."""
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
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


def levels(rgb: np.ndarray, in_black: float = 0, in_white: float = 255, gamma: float = 1.0,
           out_black: float = 0, out_white: float = 255) -> np.ndarray:
    """Photoshop Levels over all channels: input black and white points in 0..255, midtone gamma (> 1
    brightens), output range (out_black > out_white inverts)."""
    if (in_black, in_white, gamma, out_black, out_white) == NEUTRAL:
        return rgb
    if in_black >= in_white:
        raise ValueError('input black must be below input white')
    t = np.clip((rgb * 255 - in_black) / (in_white - in_black), 0, 1) ** (1 / gamma)
    return ((out_black + t * (out_white - out_black)) / 255).astype(np.float32)


# ----- Levels per channel, Curves, and their eyedroppers ----------------------------------------------------------
# One tone node (ops.levels), Levels or Curves. Both apply the composite (RGB) first, then each channel's own on the
# result, so a channel's handles and points are in the composite's output. Channel values are floats in 0..255, so an
# eyedropper lands exactly; a curve is flat (x0, y0, x1, y1, ...) in 0..255, x rising, flat beyond its end points.

NEUTRAL = (0, 255, 1.0, 0, 255)            # levels (in_black, in_white, gamma, out_black, out_white)
IDENTITY = (0.0, 0.0, 255.0, 255.0)        # a curve's points
GAMMA_RANGE = (0.1, 9.99)
MIN_GAP = 2        # in_white - in_black, and the least x step between curve points at the ends
MERGE = 4.0        # levels: a grey point this near another's input in a channel replaces it there
MAX_POINTS = 16    # of a curve converted from levels, as in Photoshop
GREY_CHROMA = 10.0  # CIELAB chroma under which a palette colour counts as grey: measured palettes are not neutral
ROLES = ('black', 'grey', 'white')


def channel_levels(rgb: np.ndarray, channels) -> np.ndarray:
    """Levels per channel: channels are three (in_black, in_white, gamma, out_black, out_white)."""
    if all(tuple(c) == NEUTRAL for c in channels):
        return rgb
    return np.stack([levels(rgb[..., i], *c) for i, c in enumerate(channels)], axis=-1).astype(np.float32)


def all_levels(rgb: np.ndarray, composite, channels) -> np.ndarray:
    """The composite levels, then each channel's."""
    return channel_levels(levels(rgb, *composite), channels)


def points(curve) -> np.ndarray:
    """A flat curve as (n, 2) points."""
    return np.asarray(curve, np.float64).reshape(-1, 2)


def flat(pts) -> tuple:
    return tuple(round(float(v), 2) for v in np.asarray(pts).ravel())


def curve_lut(curve) -> np.ndarray:
    """256 outputs in 0..1 of a curve at the inputs 0..255: monotone cubic (PCHIP: no overshoot, a local extremum
    only at a point) through its points, straight with two, flat beyond the end points."""
    p = points(curve)
    if len(p) < 2 or np.any(np.diff(p[:, 0]) <= 0):
        raise ValueError('curve points must rise in input')
    x = np.clip(np.arange(256, dtype=np.float64), p[0, 0], p[-1, 0])
    y = PchipInterpolator(p[:, 0], p[:, 1])(x) if len(p) > 2 else np.interp(x, p[:, 0], p[:, 1])
    return (np.clip(y, 0, 255) / 255).astype(np.float32)


def apply_lut(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """values in 0..1 through a 256-entry LUT, linear between entries."""
    return np.interp(values * 255, np.arange(256), lut).astype(np.float32)


def curves(rgb: np.ndarray, curves) -> np.ndarray:
    """Photoshop Curves: curves are (RGB, R, G, B) flat point tuples; the composite first, then each channel's."""
    if all(tuple(c) == IDENTITY for c in curves):
        return rgb
    out = apply_lut(rgb, curve_lut(curves[0])) if tuple(curves[0]) != IDENTITY else rgb
    return np.stack([apply_lut(out[..., i], curve_lut(c)) if tuple(c) != IDENTITY else out[..., i]
                     for i, c in enumerate(curves[1:])], axis=-1).astype(np.float32)


def level_values(x, in_black, in_white, gamma, out_black, out_white) -> np.ndarray:
    """One channel's levels at inputs x in 0..255, out in 0..255."""
    t = np.clip((np.asarray(x, np.float64) - in_black) / (in_white - in_black), 0, 1) ** (1 / gamma)
    return out_black + t * (out_white - out_black)


def levels_to_curve(in_black, in_white, gamma, out_black, out_white, tolerance: float = 0.25) -> tuple:
    """A curve through (in_black, out_black) and (in_white, out_white) with points added where it strays most from the
    levels, until it is within `tolerance` levels at every input or has MAX_POINTS."""
    pts = [(in_black, out_black), (in_white, out_white)]
    x = np.arange(np.ceil(in_black), np.floor(in_white) + 1)
    want = level_values(x, in_black, in_white, gamma, out_black, out_white)
    while len(pts) < MAX_POINTS:
        err = np.abs(curve_lut(flat(sorted(pts)))[x.astype(int)] * 255 - want)
        err[np.isin(x, [p[0] for p in pts])] = 0
        i = int(err.argmax())
        if err[i] <= tolerance:
            break
        pts.append((x[i], want[i]))
    return flat(sorted(pts))


def levels_to_curves(composite, channels) -> tuple:
    """The (RGB, R, G, B) curves that do what the levels do."""
    return tuple(levels_to_curve(*c) for c in (composite, *channels))


SAMPLE = 1   # px around the eyedropper's pixel it averages: 3x3


def sample(rgb: np.ndarray, y: int, x: int, radius: int = SAMPLE) -> np.ndarray:
    """The mean colour of the (2 radius + 1)² pixels around (y, x), cut at the edges."""
    h, w = rgb.shape[:2]
    return rgb[max(y - radius, 0):min(y + radius + 1, h), max(x - radius, 0):min(x + radius + 1, w)].reshape(-1, 3).mean(0)


def palette_roles(rgb: np.ndarray) -> tuple:
    """The (black, grey, white) indexes of a palette (N, 3) in 0..1: the darkest and the lightest (CIELAB L*) of its
    greys (chroma under GREY_CHROMA; all colours when it has none), and the grey between them nearest L* 50."""
    lab = rgb2lab(np.asarray(rgb, np.float32)[None])[0]
    L, chroma = lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])
    greys = np.flatnonzero(chroma < GREY_CHROMA)
    greys = greys if len(greys) else np.arange(len(rgb))
    black, white = int(greys[L[greys].argmin()]), int(greys[L[greys].argmax()])
    middle = [i for i in greys if L[i] != L[black] and L[i] != L[white]] or list(greys)
    return black, int(min(middle, key=lambda i: abs(L[i] - 50))), white


def levels_pick(composite, channels, role: str, sample, target) -> tuple:
    """The channels' levels after an eyedropper click: sample (the node's input) and target colours in 0..1. Black
    and white set each channel's input end to the sample (after the composite) and its output end to the target;
    grey solves each channel's gamma. What cannot be reached is clamped (all_levels shows how near it came)."""
    s = level_values(np.asarray(sample) * 255, *composite)
    t = np.asarray(target, np.float64) * 255
    out = []
    for c, (ib, iw, g, ob, ow) in enumerate(channels):
        if role == 'black':
            ib, ob = min(s[c], iw - MIN_GAP), t[c]
        elif role == 'white':
            iw, ow = max(s[c], ib + MIN_GAP), t[c]
        else:
            x = (s[c] - ib) / (iw - ib)
            u = (t[c] - ob) / (ow - ob) if ow != ob else np.nan
            if 0 < x < 1 and 0 < u < 1:
                g = np.log(x) / np.log(u)
            elif 0 < x < 1 and not np.isnan(u):
                g = GAMMA_RANGE[int(u >= 1)]
            g = min(max(g, GAMMA_RANGE[0]), GAMMA_RANGE[1])
        out.append((float(ib), float(iw), float(g), float(ob), float(ow)))   # unrounded: a high gamma lifts the least miss
    return tuple(out)


def levels_clear(channels, role: str) -> tuple:
    """The channels' levels without an eyedropper's setting: black and white put each channel's input and output end
    back, grey its gamma."""
    k = {'black': (0, 3), 'white': (1, 4), 'grey': (2,)}[role]
    return tuple(tuple(float(NEUTRAL[i]) if i in k else float(v) for i, v in enumerate(c)) for c in channels)


# A grey point group of Curves, one per grey eyedropper click: its (sample r, g, b; target r, g, b) in 0..255, the
# target's palette index, the input of its point in the R, G and B curves (-1 once that point is gone) and REPLACED
# bits: which channels' points it put in place of another's.
PICK_X = slice(7, 10)
PICK_REPLACED = 10


def _composite(curves, rgb01) -> np.ndarray:
    """rgb in 0..1 through the composite curve, in 0..255."""
    return apply_lut(np.asarray(rgb01, np.float32), curve_lut(curves[0])) * 255


def _forget(picks, c: int, xs) -> tuple:
    """The picks with their point in channel c (1..3) at an input in xs marked gone."""
    out = []
    for p in picks:
        p = list(p)
        if p[PICK_X][c - 1] in xs:
            p[PICK_X.start + c - 1] = -1.0
        out.append(tuple(p))
    return tuple(p for p in out if any(x >= 0 for x in p[PICK_X]))


def curves_end(curves, picks, role: str, sample, target) -> tuple:
    """(curves, picks) after the black or white eyedropper: each channel's first (last) point moved to the sample
    after the composite curve, output the target; points beyond it go."""
    s, t = _composite(curves, sample), np.asarray(target, np.float64) * 255
    new = [curves[0]]
    for c in (1, 2, 3):
        p = points(curves[c])
        if role == 'black':
            x = round(min(s[c - 1], p[-1, 0] - MIN_GAP), 2)
            pts, gone = np.vstack([[x, t[c - 1]], p[p[:, 0] > x]]), p[p[:, 0] <= x, 0]
        else:
            x = round(max(s[c - 1], p[0, 0] + MIN_GAP), 2)
            pts, gone = np.vstack([p[p[:, 0] < x], [x, t[c - 1]]]), p[p[:, 0] >= x, 0]
        picks = _forget(picks, c, list(gone))
        new.append(flat(pts))
    return tuple(new), picks


def curves_grey(curves, picks, sample, target, index: int) -> tuple:
    """(curves, picks) with a grey point group added: in each of R, G, B the point (sample after the composite
    curve -> target). A point of that channel within MERGE of its input is replaced (and the pick notes it)."""
    s, t = _composite(curves, sample), np.asarray(target, np.float64) * 255
    new, xs, replaced = [curves[0]], [], 0
    for c in (1, 2, 3):
        p = points(curves[c])
        x = round(float(s[c - 1]), 2)
        near = np.abs(p[:, 0] - x) < MERGE
        if near.any():
            j = int(np.abs(p[:, 0] - x).argmin())
            picks = _forget(picks, c, [p[j, 0]])
            p = np.delete(p, j, axis=0)
            replaced |= 1 << (c - 1)
        p = np.vstack([p, [x, t[c - 1]]])
        new.append(flat(p[p[:, 0].argsort()]))
        xs.append(x)
    pick = flat(np.concatenate([np.asarray(sample) * 255, t])) + (float(index), *xs, float(replaced))
    return tuple(new), picks + (pick,)


def curves_drop(curves, picks, k: int) -> tuple:
    """(curves, picks) without grey point group k and its points; a channel left with one point goes straight."""
    new = list(curves)
    for c, x in enumerate(picks[k][PICK_X], 1):
        if x < 0:
            continue
        p = points(curves[c])
        p = p[p[:, 0] != x]
        new[c] = flat(p) if len(p) >= 2 else IDENTITY
    return tuple(new), picks[:k] + picks[k + 1:]


def curves_clear(curves, picks, role: str) -> tuple:
    """(curves, picks) without the black or white eyedropper's ends: in each channel the lowest (highest) point no
    grey point group owns goes, and the end is back at 0, 0 (255, 255)."""
    end = (0.0, 0.0) if role == 'black' else (255.0, 255.0)
    new = [curves[0]]
    for c in (1, 2, 3):
        p = points(curves[c])
        owned = {pick[PICK_X][c - 1] for pick in picks}
        free = [j for j in range(len(p)) if p[j, 0] not in owned]
        if free:
            p = np.delete(p, free[0] if role == 'black' else free[-1], axis=0)
        if end[0] not in p[:, 0]:
            p = np.vstack([p, [end]])
        new.append(flat(p[p[:, 0].argsort()]))
    return tuple(new), picks


def curve_edit(curves, picks, c: int, old_x: float, new) -> tuple:
    """(curves, picks) after the point of curve c at input old_x moved to new (x, y), or was removed (new None): the
    grey point group it belonged to follows it, or loses it."""
    p = points(curves[c])
    p = p[p[:, 0] != old_x]
    if new is not None:
        p = np.vstack([p, [round(new[0], 2), round(new[1], 2)]])
        p = p[p[:, 0].argsort()]
    out = list(curves)
    out[c] = flat(p)
    if c == 0:
        return tuple(out), picks
    moved = []
    for q in picks:
        q = list(q)
        if q[PICK_X.start + c - 1] == old_x:
            q[PICK_X.start + c - 1] = -1.0 if new is None else round(new[0], 2)
        moved.append(tuple(q))
    return tuple(out), tuple(q for q in moved if any(x >= 0 for x in q[PICK_X]))


def pick_conflicts(curves, pick) -> int:
    """Bits of the channels where the pick's point breaks the curve's monotonicity: another point lies above it at a
    lower input or below it at a higher (the other way round in a falling curve)."""
    bits = 0
    for c, x in enumerate(pick[PICK_X], 1):
        p = points(curves[c])
        if x < 0 or x not in p[:, 0]:
            continue
        y = p[p[:, 0] == x][0, 1]
        rising = 1 if p[-1, 1] >= p[0, 1] else -1
        if np.any((p[:, 0] - x) * (p[:, 1] - y) * rising < 0):
            bits |= 1 << (c - 1)
    return bits


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
