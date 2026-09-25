"""Pure image ops. Colour ops (`exposure`, `contrast`, `vibrance`, `curves`, `brightness`) work over uint8
(N,H,W,3) rgb stacks, in float internally, returning uint8. `crop_resize` works over uint8 (N,H,W,4) rgba
stacks (`Image.rgba`) and resamples with premultiplied alpha so transparent pixels do not fringe.

No opencv, no scikit-image: PIL for resizing, numpy for colour maths (sRGB<->Lab ported by hand).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _to_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def crop_resize(rgba: np.ndarray, width: int = 256, height: int = 192,
                 mode: str = "cover", anchor: Tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Resize each frame to (height, width). `rgba` is `(N,H,W,4) uint8`; resampling goes through PIL's
    premultiplied `RGBa` mode so a transparent region does not bleed a dark fringe into visible pixels.
    `cover` scales to fill and crops at `anchor` fractions (dizher `reshaper.py`'s case, generalised from
    a fixed centre crop to any anchor); `contain` letterboxes on opaque black (what an unlit part of a
    screen shows); `stretch` ignores aspect ratio."""
    out = np.zeros((rgba.shape[0], height, width, 4), dtype=np.uint8)
    out[..., 3] = 255
    for i, frame in enumerate(rgba):
        image = Image.fromarray(np.ascontiguousarray(frame), "RGBA").convert("RGBa")
        w, h = image.size
        if mode == "stretch":
            out[i] = np.asarray(image.resize((width, height), Image.LANCZOS).convert("RGBA"))
            continue
        factor = max(width / w, height / h) if mode == "cover" else min(width / w, height / h)
        new_w, new_h = max(1, round(w * factor)), max(1, round(h * factor))
        resized = np.asarray(image.resize((new_w, new_h), Image.LANCZOS).convert("RGBA"))
        if mode == "cover":
            x0 = round((new_w - width) * anchor[0])
            y0 = round((new_h - height) * anchor[1])
            out[i] = resized[y0:y0 + height, x0:x0 + width]
        else:  # contain
            x0 = round((width - new_w) * anchor[0])
            y0 = round((height - new_h) * anchor[1])
            out[i, y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


def exposure(rgb: np.ndarray, stops: float) -> np.ndarray:
    """dizher `ExposureFilter`: gamma = exp(-exposure) applied directly to (0..1) rgb."""
    x = _to_float(rgb)
    gamma = np.exp(-stops)
    return _to_uint8(x ** gamma)


# ----- sRGB <-> CIE Lab (D65), hand-rolled to avoid scikit-image -------------------------------------

_XYZ_FROM_RGB = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _f(t):
    delta = 6 / 29
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)


def _f_inv(t):
    delta = 6 / 29
    return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))


def rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(rgb01.astype(np.float64))
    xyz = linear @ _XYZ_FROM_RGB.T
    xyz_n = xyz / _D65_WHITE
    fx, fy, fz = _f(xyz_n[..., 0]), _f(xyz_n[..., 1]), _f(xyz_n[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xyz_n = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=-1)
    xyz = xyz_n * _D65_WHITE
    linear = xyz @ _RGB_FROM_XYZ.T
    return np.clip(_linear_to_srgb(linear), 0, 1)


def contrast(rgb: np.ndarray, amount: float) -> np.ndarray:
    """dizher `ContrastFilter`: sigmoid (amount>0) / inverse-sigmoid (amount<=0) on Lab L*, chroma untouched."""
    if amount == 0:
        return rgb.copy()
    lab = rgb_to_lab(_to_float(rgb))
    x = lab[..., 0] / 100
    if amount > 0:
        gain = np.exp(amount)
        a = (np.exp(gain / 2) + 1) / (np.exp(gain / 2) - 1)
        y = (1 / (1 + np.exp(-(x - 0.5) * gain)) - 0.5) * a + 0.5
    else:
        a = 1 - amount / 5
        y = (x - 0.5) / (a * a) + 0.5
    lab[..., 0] = np.clip(y, 0, 1) * 100
    return _to_uint8(lab_to_rgb(lab))


def _rgb_to_hsv(rgb01: np.ndarray) -> np.ndarray:
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    maxc, minc = np.max(rgb01, axis=-1), np.min(rgb01, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc == 0, 0, delta / np.where(maxc == 0, 1, maxc))
    rc = np.where(delta == 0, 0, (maxc - r) / np.where(delta == 0, 1, delta))
    gc = np.where(delta == 0, 0, (maxc - g) / np.where(delta == 0, 1, delta))
    bc = np.where(delta == 0, 0, (maxc - b) / np.where(delta == 0, 1, delta))
    h = np.select(
        [maxc == minc, maxc == r, maxc == g],
        [np.zeros_like(maxc), (bc - gc), 2.0 + (rc - bc)],
        default=4.0 + (gc - rc),
    )
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = (i.astype(np.int64) % 6)[..., np.newaxis]
    choices = np.stack([
        np.stack([v, t, p], axis=-1), np.stack([q, v, p], axis=-1), np.stack([p, v, t], axis=-1),
        np.stack([p, q, v], axis=-1), np.stack([t, p, v], axis=-1), np.stack([v, p, q], axis=-1),
    ])
    return np.take_along_axis(choices, i[np.newaxis].repeat(6, axis=0), axis=0)[0]


def vibrance(rgb: np.ndarray, vibrance: float, saturation: float) -> np.ndarray:
    """dizher `VibeSatFilter`: reshape V*S in HSV by vibrance (power curve, protects already-saturated
    pixels less) then a flat saturation gain, restoring the original Lab luma afterwards."""
    x = _to_float(rgb)
    hsv = _rgb_to_hsv(x)
    vs = np.clip(hsv[..., 1] * hsv[..., 2], 0, 1)
    vibrance_power = np.clip(np.exp(vibrance), 0.001, 100)
    vs = np.clip(vs ** (1 / vibrance_power), 0, 1)
    vs = vs * np.exp(saturation)
    hsv = hsv.copy()
    v_safe = np.where(hsv[..., 2] == 0, 1, hsv[..., 2])
    hsv[..., 1] = np.clip(vs / v_safe, 0, 1)
    result = _hsv_to_rgb(hsv)
    result_lab = rgb_to_lab(result)
    result_lab[..., 0] = rgb_to_lab(x)[..., 0]
    return _to_uint8(lab_to_rgb(result_lab))


def curves(rgb: np.ndarray, points: Tuple[Tuple[float, float], ...] = (), channel: str = "rgb") -> np.ndarray:
    """`np.interp` over a 256-entry LUT built from control points in 0..1 domain; identity for `()`."""
    if not points:
        return rgb.copy()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lut = _to_uint8(np.interp(np.linspace(0, 1, 256), xs, ys))
    out = rgb.copy()
    channels = {"r": [0], "g": [1], "b": [2], "rgb": [0, 1, 2]}[channel]
    out[..., channels] = lut[rgb[..., channels]]
    return out


def brightness(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Simple additive brightness in 0..1 (no dizher equivalent)."""
    return _to_uint8(_to_float(rgb) + amount)


def levels(rgb: np.ndarray, in_min: int = 0, in_max: int = 255, midpoint: float = 0.5,
           out_min: int = 0, out_max: int = 255) -> np.ndarray:
    """Clipped piecewise-linear levels; midpoint is relative to the input range.

    At midpoint 0 or 1 the corresponding half of the curve becomes a step.
    Output bounds may be reversed to invert the image.
    """
    if any(not isinstance(v, (int, np.integer)) or not 0 <= v <= 255
           for v in (in_min, in_max, out_min, out_max)):
        raise ValueError("level bounds must be integers in 0..255")
    if in_min >= in_max:
        raise ValueError("in_min must be less than in_max")
    if not 0 <= midpoint <= 1:
        raise ValueError("midpoint must be in 0..1")
    center = in_min + midpoint * (in_max - in_min)
    return curves(rgb, ((in_min / 255, out_min / 255),
                        (center / 255, (out_min + out_max) / 510),
                        (in_max / 255, out_max / 255)))
