"""256x192 rgb -> ZX Spectrum ink/paper/bright attrs and bitmap. Ported from dizher's `Converter`
(`converter/zxconverter.py`, `converter/metrics.py`, `converter/ssim.py`) and the `dither` package,
as pure vectorized functions over a leading frame axis N.

`fit_attrs` brute-forces, per 8x8 attribute block, the best of the 72 same-brightness ZX colour pairs
(36 non-bright pairs + 36 bright pairs, including flat i==i pairs) by a weighted per-pixel error metric
(Luma + Chroma + Smoothness), exactly as dizher's `MetricTuner.apply()` picks `argmin` of per-block MSE.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from mokit.image import _to_float, rgb_to_lab
from mokit.types import Timing
from mokit.zx.screen import ZXPalette, Screen, encode_scr_attrs

_RGB_LUMINANCE_709 = np.array([0.2126, 0.7152, 0.0722])

# Same 15-colour table, attr nibbles and quantiser as tools/png2scr.py (kept in step manually so this
# module has no import-time dependency on the tools/ tree, which does not ship with the mokit package).
_PNG2SCR_COLORS = (
    (0, 0, 0),
    (0, 0, 205), (0, 0, 255),
    (205, 0, 0), (255, 0, 0),
    (205, 0, 205), (255, 0, 255),
    (0, 205, 0), (0, 255, 0),
    (0, 205, 205), (0, 255, 255),
    (205, 205, 0), (255, 255, 0),
    (205, 205, 205), (255, 255, 255),
)
_PNG2SCR_ATTR_I = (0x00, 0x01, 0x01 | 0x40, 0x02, 0x02 | 0x40,
                   0x03, 0x03 | 0x40, 0x04, 0x04 | 0x40, 0x05, 0x05 | 0x40,
                   0x06, 0x06 | 0x40, 0x07, 0x07 | 0x40)
_PNG2SCR_ATTR_P = (0x00, 0x08, 0x08 | 0x40, 0x10, 0x10 | 0x40,
                   0x18, 0x18 | 0x40, 0x20, 0x20 | 0x40, 0x28, 0x28 | 0x40,
                   0x30, 0x30 | 0x40, 0x38, 0x38 | 0x40)


def _png2scr_nearest_level(level: int) -> int:
    threshold = 255 - (255 - 205) / 2
    if level > threshold:
        return 255
    elif level >= 205 / 2:
        return 205
    return 0


def _png2scr_nearest_color(color: tuple) -> tuple:
    return tuple(_png2scr_nearest_level(c) for c in color)


def _png2scr_transform_charblock(colors: list, bitmap: list) -> tuple:
    """Choose ZX ink/paper for one 8x8 block and minimize paper changes (tools/png2scr.py verbatim)."""
    if len(colors) == 1:
        return [colors[0], colors[0]], [0] * 8
    paper, ink = colors
    if _png2scr_nearest_color(paper) != _PNG2SCR_COLORS[0] and _png2scr_nearest_color(ink) == _PNG2SCR_COLORS[0]:
        return [ink, paper], [~row & 0xff for row in bitmap]
    return [paper, ink], bitmap


def _lrgb_to_luminance(lrgb: np.ndarray) -> np.ndarray:
    return np.sum(lrgb * _RGB_LUMINANCE_709, axis=-1)


def _palette_pairs(palette: ZXPalette) -> Tuple[np.ndarray, np.ndarray]:
    """72 (paper, ink) index pairs restricted to one brightness half each (dizher `ZXPalette.iter_idxs_pairs`),
    as float 0..1 colours; and the flat palette index pairs (P,2) for `to_screen`."""
    idx_pairs = [(i1, i2) for lo in (0, 8) for i1 in range(lo, lo + 8) for i2 in range(i1, lo + 8)]
    idx_pairs = np.array(idx_pairs, dtype=np.int64)
    colors = (palette.rgb[idx_pairs] / 255.0).astype(np.float32)  # (P, 2, 3)
    return idx_pairs, colors


def _box_blur3(x: np.ndarray) -> np.ndarray:
    """3x3 box blur (stand-in for dizher's cv2.GaussianBlur(ksize=3)), edges replicated, over the
    last-3 axes (..., H, W, C)."""
    padded = np.pad(x, [(0, 0)] * (x.ndim - 3) + [(1, 1), (1, 1), (0, 0)], mode="edge")
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            out += padded[..., dy:dy + x.shape[-3], dx:dx + x.shape[-2], :]
    return out / 9.0


@dataclass(frozen=True)
class AttrFit:
    levels: np.ndarray   # (N,192,256) float32, 0..1 blend between paper (0) and ink (1)
    ink: np.ndarray      # (N,24,32) uint8 palette colour index 0..7
    paper: np.ndarray    # (N,24,32) uint8 palette colour index 0..7
    bright: np.ndarray   # (N,24,32) bool


def fit_attrs(rgb: np.ndarray, weights: Tuple[float, float, float] = (1.0, 1.0, 0.5),
              palette: ZXPalette = None, gamma: float = 2.2) -> AttrFit:
    """Best duocolor (paper/ink/bright) per 8x8 block, dizher `fit_duocolors` + `MetricTuner`."""
    if rgb.shape[1:3] != (192, 256):
        raise ValueError(f"expected 256x192 frames, got {rgb.shape}")
    palette = palette or ZXPalette()
    n, h, w = rgb.shape[0], 192, 256
    rows, cols = h // 8, w // 8

    image_rgb = _to_float(rgb)                          # (N,H,W,3)
    image_lrgb = image_rgb.astype(np.float64) ** gamma
    image_luma = _lrgb_to_luminance(image_lrgb)          # (N,H,W)

    idx_pairs, colors = _palette_pairs(palette)          # (P,2) int, (P,2,3) float
    p = len(idx_pairs)

    c1, c2 = colors[:, 0].astype(np.float64), colors[:, 1].astype(np.float64)   # (P,3) each
    c1_lrgb, c2_lrgb = c1 ** gamma, c2 ** gamma
    c1_luma, c2_luma = _lrgb_to_luminance(c1_lrgb), _lrgb_to_luminance(c2_lrgb)  # (P,)
    swap = c2_luma < c1_luma
    c1_lrgb2 = np.where(swap[:, None], c2_lrgb, c1_lrgb)
    c2_lrgb2 = np.where(swap[:, None], c1_lrgb, c2_lrgb)
    c1_luma2 = np.minimum(c1_luma, c2_luma)
    c2_luma2 = np.maximum(c1_luma, c2_luma)
    lum_range = c2_luma2 - c1_luma2                      # (P,)
    color_vec = c2_lrgb2 - c1_lrgb2                       # (P,3)

    # per pair, per pixel level (fraction of c2 == the paper/ink whose luma is c1..c2's "amount")
    safe_range = np.where(lum_range > 0, lum_range, 1.0)
    amount = (image_luma[np.newaxis] - c1_luma2[:, None, None, None]) / safe_range[:, None, None, None]
    amount = np.clip(amount, 0, 1)
    amount = np.where((lum_range > 0)[:, None, None, None], amount, 0.0)
    # amount was computed against (c1=darker..c2=brighter); dizher flips it back to (original c1..c2) when swapped
    amount_orig = np.where(swap[:, None, None, None], 1 - amount, amount)

    recolored_lrgb = c1_lrgb2[:, None, None, None] + amount[..., None] * color_vec[:, None, None, None]
    recolored_rgb = np.clip(recolored_lrgb, 0, None) ** (1 / gamma)   # (P,N,H,W,3)
    recolored_luma = _lrgb_to_luminance(recolored_lrgb)               # (P,N,H,W)

    # ----- metrics (Luma, Chroma, Smoothness), dizher converter/metrics.py -----
    luma_metric = np.abs((recolored_luma ** (1 / gamma)) - (image_luma[np.newaxis] ** (1 / gamma)))  # (P,N,H,W)

    blurred_image = _box_blur3(image_rgb)
    image_lab = rgb_to_lab(blurred_image)                             # (N,H,W,3)
    recolored_blurred = np.stack([_box_blur3(recolored_rgb[i]) for i in range(p)])
    recolored_lab = np.stack([rgb_to_lab(recolored_blurred[i]) for i in range(p)])
    diff_a = (image_lab[np.newaxis, ..., 1] - recolored_lab[..., 1]) / 128.0
    diff_b = (image_lab[np.newaxis, ..., 2] - recolored_lab[..., 2]) / 128.0
    chroma_metric = np.sqrt(diff_a ** 2 + diff_b ** 2)                 # (P,N,H,W)

    color_pair_luma = _lrgb_to_luminance(colors.astype(np.float64) ** gamma) ** (1 / gamma)  # (P,2)
    luma_dist = np.abs(color_pair_luma[:, 1] - color_pair_luma[:, 0])   # (P,)
    smoothness_metric = np.broadcast_to(luma_dist[:, None, None, None], (p, n, h, w)).astype(np.float64)

    w_luma, w_chroma, w_smooth = weights
    integral = luma_metric * w_luma + chroma_metric * w_chroma + smoothness_metric * w_smooth  # (P,N,H,W)

    # reshape to charblocks and MSE per block per pair: (P,N,rows,cols,8,8)
    blocks = integral.reshape(p, n, rows, 8, cols, 8).transpose(0, 1, 2, 4, 3, 5)
    mse = ((blocks * 255) ** 2).sum(axis=(4, 5)) / 64.0    # (P,N,rows,cols)
    best = mse.argmin(axis=0)                              # (N,rows,cols)

    best_idx_pairs = idx_pairs[best]                        # (N,rows,cols,2) flat palette indices
    paper_flat, ink_flat = best_idx_pairs[..., 0], best_idx_pairs[..., 1]
    bright = paper_flat >= 8
    paper = (paper_flat % 8).astype(np.uint8)
    ink = (ink_flat % 8).astype(np.uint8)

    n_idx, rows_idx, cols_idx = np.meshgrid(np.arange(n), np.arange(rows), np.arange(cols), indexing="ij")
    block_amount = amount_orig.reshape(p, n, rows, 8, cols, 8).transpose(0, 1, 2, 4, 3, 5)  # (P,N,rows,cols,8,8)
    levels_blocks = block_amount[best, n_idx, rows_idx, cols_idx]     # (N,rows,cols,8,8)
    levels = levels_blocks.transpose(0, 1, 3, 2, 4).reshape(n, h, w).astype(np.float32)

    return AttrFit(levels=levels, ink=ink, paper=paper, bright=bright)


_BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]) / 16.0 + 1.0 / 32.0


def dither(levels: np.ndarray, kind: str = "bayer") -> np.ndarray:
    """levels: (N,H,W) float 0..1 -> (N,H,W) bool ('on' = ink)."""
    n, h, w = levels.shape
    if kind == "bayer":
        threshold = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
        return levels > threshold[np.newaxis]
    if kind == "blue_noise":
        noise = _blue_noise()
        tiled = np.tile(noise, (h // noise.shape[0] + 1, w // noise.shape[1] + 1))[:h, :w]
        return levels > tiled[np.newaxis]
    if kind == "stucki":
        return np.stack([_stucki_frame(levels[i]) for i in range(n)])
    raise ValueError(f"unknown dither kind: {kind!r}")


_BLUE_NOISE_CACHE = None


def _blue_noise() -> np.ndarray:
    global _BLUE_NOISE_CACHE
    if _BLUE_NOISE_CACHE is None:
        path = Path(__file__).parent / "data" / "blue_noise.png"
        image = Image.open(path).convert("L")
        # +0.5/256 centres each level strictly inside (0,1), so levels==0 or ==1 dither to all-off/all-on
        _BLUE_NOISE_CACHE = (np.asarray(image, dtype=np.float32) + 0.5) / 256.0
    return _BLUE_NOISE_CACHE


# ponytail: pure-Python loop, ~1 s/frame; numba or a vectorized serpentine if it matters
def _stucki_frame(level: np.ndarray) -> np.ndarray:
    offsets = [(0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
               (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
    weights = np.array([8, 4, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1], dtype=np.float64)
    weights = weights / weights.sum()
    rows, cols = level.shape
    buf = level.astype(np.float64).copy()
    out = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            value = buf[i, j]
            on = value >= 0.5
            out[i, j] = on
            d = value - (1.0 if on else 0.0)
            for (di, dj), wgt in zip(offsets, weights):
                ii, jj = i + di, j + dj
                if 0 <= ii < rows and 0 <= jj < cols:
                    buf[ii, jj] += d * wgt
    return out


def optimize_bright(fit: AttrFit, pixels: np.ndarray, rgb: np.ndarray) -> AttrFit:
    raise NotImplementedError("port dizher/converter/ssim.py")


def to_screen(pixels: np.ndarray, fit: AttrFit, timing: Timing) -> Screen:
    """pixels: (N,192,256) bool 'ink' mask; attrs byte = bright<<6 | paper<<3 | ink."""
    attrs = encode_scr_attrs(
        ink=fit.ink, paper=fit.paper, bright=fit.bright,
        flash=np.zeros_like(fit.bright),
    ).reshape(fit.bright.shape)
    return Screen(pixels=pixels, attrs=attrs, timing=timing)


def quantize_exact(rgb: np.ndarray, timing: Timing) -> Screen:
    """Pre-quantised pixel art: at most 2 exact colours per 8x8 block (else ValueError naming the
    block), `_png2scr_transform_charblock` picks paper/ink and minimises paper flips, `_png2scr_nearest_color`
    snaps each to the nearest ZX level. Same algorithm as `tools/png2scr.py` (left untouched)."""
    if rgb.shape[1:3] != (192, 256):
        raise ValueError(f"expected 256x192 frames, got {rgb.shape}")
    n = rgb.shape[0]
    c2i = {c: i for c, i in zip(_PNG2SCR_COLORS, _PNG2SCR_ATTR_I)}
    c2p = {c: i for c, i in zip(_PNG2SCR_COLORS, _PNG2SCR_ATTR_P)}

    out_pixels = np.zeros((n, 192, 256), dtype=bool)
    out_attrs = np.zeros((n, 24, 32), dtype=np.uint8)
    for f in range(n):
        for row in range(24):
            for col in range(32):
                block = rgb[f, row * 8:row * 8 + 8, col * 8:col * 8 + 8]
                flat = [tuple(px) for line in block.tolist() for px in line]
                colors = list(dict.fromkeys(flat))
                if len(colors) > 2:
                    raise ValueError(f"more than 2 colors in an attribute block in ({col * 8}, {row * 8})")
                bitmap = [
                    sum((color != colors[0]) << (7 - i) for i, color in enumerate(flat[j * 8:j * 8 + 8]))
                    for j in range(8)
                ]
                attr_colors, bitmap = _png2scr_transform_charblock(colors, bitmap)
                paper_n, ink_n = map(_png2scr_nearest_color, attr_colors)
                out_attrs[f, row, col] = c2p[paper_n] | c2i[ink_n]
                bits = np.unpackbits(np.array(bitmap, dtype=np.uint8)[:, None], axis=1).astype(bool)
                out_pixels[f, row * 8:row * 8 + 8, col * 8:col * 8 + 8] = bits
    return Screen(pixels=out_pixels, attrs=out_attrs, timing=timing)
