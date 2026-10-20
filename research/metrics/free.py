"""R1, the free conversions: a picture dithered to a fixed palette with any palette colour in any pixel (no attribute
constraint), on the 256x192 grid of square pixels. See PLAN.md, "The free conversions".

    python research/metrics/free.py IMAGE [IMAGE...] --out DIR

writes every palette x method x space as PNG, the prepared source, and a contact sheet per palette.

Methods (METHODS): `nearest` (no dither, the control), `knoll-bayer` and `knoll-blue` (Knoll's pattern dithering with
an 8x8 Bayer map or a blue-noise mask), `fs` and `jjn` (Floyd-Steinberg and Jarvis-Judice-Ninke error diffusion,
serpentine). Spaces (SPACES), what is mixed and where the nearest colour is looked for:

- `linear`: the target, the errors and their sums in linear-light RGB; the nearest colour by Euclidean distance there.
- `srgb`: the same in gamma-encoded sRGB (0..1), as most dithering code does.
- `lab`: the target, the errors and their sums in linear-light RGB (as `linear`: the eye mixes dots in linear light);
  only the nearest-colour search maps the wanted value (target + error, after the clamp) to CIELAB (D65) and takes
  the palette colour at the least Euclidean distance there (CIE76).

The clamp (`clamp`, both Knoll and error diffusion): the wanted value, target plus the accumulated or diffused error,
is clamped per channel to the palette's range in the mixing space ('palette', the default; 'unit' clamps to 0..1 as
Yliluoma's listing does; 'none'). The error then is the clamped value minus the chosen colour, so a part of the
target the palette cannot reach in any channel (blue on CGA palette 0) is dropped instead of carried, and in error
diffusion the value at a pixel stays within the target plus the palette's extent: it cannot run away.

The sRGB transfer function is the exact IEC 61966-2-1 curve, not the app's 2.2 power.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from dizher.halftoning.noise import blue_noise
    from dizher.halftoning.ordered.matrices import bayer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from dizher.halftoning.noise import blue_noise
    from dizher.halftoning.ordered.matrices import bayer

try:   # error diffusion is sequential: numba compiles its loop (pip install numba); without it, plain Python, ~100x slower
    from numba import njit
    NUMBA = True
except ImportError:
    NUMBA = False

    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)


# --- palettes, sRGB bytes ---

def zx_palette(not_bright=205, bright=255):
    """The 15 ZX Spectrum colours as platforms/zxspectrum builds them (index bits 0 blue, 1 red, 2 green; 3 bright),
    the two blacks one colour: black, then non-bright blue..white, then bright blue..white. The user's reference
    emulator is Pulsar: its values replace these when known (pass its levels, or put its 15 colours in PALETTES)."""
    i = np.arange(16)
    rgb = np.stack([(i >> 1) & 1, (i >> 2) & 1, i & 1], axis=-1) * np.where(i >= 8, bright, not_bright)[:, None]
    return np.delete(rgb, 8, axis=0)


# Pepto's measured VIC-II palette (pepto.de/projects/colorvic), copied from platforms/c64, index = VIC-II colour number.
C64 = [(0x00, 0x00, 0x00), (0xFF, 0xFF, 0xFF), (0x68, 0x37, 0x2B), (0x70, 0xA4, 0xB2),
       (0x6F, 0x3D, 0x86), (0x58, 0x8D, 0x43), (0x35, 0x28, 0x79), (0xB8, 0xC7, 0x6F),
       (0x6F, 0x4F, 0x25), (0x43, 0x39, 0x00), (0x9A, 0x67, 0x59), (0x44, 0x44, 0x44),
       (0x6C, 0x6C, 0x6C), (0x9A, 0xD2, 0x84), (0x6C, 0x5E, 0xB5), (0x95, 0x95, 0x95)]

# TMS9918(A) colours 1..15 (0 is transparent): the set in MAME's tms9928a driver and in the TMS9918 Wikipedia article,
# derived from the datasheet's Y, R-Y, B-Y levels (not a photometric measurement of a screen).
MSX1 = [(0, 0, 0), (33, 200, 66), (94, 220, 120), (84, 85, 237), (125, 118, 252), (212, 82, 77), (66, 235, 245),
        (252, 85, 84), (255, 121, 120), (212, 193, 84), (230, 206, 128), (33, 176, 59), (201, 91, 186),
        (204, 204, 204), (255, 255, 255)]

# The default EGA palette (IBM EGA reference; also the CGA's 16 RGBI colours on a monitor that shows colour 6 as brown):
# levels 0/170 plus 85 for the intensity bit, colour 6 brown (170, 85, 0).
EGA = [(0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170), (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
       (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255), (255, 85, 85), (255, 85, 255), (255, 255, 85),
       (255, 255, 255)]

# CGA 320x200, black as the background colour, the three fixed colours as RGBI (EGA indices above).
CGA = {'cga0l': (0, 2, 4, 6),     # palette 0 low: green, red, brown
       'cga0h': (0, 10, 12, 14),  # palette 0 high: light green, light red, yellow
       'cga1l': (0, 3, 5, 7),     # palette 1 low: cyan, magenta, light grey
       'cga1h': (0, 11, 13, 15)}  # palette 1 high: light cyan, light magenta, white

PALETTES = {'zx': zx_palette(), 'c64': np.array(C64), 'msx1': np.array(MSX1), 'ega': np.array(EGA),
            **{k: np.array(EGA)[list(v)] for k, v in CGA.items()}}
PALETTES = {k: v.astype(np.uint8) for k, v in PALETTES.items()}


# --- colour ---

def srgb_to_linear(x):
    """sRGB 0..1 -> linear light (IEC 61966-2-1)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# linear sRGB -> XYZ, D65; rows scaled by the white point so that white is (1, 1, 1)
RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
RGB2XYZN = RGB2XYZ / RGB2XYZ.sum(axis=1, keepdims=True)
LUMINANCE = RGB2XYZ[1]


@njit(cache=True)
def _f(t):
    # CIELAB's f; linear below the knee, so it is defined for the negative values a clamp of 'none' lets through
    return np.cbrt(t) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116


@njit(cache=True)
def _lab1(r, g, b, m):
    fx, fy, fz = (_f(m[0, 0] * r + m[0, 1] * g + m[0, 2] * b), _f(m[1, 0] * r + m[1, 1] * g + m[1, 2] * b),
                  _f(m[2, 0] * r + m[2, 1] * g + m[2, 2] * b))
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def linear_to_lab(lin):
    """(..., 3) linear-light sRGB -> CIELAB, D65."""
    t = lin @ RGB2XYZN.T
    f = np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


SPACES = ('linear', 'srgb', 'lab')


def _spaces(src, pal, space):
    """sRGB 0..1 image and palette -> the target and palette in the mixing space, the palette in the distance space,
    and the map from the mixing space to the distance space (None: the same)."""
    if space == 'srgb':
        return src, pal, pal, None
    t, p = srgb_to_linear(src), srgb_to_linear(pal)
    if space == 'linear':
        return t, p, p, None
    if space == 'lab':
        return t, p, linear_to_lab(p), linear_to_lab
    raise ValueError(space)


def _nearest(x, pd):
    """(..., 3) values in the distance space -> the index of the nearest palette colour."""
    return ((x[..., None, :] - pd) ** 2).sum(-1).argmin(-1)


def _bounds(pm, clamp):
    if clamp == 'palette':
        return pm.min(0), pm.max(0)
    if clamp == 'unit':
        return np.zeros(3), np.ones(3)
    if clamp == 'none':
        return np.full(3, -np.inf), np.full(3, np.inf)
    raise ValueError(clamp)


# --- methods ---


def threshold_index(name, shape, n):
    """The threshold map as a candidate index 0..n-1 per pixel: an 8x8 Bayer map, or the repo's 512x512 blue-noise
    mask (halftoning/noise, LDR_LLL1_0.png, 256 equally frequent levels), tiled from the top left."""
    if name == 'bayer':
        m, levels = bayer(8), 64
    elif name == 'blue':
        m, levels = np.rint(blue_noise * 255).astype(int), 256
    else:
        raise ValueError(name)
    reps = (-(-shape[0] // m.shape[0]), -(-shape[1] // m.shape[1]))
    return np.tile(m, reps)[:shape[0], :shape[1]] * n // levels


def knoll(t, pm, pd, to_dist, lum, threshold='bayer', n=16, mult=1.0, clamp='palette'):
    """Knoll's pattern dithering (Adobe; US patent 6,606,166), as in Yliluoma's article: per pixel n candidates, each
    the palette colour nearest to target + mult * (the sum of target - the colours chosen so far), clamped; the list
    sorted by lum (ties by palette index); the threshold map's value picks the entry. With mult 1 the running mean
    of the list tracks the target as closely as the palette allows (a sigma-delta); Yliluoma's listing uses n 64 with
    the 8x8 map and mult 0.09, the Photoshop-like 4x4 map n 16."""
    lo, hi = _bounds(pm, clamp)
    err = np.zeros_like(t)
    cands = np.empty((n,) + t.shape[:2], dtype=np.intp)
    for c in range(n):
        want = np.clip(t + mult * err, lo, hi)
        k = _nearest(want if to_dist is None else to_dist(want), pd)
        cands[c] = k
        err += t - pm[k]
    order = np.lexsort((np.arange(len(pm)), lum))   # palette indexes by luminance
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    keys = np.sort(rank[cands], axis=0)
    pick = threshold_index(threshold, t.shape[:2], n)
    return order[np.take_along_axis(keys, pick[None], 0)[0]]


# (dy, dx, weight) of the error, dx for a left-to-right row; mirrored on right-to-left rows
KERNELS = {
    'fs': np.array([(0, 1, 7), (1, -1, 3), (1, 0, 5), (1, 1, 1)], dtype=np.float64) / [1, 1, 16],
    'jjn': np.array([(0, 1, 7), (0, 2, 5), (1, -2, 3), (1, -1, 5), (1, 0, 7), (1, 1, 5), (1, 2, 3),
                     (2, -2, 1), (2, -1, 3), (2, 0, 5), (2, 1, 3), (2, 2, 1)], dtype=np.float64) / [1, 1, 48],
}


@njit(cache=True)
def _diffuse(t, pm, pd, lo, hi, kernel, lab, m):
    h, w, _ = t.shape
    buf = t.copy()
    out = np.empty((h, w), dtype=np.int64)
    v = np.empty(3)
    for y in range(h):
        step = 1 if y % 2 == 0 else -1
        for i in range(w):
            x = i if step == 1 else w - 1 - i
            for c in range(3):
                v[c] = min(max(buf[y, x, c], lo[c]), hi[c])
            if lab:
                q0, q1, q2 = _lab1(v[0], v[1], v[2], m)
            else:
                q0, q1, q2 = v[0], v[1], v[2]
            best, bd = 0, np.inf
            for k in range(pd.shape[0]):
                d = (q0 - pd[k, 0]) ** 2 + (q1 - pd[k, 1]) ** 2 + (q2 - pd[k, 2]) ** 2
                if d < bd:
                    best, bd = k, d
            out[y, x] = best
            for j in range(kernel.shape[0]):
                yy, xx = y + int(kernel[j, 0]), x + step * int(kernel[j, 1])
                if yy < h and 0 <= xx < w:
                    for c in range(3):
                        buf[yy, xx, c] += (v[c] - pm[best, c]) * kernel[j, 2]
    return out


def diffuse(t, pm, pd, to_dist, kernel='fs', clamp='palette'):
    """Error diffusion, serpentine (even rows left to right): the wanted value is the target plus the diffused error,
    clamped (module docstring); the nearest colour to it; its error (wanted - chosen) spread by the kernel, the part
    falling outside the image dropped."""
    lo, hi = _bounds(pm, clamp)
    return _diffuse(np.ascontiguousarray(t, dtype=np.float64), pm, pd, lo, hi, KERNELS[kernel],
                    to_dist is linear_to_lab, RGB2XYZN)


# --- the conversion ---

METHODS = ('nearest', 'knoll-bayer', 'knoll-blue', 'fs', 'jjn')


def free(source, palette, method, space='linear', **params):
    """source (H, W, 3) sRGB uint8, palette a PALETTES name or (K, 3) sRGB uint8 -> (indices (H, W), rgb (H, W, 3)
    uint8). params: knoll n, mult, clamp; fs/jjn clamp."""
    pal = PALETTES[palette] if isinstance(palette, str) else np.asarray(palette, dtype=np.uint8)
    t, pm, pd, to_dist = _spaces(source / 255.0, pal / 255.0, space)
    if method == 'nearest':
        idx = _nearest(t if to_dist is None else to_dist(t), pd)
    elif method.startswith('knoll-'):
        lum = srgb_to_linear(pal / 255.0) @ LUMINANCE
        idx = knoll(t, pm, pd, to_dist, lum, threshold=method[6:], **params)
    elif method in KERNELS:
        idx = diffuse(t, pm, pd, to_dist, kernel=method, **params)
    else:
        raise ValueError(method)
    return idx, pal[idx]


def prepare(path, size=(192, 256)):
    """An image file -> (H, W, 3) sRGB uint8: the centre crop to the size's aspect, then area resampling in linear
    light."""
    img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    h, w = img.shape[:2]
    ch, cw = min(h, round(w * size[0] / size[1])), min(w, round(h * size[1] / size[0]))
    y, x = (h - ch) // 2, (w - cw) // 2
    lin = srgb_to_linear(img[y:y + ch, x:x + cw] / 255.0).astype(np.float32)
    lin = cv2.resize(lin, size[::-1], interpolation=cv2.INTER_AREA)
    return np.rint(linear_to_srgb(lin) * 255).astype(np.uint8)


# --- the command line ---

def _sheet(tiles, cols, label_h=16):
    """[(rgb or None, label)] -> a contact sheet, labels under the tiles."""
    th, tw = next(t for t, _ in tiles if t is not None).shape[:2]
    rows = -(-len(tiles) // cols)
    sheet = Image.new('RGB', (cols * (tw + 4) + 4, rows * (th + label_h + 4) + 4), (32, 32, 32))
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=12)
    for i, (rgb, label) in enumerate(tiles):
        x, y = 4 + i % cols * (tw + 4), 4 + i // cols * (th + label_h + 4)
        if rgb is not None:
            sheet.paste(Image.fromarray(rgb), (x, y))
        draw.text((x, y + th + 1), label, fill=(230, 230, 230), font=font)
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('images', nargs='+')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--palettes', nargs='+', default=list(PALETTES), choices=list(PALETTES))
    ap.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS)
    ap.add_argument('--spaces', nargs='+', default=list(SPACES), choices=SPACES)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    times = {}
    for path in args.images:
        stem = Path(path).stem
        src = prepare(path)
        Image.fromarray(src).save(args.out / f'{stem}_source.png')
        for pal in args.palettes:
            swatch = np.repeat(np.repeat(PALETTES[pal][None], src.shape[0], 0), -(-src.shape[1] // len(PALETTES[pal])),
                               1)[:, :src.shape[1]]
            tiles = [(src, 'source'), (swatch, f'{pal}, {len(PALETTES[pal])} colours')]
            tiles += [(None, '')] * (len(args.spaces) - 2)
            for method in args.methods:
                for space in args.spaces:
                    t0 = time.perf_counter()
                    _, rgb = free(src, pal, method, space)
                    times.setdefault(method, []).append(time.perf_counter() - t0)
                    Image.fromarray(rgb).save(args.out / f'{stem}_{pal}_{method}_{space}.png')
                    tiles.append((rgb, f'{method}, {space}'))
            _sheet(tiles, max(len(args.spaces), 2)).save(args.out / f'{stem}_{pal}_sheet.png')
            print(f'{stem} {pal}', flush=True)
    for method, ts in times.items():
        print(f'{method}: median {np.median(ts):.3f} s, max {max(ts):.3f} s over {len(ts)} '
              f'(the first fs/jjn call includes numba compilation unless cached)' if method in KERNELS else
              f'{method}: median {np.median(ts):.3f} s, max {max(ts):.3f} s over {len(ts)}')
    print('numba:', NUMBA)


if __name__ == '__main__':
    main()
