"""Stage A0's synthetic sanity checks (PLAN.md, "Stage A"): each a pass/fail per candidate metric at one ppd.

Run: python research/metrics/a0.py [--ppd 12.9] [--factor 6] [--metrics cvvdp,eye] [--save]
Prints the table and writes research/metrics/a0-results.md (only for the full metric set at ppd 12.9); --save writes
the stimuli at x3 to research/metrics/data/a0/ (out of git).

All stimuli are 256x192 sRGB in 0..1, deterministic: the dithers threshold a per-pixel coverage t of the ink (the
share of ink dots, mixed with the paper in linear light by the exact sRGB EOTF) against the repo's 512x512 blue-noise
texture (halftoning/noise, LDR_LLL1), its 256x192 corner, ranked within each 8x8 cell (cell_ranked) so that a
flat coverage gives every cell the same number of dots. Dither colours are ZXPalette's (levels 205 and 255).
The check "R1 charged less than any Spectrum conversion" needs the free conversions and the converter; it is not here.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from dizher.halftoning.noise import blue_noise  # noqa: E402
from dizher.platforms.zxspectrum import ZXPalette  # noqa: E402

H, W = 192, 256
HERE = os.path.dirname(os.path.abspath(__file__))
ZX = ZXPalette().colours.astype(np.float32) / 255
COLOURS = ('black', 'blue', 'red', 'magenta', 'green', 'cyan', 'yellow', 'white')
YY, XX = np.mgrid[:H, :W]


def cell_ranked(mask: np.ndarray, n: int = 8) -> np.ndarray:
    """The mask's values replaced by their ranks within each n x n cell, (rank + 0.5) / n^2: the same dot order, and
    every cell gets exactly round(t n^2) dots of a flat coverage t, so cells of one level have one mean."""
    h, w = mask.shape
    blocks = mask.reshape(h // n, n, w // n, n).transpose(0, 2, 1, 3).reshape(h // n, w // n, n * n)
    ranks = (blocks.argsort(-1, kind='stable').argsort(-1, kind='stable') + 0.5) / (n * n)
    return ranks.reshape(h // n, w // n, n, n).transpose(0, 2, 1, 3).reshape(h, w).astype(np.float32)


BLUE = cell_ranked(blue_noise[:H, :W])                                        # dispersed dots
BLUE_2X2 = M.upscale(cell_ranked(blue_noise[:H // 2, :W // 2])[..., None], 2)[..., 0]   # 2x2 clusters, ranked in 16x16


def zx(name: str, bright: bool = True) -> np.ndarray:
    return ZX[COLOURS.index(name) + 8 * bright]


def flat(rgb) -> np.ndarray:
    return np.broadcast_to(np.asarray(rgb, np.float32), (H, W, 3)).copy()


def mix(paper, ink, t) -> np.ndarray:
    """The linear-light mixture of two sRGB colours at ink coverage t (scalar or (H, W)), in sRGB."""
    t = np.broadcast_to(np.asarray(t, np.float32), (H, W))[..., None]
    lp, li = M.srgb_to_linear(np.asarray(paper)), M.srgb_to_linear(np.asarray(ink))
    return M.linear_to_srgb(lp + t * (li - lp)).astype(np.float32)


def dither(paper, ink, t, mask=BLUE) -> np.ndarray:
    t = np.broadcast_to(np.asarray(t, np.float32), (H, W))
    return np.where((t > mask)[..., None], np.asarray(ink, np.float32), np.asarray(paper, np.float32))


def ramp(a, b) -> np.ndarray:
    """Left to right from sRGB colour a to b, interpolated in sRGB."""
    s = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    return np.broadcast_to(np.asarray(a, np.float32) + s * (np.asarray(b, np.float32) - np.asarray(a, np.float32)),
                           (H, W, 3)).copy()


def lab(srgb):
    return M._xyz2lab(M.srgb_to_linear(srgb.astype(np.float64)) @ M.SRGB2XYZ.T.astype(np.float64))


# The checks: each returns (stimuli, judge), stimuli {label: (ref, test)}, judge(scores) -> (passed, detail)

def check_tone():
    """Monotone in a tone error. References: a grey ramp, sRGB 0.1 to 0.8, and a colour ramp, sRGB (0.15, 0.25, 0.6)
    (a dull blue) to (0.8, 0.45, 0.15) (an orange), left to right; tests: the same ramp plus 0.01, 0.02, 0.04, 0.08,
    0.16 on every sRGB channel (no clipping; 0.01 is under 1 dE00, about the threshold, 0.02 about 1.5). Passes if
    the score never falls with the shift and rises strictly from 0.02 on, on both ramps (a metric may call a
    sub-threshold shift free)."""
    shifts = (0.01, 0.02, 0.04, 0.08, 0.16)
    ramps = dict(grey=ramp((0.1,) * 3, (0.8,) * 3), colour=ramp((0.15, 0.25, 0.6), (0.8, 0.45, 0.15)))
    stimuli = {f'{k} +{d}': (r, r + d) for k, r in ramps.items() for d in shifts}

    def judge(s):
        rows = {k: [s[f'{k} +{d}'] for d in shifts] for k in ramps}
        ok = all(np.all(np.diff(v) >= 0) and np.all(np.diff(v[1:]) > 0) for v in rows.values())
        return ok, '; '.join(f'{k} ' + ' < '.join(f'{x:.4g}' for x in v) for k, v in rows.items())
    return stimuli, judge


def check_shift():
    """A dither shifted by one pixel almost free. Reference: a blue-noise dither of a ramp of coverage 0 to 1, left to
    right, in bright black/white and in bright red/yellow. Tests: the same dither with the noise mask rolled one pixel
    right (the same tones, every dot moved), and, as the visible error, the dither of the coverage plus 0.05 (5% more
    ink, about 2-4 dE in mid tones) with the original mask. Passes if the shift costs under half the visible error,
    for both pairs."""
    t = np.linspace(0, 1, W, dtype=np.float32)[None].repeat(H, 0)
    stimuli = {}
    for pair in (('black', 'white'), ('red', 'yellow')):
        p, i = zx(pair[0]), zx(pair[1])
        ref = dither(p, i, t)
        stimuli[f'{"/".join(pair)} shift'] = (ref, dither(p, i, t, np.roll(BLUE, 1, axis=1)))
        stimuli[f'{"/".join(pair)} +5%'] = (ref, dither(p, i, np.clip(t + 0.05, 0, 1)))

    def judge(s):
        ratios = {k: s[f'{k} shift'] / s[f'{k} +5%'] for k in ('black/white', 'red/yellow')}
        return all(0 <= r < 0.5 for r in ratios.values()), '; '.join(f'{k} shift / +5% = {r:.3g}'
                                                                     for k, r in ratios.items())
    return stimuli, judge


def check_grey50():
    """Visibility on a flat 50% grey. Reference: flat sRGB grey of half white's luminance (0.735). Tests, all bright
    black and white at exactly or about 50% coverage: the 1-pixel checkerboard, the blue-noise dither, and a
    checkerboard of 2x2 blocks (2x2 clustered dots). Passes if checker < blue noise < 2x2 clusters."""
    ref = mix(zx('black'), zx('white'), 0.5)
    b, w = zx('black'), zx('white')
    tests = dict(checker=dither(b, w, 0.5, ((XX + YY) % 2).astype(np.float32)),
                 blue=dither(b, w, 0.5),
                 clusters=dither(b, w, 0.5, ((XX // 2 + YY // 2) % 2).astype(np.float32)))
    stimuli = {k: (ref, v) for k, v in tests.items()}

    def judge(s):
        return s['checker'] < s['blue'] < s['clusters'], ' < '.join(f'{k} {s[k]:.4g}' for k in tests)
    return stimuli, judge


def check_clash():
    """An attribute clash on a smooth surface costs more than the same colour error spread evenly. Two surfaces, each
    with two bright ZX pairs of dots of different hues, each pair's coverage set by the surface's luminance:
    - grey: flat grey of luminance 0.5; red/cyan and magenta/green at 50% (dot contrasts in luminance 0.57 and 0.43),
      both mixtures exactly the grey: the clash is in the dots' hues only;
    - salmon: the linear midpoint of red/white at t = g, (1, g, g), and magenta/yellow of the same luminance,
      (1, s, 1 - s), g from 0.42 at the top to 0.46 at the bottom: a smooth pink-orange gradient, the two mixtures
      off it in opposite hues by 2.7-5.8 dE00.
    Tests: the clash, 8x8 cells alternating the two pairs as a checkerboard of cells, and each pair alone over the
    whole surface (its colour error, if any, in one direction everywhere, as large as in its clash cells). Passes if
    the clash costs more than either pair alone, on both surfaces."""
    cells = ((XX // 8 + YY // 8) % 2 == 0)[..., None]
    lum = M.SRGB2XYZ[1]
    g = np.linspace(0.42, 0.46, H, dtype=np.float32)[:, None].repeat(W, 1)
    s = (lum[0] + (1 - lum[0]) * g - lum[0] - lum[2]) / (lum[1] - lum[2])
    salmon = (np.stack([np.ones_like(g), g, g], -1) + np.stack([np.ones_like(s), s, 1 - s], -1)) / 2
    surfaces = dict(grey=(flat(M.linear_to_srgb(np.full(3, 0.5))), (('red', 'cyan', 0.5), ('magenta', 'green', 0.5))),
                    salmon=(M.linear_to_srgb(salmon).astype(np.float32),
                            (('red', 'white', g), ('magenta', 'yellow', s))))
    stimuli = {}
    for k, (ref, pairs) in surfaces.items():
        a, b = (dither(zx(p), zx(i), t) for p, i, t in pairs)
        stimuli[f'{k} clash'] = (ref, np.where(cells, a, b))
        stimuli[f'{k} {pairs[0][0]}/{pairs[0][1]}'] = (ref, a)
        stimuli[f'{k} {pairs[1][0]}/{pairs[1][1]}'] = (ref, b)

    def judge(sc):
        ok, parts = True, []
        for k, (_, pairs) in surfaces.items():
            alone = [sc[f'{k} {p}/{i}'] for p, i, _ in pairs]
            ok &= sc[f'{k} clash'] > max(alone)
            parts.append(f'{k}: clash {sc[f"{k} clash"]:.4g} vs alone ' + ', '.join(f'{x:.4g}' for x in alone))
        return ok, '; '.join(parts)
    return stimuli, judge


def check_banding():
    """A false contour costs more than a dither of the same levels. References: a grey ramp, sRGB 0 to 1, and a yellow
    one, black to sRGB (1, 1, 0), left to right. The ZX levels on each: black, the non-bright and the bright colour
    (sRGB 0, 0.804, 1). Tests: the ramp posterised to the level nearest in CIELAB L* (three bands), and a blue-noise
    dither between the two levels around each pixel's linear luminance. Passes if the bands cost more, on both."""
    stimuli = {}
    for name in ('white', 'yellow'):
        top = zx(name)
        ref = ramp(zx('black'), top)
        levels = np.stack([zx('black'), zx(name, False), top])
        ly = M.srgb_to_linear(levels) @ M.SRGB2XYZ[1]
        y = M.srgb_to_linear(ref) @ M.SRGB2XYZ[1]
        nearest = np.abs(M.lightness(ref)[..., None] - M.lightness(levels)[None, None]).argmin(-1)
        lo = np.clip(np.searchsorted(ly, y, side='right') - 1, 0, 1)
        t = (y - ly[lo]) / (ly[lo + 1] - ly[lo])
        stimuli[f'{name} bands'] = (ref, levels[nearest])
        stimuli[f'{name} dither'] = (ref, np.where((t > BLUE)[..., None], levels[lo + 1], levels[lo]))

    def judge(s):
        ok = all(s[f'{n} bands'] > s[f'{n} dither'] for n in ('white', 'yellow'))
        return ok, '; '.join(f'{n}: bands {s[f"{n} bands"]:.4g} vs dither {s[f"{n} dither"]:.4g}'
                             for n in ('white', 'yellow'))
    return stimuli, judge


def check_dark():
    """Dots in a deep shadow. Reference: a dark grey ramp, L* 0 to 30 (luminance 0 to 0.064), left to right. Tests:
    bright white dots on black at coverage = the luminance, dispersed (blue noise), none (the shadow posterised to
    black), and clustered (blue noise at half resolution, 2x2 dots, ranked in 16x16 blocks: the same mean to 1%).
    Passes if the dispersed dots cost less than both."""
    L = np.linspace(0, 0.3, W)[None].repeat(H, 0)
    y = np.where(L > 0.08, ((L + 0.16) / 1.16) ** 3, L / 9.033).astype(np.float32)
    ref = flat(0) + M.linear_to_srgb(y)[..., None].astype(np.float32)
    b, w = zx('black'), zx('white')
    y2 = M.upscale(y.reshape(H // 2, 2, W // 2, 2).mean((1, 3))[..., None], 2)[..., 0]   # whole clusters
    tests = dict(dispersed=dither(b, w, y), empty=flat(b), clustered=dither(b, w, y2, BLUE_2X2))
    stimuli = {k: (ref, v) for k, v in tests.items()}

    def judge(s):
        return (s['dispersed'] < s['empty'] and s['dispersed'] < s['clustered'],
                ', '.join(f'{k} {s[k]:.4g}' for k in tests))
    return stimuli, judge


CHECKS = dict(tone=check_tone, shift=check_shift, grey50=check_grey50, clash=check_clash, banding=check_banding,
              dark=check_dark)
CHECK_TITLES = dict(tone='monotone in a tone error (grey, colour ramp)', shift='dither shifted 1 px < 0.5 x a 5% error',
                    grey50='50% grey: checker < blue noise < 2x2', clash='clash > same error spread evenly',
                    banding='banding > dither of its levels', dark='dark ramp: dispersed < empty, < clustered')


def clash_mean_error():
    """Max over cells of dE00 between a clash cell's linear-light mean and the surface's mean there (the stimulus's
    'within a few dE')."""
    out = {}
    stimuli, _ = check_clash()
    for k in ('grey', 'salmon'):
        ref, test = stimuli[f'{k} clash']
        cm = lambda im: M.linear_to_srgb(M.srgb_to_linear(im).reshape(H // 8, 8, W // 8, 8, 3).mean((1, 3)))
        from skimage.color import deltaE_ciede2000
        out[k] = float(deltaE_ciede2000(lab(cm(ref)), lab(cm(test))).max())
    return out


def run(names, ppd, factor):
    results, scores = {}, {}
    for check, make in CHECKS.items():
        stimuli, judge = make()
        for name in names:
            s = {k: M.score(name, ref, test, ppd, factor)[0] for k, (ref, test) in stimuli.items()}
            scores[check, name] = s
            results[check, name] = judge(s)
            print(f'{check:8} {name:16} {"pass" if results[check, name][0] else "FAIL"}  {results[check, name][1]}',
                  flush=True)
    return results, scores


def table(results, names) -> str:
    lines = ['| check | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for check in CHECKS:
        lines.append(f'| {CHECK_TITLES[check]} | ' + ' | '.join('pass' if results[check, n][0] else '**FAIL**'
                                                               for n in names) + ' |')
    lines.append('| passed | ' + ' | '.join(str(sum(results[c, n][0] for c in CHECKS)) + f'/{len(CHECKS)}'
                                              for n in names) + ' |')
    return '\n'.join(lines)


def save_stimuli(directory):
    os.makedirs(directory, exist_ok=True)
    for check, make in CHECKS.items():
        stimuli, _ = make()
        seen = {}
        for k, (ref, test) in stimuli.items():
            for im in (ref, test):
                key = im.tobytes()
                if key not in seen:
                    seen[key] = f'{check}-{k.replace("/", "-").replace(" ", "_")}{"-ref" if im is ref else ""}.png'
                    cv2.imwrite(os.path.join(directory, seen[key]),
                                np.round(M.upscale(np.clip(im, 0, 1), 3)[..., ::-1] * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppd', type=float, default=M.ZOOM_PPD['x3'])
    ap.add_argument('--factor', type=int, default=None)
    ap.add_argument('--metrics', default=','.join(M.METRICS))
    ap.add_argument('--save', action='store_true')
    args = ap.parse_args()
    names = args.metrics.split(',')
    factor = args.factor or M.device_factor(args.ppd)
    if args.save:
        save_stimuli(os.path.join(HERE, 'data', 'a0'))
    t0 = time.time()
    results, scores = run(names, args.ppd, factor)
    ref = ramp((0.1,) * 3, (0.85,) * 3)
    timing = M.runtimes(ref, dither(zx('black'), zx('white'), M.srgb_to_linear(ref[..., 0])), args.ppd, names)
    report = '\n'.join([
        f'# A0 results: synthetic sanity checks at {args.ppd} ppd, device factor {factor}',
        '',
        f'Made by `python research/metrics/a0.py` ({time.strftime("%Y-%m-%d")}, {time.time() - t0:.0f} s); the '
        'stimuli and the pass rules are in the docstrings of `a0.py`, the metrics\' configuration in `metrics.py`. '
        'Every score is lower = more similar.',
        '',
        table(results, names),
        '',
        'Clash stimuli, the largest dE00 between a cell\'s mean and the surface\'s: ' +
        ', '.join(f'{k} {v:.2f}' for k, v in clash_mean_error().items()) + '.',
        '',
        '## Scores',
        '',
        '| check | metric | result | scores |',
        '|---|---|---|---|',
        *[f'| {c} | {n} | {"pass" if results[c, n][0] else "FAIL"} | {results[c, n][1]} |'
          for c in CHECKS for n in names],
        '',
        '## Runtime on one 256x192 pair',
        '',
        'The grey ramp against its blue-noise dither, 4 CPU threads, no GPU; seconds, the first call (loading the '
        'models, writing PNGs) and the best of two more.',
        '',
        '| metric | first | steady |',
        '|---|---|---|',
        *[f'| {n} | {a:.2f} | {b:.2f} |' for n, (a, b) in timing.items()],
        ''])
    print()
    print(table(results, names))
    print(report.split('## Runtime')[1])
    if set(names) == set(M.METRICS) and args.ppd == M.ZOOM_PPD['x3'] and factor == M.device_factor(args.ppd):
        with open(os.path.join(HERE, 'a0-results.md'), 'w') as f:
            f.write(report)


if __name__ == '__main__':
    main()
