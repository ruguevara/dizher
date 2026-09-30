"""Judge metrics: each scores a finished colouring against its source, lower better, and is judged in turn by how
often it ranks the winner of a judged pair below the loser (`rank`), by fault and by picture. Candidates: the
selection energy under each method's preset, blurred opponent error at several scales, SSIM on lightness and on
colour, CIEDE2000 after blur, S-CIELAB, chroma added and hue shifted (the wrong-colour fault), pair changes across
flat seams and pairs per flat region (the noisy-colouring fault), and LPIPS and DISTS when torch is installed.

A metric is f(ctx) -> float with ctx a Case: the source and result in sRGB float, the result's pair labels and the
project's selection converter (its energy, eye kernels and seam smoothness)."""
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity

from mokit.graph import Memo

from dizher import ops
from dizher.converter.energy import METHODS, LRGB2OPP, lightness_gain

from .project import project_graph, DEFAULTS, select
from .scr import render_scr, pairs_to_labels, same_region, black
from .variants import read_variant, listing, reference_id, RANGES, variant_meta

GAMMA = 2.2
METRICS = {}


def metric(name):
    def register(f):
        METRICS[name] = f
        return f
    return register


@dataclass
class Case:
    source: np.ndarray      # (H, W, 3) sRGB float, the converter's target
    result: np.ndarray      # (H, W, 3) sRGB float, the screen
    pairs: np.ndarray       # (R, C, 2) palette indexes of the screen
    conv: object            # the project's selection converter under the default method (energy calculated)
    convs: dict             # method -> selection converter under that method's preset
    meta: dict = None       # the variant's record (its method and params) when it is one

    def labels(self, conv=None):
        return pairs_to_labels(conv or self.conv, self.pairs)


class Picture:
    """A project's source and its selection converters, shared by every variant of it."""

    def __init__(self, name, methods=tuple(METHODS)):
        self.name = name
        self.memo = Memo()
        self.convs = {}
        for m in methods:
            graph = ops.apply_preset(project_graph(name, DEFAULTS), m)
            self.convs[m] = select(graph, self.memo)
        self.conv = next(iter(self.convs.values()))
        self.source = self.conv.image_rgb
        self.energies = {}

    def case(self, variant) -> Case:
        bitmap, idx = read_variant(self.name, variant)
        return Case(self.source, render_scr(bitmap, idx, self.conv.palette), idx, self.conv, self.convs,
                    variant_meta(self.name, variant))

    def energy_set(self, method) -> 'Energies':
        """The method's energy as a function of its params, on a converter of its own."""
        if method not in self.energies:
            self.energies[method] = Energies(self.name, method, conv=self.convs[method])
        return self.energies[method]


SETUP = ('flare', 'luma_scale', 'chroma_scale', 'luma_alpha', 'chroma_alpha')   # params that rebuild the candidates' setup
WEIGHTS = ('chroma', 'coherence', 'edge', 'chroma_noise', 'luma_noise')          # params the energy applies as they are


class Energies:
    """A picture's variants' energies under one method as a function of the Metric, Eye and Select pairs params:
    the setup params (flare, the eye kernels) rebuild the candidates' energy when they change, the rest apply as they
    are. On a converter copy of its own, so the preset's converter stays as it is."""

    def __init__(self, name, method, variants=(), conv=None):
        self.name = name
        self.conv = (conv if conv is not None else Picture(name, methods=(method,)).convs[method]).copy()
        self.conv.energy.invalidate()      # the copy's own arrays: calc() fills the dicts in place
        self.labels = {}
        for v in variants:
            self.label(v)
        self.setup = None

    def label(self, variant):
        if variant not in self.labels:
            self.labels[variant] = pairs_to_labels(self.conv, read_variant(self.name, variant)[1])
        return self.labels[variant]

    def set_setup(self, params: dict):
        setup = tuple((k, float(params[k])) for k in SETUP if k in params)
        if setup != self.setup:
            c = self.conv
            for k, v in setup:
                setattr(c, k, v)
            c.gain = lightness_gain(c.image_luma, c.flare)
            c.energy.calc()
            self.setup = setup

    def set_flare(self, flare):
        self.set_setup(dict(flare=flare))

    def __call__(self, params: dict, variants=None) -> dict:
        """variant -> energy under params; a variant whose pairs the palette lacks reads nan."""
        self.set_setup(params)
        c = self.conv
        if 'chroma' in params:
            c.energy.weights['Chroma'] = params['chroma']
        for k in ('coherence', 'edge', 'luma_noise', 'chroma_noise'):
            if k in params:
                setattr(c, k, params[k])
        return {v: c.energy.energy(l) if (l >= 0).all() else float('nan')
                for v, l in ((v, self.label(v)) for v in (variants if variants is not None else self.labels))}


def parse_energy_metric(name: str):
    """'energy:Exact mixture:flare=1+luma_scale=2.8' -> (method, params); the params over the method's preset."""
    parts = name.split(':')
    assert parts[0] == 'energy' and parts[1] in METHODS, name
    params = {k: float(v) for k, v in METHODS[parts[1]].preset.items() if k in RANGES}
    for item in (parts[2].split('+') if len(parts) > 2 and parts[2] else ()):
        k, v = item.split('=')
        assert k in SETUP or k in WEIGHTS, f'{name}: {k} is not an energy param'
        params[k] = float(v)
    return parts[1], params


JUDGE_TERMS = ('opp_blur:2:nogain', 'gmsd')   # judge:<w> = the first over its median + w * the second over its median


# ----- helpers -----------------------------------------------------------------------------------------------

def lin(rgb):
    return rgb.astype(np.float32) ** GAMMA


def opp(rgb):
    return lin(rgb) @ LRGB2OPP.T


def blur(img, sigma):
    """Gaussian blur per channel, sigma px; sigma 0 is the image."""
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(np.ascontiguousarray(img, dtype=np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)


def lab(rgb):
    return rgb2lab(np.clip(rgb, 0, 1).astype(np.float64))


def blurred_srgb(rgb, sigma):
    """Blur in linear light, back to sRGB."""
    return np.clip(blur(lin(rgb), sigma), 0, 1) ** (1 / GAMMA)


def pyramid(a, levels):
    out = [a]
    for _ in range(levels - 1):
        out.append(cv2.pyrDown(out[-1]))
    return out


# ----- the energy --------------------------------------------------------------------------------------------

def _energy(method):
    def f(c: Case):
        conv = c.convs[method]
        labels = c.labels(conv)
        if (labels < 0).any():
            return float('nan')
        return conv.energy.energy(labels)
    return f


for _m in METHODS:
    METRICS[f'energy:{_m}'] = _energy(_m)


# ----- blurred error -----------------------------------------------------------------------------------------

def _opp_blur(sigma, gained=True):
    def f(c: Case):
        e = opp(c.result) - opp(c.source)
        if gained:
            e = e * lightness_gain(c.conv.image_luma, c.conv.flare)
        return float((blur(e, sigma) ** 2).sum(-1).mean())
    return f


for _s in (1, 2, 4, 8):
    METRICS[f'opp_blur:{_s}'] = _opp_blur(_s)
METRICS['opp_blur:2:nogain'] = _opp_blur(2, gained=False)


def levels_for(conv, chroma):
    """Every pair's per-pixel mixture level for the target under a chroma weight (Converter.fit_duocolors under
    that weight), cached on the converter."""
    cache = conv.__dict__.setdefault('_bench_levels', {})
    key = round(float(chroma), 4)
    if key not in cache:
        from dizher.converter.dither import duo_levels
        w = np.sqrt(np.array([1.0, chroma, chroma], dtype=np.float32))
        weighted = lambda lrgb: (lrgb.astype(np.float32) @ LRGB2OPP.T) * w
        target = weighted(conv.image_lrgb)
        levels = np.empty((len(conv.color_pairs), *conv.size), np.float32)
        for i, (c1, c2) in enumerate(conv.palette.iter_color_pairs()):
            levels[i] = duo_levels(target, weighted(c1 ** GAMMA), weighted(c2 ** GAMMA))
        cache[key] = levels
    return cache[key]


def mixture(conv, labels, levels=None):
    """What the Exact mixture energy scores: each cell's pair mixed in linear light at each pixel's level, sRGB."""
    rows, cols = np.indices(conv.size)
    idx = conv.expand_cells(labels)
    pairs = lin(conv.color_pairs[idx])                                              # (H, W, 2, 3)
    levels = conv.levels if levels is None else levels
    t = levels[idx, rows, cols][..., None].astype(np.float32)
    return (pairs[..., 0, :] + t * (pairs[..., 1, :] - pairs[..., 0, :])) ** (1 / GAMMA)


def _mix_blur(sigma, own=False):
    """The blurred opponent error, no gain, of the mixture composite of the result's pairs instead of the result:
    what the selection energy looks at, measured as the judge measures. own: the levels under the variant's own
    chroma weight (its record), as its picture was made, instead of the preset's."""
    def f(c: Case):
        conv = c.convs.get('Exact mixture', c.conv)
        labels = c.labels(conv)
        if (labels < 0).any():
            return float('nan')
        levels = None
        if own:
            chroma = (c.meta or {}).get('params', {}).get('chroma', conv.energy.weights['Chroma'])
            levels = levels_for(conv, chroma)
        e = opp(mixture(conv, labels, levels)) - opp(c.source)
        return float((blur(e, sigma) ** 2).sum(-1).mean())
    return f


for _s in (1, 2):
    METRICS[f'mix_blur:{_s}'] = _mix_blur(_s)
    METRICS[f'mix_blur:{_s}:own'] = _mix_blur(_s, own=True)


def _eye_trunc(scale):
    """The opponent error through the converter's own kernels at a scale (alpha 2, support capped at half a cell as
    in pair selection), no gain: the judge's measure with the energy's truncation."""
    def f(c: Case):
        from dizher.converter.eye import eye_kernel
        radius = min(c.conv.cell) // 2
        h = eye_kernel(scale, 2.0, max_radius=radius)
        e = opp(c.result) - opp(c.source)
        b = np.stack([cv2.filter2D(np.ascontiguousarray(e[..., k]), -1, h, borderType=cv2.BORDER_REFLECT_101)
                      for k in range(3)], axis=-1)
        return float((b ** 2).sum(-1).mean())
    return f


for _sc in (1.4, 2.8):
    METRICS[f'eye_trunc:{_sc}'] = _eye_trunc(_sc)


@metric('scielab')
def scielab(c: Case):
    """S-CIELAB: opponent channels blurred as the eye model does (the converter's kernels), CIEDE2000 between them."""
    a, b = c.conv.eye_view(c.source), c.conv.eye_view(c.result)
    return float(deltaE_ciede2000(lab(a), lab(b)).mean())


def de2000_masked(a_rgb, b_rgb, mask, sigma) -> float:
    """Mean CIEDE2000 after a blur over the masked pixels; nan for an empty mask."""
    if not mask.any():
        return float('nan')
    d = deltaE_ciede2000(lab(blurred_srgb(a_rgb, sigma)), lab(blurred_srgb(b_rgb, sigma)))
    return float(d[mask].mean())


def _de2000(sigma):
    def f(c: Case):
        return float(deltaE_ciede2000(lab(blurred_srgb(c.source, sigma)), lab(blurred_srgb(c.result, sigma))).mean())
    return f


for _s in (1, 2, 4):
    METRICS[f'de2000:{_s}'] = _de2000(_s)


# ----- structure ---------------------------------------------------------------------------------------------

def _ssim_channels(a, b, data_range):
    return float(np.mean([structural_similarity(a[..., k], b[..., k], data_range=data_range, win_size=7,
                                                gaussian_weights=True, sigma=1.5) for k in range(a.shape[-1])]))


def _ssim(channels, sigma=0, levels=1):
    """1 - mean SSIM over the CIELAB channels (a slice), the images first blurred sigma px in linear light and
    compared over a pyramid of levels."""
    def f(c: Case):
        a, b = lab(blurred_srgb(c.source, sigma)), lab(blurred_srgb(c.result, sigma))
        a, b = a[..., channels], b[..., channels]
        rng = 100.0 if channels == slice(0, 1) else 255.0
        return 1 - float(np.mean([_ssim_channels(x, y, rng) for x, y in zip(pyramid(a, levels), pyramid(b, levels))]))
    return f


METRICS['ssim_L'] = _ssim(slice(0, 1))
METRICS['ssim_L:2'] = _ssim(slice(0, 1), sigma=2)
METRICS['msssim_L'] = _ssim(slice(0, 1), levels=4)
METRICS['ssim_ab'] = _ssim(slice(1, 3))
METRICS['msssim_ab'] = _ssim(slice(1, 3), levels=4)
METRICS['msssim_Lab'] = _ssim(slice(0, 3), levels=4)


# ----- the wrong-colour fault ----------------------------------------------------------------------------------

def _chroma_added(sigma):
    """Chroma the result has where the source has less, after a blur: a hue where the original had none."""
    def f(c: Case):
        a, b = lab(blurred_srgb(c.source, sigma)), lab(blurred_srgb(c.result, sigma))
        ca, cb = np.hypot(a[..., 1], a[..., 2]), np.hypot(b[..., 1], b[..., 2])
        return float(np.maximum(0, cb - ca).mean())
    return f


for _s in (2, 4):
    METRICS[f'chroma_added:{_s}'] = _chroma_added(_s)


def _hue_shift(sigma):
    """Hue turned between source and result after a blur, weighted by the smaller chroma (a grey has no hue)."""
    def f(c: Case):
        a, b = lab(blurred_srgb(c.source, sigma)), lab(blurred_srgb(c.result, sigma))
        ca, cb = np.hypot(a[..., 1], a[..., 2]), np.hypot(b[..., 1], b[..., 2])
        dh = np.abs(np.arctan2(a[..., 2], a[..., 1]) - np.arctan2(b[..., 2], b[..., 1]))
        dh = np.minimum(dh, 2 * np.pi - dh)
        return float((np.minimum(ca, cb) * dh).mean())
    return f


for _s in (2, 4):
    METRICS[f'hue_shift:{_s}'] = _hue_shift(_s)


# ----- the noisy-colouring fault -------------------------------------------------------------------------------

def flat_seams(conv):
    """(R-1, C) and (R, C-1) weights 0..1: 1 where the source is flat across the seam (the energy's smoothness)."""
    return conv.energy.seam_smoothness()


@metric('label_noise')
def label_noise(c: Case):
    """Pair changes across seams where the source is flat, per seam."""
    s = np.sort(black(c.pairs), axis=-1)
    Lh, Lv = flat_seams(c.conv)
    ch = ~same_region(s[1:].reshape(-1, 2), s[:-1].reshape(-1, 2)).reshape(Lh.shape)
    cv = ~same_region(s[:, 1:].reshape(-1, 2), s[:, :-1].reshape(-1, 2)).reshape(Lv.shape)
    return float(((Lh * ch).sum() + (Lv * cv).sum()) / (Lh.size + Lv.size))


@metric('region_pairs')
def region_pairs(c: Case):
    """Regions of the source: cells joined across flat seams (smoothness over 0.5). Over regions of 4 cells and
    more, the distinct pairs beyond one, weighted by size, per cell."""
    Lh, Lv = flat_seams(c.conv)
    R, C = c.pairs.shape[:2]
    parent = np.arange(R * C)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in range(R - 1):
        for col in range(C):
            if Lh[r, col] > 0.5:
                parent[find(r * C + col)] = find((r + 1) * C + col)
    for r in range(R):
        for col in range(C - 1):
            if Lv[r, col] > 0.5:
                parent[find(r * C + col)] = find(r * C + col + 1)
    roots = np.array([find(i) for i in range(R * C)])
    s = np.sort(black(c.pairs), axis=-1).reshape(-1, 2)
    total = 0.0
    for root in np.unique(roots):
        members = np.nonzero(roots == root)[0]
        if len(members) < 4:
            continue
        distinct = len({tuple(p) for p in s[members].tolist()})
        total += (distinct - 1) * len(members)
    return total / (R * C)


# ----- learned metrics (optional) -------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _torch_model(kind):
    """(torch, model, inputs in -1..1) or None without the bench extra; the failure is printed once."""
    try:
        import torch
        if kind == 'dists':
            import os
            import DISTS_pytorch
            from DISTS_pytorch import DISTS
            model = DISTS(load_weights=False)     # the package looks for its weights in sys.prefix, they are beside it
            weights = torch.load(os.path.join(os.path.dirname(DISTS_pytorch.__file__), 'weights.pt'))
            model.alpha.data, model.beta.data = weights['alpha'], weights['beta']
            return torch, model.eval(), False
        if kind in ('haarpsi', 'gmsd'):
            from piqa import HaarPSI, GMSD
            return torch, (HaarPSI if kind == 'haarpsi' else GMSD)().eval(), False
        import lpips
        return torch, lpips.LPIPS(net=kind, verbose=False).eval(), True
    except Exception as e:                        # noqa: BLE001  an optional dependency: the metric reads n/a
        print(f'{kind}: {e.__class__.__name__}: {e}')
        return None


def _learned(kind):
    def f(c: Case):
        loaded = _torch_model(kind)
        if loaded is None:
            return float('nan')
        torch, model, signed = loaded
        with torch.no_grad():
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))[None].astype(np.float32))
            a, b = (t(c.source), t(c.result)) if not signed else (t(c.source) * 2 - 1, t(c.result) * 2 - 1)
            value = float(model(a, b))
            return 1 - value if kind == 'haarpsi' else value      # HaarPSI is a similarity, 1 for the same picture
    return f


for _k in ('alex', 'vgg'):
    METRICS[f'lpips_{_k}'] = _learned(_k)
for _k in ('dists', 'haarpsi', 'gmsd'):
    METRICS[_k] = _learned(_k)


# ----- ranking the metrics by the judgments ---------------------------------------------------------------------

DISTANCES = ('de2000:4', 'scielab', 'opp_blur:2', 'msssim_ab', 'hue_shift:2')   # a result against the reference picture


def distances(conv, reference_rgb, result_rgb, pairs, metrics=DISTANCES) -> dict:
    """metric -> distance between two pictures of one project (conv its selection converter, for the eye kernels and
    the lightness gain): the judge metrics with the reference picture as the source."""
    c = Case(reference_rgb, result_rgb, pairs, conv, {})
    return {m: METRICS[m](c) for m in metrics}


def scores(picture: Picture, variants, metrics=None) -> dict:
    """variant -> metric -> value. Besides METRICS: 'energy:<method>:<k=v+k=v>' (parse_energy_metric) and 'judge:<w>',
    the composite of JUDGE_TERMS, each term over its median across the variants scored, the second weighted w."""
    metrics = metrics or list(METRICS)
    plain = [m for m in metrics if m in METRICS]
    judges = [m for m in metrics if m.startswith('judge:')]
    energies = [m for m in metrics if m.startswith('energy:') and m not in METRICS]
    unknown = [m for m in metrics if m not in plain + judges + energies]
    assert not unknown, f'unknown metrics {unknown}'
    terms = [t for t in JUDGE_TERMS if judges and t not in plain]
    out = {}
    for v in variants:
        c = picture.case(v)
        out[v] = {m: METRICS[m](c) for m in plain + terms}
    for m in energies:
        method, params = parse_energy_metric(m)
        for v, e in picture.energy_set(method)(params, variants).items():
            out[v][m] = e
    if judges:
        a, b = JUDGE_TERMS
        med = lambda t: np.nanmedian([out[v][t] for v in variants]) or 1.0
        ma, mb = med(a), med(b)
        for m in judges:
            w = float(m.split(':')[1])
            for v in variants:
                out[v][m] = out[v][a] / ma + w * out[v][b] / mb
        for v in variants:
            for t in terms:
                del out[v][t]
    return out


def judged_pairs(judgments: dict, by: str) -> list:
    """(winner, loser, tags) of every pair the judge decided (not same)."""
    out = []
    for p in judgments['pairs']:
        v = p['verdicts'].get(by)
        if v and v['verdict'] in ('a', 'b'):
            w, l = (p['a'], p['b']) if v['verdict'] == 'a' else (p['b'], p['a'])
            out.append((w, l, tuple(v.get('tags', ()))))
    return out


def agreement(pairs, values: dict, metrics) -> dict:
    """metric -> (share of pairs where the winner scores lower, ties half; pairs counted), over pairs whose both
    variants have a value."""
    out = {}
    for m in metrics:
        hits, n = 0.0, 0
        for w, l, _ in pairs:
            a, b = values[w][m], values[l][m]
            if np.isnan(a) or np.isnan(b):
                continue
            n += 1
            hits += 1.0 if a < b else 0.5 if a == b else 0.0
        out[m] = (hits / n if n else float('nan'), n)
    return out


def reference_rank(values: dict, metrics, rid) -> dict:
    """metric -> share of variants the metric puts below the reference (0: the reference is best)."""
    out = {}
    ref = values.get(rid)
    if ref is None:
        return {m: (float('nan'), 0) for m in metrics}
    others = [v for k, v in values.items() if k != rid]
    for m in metrics:
        vals = [v[m] for v in others if not np.isnan(v[m])]
        out[m] = (sum(x < ref[m] for x in vals) / len(vals) if vals and not np.isnan(ref[m]) else float('nan'), len(vals))
    return out


def rank(names, judgments_of, by='user', metrics=None, with_reference=True, log=print) -> dict:
    """Every metric's agreement with the judge over the named pictures: overall, by fault tag and by picture,
    and the reference's rank among each picture's variants. judgments_of(name) -> judgments dict."""
    metrics = metrics or list(METRICS)
    tags = {}
    per_picture, all_pairs, all_values, ref_ranks = {}, [], {}, {}
    for name in names:
        pairs = judged_pairs(judgments_of(name), by)
        variants = sorted({v for w, l, _ in pairs for v in (w, l)})
        if with_reference:
            variants = sorted(set(variants) | set(listing(name)))
        if not variants:
            continue
        log(f'{name}: {len(pairs)} judged pairs, {len(variants)} variants')
        picture = Picture(name)
        values = scores(picture, variants, metrics)
        per_picture[name] = agreement(pairs, values, metrics)
        all_pairs += [(f'{name}/{w}', f'{name}/{l}', t) for w, l, t in pairs]
        all_values.update({f'{name}/{v}': s for v, s in values.items()})
        rid = reference_id(name)
        if with_reference and rid in values:
            ref_ranks[name] = reference_rank(values, metrics, rid)
        for w, l, t in pairs:
            for tag in t:
                tags.setdefault(tag, []).append((f'{name}/{w}', f'{name}/{l}', t))
    overall = agreement(all_pairs, all_values, metrics)
    by_tag = {tag: agreement(ps, all_values, metrics) for tag, ps in tags.items()}
    return dict(overall=overall, by_tag=by_tag, by_picture=per_picture, reference=ref_ranks, values=all_values)


def print_rank(result, metrics=None) -> None:
    metrics = metrics or list(result['overall'])
    tags = list(result['by_tag'])
    refs = result['reference']
    print(f"{'metric':22} {'all':>9} " + ' '.join(f'{t:>9}' for t in tags) + f" {'ref':>9}")
    for m in sorted(metrics, key=lambda m: -np.nan_to_num(result['overall'][m][0], nan=-1)):
        cell = lambda a: f'{a[0]:6.2f}/{a[1]:<3d}' if not np.isnan(a[0]) else f"{'n/a':>9}"
        ref = [refs[n][m][0] for n in refs if not np.isnan(refs[n][m][0])]
        print(f'{m:22} {cell(result["overall"][m])} ' + ' '.join(cell(result['by_tag'][t][m]) for t in tags)
              + (f' {np.mean(ref):6.2f}/{len(ref):<3d}' if ref else f" {'n/a':>9}"))
    print('cells: agreement with the judge / pairs counted; ref: share of variants the metric puts below the '
          'reference, averaged over pictures with one (0 is best)')
