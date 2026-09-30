"""Candidate metrics of a rendered colouring X against the tuned picture T, each over a pixel mask, lower is better.

Compare colourings through a blur of about a pixel, never as raw pixels: the user judges the colouring, not the
dither's grain. At their viewing (the app at 2x-3x on a 21" 2560x1440 monitor at ~65 cm, ~31 Spectrum pixels per
degree at 2x, ~21 at 3x) a pixel is ~2.3' and the dots are visible, so S-CIELAB at that viewing (Zhang & Wandell 1996)
keeps them: a re-dither of the same colouring moves it 5.8 dE, against 2.0 through the converter's eye
(`project_eye`, a Gaussian of 1 px), where a painted change is ~25. LPIPS and DISTS are a coin toss raw and agree with
the user through that eye (eyes.py). Metrics that blur or average on their own (the seams at 2 px, cell means, the
coarse ones) need no eye in front."""
import cv2
import numpy as np

from dizher.converter.eye import eye_blur

PPD = 26.0   # Spectrum pixels per degree of visual angle, between 2x and 3x

SRGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                     [0.2126, 0.7152, 0.0722],
                     [0.0193, 0.1192, 0.9505]])
XYZ2OPP = np.array([[0.279, 0.72, -0.107],
                    [-0.449, 0.29, -0.077],
                    [0.086, -0.59, 0.501]])
WHITE = SRGB2XYZ @ np.ones(3)
# S-CIELAB's spatial filters: per opponent channel, (weight, spread in degrees) of Gaussians exp(-r^2 / s^2)
FILTERS = (((1.00327, 0.05), (0.114416, 0.225), (-0.117686, 7.0)),
           ((0.616725, 0.0685), (0.383275, 0.826)),
           ((0.567885, 0.0920), (0.432115, 0.6451)))


def linear(srgb):
    s = np.asarray(srgb, np.float64)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def xyz2lab(xyz):
    t = xyz / WHITE
    f = np.where(t > (6 / 29) ** 3, np.cbrt(np.maximum(t, 0)), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def blur(img, sigma):
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101) \
        if sigma > 0 else img


def seen_xyz(srgb, ppd=PPD):
    """XYZ of the image as the eye sees it at ppd pixels per degree: opponent channels each filtered by their sum of
    Gaussians (S-CIELAB), back to XYZ."""
    opp = linear(srgb) @ SRGB2XYZ.T @ XYZ2OPP.T
    out = np.zeros_like(opp)
    for k, parts in enumerate(FILTERS):
        for w, s in parts:
            out[..., k] += w * blur(opp[..., k], s * ppd / np.sqrt(2))
    return out @ np.linalg.inv(XYZ2OPP).T


def scielab(srgb, ppd=PPD):
    """CIELAB of the image as the eye sees it at ppd pixels per degree."""
    return xyz2lab(seen_xyz(srgb, ppd))


def encode(lin):
    """sRGB 0..1 of linear RGB, clipped."""
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


def seen_srgb(srgb, ppd=PPD):
    """sRGB 0..1 of the image as the eye sees it, clipped: what a network metric is shown."""
    return encode(seen_xyz(srgb, ppd) @ np.linalg.inv(SRGB2XYZ).T)


def lab_blurred(srgb, sigma):
    """CIELAB of the image blurred in linear light, one Gaussian for all channels."""
    return xyz2lab(blur(linear(srgb), sigma) @ SRGB2XYZ.T)


def chroma(lab):
    return np.hypot(lab[..., 1], lab[..., 2])


def hue_angle(x, t):
    """|hue angle difference| in radians, 0..pi."""
    d = np.arctan2(x[..., 2], x[..., 1]) - np.arctan2(t[..., 2], t[..., 1])
    return np.abs((d + np.pi) % (2 * np.pi) - np.pi)


def masked_mean(v, mask):
    return float(v[mask].mean())


def seam_excess(L, T, mask, cell=8):
    """Mean over the cell seams inside mask of how much more X steps across the seam than the picture does."""
    xs = np.arange(cell, L.shape[1], cell)
    ys = np.arange(cell, L.shape[0], cell)
    step = lambda a, axis, at: np.linalg.norm(np.take(a, at, axis) - np.take(a, at - 1, axis), axis=-1)
    vals = []
    for axis, at in ((1, xs), (0, ys)):
        ex = np.maximum(step(L, axis, at) - step(T, axis, at), 0)
        m = np.take(mask, at, axis) & np.take(mask, at - 1, axis)
        vals.append(ex[m])
    v = np.concatenate(vals)
    return float(v.mean()) if v.size else 0.0


def cell_means(lab_img, cell=8):
    R, C = lab_img.shape[0] // cell, lab_img.shape[1] // cell
    return lab_img.reshape(R, cell, C, cell, -1).mean((1, 3))


def cells_of(img, cell=8):
    """(R, C, cell * cell, channels): each cell's pixels."""
    R, C = img.shape[0] // cell, img.shape[1] // cell
    return img.reshape(R, cell, C, cell, -1).transpose(0, 2, 1, 3, 4).reshape(R, C, cell * cell, -1)


def cell_mask(mask, cell=8):
    """(R, C) the cells mostly inside the pixel mask."""
    return cell_means(mask[..., None].astype(float), cell)[..., 0] > 0.5


def mean_lab(srgb, cell=8):
    """(R, C, 3) CIELAB of each cell's mean colour in linear light."""
    return xyz2lab(cell_means(linear(srgb), cell) @ SRGB2XYZ.T)


def two_colour_share(X, mask, cell=8):
    """Share of the cells in mask that show two colours rather than one."""
    q = (cells_of(X, cell) * 255).round().astype(np.int64)
    code = np.sort((q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2], axis=-1)
    two = (code[..., 1:] != code[..., :-1]).any(-1)
    return float(two[cell_mask(mask, cell)].mean())


def cell_spread(lab, cell=8):
    """(R, C) RMS distance of each cell's pixels from the cell's mean, in the given CIELAB image."""
    px = cells_of(lab, cell)
    return np.sqrt(((px - px.mean(2, keepdims=True)) ** 2).sum(-1).mean(-1))


# the hues of the Spectrum's chromatic colours, CIELAB angles: red, yellow, green, cyan, blue, magenta
ZX_HUES = np.arctan2(*(lambda lab: (lab[:, 2], lab[:, 1]))(xyz2lab(linear(np.array(
    [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]], float)) @ SRGB2XYZ.T)))


def hue_family(lab, neutral=12.0):
    """Index of the nearest Spectrum hue, -1 where the colour is near grey (chroma < neutral)."""
    d = np.abs((np.arctan2(lab[..., 2], lab[..., 1])[..., None] - ZX_HUES + np.pi) % (2 * np.pi) - np.pi)
    return np.where(chroma(lab) < neutral, -1, d.argmin(-1))


def hue_family_miss(X, T, mask, cell=8):
    """Share of the cells in mask whose mean colour falls in another hue family (or grey) than the picture's."""
    m = cell_mask(mask, cell)
    return float((hue_family(mean_lab(X, cell)) != hue_family(mean_lab(T, cell)))[m].mean())


def neighbour_excess(X, T, mask, cell=8):
    """Cells' mean colours (linear light, then CIELAB): over 4-neighbour cells both in mask, how much more X changes
    from cell to cell than the picture does: blocks and noise where the picture is smooth."""
    mx = xyz2lab(cell_means(linear(X), cell) @ SRGB2XYZ.T)
    mt = xyz2lab(cell_means(linear(T), cell) @ SRGB2XYZ.T)
    mc = cell_means(mask[..., None].astype(float), cell)[..., 0] > 0.5
    vals = []
    for d in ((1, 0), (0, 1)):
        a = (slice(d[0], None), slice(d[1], None))
        b = (slice(None, mx.shape[0] - d[0]), slice(None, mx.shape[1] - d[1]))
        ex = np.maximum(np.linalg.norm(mx[a] - mx[b], axis=-1) - np.linalg.norm(mt[a] - mt[b], axis=-1), 0)
        vals.append(ex[mc[a] & mc[b]])
    v = np.concatenate(vals)
    return float(v.mean()) if v.size else 0.0


def ssim_map(x, y, sigma=1.5):
    C1, C2 = (0.01 * 100) ** 2, (0.03 * 100) ** 2   # L* spans 0..100
    mx, my = blur(x, sigma), blur(y, sigma)
    vx, vy = blur(x * x, sigma) - mx ** 2, blur(y * y, sigma) - my ** 2
    cov = blur(x * y, sigma) - mx * my
    return ((2 * mx * my + C1) * (2 * cov + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))


def ms_dssim(X, T, mask, channel=0, scales=(2, 4, 8)):
    """1 - SSIM of one CIELAB channel, the images first averaged over s x s pixels in linear light, over the scales."""
    out = []
    for s in scales:
        h, w = X.shape[0] // s, X.shape[1] // s
        down = lambda img: xyz2lab(cv2.resize(linear(img).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                                   @ SRGB2XYZ.T)[..., channel].astype(np.float32)
        m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
        if m.any():
            out.append(1 - masked_mean(ssim_map(down(X), down(T)), m))
    return float(np.mean(out)) if out else 0.0


_DEEP = {}


def deep(X, T, mask, name):
    """LPIPS or DISTS (piq, VGG features) over the bounding box of mask, both images at 2x as the app shows them.
    Research only: needs torch and piq, not the app's dependencies."""
    import piq
    import torch
    if name not in _DEEP:
        torch.set_num_threads(1)   # one per worker process
        _DEEP[name] = {'lpips': piq.LPIPS, 'dists': piq.DISTS}[name]().eval()
    ys, xs = np.nonzero(mask)
    crop = lambda a: torch.from_numpy(np.ascontiguousarray(np.repeat(np.repeat(
        a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 2, 0), 2, 1))).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        return float(_DEEP[name](crop(X), crop(T)))


def project_eye(srgb):
    """The converter's default eye: a Gaussian of sigma 1 px on linear RGB. The network metrics see the dither
    through it: raw, LPIPS and DISTS are a coin toss on the painted segments; blurred 0.75-1.5 px they agree with the
    user best (eyes.py), and this one was fixed before the grid."""
    return encode(eye_blur(linear(srgb).astype(np.float32), 1.4, 2.0))


def seen_change(X, Y, mask):
    """Mean dE between X and Y through the converter's eye over mask: how visible a change is."""
    lab = lambda img: xyz2lab(linear(project_eye(img)) @ SRGB2XYZ.T)
    return masked_mean(np.linalg.norm(lab(X) - lab(Y), axis=-1), mask)


def gradient(lab):
    """(H, W) gradient magnitude of a CIELAB image, all three channels (Sobel, per pixel)."""
    g = [cv2.Sobel(lab[..., k].astype(np.float32), cv2.CV_32F, dx, 1 - dx, ksize=3) for k in range(3) for dx in (0, 1)]
    return np.sqrt(sum(x * x for x in g)) / 8


def structure(ex, et, mask) -> dict:
    """Torch-free structure terms of X against T, both CIELAB through the converter's eye: what the picture has that X
    lacks (the counterpart of the seam excess, which a flat colouring wins), GMSD and SSIM of lightness."""
    out = {}
    for s in (0, 1, 2):
        gx, gt = (gradient(blur(ex, s)), gradient(blur(et, s)))
        out[f'detail_deficit_{s}'] = masked_mean(np.maximum(gt - gx, 0), mask)
    lx, lt = gradient(ex[..., :1].repeat(3, -1)) / np.sqrt(3), gradient(et[..., :1].repeat(3, -1)) / np.sqrt(3)
    c = 170.0 / 255 * 100   # GMSD's constant for 0..255, in L* units
    gms = (2 * lx * lt + c) / (lx ** 2 + lt ** 2 + c)
    out['gmsd_eye'] = float(gms[mask].std())
    out['gmsm_eye'] = 1 - float(gms[mask].mean())   # the mean similarity: GMSD's spread rates a black screen well
    out['dssim_eye'] = 1 - masked_mean(ssim_map(ex[..., 0], et[..., 0]), mask)
    return out


def colourfulness(srgb, mask):
    """Hasler & Suesstrunk's colourfulness of the pixels in mask (sRGB 0..1)."""
    r, g, b = (srgb[..., k][mask] * 255 for k in range(3))
    rg, yb = r - g, (r + g) / 2 - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def drain(X, T, mask, cell=8, grey=12.0, chromatic=15.0) -> dict:
    """What a colouring drained of colour loses: the share of cells showing only near-grey colours where the picture's
    cell is chromatic, chroma lost through the converter's eye, and colourfulness below the picture's."""
    px = cells_of(X, cell)
    lab = xyz2lab(linear(px) @ SRGB2XYZ.T)
    all_grey = (chroma(lab) < grey).all(-1)                       # (R, C) every pixel near grey
    m = cell_mask(mask, cell)
    colourful = (chroma(mean_lab(T, cell)) > chromatic) & m
    ex, et = (xyz2lab(linear(project_eye(img)) @ SRGB2XYZ.T) for img in (X, T))
    return {'grey_share': float((all_grey & colourful).sum() / max(m.sum(), 1)),
            'chroma_deficit_eye': masked_mean(np.maximum(chroma(et) - chroma(ex), 0), mask),
            'colourfulness_deficit': max(colourfulness(project_eye(T), mask) - colourfulness(project_eye(X), mask), 0.0)}


def all_metrics(X, T, mask) -> dict:
    """Every candidate metric of X against T over mask."""
    sx, st = scielab(X), scielab(T)
    dE = np.linalg.norm(sx - st, axis=-1)
    cx, ct = chroma(sx), chroma(st)
    dL = np.abs(sx[..., 0] - st[..., 0])
    dC = cx - ct
    dH = np.sqrt(np.maximum(dE ** 2 - (sx[..., 0] - st[..., 0]) ** 2 - dC ** 2, 0))
    fx, ft = lab_blurred(X, 0.5), lab_blurred(T, 0.5)
    b2x, b2t = lab_blurred(X, 2.0), lab_blurred(T, 2.0)
    rmse = np.linalg.norm(blur(linear(X), 1.0) - blur(linear(T), 1.0), axis=-1)
    cm = cell_mask(mask)
    seams = {}
    for s in (1.0, 2.0, 4.0):
        bx, bt = (b2x, b2t) if s == 2.0 else (lab_blurred(X, s), lab_blurred(T, s))
        seams[f'seam_L_{s:g}'] = seam_excess(bx[..., :1], bt[..., :1], mask)
        seams[f'seam_ab_{s:g}'] = seam_excess(bx[..., 1:], bt[..., 1:], mask)
    return {
        'scielab_dE': masked_mean(dE, mask),
        'scielab_dE_p95': float(np.percentile(dE[mask], 95)),
        'scielab_dL': masked_mean(dL, mask),
        'scielab_dH': masked_mean(dH, mask),
        'chroma_excess': masked_mean(np.maximum(dC, 0), mask),
        'chroma_excess_neutral': masked_mean(np.maximum(dC, 0) * np.exp(-ct / 15), mask),
        'chroma_deficit': masked_mean(np.maximum(-dC, 0), mask),
        'hue_angle': masked_mean(hue_angle(sx, st) * ct / (ct + 20), mask),
        'fine_chroma_excess': masked_mean(np.maximum(chroma(fx) - chroma(ft), 0) * np.exp(-chroma(ft) / 15), mask),
        'seam_excess': seam_excess(b2x, b2t, mask),
        'neighbour_excess': neighbour_excess(X, T, mask),
        'ms_dssim_L': ms_dssim(X, T, mask, 0),
        'ms_dssim_a': ms_dssim(X, T, mask, 1, (4, 8)),
        'ms_dssim_b': ms_dssim(X, T, mask, 2, (4, 8)),
        'blur_rmse': masked_mean(rmse, mask),
        **seams,
        'two_colour_share': two_colour_share(X, mask),
        'dot_contrast': float(cell_spread(xyz2lab(linear(X) @ SRGB2XYZ.T))[cm].mean()),
        'texture_excess': float(np.maximum(cell_spread(sx) - cell_spread(st), 0)[cm].mean()),
        'hue_family_miss': hue_family_miss(X, T, mask),
        'blur_dE_4': masked_mean(np.linalg.norm(lab_blurred(X, 4) - lab_blurred(T, 4), axis=-1), mask),
        'blur_dE_8': masked_mean(np.linalg.norm(lab_blurred(X, 8) - lab_blurred(T, 8), axis=-1), mask),
        'lpips': deep(X, T, mask, 'lpips'),
        'dists': deep(X, T, mask, 'dists'),
        **structure(xyz2lab(linear(project_eye(X)) @ SRGB2XYZ.T), xyz2lab(linear(project_eye(T)) @ SRGB2XYZ.T), mask),
        'ms_dssim_eye': ms_dssim(project_eye(X), project_eye(T), mask, 0, (1, 2, 4)),
        **drain(X, T, mask),
        'lpips_eye': deep(project_eye(X), project_eye(T), mask, 'lpips'),
        'dists_eye': deep(project_eye(X), project_eye(T), mask, 'dists'),
    }


# the full judge v3 (fit.py, 2026-10-01): LPIPS through the converter's eye and the lightness seam step, both picked by
# the data once the first optimum's counterexamples joined the labels; LPIPS held at the heaviest weight that costs no
# agreement. Fitted on the painted segments, the user's votes and the counterexamples, all visible through the eye
JUDGE = {
    'lpips_eye': 16.41,
    'seam_L_1': 1.459,
}


# the fast judge v3 (fit.py --fast): the seam steps in lightness and colour, anchored by the share of cells shown
# grey where the picture is chromatic (the first optimum drained the colour); numpy and OpenCV only, for the app
JUDGE_FAST = {
    'seam_ab_1': 0.4145,
    'seam_L_1': 1.234,
    'chroma_deficit_eye': 0.004768,
    'grey_share': 13.95,
}


def _scielab_terms(X, T, mask):
    sx, st = scielab(X), scielab(T)
    dE, ct = np.linalg.norm(sx - st, axis=-1), chroma(st)
    return {'scielab_dE': masked_mean(dE, mask), 'chroma_deficit': masked_mean(np.maximum(ct - chroma(sx), 0), mask),
            'hue_angle': masked_mean(hue_angle(sx, st) * ct / (ct + 20), mask)}


_eye_lab = lambda img: xyz2lab(linear(project_eye(img)) @ SRGB2XYZ.T)
# one metric at a time, for searches that score many candidates (all_metrics computes every one); the rest fall back
# to all_metrics
TERMS = {
    'seam_excess': lambda X, T, m: seam_excess(lab_blurred(X, 2.0), lab_blurred(T, 2.0), m),
    **{f'seam_{part}_{s:g}': (lambda s, sl: lambda X, T, m: seam_excess(lab_blurred(X, s)[..., sl],
                                                                        lab_blurred(T, s)[..., sl], m))(s, sl)
       for s in (1.0, 2.0, 4.0) for part, sl in (('L', slice(0, 1)), ('ab', slice(1, 3)))},
    'neighbour_excess': neighbour_excess,
    'ms_dssim_eye': lambda X, T, m: ms_dssim(project_eye(X), project_eye(T), m, 0, (1, 2, 4)),
    **{k: (lambda k: lambda X, T, m: structure(_eye_lab(X), _eye_lab(T), m)[k])(k)
       for k in ('detail_deficit_0', 'detail_deficit_1', 'detail_deficit_2', 'gmsd_eye', 'gmsm_eye', 'dssim_eye')},
    **{k: (lambda k: lambda X, T, m: drain(X, T, m)[k])(k)
       for k in ('grey_share', 'chroma_deficit_eye', 'colourfulness_deficit')},
    **{k: (lambda k: lambda X, T, m: _scielab_terms(X, T, m)[k])(k) for k in ('scielab_dE', 'chroma_deficit', 'hue_angle')},
    'hue_family_miss': hue_family_miss,
    'lpips_eye': lambda X, T, m: deep(project_eye(X), project_eye(T), m, 'lpips'),
    'dists_eye': lambda X, T, m: deep(project_eye(X), project_eye(T), m, 'dists'),
}
BATCHED = {'lpips_eye': 'lpips', 'dists_eye': 'dists'}   # network terms, one call for many candidates


def term(name, X, T, mask):
    return TERMS[name](X, T, mask) if name in TERMS else all_metrics(X, T, mask)[name]


def deep_batch(Xs, T, mask, name):
    """LPIPS or DISTS of each of Xs against T, one network call; as deep() for each."""
    import piq
    import torch
    key = (name, 'batch')
    if key not in _DEEP:
        torch.set_num_threads(1)
        _DEEP[key] = {'lpips': piq.LPIPS, 'dists': piq.DISTS}[name](reduction='none').eval()
    ys, xs = np.nonzero(mask)
    crop = lambda a: np.repeat(np.repeat(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 2, 0), 2, 1)
    xb = torch.from_numpy(np.stack([crop(x) for x in Xs])).permute(0, 3, 1, 2).float()
    tb = torch.from_numpy(crop(T)[None].copy()).permute(0, 3, 1, 2).float().expand_as(xb)
    with torch.no_grad():
        return _DEEP[key](xb, tb).reshape(-1).numpy().astype(float)


def score(judge, X, T, mask=None) -> float:
    """A judge (metric -> weight) of the rendered colouring X against the tuned picture T over mask (the whole picture
    by default), lower is better."""
    mask = np.ones(X.shape[:2], bool) if mask is None else mask
    return sum(w * term(k, X, T, mask) for k, w in judge.items())


def score_batch(judge, Xs, T, mask) -> np.ndarray:
    """score() of each of Xs, the network terms in one call."""
    out = np.zeros(len(Xs))
    for k, w in judge.items():
        if k in BATCHED:
            out += w * deep_batch([project_eye(x) for x in Xs], project_eye(T), mask, BATCHED[k])
        else:
            out += w * np.array([term(k, x, T, mask) for x in Xs])
    return out


def judge_fast_score(X, T, mask=None) -> float:
    """The fast judge (JUDGE_FAST), lower is better: numpy and OpenCV only."""
    return score(JUDGE_FAST, X, T, mask)


def judge_score(X, T, mask=None) -> float:
    """The full judge (JUDGE), lower is better."""
    return score(JUDGE, X, T, mask)
