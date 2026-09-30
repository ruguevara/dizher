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
        'lpips_eye': deep(project_eye(X), project_eye(T), mask, 'lpips'),
        'dists_eye': deep(project_eye(X), project_eye(T), mask, 'dists'),
    }


# judge v2 (fit.py, 2026-09-30): the seam term, the one the data hold to, and LPIPS through the converter's eye as the
# anchor that keeps a smooth but wrong colouring from winning, at the heaviest weight that costs no agreement with the
# user; fitted on the painted segments and the user's votes
JUDGE = {
    'seam_excess': 0.8827,
    'lpips_eye': 27.06,
}


def judge_score(X, T, mask=None) -> float:
    """The judge of the rendered colouring X against the tuned picture T over mask (the whole picture by default),
    lower is better; the lower of two sides by 1 is preferred at odds of about e to 1."""
    m = all_metrics(X, T, np.ones(X.shape[:2], bool) if mask is None else mask)
    return sum(w * m[k] for k, w in JUDGE.items())
