"""PLAN step 6.4: the selection energy's weights fitted to the artists (the zxart.ee top 100, zxart.py), the seam
score's terms among them, pixels and all. The artist's screen through the user's eye is the target (as the recovery
test, zxart.recover); the converter's own tables at that target give, for every cell and every pair it could take,
the unary (the eye-model colour error), the eye-model seam terms, the coherence term and the seam score's terms (E at
the border's pixels, M, the solidity step, each times the lightness Y). The artist's pair must beat the alternatives
the colour allows (the ALTS lowest by the unary) when the neighbours keep the artist's pairs: a ranking loss on the
differences, the unary's weight 1 (the unit), the rest >= 0. So every weight comes out in the energy's own units: the
artists' implied coherence, and the seam score's weight against the unary.

    python research/pairs/zxfit.py [K]    the fit on all 100 screens at K times the eye's sigmas (1), and with half
                                          the screens out: each energy's pairwise accuracy on the other half (the
                                          artist's pair below the alternative); then the fitted score terms on the
                                          user's own data (seamfit's renders and rounds), against seamfit.SEAM
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seamfit   # noqa: E402
import seams   # noqa: E402
import selection   # noqa: E402
import zxart   # noqa: E402
from common import Project   # noqa: E402
from metrics import SRGB2XYZ, linear, xyz2lab   # noqa: E402
from dizher.converter.energy import GROUPS, OFFSETS, SEAM_COST, _ranges, surface_binding   # noqa: E402

FEATURES = ['unary', 'eye seam Luma', 'eye seam Chroma', 'coherence', 'E switch L', 'M cell means ab',
            'S solid step L', 'E switch L x Y', 'M cell means ab x Y', 'S solid step L x Y']
IN_SEAMFIT = {'E switch L': 'E switch L', 'M cell means ab': 'M cell means ab', 'S solid step L': 'S solid step L',
              'E switch L x Y': 'E switch L x Y cell lightness L',
              'M cell means ab x Y': 'M cell means ab x Y cell lightness L',
              'S solid step L x Y': 'S solid step L x Y cell lightness L'}
ALTS = 16   # alternatives per cell: the pairs lowest by the unary after the artist's


def terms(c, cells: dict, offset) -> dict:
    """selection.table's parts, (R', C', P, P) each: E, M, S and Y of the border between cell (r, c) at pair p and
    its neighbour at offset at pair q."""
    dr, dc = offset
    near, far = ('bottom', 'top') if dr else ('right', 'left')
    a = (slice(None), slice(0, cells['L'].shape[1] - dr), slice(0, cells['L'].shape[2] - dc))
    b = (slice(None), slice(dr, None), slice(dc, None))
    pick = lambda k, s: np.moveaxis(cells[k][s], 0, 2)
    ink = pick(near, a)[..., :, None] + pick(far, b)[..., None, :]
    dLi = np.abs(cells['ends'][:, None, 1] - cells['ends'][None, :, 1])
    dLp = np.abs(cells['ends'][:, None, 0] - cells['ends'][None, :, 0])
    E = (ink * dLi + (16 - ink) * dLp) / 16
    abT = xyz2lab(linear(c.image_rgb).reshape(cells['L'].shape[1], 8, cells['L'].shape[2], 8, 3).mean((1, 3))
                  @ SRGB2XYZ.T)[..., 1:]
    stepT = np.linalg.norm(abT[a[1:]] - abT[b[1:]], axis=-1)[..., None, None]
    M = np.maximum(np.linalg.norm(pick('ab', a)[..., :, None, :] - pick('ab', b)[..., None, :, :], axis=-1) - stepT, 0)
    Y = (pick('L', a)[..., :, None] + pick('L', b)[..., None, :]) / 200
    S = np.abs(pick('solid', a)[..., :, None] - pick('solid', b)[..., None, :])
    same = np.eye(E.shape[-1], dtype=bool)
    out = dict(E=E, M=M, S=S, Y=Y)
    for k in ('E', 'M', 'S'):
        out[k][..., same] = 0
    return {k: v.astype(np.float32) for k, v in out.items()}


def around(T: np.ndarray, labels: np.ndarray, offset) -> np.ndarray:
    """(R, C, P) the sum over the cells' neighbours at this offset (both roles) of T[border, p, the neighbour's
    label]: the pairwise cost a cell at pair p would pay with the neighbours as labelled."""
    R, C = labels.shape
    dr, dc = offset
    a, b = _ranges(dr, dc, R, C), _ranges(-dr, -dc, R, C)
    G = np.zeros((R, C, T.shape[-1]), np.float32)
    G[a] += np.take_along_axis(T, labels[b][..., None, None], axis=-1)[..., 0]           # (r, c) first, at p
    G[b] += np.take_along_axis(T, labels[a][..., None, None], axis=-2)[..., 0, :]        # (r, c) second, at p
    return G


def job(args) -> np.ndarray:
    """(n, k) feature differences, alternative minus the artist's, for one screen: each cell, the ALTS pairs lowest
    by the unary besides the artist's."""
    f, K = args
    bitmap, idx = zxart.decode(f.read_bytes())
    png = zxart.OUT / ('recover' if K == 1 else f'recover-x{K:g}') / f'{f.stem}.png'
    assert png.exists(), f'{png}: run zxart.py recover {K:g} first'
    p = Project(str(png), optimise={'enabled': False})
    c, e = p.selection, p.selection.energy
    key = {tuple(sorted(pair)): l for l, pair in enumerate(p.pairs.tolist())}
    labels = np.array([[key[tuple(sorted(cell))] for cell in row] for row in idx.tolist()])
    R, C = labels.shape
    D = e.unary()
    if c.surface > 0:
        D = D + surface_binding(D, c.image_rgb, c.cell, c.surface)
    F = [np.moveaxis(D, 0, -1)]                                                        # (R, C, P)
    for g in GROUPS:
        F.append(sum(around(e.S[g][off], labels, off) for off in OFFSETS))
    Lh, Lv = e.seam_smoothness()
    V = c.pair_dissimilarity
    F.append(SEAM_COST * (around(V[None, None] * Lh[..., None, None], labels, (1, 0))
                          + around(V[None, None] * Lv[..., None, None], labels, (0, 1))))
    cells = selection.per_cell(c)
    parts = {k: np.zeros((R, C, len(p.pairs)), np.float32) for k in ('E', 'M', 'S', 'EY', 'MY', 'SY')}
    for off in ((1, 0), (0, 1)):
        t = terms(c, cells, off)
        for k in ('E', 'M', 'S'):
            parts[k] += around(t[k], labels, off)
            parts[k + 'Y'] += around(t[k] * t['Y'], labels, off)
    F += [parts[k] for k in ('E', 'M', 'S', 'EY', 'MY', 'SY')]
    F = np.stack(F, -1)                                                                # (R, C, P, k)
    mine = np.take_along_axis(F, labels[..., None, None], axis=2)                      # (R, C, 1, k)
    d = F - mine
    d[np.arange(R)[:, None], np.arange(C)[None], labels] = np.inf                       # the artist's out
    order = np.argsort(d[..., 0], axis=-1)[..., :ALTS]
    return np.take_along_axis(d, order[..., None], axis=2).reshape(-1, F.shape[-1])


CURRENT = np.array([1, 1, 1, 2] + [0] * 6, float)   # the current energy: unary, eye seams, Halftoned's coherence 2
TAUS = (0.02, 0.1, 0.5)   # the bounded loss's temperature, in units of the median unary difference


def fit(d: np.ndarray, tau: float) -> np.ndarray:
    """w >= 0, the unary's 1: a bounded ranking loss over the differences, sigmoid(-margin / tau), so the rows no
    energy can rank (the artist's content choices) do not drag the rest; from the current energy's weights."""
    scale = d.std(0) + 1e-9
    x = d / scale
    t = tau * np.median(d[:, 0])
    loss = lambda w: (1 / (1 + np.exp(np.clip((x @ w) / t, -50, 50)))).mean() + 1e-4 * w[1:] @ w[1:]
    w0 = CURRENT * scale
    bounds = [(scale[0], scale[0])] + [(0, None)] * (d.shape[1] - 1)
    return minimize(loss, w0, method='L-BFGS-B', bounds=bounds).x / scale


def accuracy(w, d) -> float:
    """The share of alternatives above the artist's pair by this energy."""
    return float((d @ w > 0).mean())


def main():
    K = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    files = sorted(zxart.OUT.glob('*.scr'))
    with Pool(8) as pool:
        ds = pool.map(job, [(f, K) for f in files])
    d = np.concatenate(ds)
    print(f'{len(ds)} screens at x{K:g}, {len(d)} (cell, alternative) rows, {ALTS} alternatives a cell')
    rng = np.random.default_rng(0)
    half = rng.permutation(len(ds)) < len(ds) // 2
    unary = np.eye(len(FEATURES))[0]
    score = CURRENT.copy()
    for f, k in (('E switch L x Y', 'E switch L x Y cell lightness L'), ('M cell means ab', 'M cell means ab'),
                 ('S solid step L x Y', 'S solid step L x Y cell lightness L')):
        score[FEATURES.index(f)] = selection.WEIGHT * zxart.SCALE * seamfit.SEAM[k]
    coh6 = CURRENT.copy()
    coh6[3] = 6
    print('\npairwise accuracy (the artist\'s pair below the alternative) on all screens: '
          f'the unary alone {accuracy(unary, d):.4f}; the current energy at coherence 0 {accuracy(CURRENT * [1, 1, 1, 0, 1, 1, 1, 1, 1, 1], d):.4f}, '
          f'2 {accuracy(CURRENT, d):.4f}, 6 {accuracy(coh6, d):.4f}; with the seam score at 16 {accuracy(score, d):.4f}')
    print('the fit at each temperature: on all screens, and fitted on half the screens, scored on the other half')
    fits = {}
    for tau in TAUS:
        w = fit(d, tau)
        fits[tau] = w
        out = []
        for sel in (~half, half):
            w_in = fit(np.concatenate([x for x, s in zip(ds, sel) if s]), tau)
            out.append(accuracy(w_in, np.concatenate([x for x, s in zip(ds, sel) if not s])))
        print(f'  tau {tau:4}: all {accuracy(w, d):.4f}; held out {out[0]:.4f} {out[1]:.4f}; weights '
              + ' '.join(f'{f.split()[0][:3]}{v:.3g}' for f, v in zip(FEATURES, w)))
    tau = max(TAUS, key=lambda t: accuracy(fits[t], d))
    w = fits[tau]
    print(f'\nthe weights in the energy\'s units (the unary 1), all screens, tau {tau}:')
    for f, v in zip(FEATURES, w):
        print(f'  {f:22} {v:8.4f}')
    # the fitted score terms on the user's own data
    rs, ps = seamfit.renders(), seamfit.rounds()
    scale = np.concatenate([r['F'] for r in rs]).std(0) + 1e-9
    ours = np.zeros(len(seamfit.FEATURES))
    for f, k in IN_SEAMFIT.items():
        ours[seamfit.FEATURES.index(k)] = w[FEATURES.index(f)]
    theirs = np.zeros(len(seamfit.FEATURES))
    for k, v in seamfit.SEAM.items():
        theirs[seamfit.FEATURES.index(k)] = v
    print('\nthe seam score\'s terms on the user\'s data (renders AUC, top k; rounds rho):')
    print(seamfit.line("  fitted to the artists", seamfit.scores(ours * scale, rs, ps, scale)))
    print(seamfit.line("  fitted to the user (SEAM)", seamfit.scores(theirs * scale, rs, ps, scale)))
    print('\nthe seam score\'s terms as weights of the score (SEAM\'s units: the selection at WEIGHT x SCALE per unit):')
    for f, k in IN_SEAMFIT.items():
        own = seamfit.SEAM.get(k, 0) * selection.WEIGHT * zxart.SCALE
        print(f'  {f:22} the artists {w[FEATURES.index(f)]:.4f}   the user at 16 {own:.4f}')


if __name__ == '__main__':
    main()
