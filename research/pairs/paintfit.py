"""The selection energy's weights fitted to the user's hand-painted projects (zxfit.py's inverse optimisation, the
painted cells as the target instead of the artists' screens). The painted cells are the Overpaint overrides of the
projects in tests/images (common.painted_projects); the neighbours keep the painting's labels; the painted pair must
beat ALL the other pairs (--alts K: the K lowest by the unary) on zxfit.FEATURES' differences, the unary's weight 1,
the rest >= 0 (zxfit.fit, the same TAUS). Per-picture feature differences are cached in data/paintfit/NAME.npz.

    python research/pairs/paintfit.py fit [--alts K]       leave one picture out: per picture the cells, the current
                                                           energy's pairwise accuracy at coherence 2 and 6, the fit
                                                           (on the other pictures) held out; then the weights on all
    python research/pairs/paintfit.py recover [--alts K]   the DP at the current energy (coherence 2) and at each
                                                           picture's leave-one-out weights: the share of painted cells
                                                           whose selected pair shows the painted colours (pair_bench's
                                                           match: blacks one, a solid cell takes any pair holding it)
    python research/pairs/paintfit.py search               a small pairwise energy fitted by recovery under the DP: the
                                                           unary alone plus Lh/Lv x Y^g x SEAM_COST x (a V + b no shared
                                                           colour + c dim-bright mix); grid coordinate descent, leave one
                                                           picture out; renders in data/paintfit/renders/
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection   # noqa: E402
from common import DATA, Project, painted_projects   # noqa: E402
from zxfit import CURRENT, FEATURES, TAUS, accuracy, around, fit, terms   # noqa: E402
from dizher.converter.energy import GROUPS, OFFSETS, SEAM_COST, optimise, surface_binding   # noqa: E402

CACHE = DATA / 'paintfit'


def project(name):
    return Project(name, optimise={'enabled': False})


def features(p, labels) -> np.ndarray:
    """(R, C, P, 10) zxfit's features of every cell at every pair, the neighbours at labels."""
    c, e = p.selection, p.selection.energy
    R, C = labels.shape
    D = e.unary()
    if c.surface > 0:
        D = D + surface_binding(D, c.image_rgb, c.cell, c.surface)
    F = [np.moveaxis(D, 0, -1)]
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
    return np.stack(F + [parts[k] for k in ('E', 'M', 'S', 'EY', 'MY', 'SY')], -1)


def job(name):
    """(d, rank): (n, 10) differences alternative minus the painted pair for each painted cell, all other pairs, and
    each row's rank of the alternative by the unary within its cell; cached."""
    f = CACHE / f'{name}.npz'
    if f.exists():
        z = np.load(f)
        return z['d'], z['rank']
    p = project(name)
    labels = p.labels()
    cells = sorted({(r, c) for r, c, *_ in p.graph['overpaint'].params.overrides})
    F = features(p, labels)
    ds, ranks = [], []
    for r, c in cells:
        d = F[r, c] - F[r, c, labels[r, c]]
        d = np.delete(d, labels[r, c], 0)
        ds.append(d)
        ranks.append(np.argsort(np.argsort(d[:, 0])))
    CACHE.mkdir(parents=True, exist_ok=True)
    d, rank = np.concatenate(ds), np.concatenate(ranks)
    np.savez_compressed(f, d=d, rank=rank)
    return d, rank


def best_fit(d):
    """The fit at the temperature with the best accuracy on its own rows."""
    fits = [fit(d, tau) for tau in TAUS]
    return max(fits, key=lambda w: accuracy(w, d))


def loo(data, alts):
    """Per picture the weights fitted on the others (the K lowest alternatives by the unary)."""
    keep = [d[r < alts] for d, r in data]
    return [best_fit(np.concatenate(keep[:i] + keep[i + 1:])) for i in range(len(data))], keep


def recover_job(args):
    """Share of painted cells the DP selects with each weight vector (the 10 features) as the painting."""
    name, ws = args
    p = project(name)
    c, e = p.selection, p.selection.energy
    labels = p.labels()
    cells = sorted({(r, c_) for r, c_, *_ in p.graph['overpaint'].params.overrides})
    black = lambda x: np.where(np.asarray(x) % 8 == 0, 0, x)
    shows = np.sort(black(p.pairs), axis=-1)            # label -> its two colours, blacks one
    D = e.unary()
    if c.surface > 0:
        D = D + surface_binding(D, c.image_rgb, c.cell, c.surface)
    Lh, Lv = e.seam_smoothness()
    cc = selection.per_cell(c)
    tt = {off: terms(c, cc, off) for off in ((0, 1), (1, 0))}
    out = []
    for w in ws:
        S = {off: sum(w[1 + i] * e.S[g][off] for i, g in enumerate(GROUPS)) for off in OFFSETS}
        for off, t in tt.items():
            S[off] = S[off] + w[4] * t['E'] + w[5] * t['M'] + w[6] * t['S'] + t['Y'] * (
                w[7] * t['E'] + w[8] * t['M'] + w[9] * t['S'])
        got = optimise(D, S, c.pair_dissimilarity, Lh, Lv, w[3])
        ok = []
        for r, col in cells:
            ref, mine = shows[labels[r, col]], shows[got[r, col]]
            ok.append((ref == mine).all() if ref[0] != ref[1] else ref[0] in mine)
        out.append(float(np.mean(ok)))
    return name, out


GRID = (0, 0.03, 0.1, 0.3, 1, 3)   # weights a, b, c in SEAM_COST units
RENDER = ('andy', 'diver-sunset', 'autumn', 'RC1')
_tab, _scores = {}, {}


def tables(name):
    """(cached) D, the pair tables V, no-shared, dim-bright, the seams' Lh/Lv x Y^g, the labels and the painted cells."""
    f = CACHE / f'tab-{name}.npz'
    if not f.exists():
        p = project(name)
        c, e = p.selection, p.selection.energy
        D = e.unary()
        if c.surface > 0:
            D = D + surface_binding(D, c.image_rgb, c.cell, c.surface)
        bl = np.where(p.pairs % 8 == 0, 0, p.pairs)
        ns = ~(bl[:, None, :, None] == bl[None, :, None, :]).any((2, 3))
        bright = (bl >= 8).any(1)
        Lh, Lv = e.seam_smoothness()
        Y = cv2.cvtColor(c.image_rgb.astype(np.float32), cv2.COLOR_RGB2Lab)[..., 0]
        Y = Y.reshape(D.shape[1], 8, D.shape[2], 8).mean((1, 3)) / 100
        Yh, Yv = (Y[1:] + Y[:-1]) / 2, (Y[:, 1:] + Y[:, :-1]) / 2
        cells = np.array(sorted({(r, c_) for r, c_, *_ in p.graph['overpaint'].params.overrides}))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, D=D, V=c.pair_dissimilarity, ns=ns, db=bright[:, None] != bright[None, :], Lh=Lh,
                            Lv=Lv, Lh1=Lh * Yh, Lv1=Lv * Yv, cur=p.labels(overrides=()), paint=p.labels(), cells=cells,
                            shows=np.sort(bl, -1))
    return _tab.setdefault(name, dict(np.load(f)))


def match(t, got):
    """Share of the painted cells whose pair in got shows the painting's colours (blacks one; solid takes any)."""
    ref, mine = (t['shows'][l[t['cells'][:, 0], t['cells'][:, 1]]] for l in (t['paint'], got))
    return float(np.where(ref[:, 0] == ref[:, 1], (mine == ref[:, :1]).any(1), (ref == mine).all(1)).mean())


def select(t, w, g):
    """The DP of the small energy: the unary and the borders' cost (a V + b no-shared + c dim-bright) x Lh/Lv x Y^g."""
    M = (SEAM_COST * (w[0] * t['V'] + w[1] * t['ns'] + w[2] * t['db']))[None, None].astype(np.float32)
    S = {off: M * t[L + '1' * g][..., None, None] for off, L in (((1, 0), 'Lh'), ((0, 1), 'Lv'))}
    S.update({off: np.zeros((1, 1) + M.shape[2:], np.float32) for off in ((1, 1), (1, -1))})
    S = {off: (v if off in ((1, 0), (0, 1)) else np.broadcast_to(v, (t['D'].shape[1] - 1, t['D'].shape[2] - 1, *v.shape[2:]))) for off, v in S.items()}
    return optimise(t['D'], S, t['V'], t['Lh'], t['Lv'], 0.0)


def score_job(a):
    name, w, g = a
    return match(tables(name), select(tables(name), w, g))


def search_job(a):
    name, w, g = a
    p = project(name)
    t = tables(name)
    X = [p.target(), p.render(t['cur'], optimise=dict(enabled=False)),
         p.render(select(t, w, g), optimise=dict(enabled=False)), p.render(t['paint'], optimise=dict(enabled=False))]
    sheet = np.round(np.concatenate(X, 1) * 255).astype(np.uint8).repeat(3, 0).repeat(3, 1)
    (CACHE / 'renders').mkdir(exist_ok=True)
    cv2.imwrite(str(CACHE / 'renders' / f'{name}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def search():
    names = painted_projects()
    with Pool(4) as pool:
        pool.map(tables, names)
        def scores(w, g):
            if (w, g) not in _scores:
                t0 = time.time()
                _scores[w, g] = np.array(pool.map(score_job, [(n, w, g) for n in names]))
                _scores[w, g, 't'] = time.time() - t0
            return _scores[w, g]
        def descend(idx):
            best, bg = None, 0
            for g in (0, 1):
                w = (1, 0, 0)
                for _ in range(2):
                    for k in range(3):
                        for v in GRID:
                            x = w[:k] + (v,) + w[k + 1:]
                            if scores(x, g)[idx].mean() > scores(w, g)[idx].mean() + 1e-9:
                                w = x
                if best is None or scores(w, g)[idx].mean() > scores(best, bg)[idx].mean():
                    best, bg = w, g
            return best, bg
        cur = [match(tables(n), tables(n)['cur']) for n in names]
        folds = [descend([j for j in range(len(names)) if j != i]) for i in range(len(names))]
        print(f'{"picture":16}{"current":>8}{"LOO":>7}   weights a b c, g (fitted on the other 8)')
        for i, n in enumerate(names):
            w, g = folds[i]
            print(f'{n:16}{cur[i]:8.3f}{scores(w, g)[i]:7.3f}   {w}, g={g}')
        allw, allg = descend(list(range(len(names))))
        print(f'\nmean: current {np.mean(cur):.3f}, LOO {np.mean([scores(*f)[i] for i, f in enumerate(folds)]):.3f}')
        print(f'all pictures: a b c {allw}, g={allg}: per picture {np.round(scores(allw, allg), 3).tolist()}, '
              f'mean {scores(allw, allg).mean():.3f}')
        ts = [v for k, v in _scores.items() if len(k) == 3]
        print(f'{len(ts)} evaluations (9 pictures each, Pool 4), {np.mean(ts):.1f} s each')
        pool.map(search_job, [(n, *folds[names.index(n)]) for n in RENDER])   # the held-out weights of each


def main():
    mode = sys.argv[1]
    if mode == 'search':
        return search()
    alts = int(sys.argv[sys.argv.index('--alts') + 1]) if '--alts' in sys.argv else 10 ** 9
    names = painted_projects()
    with Pool(4) as pool:
        data = pool.map(job, names)
        ws, keep = loo(data, alts)
        coh6 = CURRENT.copy()
        coh6[3] = 6
        if mode == 'fit':
            print(f'{"picture":16}{"cells":>6}{"cur@2":>8}{"cur@6":>8}{"fit-LOO":>9}')
            for n, (d, r), w, k in zip(names, data, ws, keep):
                print(f'{n:16}{int((r == 0).sum()):6}{accuracy(CURRENT, k):8.4f}{accuracy(coh6, k):8.4f}'
                      f'{accuracy(w, k):9.4f}')
            w = best_fit(np.concatenate(keep))
            print(f'\nweights on all {len(names)} pictures ({sum(len(k) for k in keep)} rows):')
            for f, v in zip(FEATURES, w):
                print(f'  {f:22} {v:8.4f}')
        else:
            print(f'{"picture":16}{"current@2":>10}{"fit-LOO":>9}')
            for n, (_, (a, b)) in zip(names, pool.map(recover_job, [(n, [CURRENT, w]) for n, w in zip(names, ws)])):
                print(f'{n:16}{a:10.3f}{b:9.3f}')


if __name__ == '__main__':
    main()
