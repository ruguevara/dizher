"""Prior pairs from artists (IDEAS, open): the zxart.ee top 100 standard screens by votes, and what their cell
borders look like in the seam research's terms (seams.py, seamfit.SEAM): where an artist changes the pair, how the
change is made so that it does not show, and how that depends on the lightness. A null for the pixels along the
border: the same cells with the border's dots at the cell's own mean level (E at the mean ink count), so the excess of
the real border over it says how much the artist hides at the border itself.

    python research/pairs/zxart.py fetch    the top N (TOP) standard pictures by votes: the index to
                                            rounds/zxart/top100.json (in git), the screens to data/zxart/ID.scr
    python research/pairs/zxart.py stats    the priors: the pairs used, the borders (same set, a shared colour, none)
                                            by lightness, the seam terms on the changed borders against the null,
                                            the bright/dim switches; the per-border table to data/zxart/seams.npz
    python research/pairs/zxart.py recover [K [W [artists]]]   the recovery test: each screen through the user's eye (views.seen2
                                            at 1x; K times its sigmas, 1) as a new project's source, Select pairs at the defaults without DBS; per
                                            cell the shown set against the artist's, per border the change rate, the
                                            shared colour, E and the score, the converter against the artist on the
                                            same borders; data/zxart/recover[-xK[-sharedW]]/ID.png, the table ID.npz.
                                            W: a shared-colour term (step 6.1) added to the selection: a change of
                                            pair that keeps no colour of the neighbour's costs W x coherence x the
                                            seam's smoothness weight (0 across the target's edges); 100: as good as
                                            forbidden on flat seams. artists (step 6.2): the coherence term's V
                                            replaced by the artists' transition table (stats: data/zxart/transitions.npz),
                                            -log(count / expected) clipped at 0, scaled to 0..1, by the pairs' shown
                                            sets (both colours shown); W then 0 for the term alone
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seamfit   # noqa: E402
import seams   # noqa: E402
from acuity import EYE   # noqa: E402
from common import DATA, Project   # noqa: E402
from views import seen2   # noqa: E402
from metrics import SRGB2XYZ, linear, xyz2lab   # noqa: E402
from dizher.converter.energy import GROUPS, OFFSETS, optimise, pair_dissimilarity, surface_binding   # noqa: E402
from dizher.platforms.zxspectrum import ZXPalette   # noqa: E402
from dizher.platforms.zxspectrum.scr import _row_offsets, to_scr   # noqa: E402

OUT, ROUND = DATA / 'zxart', Path(__file__).resolve().parent / 'rounds' / 'zxart'
TOP = 100
API = 'https://zxart.ee/api/export:zxPicture/filter:zxPictureType=standard/limit:100/start:{}/order:votes,desc'
UA = {'User-Agent': 'dizher research (pair-selection priors)'}
YBINS = (0, 0.25, 0.5, 0.75, 1.01)


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    index, start = [], 0
    while len(index) < TOP:
        with urllib.request.urlopen(urllib.request.Request(API.format(start), headers=UA)) as r:
            page = json.load(r)['responseData']['zxPicture']
        for p in page:
            f = OUT / f"{p['id']}.scr"
            if not f.exists():
                time.sleep(0.5)
                url = urllib.parse.quote(p['originalUrl'], safe=':/')   # some names are Cyrillic
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
                    data = r.read()
                if len(data) == 6912 + 17:   # hobeta header
                    data = data[17:]
                if len(data) != 6912:
                    print('skip', p['id'], p['title'], len(data), 'bytes')
                    continue
                f.write_bytes(data)
            index.append({k: p.get(k) for k in ('id', 'title', 'authorIds', 'year', 'rating', 'url', 'originalUrl')})
            if len(index) == TOP:
                break
        start += 100
    (ROUND / 'top100.json').write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print(len(index), 'screens in', OUT)


def decode(data: bytes):
    """(192, 256) bool ink, (24, 32, 2) palette indexes (paper, ink), 0..7 and bright 8..15."""
    d = np.frombuffer(data, np.uint8)
    bitmap = np.unpackbits(np.stack([d[o:o + 32] for o in _row_offsets()]), axis=1).astype(bool)
    a = d[6144:6912].reshape(24, 32).astype(int)
    bright = (a >> 6 & 1) << 3
    return bitmap, np.stack([(a >> 3 & 7) | bright, (a & 7) | bright], -1)


def shown(bitmap, idx):
    """(24, 32) a code of the set of colours each cell shows: black's two indexes one, a colour with no pixel out,
    the two sorted (a single colour twice)."""
    cells = bitmap.reshape(24, 8, 32, 8)
    has_ink, has_paper = cells.any((1, 3)), ~cells.all((1, 3))
    merge = lambda i: np.where(i == 8, 0, i)
    paper, ink = merge(idx[..., 0]), merge(idx[..., 1])
    a = np.where(has_paper, paper, ink)
    b = np.where(has_ink, ink, paper)
    return np.minimum(a, b) * 16 + np.maximum(a, b)


def borders(bitmap, idx):
    """Per 4-neighbour border (h then v, as seams.py): the columns of its table."""
    X = np.where(bitmap[..., None], ZX[idx[..., 1]].repeat(8, 0).repeat(8, 1), ZX[idx[..., 0]].repeat(8, 0).repeat(8, 1))
    code, L = shown(bitmap, idx), seams.lab(X)
    lab_mean = xyz2lab(linear(X).reshape(24, 8, 32, 8, 3).mean((1, 3)) @ SRGB2XYZ.T)
    ends = seams.lab(ZX[idx])[..., 0]                                       # (24, 32, 2) paper, ink L*
    cells = bitmap.reshape(24, 8, 32, 8)
    solid = np.exp(-L[..., 0].reshape(24, 8, 32, 8).std((1, 3)) / seams.SOLID)
    level = cells.mean((1, 3))                                              # the cell's ink share
    out = {}
    for name, axis in (('h', 0), ('v', 1)):
        a = (slice(0, -1), slice(None)) if axis == 0 else (slice(None), slice(0, -1))
        b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        near = cells[:, 7, :, :].mean(-1) if axis == 0 else cells[:, :, :, 7].mean(1)   # the border's ink share
        far = cells[:, 0, :, :].mean(-1) if axis == 0 else cells[:, :, :, 0].mean(1)
        dLi, dLp = (np.abs(ends[a][..., k] - ends[b][..., k]) for k in (1, 0))
        E = lambda na, fb: ((na + fb) / 2 * dLi + (1 - (na + fb) / 2) * dLp)   # the switch's lightness change
        cols = dict(
            changed=code[a] != code[b],
            shared=(code[a] // 16 == code[b] // 16) | (code[a] // 16 == code[b] % 16) | (code[a] % 16 == code[b] // 16)
            | (code[a] % 16 == code[b] % 16),
            bright_only=(code[a] != code[b]) & (code[a] // 16 % 8 == code[b] // 16 % 8) & (code[a] % 8 == code[b] % 8),
            Y=(lab_mean[a][..., 0] + lab_mean[b][..., 0]) / 200,
            M=np.linalg.norm(lab_mean[a][..., 1:] - lab_mean[b][..., 1:], axis=-1),
            S=np.abs(solid[a] - solid[b]),
            E=E(near[a], far[b]), E0=E(level[a], level[b]),
            pair_a=idx[a][..., 0] * 16 + idx[a][..., 1], pair_b=idx[b][..., 0] * 16 + idx[b][..., 1],
            code_a=code[a], code_b=code[b])
        out[name] = {k: v.ravel() for k, v in cols.items()}
    return {k: np.concatenate([out['h'][k], out['v'][k]]) for k in out['h']}


ZX = ZXPalette().as_float()
W = seamfit.SEAM


def score(t):
    return (W['M cell means ab'] * t['M'] + t['Y'] * (W['E switch L x Y cell lightness L'] * t['E']
                                                       + W['S solid step L x Y cell lightness L'] * t['S']))


def stats():
    tables, cells = [], []
    for f in sorted(OUT.glob('*.scr')):
        bitmap, idx = decode(f.read_bytes())
        t = borders(bitmap, idx)
        t['pic'] = np.full(len(t['Y']), int(f.stem))
        tables.append(t)
        cells.append(shown(bitmap, idx).ravel())
    t = {k: np.concatenate([x[k] for x in tables]) for k in tables[0]}
    cells = np.concatenate(cells)
    np.savez(OUT / 'seams.npz', **t)
    n = len(cells)
    single = cells // 16 == cells % 16
    bright = (cells // 16 >= 8) | (cells % 16 >= 8)
    print(f'{len(tables)} screens, {n} cells: one colour {single.mean():.2f}, bright {bright.mean():.2f}')
    letter = lambda i: 'kbrmgcyw'[i % 8].upper() if i >= 8 else 'kbrmgcyw'[i % 8]
    pair = lambda c: f'{letter(c // 16)}/{letter(c % 16)}'
    top = np.unique(cells[~single], return_counts=True)
    order = np.argsort(-top[1])[:12]
    print('pairs shown (two colours):', ' '.join(f'{pair(top[0][i])} {top[1][i] / (~single).sum():.3f}' for i in order))

    ch, sh = t['changed'], t['shared']
    print(f"\nborders {len(ch)}: pair changes {ch.mean():.3f}; of them with a shared colour {sh[ch].mean():.3f}, "
          f"bright/dim only {t['bright_only'][ch].mean():.3f}")
    print('by lightness Y (the two cells\' mean L*/100): share of borders, the change rate, shared | change, '
          'and on the changes mean E at the border vs E0 at the cells\' levels, M, S, the score vs the null')
    sc, sc0 = score(t), score({**t, 'E': t['E0']})
    for lo, hi in zip(YBINS, YBINS[1:]):
        b = (t['Y'] >= lo) & (t['Y'] < hi)
        c = b & ch
        print(f'  Y {lo:.2f}-{min(hi, 1):.2f}: {b.mean():.2f} of borders, change {ch[b].mean():.3f}, '
              f'shared {sh[c].mean():.2f}, E {t["E"][c].mean():5.1f} vs E0 {t["E0"][c].mean():5.1f}, '
              f'M {t["M"][c].mean():5.1f}, S {t["S"][c].mean():.2f}, score {sc[c].mean():.2f} vs {sc0[c].mean():.2f}')
    for label, sel in (('shared colour', ch & sh), ('no shared colour', ch & ~sh)):
        print(f'{label}: {sel.sum()} changes, E {t["E"][sel].mean():.1f} vs E0 {t["E0"][sel].mean():.1f} '
              f'(E = 0 on {(t["E"][sel] == 0).mean():.2f}, E0 = 0 on {(t["E0"][sel] == 0).mean():.2f}), '
              f'M {t["M"][sel].mean():.1f}, score {sc[sel].mean():.2f} vs {sc0[sel].mean():.2f}')
    print('\nthe most frequent changes (shown sets a | b, count, share of changes, mean score):')
    key = np.minimum(t['code_a'], t['code_b']) * 256 + np.maximum(t['code_a'], t['code_b'])
    u, cnt = np.unique(key[ch], return_counts=True)
    for i in np.argsort(-cnt)[:15]:
        s = ch & (key == u[i])
        print(f'  {pair(u[i] // 256):>5} | {pair(u[i] % 256):<5} {cnt[i]:5d} {cnt[i] / ch.sum():.3f} {sc[s].mean():.2f}')
    transitions(t, key, u, cnt, sc, pair)


def transitions(t, key, u, cnt, sc, pair):
    """The transition prior: each change of shown sets a | b, its count over the count expected if a picture's cells
    changed pair at random (the picture's shown-set frequencies, its change rate), against the coherence term's V
    (the sets as pairs, dark to bright) and the seam score: which measure tells what artists do."""
    ch = t['changed']
    exp = np.zeros(len(u))
    for pic in np.unique(t['pic']):
        m = t['pic'] == pic
        codes, n = np.unique(np.concatenate([t['code_a'][m], t['code_b'][m]]), return_counts=True)
        f = dict(zip(codes, n / n.sum()))
        a, b = u // 256, u % 256
        pa, pb = np.array([f.get(c, 0) for c in a]), np.array([f.get(c, 0) for c in b])
        exp += ch[m].sum() * pa * pb * np.where(a != b, 2, 1)
    codes = np.unique(np.concatenate([u // 256, u % 256]))
    rgb = ZX[np.stack([codes // 16, codes % 16], -1)]                      # (n, 2, 3)
    order = np.argsort(rgb.sum(-1), axis=-1)
    V = pair_dissimilarity(np.take_along_axis(rgb, order[..., None], 1))
    at = {c: i for i, c in enumerate(codes)}
    v = np.array([V[at[k // 256], at[k % 256]] for k in u])
    mean = np.array([sc[ch & (key == k)].mean() for k in u])
    lr = np.log((cnt + 1) / (exp + 1))
    full = np.zeros((len(codes), len(codes)))
    full[[at[k // 256] for k in u], [at[k % 256] for k in u]] = lr
    full = np.maximum(full, full.T)
    np.savez(OUT / 'transitions.npz', codes=codes, lr=full)   # the table for artists_V
    w = cnt >= 30
    print(f'\ntransitions: {len(u)} kinds, {w.sum()} with 30+ counts; log(observed / expected) against V rho '
          f'{spearmanr(lr[w], v[w]).statistic:.2f}, against the score {spearmanr(lr[w], mean[w]).statistic:.2f}; '
          f'V against the score {spearmanr(v[w], mean[w]).statistic:.2f}')
    for label, idx in (('the most made', np.argsort(-np.where(w, lr, -9))[:8]), ('the most avoided', np.argsort(np.where(w, lr, 9))[:8])):
        print(f'{label} (count, expected, log ratio, V, score):')
        for i in idx:
            print(f'  {pair(u[i] // 256):>5} | {pair(u[i] % 256):<5} {cnt[i]:5d} {exp[i]:7.0f} {lr[i]:5.2f} {v[i]:.2f} {mean[i]:.2f}')


def recover_job(job) -> dict:
    """The artist's and the converter's border tables and shown sets for one screen, (file, K, folder)."""
    f, K, W, V, folder = job
    bitmap, idx = decode(f.read_bytes())
    X = np.where(bitmap[..., None], ZX[idx[..., 1]].repeat(8, 0).repeat(8, 1), ZX[idx[..., 0]].repeat(8, 0).repeat(8, 1))
    png = folder / f'{f.stem}.png'
    cv2.imwrite(str(png), cv2.cvtColor(seen2((X * 255).round().astype(np.uint8), *(s * K for s in EYE), k=1),
                                       cv2.COLOR_RGB2BGR))
    p = Project(str(png), optimise={'enabled': False})
    labels = p.selection.best_attr_indexes if not (W or V) else select_shared(p, W, V)
    idx2 = p.pairs[labels]
    X2 = p.convert() if not (W or V) else p.render(labels)
    bitmap2 = np.abs(X2 - ZX[idx2[..., 1]].repeat(8, 0).repeat(8, 1)).sum(-1) < 1e-3
    cv2.imwrite(str(png.with_name(f'{f.stem}-converter.png')), cv2.cvtColor((X2 * 255).round().astype(np.uint8),
                                                                           cv2.COLOR_RGB2BGR))
    out = dict(artist=borders(bitmap, idx), converter=borders(bitmap2, idx2),
               cells=np.stack([shown(bitmap, idx).ravel(), shown(bitmap2, idx2).ravel()]))
    np.savez(png.with_suffix('.npz'), **{f'{who}_{k}': v for who in ('artist', 'converter') for k, v in out[who].items()},
             cells=out['cells'])
    return out


def artists_V(pairs: np.ndarray) -> np.ndarray:
    """(P, P) the artists' transition table as the coherence term's V: -log(count / expected) of two shown sets
    side by side, clipped at 0 (made more than expected: free), scaled to 0..1; a set the artists never show costs
    the most. The pairs by their shown sets with both colours shown, black's two indexes one."""
    z = np.load(OUT / 'transitions.npz')
    cost = np.clip(-z['lr'], 0, None)
    cost /= cost.max()
    at = {int(c): i for i, c in enumerate(z['codes'])}
    col = np.where(pairs == 8, 0, pairs)
    code = col.min(1) * 16 + col.max(1)
    i = np.array([at.get(int(c), -1) for c in code])
    V = np.where((i[:, None] >= 0) & (i[None] >= 0), cost[i[:, None], i[None]], 1.0)
    V[code[:, None] == code[None]] = 0
    return V.astype(np.float32)


def select_shared(p: Project, W: float, V: bool = False) -> np.ndarray:
    """Select pairs' DP at the project's settings (as selection.select) plus the shared-colour term: W x coherence
    on each flat seam whose two pairs share no colour (black's two indexes one), times the seam's smoothness; V: the
    artists' table in place of the pairs' attribute distance."""
    c = p.selection
    e, w = c.energy, c.energy.weights
    D = e.unary()
    if c.surface > 0:
        D = D + surface_binding(D, c.image_rgb, c.cell, c.surface)
    S = {off: sum(w[g] * e.S[g][off] for g in GROUPS) for off in OFFSETS}
    Lh, Lv = e.seam_smoothness()
    col = np.where(p.pairs == 8, 0, p.pairs)                                        # (P, 2)
    cost = (W * c.coherence * ~(col[:, None, :, None] == col[None, :, None, :]).any((2, 3))).astype(np.float32)
    S[(1, 0)] = S[(1, 0)] + Lh[..., None, None] * cost
    S[(0, 1)] = S[(0, 1)] + Lv[..., None, None] * cost
    return optimise(D, S, artists_V(p.pairs) if V else c.pair_dissimilarity, Lh, Lv, c.coherence)


def recover():
    K = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    V = len(sys.argv) > 4 and sys.argv[4] == 'artists'
    folder = OUT / ('recover' + (f'-x{K:g}' if K != 1 else '') + (f'-shared{W:g}' if W else '') + ('-artists' if V else ''))
    folder.mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        res = pool.map(recover_job, [(f, K, W, V, folder) for f in sorted(OUT.glob('*.scr'))])
    print(f'blur x{K:g}' + (f', shared-colour term {W:g} x coherence' if W else '') + (", the artists' V" if V else ''))
    cells = np.concatenate([r['cells'] for r in res], 1)
    T = {who: {k: np.concatenate([r[who][k] for r in res]) for k in res[0][who]} for who in ('artist', 'converter')}
    print(f"{len(res)} screens; cells with the artist's shown set {np.mean(cells[0] == cells[1]):.3f}")
    a, c = T['artist'], T['converter']
    for who, t in T.items():
        ch, sc = t['changed'], score(t)
        print(f"{who:>9}: changes {ch.mean():.3f}, shared {t['shared'][ch].mean():.3f}, bright/dim only "
              f"{t['bright_only'][ch].mean():.3f}; on the changes E {t['E'][ch].mean():5.1f} (E = 0 on "
              f"{(t['E'][ch] == 0).mean():.2f}), M {t['M'][ch].mean():5.1f}, S {t['S'][ch].mean():.2f}, "
              f"score {sc[ch].mean():.2f}; no shared colour: {(ch & ~t['shared']).mean():.3f} of borders, "
              f"score {sc[ch & ~t['shared']].mean():.2f}")
    both, only_c, only_a = a['changed'] & c['changed'], ~a['changed'] & c['changed'], a['changed'] & ~c['changed']
    sc = score(c)
    print(f"borders: both change {both.mean():.3f}, the converter alone {only_c.mean():.3f} (its score there "
          f"{sc[only_c].mean():.2f}, no shared colour {(~c['shared'][only_c]).mean():.2f}), the artist alone "
          f"{only_a.mean():.3f}")
    for lo, hi in zip(YBINS, YBINS[1:]):
        b = (a['Y'] >= lo) & (a['Y'] < hi)
        print(f"  Y {lo:.2f}-{min(hi, 1):.2f}: change artist {a['changed'][b].mean():.3f} converter "
              f"{c['changed'][b].mean():.3f}, shared {a['shared'][b & a['changed']].mean():.2f} vs "
              f"{c['shared'][b & c['changed']].mean():.2f}, score {score(a)[b & a['changed']].mean():.2f} vs "
              f"{sc[b & c['changed']].mean():.2f}")


def check():
    """decode inverts to_scr; shown merges the blacks and drops a colour with no pixel."""
    rng = np.random.default_rng(0)
    bitmap, idx = rng.random((192, 256)) < 0.5, rng.integers(0, 16, (24, 32, 2))
    idx[..., 1] = idx[..., 0] // 8 * 8 + idx[..., 1] % 8   # one brightness per cell, as the attribute byte holds
    b, i = decode(to_scr(bitmap, idx))
    assert (b == bitmap).all() and (i == idx).all()
    bitmap[:8, :8], idx[0, 0] = True, (8, 10)                    # all ink: bright red alone
    bitmap[:8, 8:16], idx[0, 1] = False, (3, 8)                  # all paper: magenta alone
    bitmap[:8, 16:24], idx[0, 2] = rng.random((8, 8)) < 0.5, (8, 7)   # bright black paper, white ink
    assert shown(bitmap, idx)[0, :3].tolist() == [10 * 17, 3 * 17, 7]
    print('ok')


if __name__ == '__main__':
    {'fetch': fetch, 'stats': stats, 'recover': recover, 'check': check}[sys.argv[1]]()
