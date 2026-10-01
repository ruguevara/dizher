"""Fit the judge: a logistic regression on the differences of the candidate metrics between the two sides of a pair.

    python research/pairs/fit.py          needs data/metrics.csv (score.py) and data/variants/*.json (variants.py)
    python research/pairs/fit.py --fast   the same without the torch metrics (DEEP), anchored by a structure term,
                                          for metrics.JUDGE_FAST; also how often it agrees with metrics.JUDGE

Labels, four sources, each weighted to the same total:
  B    the 229 painted segments: the painting (A) better than Select pairs' cells (B)
  C    the same segments: the painting better than the next best pair by the energy (C), assumed
       (B and C only where the change is plainly more than the dither's: VISIBLE; the user's votes all count)
  cal  the user's decisive votes in calibration round 1, both ways, repeats included
  O    optimum.py's counterexamples: the painting better than itself with a segment from a judge's optimum, assumed
       (visible changes only, as B and C; where the user judged a round's optima, rounds/optN/verdicts.json, only
       the pictures whose optimum they found worse than the painting)
and one only reported, not trained on (weight 0):
  F    flat.py's counterexamples: the painting better than its segment all on one pair, assumed
P(A better) = sigmoid(sum_k w_k (f_k(other) - f_k(A)) / s_k), s_k the RMS of that difference, no intercept, L2.
Every metric is an error, so its weight is >= 0, but for the FREE ones, whose better direction is not known: unbounded,
the fit learns "further from the picture is better" from these one-sided labels (the painting always leaves the
colorimetric optimum). Terms are added one at a time, up to MAX_TERMS, the one that most raises the
leave-one-image-out accuracy (the mean over the three sources) each time; the selection is repeated on BOOT
resamples of the images, and the judge keeps the terms picked in at least half of them (the first alone if none), as
the others come and go with the images drawn (a step gains about one vote). The labels hold nothing against a colouring
that is smooth but wrong, so a fidelity term (ANCHORS) is then fixed at a weight that costs at most SLACK
of that accuracy and keeps a black screen last on every whole picture, the one sharing most of the user's verdicts
on the last round's optimum (rounds/optN/verdicts.json). Prints the single metrics, the selection path, the anchor sweep, the judge's weights in the
metrics' own units for metrics.JUDGE, and the judge on whole pictures."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA   # noqa: E402

HERE = Path(__file__).resolve().parent
MAX_TERMS, L2, BOOT = 4, 1.0, 30
VISIBLE = 2   # a B or C pair counts when its change is at least this many times a re-dither's
DEEP = ('lpips', 'dists', 'lpips_eye', 'dists_eye')   # torch: out of the fast judge, which the app can run
COLOUR = ('grey_share', 'chroma_deficit_eye', 'colourfulness_deficit', 'hue_angle', 'hue_family_miss', 'chroma_deficit')
# ^ what the first optima gave up
FAST_ANCHORS = ('detail_deficit_1', 'detail_deficit_2', 'gmsd_eye', 'gmsm_eye', 'dssim_eye', 'ms_dssim_eye',
                'scielab_dE') + COLOUR
ANCHORS = ('lpips_eye', 'dists_eye', 'scielab_dE', 'blur_dE_4', 'blur_dE_8', 'blur_rmse', 'seam_ab_1', 'neighbour_excess') \
    + COLOUR   # the colour seams and cell-to-cell change: the full judge's round 3 optimum was noisy in colour
SLACK = 0.02   # the accuracy a fidelity anchor may cost
FREE = ('two_colour_share', 'dot_contrast')   # no known better direction; every other metric is an error, weight >= 0
TRAIN, SOURCES = ('B', 'C', 'cal', 'O'), ('B', 'C', 'cal', 'O', 'F')


def load():
    """(features, rows): a row is (source, image, {feature: f(other) - f(A)}, y), y = 1 when A is better."""
    table = {}
    for r in csv.DictReader(open(DATA / 'metrics.csv')):
        table.setdefault(r['id'], {})[r['variant']] = r
    variants = {m['id']: m for f in (DATA / 'variants').glob('*.json') for m in json.loads(f.read_text())}
    features = [k for k in next(iter(variants.values()))['A'] if k in next(iter(table.values()))['A']]
    diff = lambda a, o: {k: float(o[k]) - float(a[k]) for k in features}
    rows = []
    noise = np.median([float(t['N']['change']) for t in table.values() if 'N' in t])
    hidden = 0
    for pid, t in table.items():
        for side in 'BC':
            if float(t[side]['change']) < VISIBLE * noise:   # the eye cannot tell it from a re-dither: no label
                hidden += 1
                continue
            rows.append((side, t['A']['image'], diff(t['A'], t[side]), 1))
    print(f'B and C pairs within {VISIBLE}x the dither noise ({noise:.1f} dE through the eye), left out: {hidden}')
    key = json.loads((HERE / 'rounds' / 'cal1' / 'key.json').read_text())
    votes = json.loads((HERE / 'rounds' / 'cal1' / 'votes.json').read_text())
    for name, k in key.items():
        v = votes[name]['verdict']   # -1 the left sheet, 1 the right, 0 even
        if v == 0:
            continue
        won = (v == 1) == (k['painting'] == 2)
        m = variants.get(k['id']) if k['kind'] == 'kept' else \
            {'A': table[k['id']]['A'], 'B': table[k['id']]['B'], 'cells': int(table[k['id']]['A']['cells'])}
        if m is None or m['cells'] != k['cells']:   # the rebuild no longer makes the pair the user saw
            print(f"cal: {name} {k['id']} not rebuilt as voted on, left out")
            continue
        rows.append(('cal', k['id'].split('/')[0], diff(m['A'], m['B']), 1 if won else 0))
    hidden = 0
    for f in sorted((DATA / 'optimum').glob('r*/pairs-*.json')):
        verdicts = HERE / 'rounds' / f'opt{f.parent.name[1:]}' / 'verdicts.json'
        verdicts = json.loads(verdicts.read_text()) if verdicts.exists() else {}
        name = f.stem.removeprefix('pairs-')
        if name in verdicts and verdicts[name][1] != '-':   # the user did not find this optimum worse than the painting
            continue
        for m in json.loads(f.read_text()):
            if m['change'] < VISIBLE * noise:
                hidden += 1
            else:
                rows.append(('O', m['image'], diff(m['A'], m['B']), 1))
    print(f'O pairs within {VISIBLE}x the dither noise, left out: {hidden}')
    for f in (DATA / 'flat').glob('*.json'):
        rows += [('F', m['image'], diff(m['A'], m['B']), 1) for m in json.loads(f.read_text())]
    return features, rows


def fit(X, y, w, bounds):
    """Weights of the L2 logistic regression without intercept, each within its (low, high) bounds."""
    s = 2 * y - 1

    def loss(b):
        z = s * (X @ b)
        p = 1 / (1 + np.exp(z))
        return (w * np.logaddexp(0, -z)).sum() + L2 / 2 * b @ b, -(X.T @ (w * s * p)) + L2 * b

    return minimize(loss, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B', bounds=bounds).x


def loio(X, y, w, src, images, bounds):
    """Leave-one-image-out accuracy per source."""
    right = np.zeros(len(y))
    for im in np.unique(images):
        out = images == im
        b = fit(X[~out], y[~out], w[~out], bounds)
        right[out] = ((X[out] @ b > 0) == (y[out] == 1))
    return {s: right[src == s].mean() for s in SOURCES if (src == s).any()}


def select(X, y, src, images, bound):
    """Forward selection by the leave-one-image-out mean accuracy over TRAIN: (chosen, best, [(term, mean, acc)])."""
    w = np.array([1 / (src == s).sum() if s in TRAIN else 0.0 for s in src])
    chosen, best, path = [], 0.0, []
    while len(chosen) < MAX_TERMS:
        trials = []
        for j in range(X.shape[1]):
            if j not in chosen:
                acc = loio(X[:, chosen + [j]], y, w, src, images, [bound[i] for i in chosen + [j]])
                trials.append((np.mean([acc[s] for s in TRAIN]), j, acc))
        score, j, acc = max(trials, key=lambda t: t[0])
        if score <= best:
            break
        chosen.append(j)
        best = score
        path.append((j, score, acc))
    return chosen, best, path


def main(fast=False):
    features, rows = load()
    if fast:
        features = [k for k in features if k not in DEEP]
    anchors = FAST_ANCHORS if fast else ANCHORS
    src = np.array([r[0] for r in rows])
    images = np.array([r[1] for r in rows])
    D = np.array([[r[2][k] for k in features] for r in rows])
    y = np.array([r[3] for r in rows])
    w = np.array([1 / (src == s).sum() if s in TRAIN else 0.0 for s in src])
    scale = np.sqrt((D ** 2).mean(0)) + 1e-12
    X = D / scale
    bound = [(None, None) if k in FREE else (0, None) for k in features]
    counts = {s: int((src == s).sum()) for s in SOURCES}
    print(f"rows: {counts}, cal: A better {int(y[src == 'cal'].sum())} of {counts['cal']}\n")

    head = ''.join(f' {s:>5}' for s in SOURCES if counts[s])
    row = lambda acc: ''.join(f' {acc[s]:5.2f}' for s in SOURCES if s in acc)
    print(f"{'single metric, lower better':28}{head}")
    for j, k in enumerate(features):
        print(f"{k:28}" + row({s: np.where(D[src == s, j] == 0, 0.5, (D[src == s, j] > 0) == (y[src == s] == 1)).mean()
                                 for s in SOURCES if counts[s]}))   # a tie counts half

    print(f"\nforward selection, leave one image out, mean over {TRAIN}:\n{'+ term':28}{head}  mean")
    chosen, best, path = select(X, y, src, images, bound)
    for j, score, acc in path:
        print(f"{features[j]:28}{row(acc)} {score:5.2f}")

    rng, picks = np.random.default_rng(0), []
    names = np.unique(images)
    for _ in range(BOOT):   # the selection again on images drawn with replacement
        idx = np.concatenate([np.nonzero(images == im)[0] for im in rng.choice(names, len(names))])
        picks.append([features[j] for j in select(X[idx], y[idx], src[idx], images[idx], bound)[0]])
    print(f"\nthe selection on {BOOT} resamples of the images, how often each term is picked (first / at all):")
    for k in sorted({k for p in picks for k in p}, key=lambda k: -sum(k in p for p in picks)):
        print(f"{k:28} {sum(p[0] == k for p in picks):3} {sum(k in p for p in picks):4}")
    # the terms that come and go with the images drawn are left out
    stable = sorted({k for p in picks for k in p if sum(k in q for q in picks) >= BOOT / 2},
                    key=lambda k: -sum(k in p for p in picks))
    stable = stable or [max(set(p[0] for p in picks), key=lambda k: sum(p[0] == k for p in picks))]
    chosen = [features.index(k) for k in stable]
    acc = loio(X[:, chosen], y, w, src, images, [bound[i] for i in chosen])
    best = np.mean([acc[s] for s in TRAIN])
    print(f"\nthe judge's stable terms: {', '.join(stable)},{row(acc)} {best:5.2f}")

    W = whole_metrics()
    print(f"\na fidelity anchor at a fixed standardised weight, the rest refitted; on the 9 whole pictures, black last, "
          f"the painting over Select pairs and over the last optimum:\n{'anchor':18} {'weight':>6}{head}  mean black paint"
          f"   opt user")
    trials = []
    for k in [k for k in anchors if features.index(k) not in chosen]:   # a stable term is no anchor
        a = features.index(k)
        cols = chosen + [a] if a not in chosen else chosen
        for v in (0.1, 0.2, 0.3, 0.5, 1.0):
            acc = loio(X[:, cols], y, w, src, images, [bound[i] if i != a else (v, v) for i in cols])
            score = np.mean([acc[s] for s in TRAIN])
            bnds = [bound[i] if i != a else (v, v) for i in cols]
            full = {features[j]: bj / scale[j] for j, bj in zip(cols, fit(X[:, cols], y, w, bnds))}
            black, paint, opt, user = gates(full, W)
            print(f"{k:18} {v:6.1f}{row(acc)} {score:5.2f} {black:5} {paint:5} {opt:5} {user:4}")
            if black == len(W):   # a judge that ever rates a black screen above a render is broken
                trials.append((k, v, score, cols, user))

    # within SLACK of the best mean, the one that shares most of the user's verdicts on the last optimum, then the
    # best mean; none within: the best scoring one
    ok = [t for t in trials if t[2] >= best - SLACK]
    k, v, score, cols, user = max(ok, key=lambda t: (t[4], t[2])) if ok else max(trials, key=lambda t: (t[2], -t[1]))
    a = features.index(k)
    b = fit(X[:, cols], y, w, [bound[i] if i != a else (v, v) for i in cols])
    judge = {features[j]: bj / scale[j] for j, bj in zip(cols, b) if bj}
    print(f"\nchosen anchor {k} at {v}, mean {score:.2f}, {user} of the user's verdicts ("
          f"{'within' if ok else 'none within'} {SLACK} of the best mean)\nJUDGE = {{")
    for f, wt in judge.items():
        print(f"    '{f}': {wt:.4g},")
    print('}')
    if fast:
        from metrics import JUDGE
        full = {k: D[:, features.index(k)] if k in features else np.array([r[2][k] for r in rows]) for k in JUDGE}
        a = sum(wt * full[k] for k, wt in JUDGE.items()) > 0
        b = sum(wt * D[:, features.index(k)] for k, wt in judge.items()) > 0
        print('agreement with the full judge (metrics.JUDGE): ' +
              ', '.join(f'{s} {(a == b)[src == s].mean():.2f}' for s in SOURCES if counts[s]))
    print(f"\nwhole pictures, judge score (lower better):\n{'image':16} {'painting':>9} {'Select':>9} {'black':>9} "
          f"{'optimum':>9}")
    for name, m in W.items():
        a, h, k, o = (sum(wt * r[f] for f, wt in judge.items()) for r in m)
        print(f"{name:16} {a:9.3f} {h:9.3f} {k:9.3f} {o:9.3f}")


def whole_metrics():
    """{image: the metrics of the painting, of Select pairs alone, of a black screen and of the last round's optimum}
    over the whole picture."""
    from metrics import all_metrics
    last = max((DATA / 'optimum').glob('r*'), key=lambda p: int(p.name[1:]))
    out = {}
    for path in sorted(DATA.glob('*.npz')):
        z = np.load(path)
        T, mask = z['target'].astype(np.float32) / 255, np.ones(z['target'].shape[:2], bool)
        f = lambda a: a.astype(np.float32) / 255
        out[path.stem] = [all_metrics(X, T, mask) for X in (f(z['A']), f(z['H']), np.zeros_like(T),
                                                            f(np.load(last / path.name)['O']))]
    return out


def gates(judge, W):
    """On how many whole pictures the judge rates a black screen worst, the painting over Select pairs, and the
    painting over the last optimum; and of the user's decisive verdicts on the last optimum (against Select pairs and
    against the painting), how many it shares."""
    s = {name: [sum(wt * r[f] for f, wt in judge.items()) for r in m] for name, m in W.items()}
    last = max((DATA / 'optimum').glob('r*'), key=lambda p: int(p.name[1:]))
    path = HERE / 'rounds' / f'opt{last.name[1:]}' / 'verdicts.json'
    shared = 0
    if path.exists():
        v = json.loads(path.read_text())
        for name, (a, h, k, o) in s.items():
            for verdict, other in zip(v.get(name, ()), (h, a)):
                shared += (verdict == '+' and o < other) or (verdict == '-' and o > other)
    return (sum(k > max(a, h) for a, h, k, o in s.values()), sum(a < h for a, h, k, o in s.values()),
            sum(a < o for a, h, k, o in s.values()), shared)


if __name__ == '__main__':
    main('--fast' in sys.argv)
