"""Fit the judge: a logistic regression on the differences of the candidate metrics between the two sides of a pair.

    python research/pairs/fit.py          needs data/metrics.csv (score.py) and data/variants/*.json (variants.py)

Labels, three sources, each weighted to the same total:
  B    the 229 painted segments: the painting (A) better than Select pairs' cells (B)
  C    the same segments: the painting better than the next best pair by the energy (C), assumed
  cal  the user's decisive votes in calibration round 1, both ways, repeats included
and one only reported, not trained on (weight 0):
  F    flat.py's counterexamples: the painting better than its segment all on one pair, assumed
P(A better) = sigmoid(sum_k w_k (f_k(other) - f_k(A)) / s_k), s_k the RMS of that difference, no intercept, L2.
Every metric is an error, so its weight is >= 0, but for the FREE ones, whose better direction is not known: unbounded,
the fit learns "further from the picture is better" from these one-sided labels (the painting always leaves the
colorimetric optimum). Terms are added one at a time, up to MAX_TERMS, the one that most raises the
leave-one-image-out accuracy (the mean over the three sources) each time. The labels hold nothing against a colouring
that is smooth but wrong, so a fidelity term (ANCHORS) is then fixed at the heaviest weight that costs at most SLACK
of that accuracy. Prints the single metrics, the selection path, the anchor sweep, the judge's weights in the
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
MAX_TERMS, L2 = 4, 1.0
ANCHORS = ('scielab_dE', 'blur_dE_4', 'blur_dE_8', 'blur_rmse', 'hue_family_miss')
SLACK = 0.02   # the accuracy a fidelity anchor may cost
FREE = ('two_colour_share', 'dot_contrast')   # no known better direction; every other metric is an error, weight >= 0
TRAIN, SOURCES = ('B', 'C', 'cal'), ('B', 'C', 'cal', 'F')


def load():
    """(features, rows): a row is (source, image, {feature: f(other) - f(A)}, y), y = 1 when A is better."""
    table = {}
    for r in csv.DictReader(open(DATA / 'metrics.csv')):
        table.setdefault(r['id'], {})[r['variant']] = r
    variants = {m['id']: m for f in (DATA / 'variants').glob('*.json') for m in json.loads(f.read_text())}
    features = [k for k in next(iter(variants.values()))['A'] if k in next(iter(table.values()))['A']]
    diff = lambda a, o: {k: float(o[k]) - float(a[k]) for k in features}
    rows = []
    for pid, t in table.items():
        for side in 'BC':
            rows.append((side, t['A']['image'], diff(t['A'], t[side]), 1))
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


def main():
    features, rows = load()
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
        print(f"{k:28}" + row({s: ((D[src == s, j] > 0) == (y[src == s] == 1)).mean() for s in SOURCES if counts[s]}))

    print(f"\nforward selection, leave one image out, mean over {TRAIN}:\n{'+ term':28}{head}  mean")
    chosen, best = [], 0.0
    while len(chosen) < MAX_TERMS:
        trials = []
        for j in range(len(features)):
            if j not in chosen:
                acc = loio(X[:, chosen + [j]], y, w, src, images, [bound[i] for i in chosen + [j]])
                trials.append((np.mean([acc[s] for s in TRAIN]), j, acc))
        score, j, acc = max(trials)
        if score <= best:
            break
        chosen.append(j)
        best = score
        print(f"{features[j]:28}{row(acc)} {score:5.2f}")

    print(f"\na fidelity anchor at a fixed standardised weight, the rest refitted:\n{'anchor':18} {'weight':>6}{head}  mean")
    anchors = {}
    for k in ANCHORS:
        a = features.index(k)
        cols = chosen + [a] if a not in chosen else chosen
        for v in (0.1, 0.2, 0.3, 0.5, 1.0):
            acc = loio(X[:, cols], y, w, src, images, [bound[i] if i != a else (v, v) for i in cols])
            score = np.mean([acc[s] for s in TRAIN])
            print(f"{k:18} {v:6.1f}{row(acc)} {score:5.2f}")
            if score >= best - SLACK:
                anchors[k] = (v, score, cols)

    k, (v, score, cols) = max(anchors.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    a = features.index(k)
    b = fit(X[:, cols], y, w, [bound[i] if i != a else (v, v) for i in cols])
    judge = {features[j]: bj / scale[j] for j, bj in zip(cols, b) if bj}
    print(f"\nchosen anchor {k} at {v} (the heaviest within {SLACK} of the best mean)\nJUDGE = {{")
    for f, wt in judge.items():
        print(f"    '{f}': {wt:.4g},")
    print('}')
    whole(judge)


def whole(judge):
    """The judge on whole pictures: the painting against Select pairs alone and against a black screen."""
    from metrics import all_metrics
    print(f"\nwhole pictures, judge score (lower better):\n{'image':16} {'painting':>9} {'Select':>9} {'black':>9}")
    for path in sorted(DATA.glob('*.npz')):
        z = np.load(path)
        T, mask = z['target'].astype(np.float32) / 255, np.ones(z['target'].shape[:2], bool)
        m = [all_metrics(X, T, mask) for X in (z['A'].astype(np.float32) / 255, z['H'].astype(np.float32) / 255,
                                                np.zeros_like(T))]
        a, h, k = (sum(wt * r[f] for f, wt in judge.items()) for r in m)
        print(f"{path.stem:16} {a:9.3f} {h:9.3f} {k:9.3f}")


if __name__ == '__main__':
    main()
