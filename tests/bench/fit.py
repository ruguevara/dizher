"""The selection energy fitted to the judgments: the Metric and Select pairs values under which the energy puts the
winner of a judged pair below the loser most often. The energy of a labelling is cheap once the candidates are set
up, and only flare changes that setup, so flare runs on a grid and the other params over a random sample within
variants.RANGES; the best is refined by Nelder-Mead on a soft agreement. Leave-one-picture-out tells how much of the
agreement is fit to the pictures at hand."""
import numpy as np
from scipy.optimize import minimize

from dizher.converter.energy import METHODS, lightness_gain

from .metrics import Picture, judged_pairs
from .scr import pairs_to_labels
from .variants import RANGES, read_variant

FLARES = (0.03, 0.1, 0.3)
FREE = ('chroma', 'coherence', 'edge', 'chroma_noise', 'luma_noise')


class Energies:
    """Every judged variant's energy of one picture under one method, as a function of the params."""

    def __init__(self, name, method, variants):
        self.picture = Picture(name, methods=(method,))
        self.conv = self.picture.convs[method]
        self.labels = {v: pairs_to_labels(self.conv, read_variant(name, v)[1]) for v in variants}
        self.flare = None

    def set_flare(self, flare):
        if flare != self.flare:
            c = self.conv
            c.flare = flare
            c.gain = lightness_gain(c.image_luma, flare)
            c.energy.calc()
            self.flare = flare

    def __call__(self, params: dict) -> dict:
        """variant -> energy under params (flare included)."""
        self.set_flare(params['flare'])
        c = self.conv
        c.energy.weights['Chroma'] = params['chroma']
        c.coherence, c.edge = params['coherence'], params['edge']
        c.luma_noise, c.chroma_noise = params['luma_noise'], params['chroma_noise']
        return {v: c.energy.energy(l) if (l >= 0).all() else float('nan') for v, l in self.labels.items()}


def agreement(energies: dict, pairs, soft=False) -> float:
    """Share of pairs whose winner has the lower energy (ties half); soft: a sigmoid of the relative difference,
    for the refinement."""
    hits, n = 0.0, 0
    for w, l, _ in pairs:
        a, b = energies[w], energies[l]
        if np.isnan(a) or np.isnan(b):
            continue
        n += 1
        if soft:
            hits += 1 / (1 + np.exp(-8 * (b - a) / (abs(a) + abs(b) + 1e-9)))
        else:
            hits += 1.0 if a < b else 0.5 if a == b else 0.0
    return hits / n if n else float('nan')


def sample(n, seed, flares=FLARES):
    rng = np.random.default_rng(seed)
    keys = list(FREE)
    u = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in keys], axis=1)
    out = []
    for i in range(n):
        p = {k: float(lo + (hi - lo) * u[i, j]) for j, k in enumerate(keys) for (lo, hi) in [RANGES[k]]}
        p['flare'] = float(flares[i % len(flares)])
        out.append(p)
    return out


def objective(params, energies_by_picture: dict, pairs_by_picture: dict, soft=False) -> float:
    """Agreement over all pictures' pairs, pairs weighted equally."""
    values = {}
    for name, e in energies_by_picture.items():
        values.update({f'{name}/{v}': x for v, x in e(params).items()})
    pairs = [(f'{n}/{w}', f'{n}/{l}', t) for n, ps in pairs_by_picture.items() for w, l, t in ps]
    return agreement(values, pairs, soft)


def clip(params):
    return {k: float(np.clip(v, *RANGES[k])) for k, v in params.items()}


def refine(params, energies, pairs, steps=60):
    """Nelder-Mead on the soft agreement from params, flare fixed; returns the params and their hard agreement."""
    keys = list(FREE)
    x0 = np.array([params[k] for k in keys])
    scale = np.array([hi - lo for k in keys for (lo, hi) in [RANGES[k]]])
    def f(x):
        p = clip(dict(zip(keys, x0 + x * scale), flare=params['flare']))
        return -objective(p, energies, pairs, soft=True)
    r = minimize(f, np.zeros(len(keys)), method='Nelder-Mead', options=dict(maxfev=steps, xatol=1e-3, fatol=1e-4))
    best = clip(dict(zip(keys, x0 + r.x * scale), flare=params['flare']))
    return best, objective(best, energies, pairs)


def fit(names, judgments_of, by='user', method=None, n=150, seed=0, holdout=True, log=print) -> dict:
    """method -> dict(params, agreement, pairs, preset_agreement, holdout: name -> agreement of the params fitted
    without that picture on its pairs)."""
    out = {}
    for m in ([method] if method else list(METHODS)):
        pairs_by, energies_by = {}, {}
        for name in names:
            pairs = judged_pairs(judgments_of(name), by)
            if not pairs:
                continue
            variants = sorted({v for w, l, _ in pairs for v in (w, l)})
            pairs_by[name] = pairs
            energies_by[name] = Energies(name, m, variants)
        if not pairs_by:
            continue
        total = sum(len(p) for p in pairs_by.values())
        log(f'{m}: {total} pairs over {len(pairs_by)} pictures')
        candidates = sample(n, seed)
        preset = {k: float(v) for k, v in METHODS[m].preset.items() if k in RANGES}
        candidates.append(preset)
        scored = [(objective(p, energies_by, pairs_by), p) for p in candidates]
        scored.sort(key=lambda t: -t[0])
        best_params, best = refine(scored[0][1], energies_by, pairs_by)
        if best < scored[0][0]:
            best, best_params = scored[0]
        result = dict(params=best_params, agreement=best, pairs=total, preset_agreement=objective(preset, energies_by, pairs_by),
                      preset=preset, holdout={})
        log(f'{m}: preset {result["preset_agreement"]:.2f}, fitted {best:.2f} '
            + ' '.join(f'{k}={v:.3g}' for k, v in best_params.items()))
        if holdout and len(pairs_by) > 1:
            for held in pairs_by:
                train_e = {k: v for k, v in energies_by.items() if k != held}
                train_p = {k: v for k, v in pairs_by.items() if k != held}
                top = max(candidates, key=lambda p: objective(p, train_e, train_p))
                result['holdout'][held] = objective(top, {held: energies_by[held]}, {held: pairs_by[held]})
                log(f'  without {held}: {result["holdout"][held]:.2f} on its {len(pairs_by[held])} pairs')
        out[m] = result
    return out
