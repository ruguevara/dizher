"""Variants of a picture's colouring: the selection run with sampled Metric and Select pairs values under each method,
finished as the app would, each saved as a screen with its params. Judged variants live in tests/images/NAME/variants/
(in git), the rest in the project's cache/variants/ (regenerated); the reference is the variant 'reference'."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from mokit.graph import Memo

from dizher.converter.energy import METHODS

from .project import IMAGES, convert, reference_file
from .scr import read_scr

RANGES = dict(chroma=(0.5, 3.0), coherence=(0.0, 8.0), edge=(0.05, 0.4), chroma_noise=(0.0, 0.1),
              luma_noise=(0.0, 0.1), flare=(0.0, 0.3))
REFERENCE = 'reference'


def variant_id(params: dict) -> str:
    """8 hex digits of the params (method included), rounded so a reprint of the JSON gives the same id."""
    canon = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(params.items())}
    return hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:8]


def presets() -> list:
    return [dict(method=m, **{k: float(v) for k, v in s.preset.items() if k in RANGES}) for m, s in METHODS.items()]


def sample(n: int, seed: int = 0, methods=tuple(METHODS)) -> list:
    """n param sets: a Latin hypercube over RANGES, the methods in turn, plus each method's preset."""
    rng = np.random.default_rng(seed)
    keys = list(RANGES)
    u = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in keys], axis=1)     # (n, k) in 0..1
    out = []
    for i in range(n):
        p = {k: float(lo + (hi - lo) * u[i, j]) for j, (k, (lo, hi)) in enumerate(RANGES.items())}
        p['method'] = methods[i % len(methods)]
        out.append(p)
    return presets() + out


def folders(name):
    return IMAGES / name / 'variants', IMAGES / name / 'cache' / 'variants'


def variant_file(name, variant) -> Path:
    """A variant's screen by id (kept, else cached), 'reference', or a path."""
    if variant == REFERENCE:
        return reference_file(name)
    for folder in folders(name):
        f = folder / f'{variant}.scr'
        if f.exists():
            return f
    f = Path(variant)
    assert f.exists(), f'no variant {variant} of {name}'
    return f


def variant_meta(name, variant) -> dict:
    f = variant_file(name, variant).with_suffix('.json')
    return json.loads(f.read_text()) if f.exists() else dict(id=variant, params={})


def read_variant(name, variant):
    """(bitmap, (R, C, 2) palette indexes)."""
    return read_scr(variant_file(name, variant))


def listing(name) -> dict:
    """id -> meta of every variant of a picture, kept and cached; the reference when there is one."""
    out = {}
    for folder in folders(name):
        for f in sorted(folder.glob('*.json')):
            meta = json.loads(f.read_text())
            out.setdefault(meta['id'], meta)
    if reference_file(name).exists():
        out.setdefault(REFERENCE, dict(id=REFERENCE, params={}))
    return out


def keep(name, variant) -> None:
    """A cached variant copied to the kept folder (judged variants stay in git)."""
    kept, cache = folders(name)
    if variant == REFERENCE or (kept / f'{variant}.scr').exists():
        return
    kept.mkdir(parents=True, exist_ok=True)
    for ext in ('.scr', '.json'):
        shutil.copy(cache / f'{variant}{ext}', kept / f'{variant}{ext}')


def generate(name, params_list, optimise=True, memo=None, log=print) -> list:
    """Every param set converted and saved to the cache (skipped when its screen is there); returns the ids."""
    _, cache = folders(name)
    cache.mkdir(parents=True, exist_ok=True)
    memo = memo if memo is not None else Memo()
    ids = []
    for params in params_list:
        vid = variant_id(dict(params, optimised=optimise))
        ids.append(vid)
        if (cache / f'{vid}.scr').exists() or (folders(name)[0] / f'{vid}.scr').exists():
            continue
        params = dict(params)
        method = params.pop('method')
        _, conv, final, seconds = convert(name, method, memo, optimise, **params)
        final.save(str(cache / f'{vid}.scr'))
        (cache / f'{vid}.json').write_text(json.dumps(dict(
            id=vid, name=name, method=method, params=params, optimised=optimise, seconds=round(seconds, 1),
            made=time.strftime('%Y-%m-%d')), indent=1))
        log(f'{name} {vid} {method} ' + ' '.join(f'{k}={v:.3g}' for k, v in params.items()) + f' {seconds:.1f} s')
    return ids


def distance(a: dict, b: dict) -> float:
    """How far apart two variants' settings are: 1 for another method, plus the normalised param distance."""
    pa, pb = a.get('params', {}), b.get('params', {})
    d = float(a.get('method') != b.get('method'))
    d += np.sqrt(sum(((pa.get(k, lo) - pb.get(k, lo)) / (hi - lo)) ** 2 for k, (lo, hi) in RANGES.items()))
    return d
