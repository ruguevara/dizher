"""After session E: Select pairs on an edge-preserving smoothed picture, so the cells of one surface see one colour and
pick one pair instead of a patchwork; DBS still dithers toward the original (the labels rendered by the base pipeline).

    python research/pairs/smooth.py make      per picture the base (Exact mixture preset) and each of SMOOTH, in
                                              data/smooth/NAME.npz; a sheet per picture data/smooth/NAME.png to look at
    python research/pairs/smooth.py user V    blind sheets for the user, the base against variant V (an index of
                                              SMOOTH), in data/smooth/user/, key rounds/smooth/user-key.json;
                                              vote with dp.py vote --round smooth

Smoothing: a bilateral filter in CIELAB, spatial sigma in cells, colour sigma in dE: inside a surface (steps below the
colour sigma) the picture averages over about a cell or two, across an edge it does not."""
import json
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Project   # noqa: E402
from tune import setting   # noqa: E402
from views import GAP, up   # noqa: E402
from mokit.graph import evaluate   # noqa: E402
from dizher import ops   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'smooth', HERE / 'rounds' / 'smooth'
PICTURES = ('RC1', 'anubis', 'autumn', 'diver-sunset', 'jojo', 'rocket-rackoon', 'vangog')   # the held-out pictures stay out
SMOOTH = ((1.0, 20.0), (2.0, 20.0), (2.0, 40.0))   # (cells, dE)


def smoothed(picture: np.ndarray, cells: float, colour: float) -> np.ndarray:
    lab = cv2.cvtColor(picture.astype(np.float32), cv2.COLOR_RGB2Lab)
    lab = cv2.bilateralFilter(lab, -1, colour, cells * 8)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def labels(p: Project, picture: np.ndarray) -> np.ndarray:
    """Select pairs' labels for this picture at the project's settings."""
    g, m = p.graph, p.memo
    prep = ops.prepare(picture, *(evaluate(g, nid, m) for nid in ('target', 'metric', 'eye', 'halftoner')))
    return ops.select_pairs(prep, **asdict(g['select'].params)).best_attr_indexes.copy()


def make_job(name: str):
    p = Project(name, **setting({}))
    base = p.selection.best_attr_indexes.copy()
    picture = evaluate(p.graph, 'detail', p.memo)
    assert (labels(p, picture) == base).all()
    out = dict(target=p.target(), base=p.render(base))
    changed = []
    for i, (cells, colour) in enumerate(SMOOTH):
        L = labels(p, smoothed(picture, cells, colour))
        out[f's{i}'] = p.render(L)
        changed.append(int((L != base).sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    out = {k: np.round(v * 255).astype(np.uint8) for k, v in out.items()}
    np.savez_compressed(OUT / f'{name}.npz', **out)
    gap = np.full((384, GAP, 3), 90, np.uint8)
    row = [x for k in ('target', 'base', *(f's{i}' for i in range(len(SMOOTH)))) for x in (up(out[k], 2), gap)][:-1]
    cv2.imwrite(str(OUT / f'{name}.png'), cv2.cvtColor(np.concatenate(row, 1), cv2.COLOR_RGB2BGR))
    return name, changed


def make():
    with Pool(len(PICTURES)) as pool:
        for name, changed in pool.imap_unordered(make_job, PICTURES):
            print(f'{name}: cells changed ' + ', '.join(f'{s}: {n}' for s, n in zip(SMOOTH, changed)), flush=True)


def user(v: int, seed=0):
    """Per picture one sheet: the picture, then base and variant v in a random order, raw 3x as the app shows them."""
    rng = np.random.default_rng(seed)
    out = OUT / 'user'
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    for i, name in enumerate(rng.permutation(PICTURES)):
        z = np.load(OUT / f'{name}.npz')
        side = int(rng.integers(1, 3))   # the variant's side
        a, b = (z[f's{v}'], z['base'])[::1 if side == 1 else -1]
        gap = np.full((576, GAP, 3), 90, np.uint8)
        sheet = np.concatenate([up(z['target'], 3), gap, up(a, 3), gap, up(b, 3)], 1)
        cv2.imwrite(str(out / f'u{i + 1:02d}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        key[f'u{i + 1:02d}'] = dict(id=str(name), smooth=SMOOTH[v], x=side)
    ROUND.mkdir(parents=True, exist_ok=True)
    (ROUND / 'user-key.json').write_text(json.dumps(key, indent=1))
    print(f'{len(key)} sheets in {out}')


if __name__ == '__main__':
    {'make': make, 'user': lambda: user(int(sys.argv[2]))}[sys.argv[1]]()
