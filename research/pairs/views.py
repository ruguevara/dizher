"""Images of a local pair for judging by eye, blind: the sides in a seeded random order.

    python research/pairs/views.py OUT [N]     N pairs sampled evenly over the images (all by default), with a key

Per pair, two PNGs, one with the sides swapped, under shuffled names: on top the tuned picture's crop, colouring 1's
and colouring 2's, each at 2-6x nearest; below, the two whole screens at 2x with the judged cells outlined. Integer
zoom only, so the dots stay dots. Null pairs (the painting against itself from another halftone origin) come along,
one in eight, so a judge that sees a difference in any two renders shows."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, window   # noqa: E402

GAP = 8


def up(img, k):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def outline(img, cells, k, colour=(255, 255, 255)):
    out = np.ascontiguousarray(img.copy())
    edge = cv2.dilate(cells.astype(np.uint8), np.ones((3, 3), np.uint8)) - cells.astype(np.uint8)
    for r, c in zip(*np.nonzero(cells)):
        for dr, dc, a, b in ((-1, 0, (0, 0), (1, 0)), (1, 0, (0, 1), (1, 1)), (0, -1, (0, 0), (0, 1)), (0, 1, (1, 0), (1, 1))):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < cells.shape[0] and 0 <= cc < cells.shape[1]) or not cells[rr, cc]:
                p = (c * 8 * k + a[0] * (8 * k - 1), r * 8 * k + a[1] * (8 * k - 1))
                q = (c * 8 * k + b[0] * (8 * k - 1), r * 8 * k + b[1] * (8 * k - 1))
                cv2.line(out, p, q, (0, 0, 0), 3)
                cv2.line(out, p, q, colour, 1)
    return out


def sheet(T, X1, X2, cells):
    mask = window(cells, 1)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    k = max(2, min(6, int(600 / max(y1 - y0, x1 - x0))))
    crops = [up(img[y0:y1, x0:x1], k) for img in (T, X1, X2)]
    h = crops[0].shape[0]
    gap = np.full((h, GAP, 3), 90, np.uint8)
    top = np.concatenate([crops[0], gap, crops[1], gap, crops[2]], 1)
    full = [outline(up(img, 2), cells, 2) for img in (X1, X2)]
    bottom = np.concatenate([full[0], np.full((384, GAP, 3), 90, np.uint8), full[1]], 1)
    w = max(top.shape[1], bottom.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)), constant_values=90)
    return np.concatenate([pad(top), np.full((GAP, w, 3), 90, np.uint8), pad(bottom)], 0)


def main(out: Path, n=None, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pairs = []
    for path in sorted(DATA.glob('*.npz')):
        for m in json.loads(path.with_suffix('.json').read_text()):
            pairs.append((path, m))
    if n is not None and n < len(pairs):   # evenly over the images: round robin through each image's shuffled pairs
        by = {}
        for p in pairs:
            by.setdefault(p[0], []).append(p)
        for v in by.values():
            rng.shuffle(v)
        pairs = [v[i] for i in range(max(map(len, by.values()))) for v in by.values() if i < len(v)][:n]
    jobs = [(path, m, 'user') for path, m in pairs]
    nulls = [(path, m, 'null') for path, m in pairs if m['null']]
    jobs += nulls[:max(1, len(pairs) // 8)]
    names = [f'v{i:03d}' for i in rng.permutation(2 * len(jobs))]
    key, cache = {}, {}
    for j, (path, m, kind) in enumerate(jobs):
        z = cache.setdefault(path, np.load(path))
        s = m['segment']
        cells = z['diff'] & (z['segments'] == s)
        other = z['N'] if kind == 'null' else z[f'B{s}']
        for side, (X1, X2) in ((1, (z['A'], other)), (2, (other, z['A']))):
            name = names.pop()
            cv2.imwrite(str(out / f'{name}.png'), cv2.cvtColor(sheet(z['target'], X1, X2, cells), cv2.COLOR_RGB2BGR))
            key[name] = dict(id=m['id'], kind=kind, user=side, cells=m['cells'])
    (out / 'key.json').write_text(json.dumps(dict(sorted(key.items())), indent=1))
    print(f'{len(key)} sheets of {len(jobs)} pairs in {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
