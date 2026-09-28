"""ZX Spectrum .scr files as the bench sees them: bitmap and (paper, ink) indexes, the colours each cell shows,
agreement with a reference cell by cell and seam by seam."""
from pathlib import Path

import numpy as np

from dizher.platforms.zxspectrum.scr import _row_offsets

NAMES = 'k b r m g c y w'.split()


def pair_name(p, i) -> str:
    """Palette indexes -> 'k/Y': letters k b r m g c y w, capitals bright."""
    n = lambda x: NAMES[x % 8].upper() if x >= 8 else NAMES[x % 8]
    return f'{n(p)}/{n(i)}'


def parse_pair(s: str) -> tuple:
    """'k/Y' -> (0, 14)."""
    index = lambda ch: 'kbrmgcyw'.index(ch.lower()) + (8 if ch.isupper() else 0)
    return tuple(index(ch) for ch in s.split('/'))


def read_scr(path):
    """(192, 256) bool ink bitmap, (24, 32, 2) palette indexes (paper, ink)."""
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    assert data.size == 6912, f'{path}: {data.size} bytes, not a 6912-byte screen'
    rows = np.stack([data[off:off + 32] for off in _row_offsets()])
    bitmap = np.unpackbits(rows, axis=1).astype(bool)
    attrs = data[6144:6912].reshape(24, 32).astype(int)
    bright = (attrs >> 6 & 1) * 8
    return bitmap, np.stack([(attrs >> 3 & 7) + bright, (attrs & 7) + bright], axis=-1)


def has_flash(path) -> bool:
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return bool((data[6144:6912] & 0x80).any())


def black(i):
    """The two blacks are one colour."""
    return np.where(np.asarray(i) % 8 == 0, 0, i)


def shown(bitmap, pairs, cell=(8, 8)):
    """(R, C, 2) the colours each cell shows, blacks unified; a solid cell repeats its colour."""
    h, w = cell
    R, C = pairs.shape[:2]
    ink = bitmap.reshape(R, h, C, w).sum(axis=(1, 3))
    paper, inkc = black(pairs[..., 0]), black(pairs[..., 1])
    first = np.where(ink == h * w, inkc, paper)
    second = np.where(ink == 0, paper, inkc)
    return np.sort(np.stack([first, second], axis=-1), axis=-1)


def label_pairs(conv, labels):
    """(R, C, 2) sorted palette indexes, blacks unified, of a labelling."""
    idx = np.array(list(conv.palette.iter_idxs_pairs()))
    return np.sort(black(idx[labels]), axis=-1)


def pairs_to_labels(conv, pairs):
    """(R, C, 2) palette indexes (any order, either black) -> (R, C) labels into conv's pairs; -1 where the palette
    has no such pair (a colour the Target excludes)."""
    idx = np.sort(black(np.array(list(conv.palette.iter_idxs_pairs()))), axis=-1)     # (P, 2)
    key = {}
    for i, p in enumerate(idx.tolist()):
        key.setdefault(tuple(p), i)          # k/k and K/K are one pair: the first index
    want = np.sort(black(pairs), axis=-1)
    labels = np.full(pairs.shape[:2], -1, dtype=np.int64)
    for r, c in np.ndindex(labels.shape):
        labels[r, c] = key.get(tuple(want[r, c].tolist()), -1)
    return labels


def matches(ref, got):
    """(R, C) bool: got's pair is the reference's, or holds its solid colour."""
    solid = ref[..., 0] == ref[..., 1]
    same = (ref == got).all(-1)
    holds = (got == ref[..., :1]).any(-1)
    return np.where(solid, holds, same)


def same_region(a, b):
    """Two neighbouring cells read as one region: equal pairs, or a solid colour the other pair holds."""
    eq = (a == b).all(-1)
    a_solid, b_solid = a[..., 0] == a[..., 1], b[..., 0] == b[..., 1]
    return eq | (a_solid & (b == a[..., :1]).any(-1)) | (b_solid & (a == b[..., :1]).any(-1))


def seams(pairs):
    """Every 4-neighbour seam's (a, b) pairs, flattened."""
    return (np.concatenate([pairs[1:].reshape(-1, 2), pairs[:, 1:].reshape(-1, 2)]),
            np.concatenate([pairs[:-1].reshape(-1, 2), pairs[:, :-1].reshape(-1, 2)]))


def score(ref, got, painted):
    ok = matches(ref, got)
    mask = np.zeros(ok.shape, bool)
    for r, c in painted:
        mask[r, c] = True
    ra, rb = seams(ref)
    ga, gb = seams(got)
    ref_same, got_same = same_region(ra, rb), same_region(ga, gb)
    return dict(agree=ok.mean(), painted=ok[mask].mean() if mask.any() else np.nan, rest=ok[~mask].mean(),
                false_seams=(ref_same & ~got_same).sum() / max(ref_same.sum(), 1),
                missed_seams=(~ref_same & got_same).sum() / max((~ref_same).sum(), 1),
                pair_changes=(~got_same).sum())


def render_scr(bitmap, pairs, palette):
    """(192, 256, 3) float sRGB of a screen in the palette."""
    rgb = palette.as_float()
    paper = np.repeat(np.repeat(rgb[pairs[..., 0]], 8, 0), 8, 1)
    ink = np.repeat(np.repeat(rgb[pairs[..., 1]], 8, 0), 8, 1)
    return np.where(bitmap[..., None], ink, paper)
