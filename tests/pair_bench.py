"""Pair selection against hand-checked attribute maps. Run by hand, not collected by pytest:

    python tests/pair_bench.py freeze NAME...         reference.scr from the project as saved, painted cells included
    python tests/pair_bench.py run [NAME...] [--set k=v ...] [--sheets DIR]
                                                      select pairs with the default Metric, Eye and Select params (each
                                                      project keeps its Tune params, palette and halftoner) and score them
    python tests/pair_bench.py regions NAME [--out DIR] [--k N]
                                                      the cells grouped into numbered regions, for painting by region
    python tests/pair_bench.py paint NAME FILE.json   regions (and cells) to Overpaint overrides in the project

A project is tests/images/NAME/project.json; its reference is reference.scr beside it. The reference is judged per
cell as the colours it shows: a cell whose bitmap is all paper or all ink is solid, and any pair holding that colour
matches it. Pairs are unordered, and the two blacks are one colour.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from mokit.graph import Memo, evaluate
from mokit.project import load_project, save_project

from dizher import ops
from dizher.platforms.zxspectrum.scr import _row_offsets

IMAGES = Path(__file__).parent / 'images'
SET = ('anubis', 'rocket-rackoon', 'jojo')
DEFAULTS = ('metric', 'eye', 'select')      # nodes the run resets, so every image is judged under one setting
NAMES = 'k b r m g c y w'.split()


def pair_name(p, i) -> str:
    n = lambda x: NAMES[x % 8].upper() if x >= 8 else NAMES[x % 8]
    return f'{n(p)}/{n(i)}'


def project_graph(name: str, defaults=()):
    """The project's graph in the current pipeline's shape (older projects lack nodes, e.g. Overpaint)."""
    project = load_project(IMAGES / name)
    graph = ops.make_graph()
    for nid in graph.ids():
        if nid in project.graph and nid not in defaults:
            graph = graph.with_params(nid, project.graph[nid].params)
    return graph


def painted_cells(graph) -> set:
    return {(r, c) for r, c, *_ in graph['overpaint'].params.overrides}


def read_scr(path: Path):
    """(192, 256) bool ink bitmap, (24, 32, 2) palette indexes (paper, ink)."""
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    rows = np.stack([data[off:off + 32] for off in _row_offsets()])
    bitmap = np.unpackbits(rows, axis=1).astype(bool)
    attrs = data[6144:6912].reshape(24, 32).astype(int)
    bright = (attrs >> 6 & 1) * 8
    return bitmap, np.stack([(attrs >> 3 & 7) + bright, (attrs & 7) + bright], axis=-1)


def black(i):
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
    rgb = palette.as_float()
    paper = np.repeat(np.repeat(rgb[pairs[..., 0]], 8, 0), 8, 1)
    ink = np.repeat(np.repeat(rgb[pairs[..., 1]], 8, 0), 8, 1)
    return np.where(bitmap[..., None], ink, paper)


def outline(img, cells, colour, zoom):
    out = img.copy()
    for r, c in cells:
        y, x = r * 8 * zoom, c * 8 * zoom
        cv2.rectangle(out, (x, y), (x + 8 * zoom - 1, y + 8 * zoom - 1), colour, 1)
    return out


def sheet(path, target, reference, result, wrong, zoom=2):
    """target | reference | result, the cells the result gets wrong outlined in red on it."""
    up = lambda a: np.repeat(np.repeat((a.clip(0, 1) * 255).astype(np.uint8), zoom, 0), zoom, 1)
    res = outline(up(result), wrong, (255, 0, 0), zoom)
    gap = np.full((192 * zoom, 4, 3), 128, np.uint8)
    img = np.concatenate([up(target), gap, up(reference), gap, res], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def finish(graph, memo, selection):
    """The project's Halftone and Optimise run on a selection, as the app would."""
    from dizher.ops import halftone, optimise
    return optimise(halftone(selection), **vars(graph['optimise'].params))


def pair_view(path, reference, result, zoom=2):
    """reference | result at zoom, no marks: for judging by eye."""
    up = lambda a: np.repeat(np.repeat((a.clip(0, 1) * 255).astype(np.uint8), zoom, 0), zoom, 1)
    gap = np.full((192 * zoom, 6, 3), 128, np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate([up(reference), gap, up(result)], axis=1), cv2.COLOR_RGB2BGR))


def freeze(names):
    for name in names:
        graph = project_graph(name)
        t = time.time()
        conv = evaluate(graph, 'optimise', Memo())
        conv.save(str(IMAGES / name / 'reference.scr'))
        print(f'{name}: reference.scr, {len(painted_cells(graph))} painted cells, {time.time() - t:.1f} s')


def parse_sets(items):
    out = {}
    for item in items or ():
        k, v = item.split('=')
        out[k] = float(v)
    return out


def select(graph, memo, **params):
    """Select pairs with params set on whichever of the Metric, Eye and Select nodes has them."""
    for nid in DEFAULTS:
        node = graph[nid].params
        mine = {k: v for k, v in params.items() if hasattr(node, k)}
        if mine:
            graph = graph.with_params(nid, replace(node, **mine))
    unknown = [k for k in params if not any(hasattr(graph[nid].params, k) for nid in DEFAULTS)]
    assert not unknown, unknown
    return evaluate(graph, 'select', memo)


def run(names, sets, sheets=None, memos=None, quiet=False):
    rows = []
    for name in [n for n in names if (IMAGES / n / 'reference.scr').exists()]:
        graph = project_graph(name, DEFAULTS)
        memo = memos.setdefault(name, Memo()) if memos is not None else Memo()
        t = time.time()
        conv = select(graph, memo, **sets)
        bitmap, ref_idx = read_scr(IMAGES / name / 'reference.scr')
        ref = shown(bitmap, ref_idx)
        got = label_pairs(conv, conv.best_attr_indexes)
        s = score(ref, got, painted_cells(project_graph(name)))
        s['name'], s['seconds'] = name, time.time() - t
        rows.append(s)
        if sheets:
            wrong = list(zip(*np.nonzero(~matches(ref, got))))
            result = conv.snapshot(labels=conv.best_attr_indexes).dithered_result
            sheet(Path(sheets) / f'{name}.png', conv.image_rgb, render_scr(bitmap, ref_idx, conv.palette), result, wrong)
            final = finish(graph, memo, conv).dithered_result
            pair_view(Path(sheets) / f'{name}-final.png', render_scr(bitmap, ref_idx, conv.palette), final)
    if not quiet:
        print(f"{'image':16} {'agree':>6} {'painted':>7} {'rest':>6} {'false':>6} {'missed':>6} {'changes':>7} {'s':>5}")
        for s in rows:
            print(f"{s['name']:16} {s['agree']:6.3f} {s['painted']:7.3f} {s['rest']:6.3f} {s['false_seams']:6.3f} "
                  f"{s['missed_seams']:6.3f} {s['pair_changes']:7d} {s['seconds']:5.1f}")
        mean = lambda k: np.nanmean([s[k] for s in rows])
        print(f"{'mean':16} {mean('agree'):6.3f} {mean('painted'):7.3f} {mean('rest'):6.3f} {mean('false_seams'):6.3f} "
              f"{mean('missed_seams'):6.3f}")
    return rows


def regions(name, out, k):
    """Cells clustered by the target's cell mean colour (CIELAB) and position, split into connected regions."""
    graph = project_graph(name, DEFAULTS)
    conv = evaluate(graph, 'prepare', Memo())
    lab = cv2.cvtColor(conv.image_rgb.astype(np.float32), cv2.COLOR_RGB2Lab)
    R, C = 24, 32
    means = lab.reshape(R, 8, C, 8, 3).mean(axis=(1, 3))
    feats = means.reshape(-1, 3).astype(np.float32)
    _, km, _ = cv2.kmeans(feats, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5,
                          cv2.KMEANS_PP_CENTERS)
    km = km.reshape(R, C).astype(np.int32)
    labels = np.zeros((R, C), np.int32)
    n = 0
    for cluster in range(k):
        count, comp = cv2.connectedComponents((km == cluster).astype(np.uint8), connectivity=4)
        for i in range(1, count):
            labels[comp == i] = n
            n += 1
    zoom = 4
    img = np.repeat(np.repeat((conv.image_rgb.clip(0, 1) * 255).astype(np.uint8), zoom, 0), zoom, 1)
    img = np.ascontiguousarray(img)
    for r in range(R):
        for c in range(C):
            y, x = r * 8 * zoom, c * 8 * zoom
            if r + 1 < R and labels[r + 1, c] != labels[r, c]:
                cv2.line(img, (x, y + 8 * zoom - 1), (x + 8 * zoom - 1, y + 8 * zoom - 1), (255, 255, 255), 1)
            if c + 1 < C and labels[r, c + 1] != labels[r, c]:
                cv2.line(img, (x + 8 * zoom - 1, y), (x + 8 * zoom - 1, y + 8 * zoom - 1), (255, 255, 255), 1)
    for i in range(n):
        rs, cs = np.nonzero(labels == i)
        j = np.argmin((rs - rs.mean()) ** 2 + (cs - cs.mean()) ** 2)
        y, x = rs[j] * 8 * zoom + 20, cs[j] * 8 * zoom + 4
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f'{name}-regions.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.save(out / f'{name}-regions.npy', labels)
    print(f'{n} regions: {out / f"{name}-regions.png"}')


def paint(name, spec_path):
    """spec: {"regions": FILE.npy, "pairs": {"region": "P/I", ...}, "cells": [[r, c, "P/I"], ...]}, colours by letter
    (k b r m g c y w, capitals bright)."""
    spec = json.loads(Path(spec_path).read_text())
    index = lambda ch: 'kbrmgcyw'.index(ch.lower()) + (8 if ch.isupper() else 0)
    parse = lambda s: tuple(index(ch) for ch in s.split('/'))
    cells = {}
    if 'regions' in spec:
        labels = np.load(spec['regions'])
        for region, pair in spec.get('pairs', {}).items():
            for r, c in zip(*np.nonzero(labels == int(region))):
                cells[int(r), int(c)] = parse(pair)
    for r, c, pair in spec.get('cells', ()):
        cells[r, c] = parse(pair)
    project = load_project(IMAGES / name)
    graph = project.graph
    overrides = tuple(sorted((r, c, p, i) for (r, c), (p, i) in cells.items()))
    graph = graph.with_params('overpaint', replace(graph['overpaint'].params, overrides=overrides))
    save_project(project.folder, graph, project.view)
    print(f'{name}: {len(overrides)} painted cells')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('freeze', 'run', 'regions', 'paint'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--set', action='append', help='Metric, Eye or Select param=value')
    ap.add_argument('--sheets', help='folder for contact sheets')
    ap.add_argument('--out', default='.')
    ap.add_argument('--k', type=int, default=12)
    a = ap.parse_args(argv)
    if a.command == 'freeze':
        freeze(a.args)
    elif a.command == 'run':
        if a.sheets:
            Path(a.sheets).mkdir(parents=True, exist_ok=True)
        run(a.args or SET, parse_sets(a.set), a.sheets)
    elif a.command == 'regions':
        regions(a.args[0], a.out, a.k)
    else:
        paint(*a.args)


if __name__ == '__main__':
    sys.exit(main())
