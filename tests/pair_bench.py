"""Pair selection against hand-checked attribute maps. Run by hand, not collected by pytest:

    python tests/pair_bench.py freeze NAME...         the reference refrozen: the project converted as saved, painted
                                                      cells included, to cache/reference.scr (run does it when needed)
    python tests/pair_bench.py run [NAME...] [--method M] [--set k=v ...] [--sheets DIR]
                                                      each painted project converted with its painting hidden, under
                                                      its own settings (default: what the user corrected) or under a
                                                      method's tuned Metric and Select values, and scored against the
                                                      reference: cells agreeing (fixed: of the cells the user
                                                      corrected; kept: of the rest), and the distance of the picture
                                                      it shows to the reference picture (de2000:4 whole and over the
                                                      corrected cells, and the other judge metrics; the painting is
                                                      judged as a picture, not by attributes)
    python tests/pair_bench.py compare [NAME...] [--sheets DIR]
                                                      the project's own settings and every method side by side
    python tests/pair_bench.py regions NAME [--out DIR] [--k N]
                                                      the cells grouped into numbered regions, for painting by region
    python tests/pair_bench.py paint NAME FILE.json   regions (and cells) to Overpaint overrides in the project
    python tests/pair_bench.py render NAME [--out DIR] [--zoom 3] [--crop r0,c0,r1,c1 ...] [--variant ID|FILE.scr]
                                       [--method M] [--no-eye]
                                                      target | reference | result (the method's, or a variant) with cell
                                                      rulers, the cells off the reference outlined, the eye views under
                                                      them, and each crop of cells (inclusive) at twice the zoom
    python tests/pair_bench.py variants NAME... [--n 24] [--seed 0] [--fast]
                                                      the selection under sampled Metric and Select values of every
                                                      method, finished as the app would (--fast: no DBS), each a screen
                                                      in the project's cache/variants/ with its params
    python tests/pair_bench.py judge pairs NAME... [--n 8] [--seed 0] [--out DIR]
                                                      pairs of variants to judge, appended to the project's
                                                      judgments.json (their screens kept in variants/), a sheet
                                                      target | A | B per pair in DIR
    python tests/pair_bench.py judge record NAME --by user "12 a hue, 13 same, 14 b clash noise -- the sky"
                                                      a judge's verdicts (a, b or same, the faults named: hue, clash,
                                                      noise, tone, other; a note after --) into judgments.json
    python tests/pair_bench.py judge agree [NAME...]   how far the judges agree, by fault
    python tests/pair_bench.py zxart fetch [--n 50] [--rating 4]
                                                      the top standard pictures by votes from zxart.ee: their screens
                                                      to tests/images/zxart/<id>/reference.scr, the list to index.json
    python tests/pair_bench.py zxart prepare [--luma 1.0] [--chroma 2.5]
                                                      each picture's source (its screen through a wider eye blur) and
                                                      project, so run/compare zxart/<id> score against the artist
    python tests/pair_bench.py zxart stats            the pairs the artists use, how often they change, clashing hues
    python tests/pair_bench.py zxart names            the project names, for run and compare
    python tests/pair_bench.py rank [NAME...] [--by user] [--metrics m,m...] [--no-ref]
                                                      every judge metric (bench/metrics.py) by how often it puts the
                                                      winner of a judged pair below the loser: overall, by fault and
                                                      by picture, and where it ranks the reference among the variants;
                                                      --metrics may name energy:<method>:<k=v+k=v> (the energy under
                                                      those values over the preset) and judge:<w> (the composite)
    python tests/pair_bench.py fit [NAME...] [--by user] [--method M] [--n 150] [--seed 0]
                                                      the Metric and Select pairs values under which each method's
                                                      energy agrees most with the judge (bench/fit.py), with the
                                                      agreement of its preset and of a fit without each picture

A project is tests/images/NAME/project.json; its reference is the project converted as saved, painting included
(bench/project.py), or a reference.scr beside it (the zxart set). Cell agreement counts the colours a cell shows: a
cell whose bitmap is all paper or all ink is solid, and any pair holding that colour matches it; pairs are unordered,
the two blacks one colour. The picture distances need no such care.
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
from dizher.converter.energy import METHODS

sys.path.insert(0, str(Path(__file__).parent))
from bench import render as R, variants as V, judge as J, zxart as Z, metrics as M, fit as F         # noqa: E402
from bench.scr import pair_name, parse_pair, black, label_pairs, matches, score, render_scr   # noqa: E402
from bench.project import (IMAGES, DEFAULTS, project_graph, select, finish, reference,                  # noqa: E402
                           has_reference, convert_as_saved, changed_cells)
import bench.project                                                                                  # noqa: E402

JUDGED = ('anubis', 'rocket-rackoon', 'jojo', 'andy', 'david', 'vangog')   # the calibration set


def painted_set() -> tuple:
    """Every project with a reference: painted cells, or a screen of its own (not the zxart set, run by name)."""
    return tuple(sorted(d.name for d in IMAGES.iterdir()
                        if (d / 'project.json').exists() and d.name != 'zxart' and has_reference(d.name)))


def _count(got, cells, names):
    return sum(pair_name(*got[r, c]) in names for r, c in cells)


def _painted_as(ref, painted, name):
    return [(r, c) for r, c in painted if pair_name(*ref[r, c]) == name]


# Faults the user named, counted per image: (label, function of the result's pairs, the reference, painted cells).
SPOTS = {
    'anubis': (('eye yellow/3', lambda got, ref, p: _count(got, ((6, 18), (7, 18), (7, 19)), ('k/Y', 'k/y'))),
               ('magenta', lambda got, ref, p: _count(got, np.ndindex(got.shape[:2]), ('k/m', 'k/M')))),
    'jojo': (('magenta', lambda got, ref, p: _count(got, np.ndindex(got.shape[:2]), ('k/m', 'k/M'))),
             ('k/y faces grey', lambda got, ref, p: _count(got, _painted_as(ref, p, 'k/y'), ('k/W', 'k/w')))),
    'rocket-rackoon': (('R/Y kept', lambda got, ref, p: _count(got, _painted_as(ref, p, 'R/Y'), ('R/Y',))),),
}


def spots(name, ref, got, painted) -> str:
    return ', '.join(f'{label} {check(got, ref, painted)}' for label, check in SPOTS.get(name, ()))


def freeze(names):
    for name in names or painted_set():
        bench.project.freeze(name, force=True, log=print)


def parse_sets(items):
    out = {}
    for item in items or ():
        k, v = item.split('=')
        out[k] = float(v)
    return out


def scored(name, conv, final, seconds=0.0):
    """Cell agreement of a selection with the reference, and the distances of the finished picture to the reference
    picture (bench/metrics.py DISTANCES), with the spot checks. The reference grew out of the project's own
    conversion, and the user corrected only what looked wrong, so the cells that were changed and the cells that
    were kept are told apart: fixed is the share of corrected cells the selection gets as the user painted them, kept
    the share of the other cells it leaves as the user accepted, de2000_fixed the picture distance over the corrected
    cells alone."""
    bitmap, ref_idx, ref, painted = reference(name)
    got = label_pairs(conv, conv.best_attr_indexes)
    s = score(ref, got, painted)
    ok, changed = matches(ref, got), changed_cells(name)
    s['fixed'] = ok[changed].mean() if changed.any() else np.nan
    s['kept'] = ok[~changed].mean() if (~changed).any() else np.nan
    s['changed'] = int(changed.sum())
    ref_img = render_scr(bitmap, ref_idx, conv.palette)
    pairs = np.array(list(conv.palette.iter_idxs_pairs()))[conv.best_attr_indexes]
    s.update(M.distances(conv, ref_img, final.dithered_result, pairs))
    s['de2000_fixed'] = M.de2000_masked(ref_img, final.dithered_result, conv.expand_cells(changed), 4)
    s['name'], s['seconds'], s['spots'] = name, seconds, spots(name, ref, got, painted)
    return s


PROJECT = 'project'     # the project's own settings, as the user saw it before painting


def convert(name, method, memo, **sets):
    """(graph, selection, final, seconds) under a method's preset, or PROJECT: the project's own settings."""
    if method == PROJECT:
        assert not sets, 'sets apply to a method'
        return convert_as_saved(name, memo)
    graph = ops.apply_preset(project_graph(name, DEFAULTS), method)
    t = time.time()
    conv = select(graph, memo, **sets)
    return graph, conv, finish(graph, memo, conv), time.time() - t


def print_rows(rows):
    print(f"{'image':16} {'agree':>6} {'fixed':>6} {'kept':>6} {'chg':>4} {'false':>6} {'changes':>7} {'de2000':>7} "
          f"{'dE_fix':>7} {'scielab':>7} {'msssim':>7} {'s':>5}")
    for s in rows:
        print(f"{s['name']:16} {s['agree']:6.3f} {s['fixed']:6.3f} {s['kept']:6.3f} {s['changed']:4d} "
              f"{s['false_seams']:6.3f} {s['pair_changes']:7d} {s['de2000:4']:7.3f} {s['de2000_fixed']:7.3f} "
              f"{s['scielab']:7.3f} {s['msssim_ab']:7.3f} {s['seconds']:5.1f}  {s['spots']}")
    mean = lambda k: np.nanmean([s[k] for s in rows])
    print(f"{'mean':16} {mean('agree'):6.3f} {mean('fixed'):6.3f} {mean('kept'):6.3f} {mean('changed'):4.0f} "
          f"{mean('false_seams'):6.3f} {mean('pair_changes'):7.0f} {mean('de2000:4'):7.3f} {mean('de2000_fixed'):7.3f} "
          f"{mean('scielab'):7.3f} {mean('msssim_ab'):7.3f}")


def run(names, sets, sheets=None, memos=None, quiet=False, method=PROJECT):
    rows = []
    for name in [n for n in names if has_reference(n)]:
        memo = memos.setdefault(name, Memo()) if memos is not None else Memo()
        graph, conv, final, seconds = convert(name, method, memo, **sets)
        s = scored(name, conv, final, seconds)
        rows.append(s)
        if sheets:
            bitmap, ref_idx, ref, painted = reference(name)
            got = label_pairs(conv, conv.best_attr_indexes)
            wrong = list(zip(*np.nonzero(~matches(ref, got))))
            ref_img = render_scr(bitmap, ref_idx, conv.palette)
            R.save(Path(sheets) / f'{name.replace("/", "-")}.png',
                   R.sheet([('target', conv.image_rgb), ('reference', ref_img), (method, final.dithered_result)], 2,
                           marks={method: wrong}))
    if not quiet:
        print(f'== {method}')
        print_rows(rows)
    return rows


def compare(names, sheets=None):
    """The project's own settings and every method with its tuned values on each reference: cells agreeing and the
    distance of the finished picture to the reference picture. With sheets: per image, the reference | each result."""
    names = [n for n in names if has_reference(n)]
    columns = {m: {} for m in (PROJECT, *METHODS)}
    finals, palettes = {n: [] for n in names}, {}
    for name in names:
        memo = Memo()
        for method in columns:
            graph, conv, final, seconds = convert(name, method, memo)
            columns[method][name] = scored(name, conv, final, seconds)
            palettes[name] = conv.palette
            if sheets:
                finals[name].append((method, final.dithered_result))
    for label, rows in columns.items():
        print(f'== {label}')
        print_rows(list(rows.values()))
    if sheets:
        for name in names:
            bitmap, ref_idx, _, _ = reference(name)
            tiles = [('reference', render_scr(bitmap, ref_idx, palettes[name]))] + finals[name]
            R.save(Path(sheets) / f'{name.replace("/", "-")}-compare.png', R.sheet(tiles, 2))


def regions(name, out, k):
    """Cells clustered by the target's cell mean colour (CIELAB) and position, split into connected regions."""
    graph = project_graph(name, DEFAULTS)
    conv = evaluate(graph, 'prepare', Memo())
    lab = cv2.cvtColor(conv.image_rgb.astype(np.float32), cv2.COLOR_RGB2Lab)
    rows, cols = 24, 32
    means = lab.reshape(rows, 8, cols, 8, 3).mean(axis=(1, 3))
    feats = means.reshape(-1, 3).astype(np.float32)
    _, km, _ = cv2.kmeans(feats, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5,
                          cv2.KMEANS_PP_CENTERS)
    km = km.reshape(rows, cols).astype(np.int32)
    labels = np.zeros((rows, cols), np.int32)
    n = 0
    for cluster in range(k):
        count, comp = cv2.connectedComponents((km == cluster).astype(np.uint8), connectivity=4)
        for i in range(1, count):
            labels[comp == i] = n
            n += 1
    zoom, m = 4, R.MARGIN
    img = R.ruled(conv.image_rgb, zoom, grid=False)
    for r in range(rows):
        for c in range(cols):
            y, x = m + r * 8 * zoom, m + c * 8 * zoom
            if r + 1 < rows and labels[r + 1, c] != labels[r, c]:
                cv2.line(img, (x, y + 8 * zoom - 1), (x + 8 * zoom - 1, y + 8 * zoom - 1), (255, 255, 255), 1)
            if c + 1 < cols and labels[r, c + 1] != labels[r, c]:
                cv2.line(img, (x + 8 * zoom - 1, y), (x + 8 * zoom - 1, y + 8 * zoom - 1), (255, 255, 255), 1)
    for i in range(n):
        rs, cs = np.nonzero(labels == i)
        j = np.argmin((rs - rs.mean()) ** 2 + (cs - cs.mean()) ** 2)
        y, x = m + rs[j] * 8 * zoom + 20, m + cs[j] * 8 * zoom + 4
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R.save(out / f'{name}-regions.png', img)
    np.save(out / f'{name}-regions.npy', labels)
    print(f'{n} regions: {out / f"{name}-regions.png"}')


def paint(name, spec_path):
    """spec: {"regions": FILE.npy, "pairs": {"region": "P/I", ...}, "cells": [[r, c, "P/I"], ...]}, colours by letter
    (k b r m g c y w, capitals bright)."""
    spec = json.loads(Path(spec_path).read_text())
    parse = parse_pair
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


def render(name, out, zoom=3, crops=(), variant=None, eye=True, method=PROJECT):
    """target | reference | result with rulers, eye views and crops: NAME.png (NAME-ID.png for a variant)."""
    memo = Memo()
    graph, conv, final, _ = convert(name, method, memo)
    columns, marks, ref = [('target', conv.image_rgb)], {}, None
    if has_reference(name):
        bitmap, ref_idx, ref, _ = reference(name)
        columns.append(('reference', render_scr(bitmap, ref_idx, conv.palette)))
    if variant:
        bitmap, idx = V.read_variant(name, variant)
        title, got = f'variant {Path(str(variant)).stem}', np.sort(black(idx), axis=-1)
        columns.append((title, render_scr(bitmap, idx, conv.palette)))
    else:
        title, got = method, label_pairs(conv, conv.best_attr_indexes)
        columns.append((title, final.dithered_result))
    if ref is not None:
        marks[title] = list(zip(*np.nonzero(~matches(ref, got))))
    img = R.sheet(columns, zoom, conv.eye_view if eye else None, crops, marks=marks)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = name.replace('/', '-')                      # zxart/47111 -> zxart-47111
    path = out / (f'{stem}-{Path(str(variant)).stem}.png' if variant else f'{stem}.png')
    R.save(path, img)
    print(path)
    return path


def variants(names, n, seed, fast):
    for name in names:
        ids = V.generate(name, V.sample(n, seed), optimise=not fast)
        print(f'{name}: {len(ids)} variants in {V.folders(name)[1]}')


def judge(args, n, seed, out, by, zoom):
    what, rest = args[0], args[1:]
    if what == 'pairs':
        for i, name in enumerate(rest or JUDGED):
            judged = [(p['a'], p['b']) for p in J.load(name)['pairs']]
            entries = J.add_pairs(name, J.pick(name, n, seed + i, judged))     # another draw per picture
            J.make_sheets(name, entries, out, zoom)
    elif what == 'record':
        name, text = rest[0], ' '.join(rest[1:])
        print(f'{name}: {J.record(name, text, by)} verdicts by {by}')
    elif what == 'agree':
        a = J.agreement(rest or JUDGED)
        print(f"{a['pairs']} pairs judged by both: agree {a['agree']:.2f}, kappa {a['kappa']:.2f}, "
              f"opposed {a['opposed']:.2f}, same {a['same'][0]:.2f} / {a['same'][1]:.2f}")
        for t, (count, agree) in a['tags'].items():
            print(f'  {t:6} {count:3d} pairs, agree {agree:.2f}')
    else:
        raise SystemExit(f'judge {what}? pairs, record or agree')


def zxart(args, n, rating, luma, chroma):
    what = args[0] if args else 'names'
    if what == 'fetch':
        Z.fetch(n, rating)
    elif what == 'prepare':
        Z.prepare(luma_sigma=luma, chroma_sigma=chroma)
    elif what == 'stats':
        Z.print_stats(Z.stats())
    elif what == 'names':
        print(' '.join(Z.name(e['id']) for e in Z.load_index()))
    else:
        raise SystemExit(f'zxart {what}? fetch, prepare, stats or names')


def rank(names, by, metrics, with_reference):
    result = M.rank(names, J.load, by, metrics, with_reference)
    M.print_rank(result, metrics)
    for name, table in result['by_picture'].items():
        best = sorted(table.items(), key=lambda kv: -np.nan_to_num(kv[1][0], nan=-1))[:5]
        print(f'{name}: ' + ', '.join(f'{m} {a:.2f}/{n}' for m, (a, n) in best))


def fit(names, by, method, n, seed):
    F.fit(names, J.load, by, method, n, seed)


def parse_crop(s):
    r0, c0, r1, c1 = (int(x) for x in s.split(','))
    return r0, c0, r1, c1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('freeze', 'run', 'compare', 'regions', 'paint', 'render', 'variants', 'judge',
                                        'zxart', 'rank', 'fit'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--set', action='append', help='Metric, Eye or Select param=value')
    ap.add_argument('--method', default=None, choices=(PROJECT, *METHODS),
                    help="selection method; run and render default to the project's own settings, fit to each method")
    ap.add_argument('--sheets', help='folder for contact sheets')
    ap.add_argument('--out', default='.')
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--zoom', type=int, default=3)
    ap.add_argument('--crop', action='append', default=[], help='r0,c0,r1,c1 cells, inclusive (render)')
    ap.add_argument('--variant', help='a variant id or .scr file in place of the method\'s result (render)')
    ap.add_argument('--no-eye', action='store_true', help='no eye-view rows (render)')
    ap.add_argument('--n', type=int, help='variants per picture (variants: 24), pairs per picture (judge pairs: 8)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fast', action='store_true', help='variants stop at the halftone, no DBS')
    ap.add_argument('--by', default='user', help='the judge recording verdicts')
    ap.add_argument('--rating', type=float, default=4.0, help='least zxart rating (zxart fetch)')
    ap.add_argument('--metrics', help='comma-separated metric names (rank); default all')
    ap.add_argument('--no-ref', action='store_true', help='rank: judged variants only, no reference rank')
    ap.add_argument('--luma', type=float, default=Z.LUMA_SIGMA, help='px of lightness blur of a zxart source')
    ap.add_argument('--chroma', type=float, default=Z.CHROMA_SIGMA, help='px of colour blur of a zxart source')
    a = ap.parse_intermixed_args(argv)
    if a.command == 'freeze':
        freeze(a.args)
    elif a.command == 'run':
        if a.sheets:
            Path(a.sheets).mkdir(parents=True, exist_ok=True)
        run(a.args or painted_set(), parse_sets(a.set), a.sheets, method=a.method or PROJECT)
    elif a.command == 'compare':
        if a.sheets:
            Path(a.sheets).mkdir(parents=True, exist_ok=True)
        compare(a.args or painted_set(), a.sheets)
    elif a.command == 'regions':
        regions(a.args[0], a.out, a.k)
    elif a.command == 'render':
        render(a.args[0], a.out, a.zoom, [parse_crop(c) for c in a.crop], a.variant, not a.no_eye, a.method or PROJECT)
    elif a.command == 'variants':
        variants(a.args or JUDGED, a.n or 24, a.seed, a.fast)
    elif a.command == 'judge':
        judge(a.args, a.n or 8, a.seed, a.out, a.by, a.zoom if a.zoom != 3 else 2)
    elif a.command == 'zxart':
        zxart(a.args, a.n or 50, a.rating, a.luma, a.chroma)
    elif a.command == 'rank':
        rank(a.args or JUDGED, a.by, a.metrics.split(',') if a.metrics else None, not a.no_ref)
    elif a.command == 'fit':
        fit(a.args or JUDGED, a.by, a.method, a.n or 150, a.seed)
    else:
        paint(*a.args)


if __name__ == '__main__':
    sys.exit(main())
