"""The app's own ZX Spectrum conversion of a prepared 256x192 source, from Python, at chosen settings.

    python research/metrics/zx.py IMAGE [IMAGE...] --out DIR [--coherence C ...] [--style dbs checker] [--set node.param=value ...]

writes each image's prepared source (free.prepare: centre crop, area resampling in linear light) and a render per
coherence x style as PNG, and prints the run times.

`convert(source, ...)` runs the pipeline of a new project as the app opens one (Pipeline.open: ops.make_graph(), the
newest Metric method's preset, here applied once more with ops.apply_preset so a change of the defaults in ops.py
cannot drift from the preset), the source given as the picture the app would open: it is written losslessly to a
content-named PNG in the temp folder and read by the Source node, so Framing (Fill, scale 1, no shift) copies it
as it is and every Tune stage runs at its neutral defaults, exactly as for a 256x192 PNG opened in the app.

Settings, any node parameter (ops.PIPELINE's node ids, the params of their ops):

- `settings={'select.coherence': 3.0, 'halftone.dithering': 0.5}`, or the same as keywords with a double underscore:
  `select__coherence=3.0, halftone__dithering=0.5`. An unknown node or parameter is an error.
- Conveniences (each only when given, not None):
  - `coherence=C` sets `select.coherence` (Select pairs; the default, the Exact mixture preset's, is 6.0; the UI's
    slider stops at 8, the op takes any value >= 0);
  - `dbs=True/False` sets `optimise.enabled` (the Optimise stage, Direct Binary Search; on by default);
  - `dithering=D` sets `halftone.dithering` (Halftone's Dithering; 1 by default);
  - `style='dbs'` (the default) changes nothing; `style='checker'` sets `halftone.checker=True`,
    `halftone.dithering=CHECKER_DITHERING` (0.375, the middle of the 0.25-0.5 an artist uses) and
    `optimise.enabled=False`: the halftoner's (default Ordered, 'Void dispersed dots') checker style without DBS.
  Precedence, lowest first: the new project's defaults, `style`, `coherence`/`dbs`/`dithering`, `settings` and the
  double-underscore keywords. So `style='checker', dithering=0.25` is the checker style at 25%.

The result (`convert`'s dict):

- `rgb` (192, 256, 3) uint8: the final render, Optimise's `dithered_result` as Converter.save writes a PNG
  (`(dithered_result * 255).round()`), the default ZX palette at levels 205/255;
- `bitmap` (192, 256) bool: True where the pixel shows its cell's ink;
- `pairs` (24, 32, 2) int: each cell's (paper, ink) palette index, 0-15 (8-15 bright), the paper the darker;
- `labels` (24, 32) int: each cell's pair index into `pair_list`;
- `pair_list` (P, 2) int: the palette's (paper, ink) pairs in the converter's order;
- `palette` (16, 3) uint8: the palette's colours;
- `halftone_rgb` (192, 256, 3) uint8: the Halftone stage's render (before DBS; equals `rgb` when DBS is off);
- `target` (192, 256, 3) float32: the tuned picture in 0..1 the converter aims at (Detail's result);
- `graph`: the mokit graph that ran, `seconds`: the wall time of the run.

Pass one `memo` (mokit.graph.Memo) to several calls on the same source to reuse the shared stages: Prepare (~1 s)
when only Select pairs, Halftone or Optimise settings change, Select pairs when only Halftone or Optimise change.
A memo holds every stage's Converter (the candidates are ~100 MB per source), so use one per source, not one for all.
"""
import argparse
import hashlib
import sys
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from mokit.graph import Memo, evaluate
    from dizher import ops
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from mokit.graph import Memo, evaluate
    from dizher import ops

SIZE = (192, 256)
CHECKER_DITHERING = 0.375
STYLES = {
    'dbs': {},
    'checker': {'halftone.checker': True, 'halftone.dithering': CHECKER_DITHERING, 'optimise.enabled': False},
}
SOURCES = Path(tempfile.gettempdir()) / 'dizher-research-zx'   # the sources as PNG, named by their content


def new_graph():
    """A new project's graph, as the app opens an image: make_graph's defaults, the newest method's preset."""
    return ops.apply_preset(ops.make_graph(), ops.NEWEST)


def with_settings(graph, settings: dict):
    """The graph with {'node.param': value} set; an unknown node or parameter is a KeyError."""
    for name, value in settings.items():
        nid, _, param = name.partition('.')
        if nid not in graph:
            raise KeyError(f'no node {nid!r} (nodes: {", ".join(graph.ids())})')
        params = graph[nid].params
        known = [f.name for f in fields(params)]
        if param not in known:
            raise KeyError(f'{nid} has no parameter {param!r} (it has: {", ".join(known)})')
        graph = graph.with_params(nid, replace(params, **{param: value}))
    return graph


def source_path(source: np.ndarray) -> Path:
    """The source as a lossless PNG named by its content, so the Source node's key (the file's digest) is the same
    for the same pixels and a memo reuses it."""
    source = np.ascontiguousarray(source)
    if source.shape != (*SIZE, 3) or source.dtype != np.uint8:
        raise ValueError(f'source must be {SIZE + (3,)} uint8, got {source.shape} {source.dtype}')
    path = SOURCES / (hashlib.sha1(source.tobytes()).hexdigest() + '.png')
    if not path.exists():
        SOURCES.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.png')
        Image.fromarray(source).save(tmp)
        tmp.replace(path)
    return path


def as_ubyte(rgb: np.ndarray) -> np.ndarray:
    """A render in 0..1 as Converter.save writes it to a PNG."""
    return (rgb * 255).round().astype(np.uint8)


def graph_for(source: np.ndarray, coherence=None, dbs=None, dithering=None, style: str = 'dbs',
              settings: dict = None, **params):
    """The graph convert runs; see the module's docstring for the arguments."""
    if style not in STYLES:
        raise ValueError(f'style must be one of {", ".join(STYLES)}, not {style!r}')
    chosen = dict(STYLES[style])
    for name, value in (('select.coherence', coherence), ('optimise.enabled', dbs), ('halftone.dithering', dithering)):
        if value is not None:
            chosen[name] = value
    chosen.update(settings or {})
    for key, value in params.items():
        nid, sep, param = key.partition('__')
        if not sep:
            raise TypeError(f'unknown keyword {key!r}: a node parameter is node__param, e.g. select__coherence')
        chosen[f'{nid}.{param}'] = value
    graph = new_graph()
    graph = graph.with_params('source', replace(graph['source'].params, path=source_path(source)))
    return with_settings(graph, chosen)


def convert(source: np.ndarray, coherence=None, dbs=None, dithering=None, style: str = 'dbs', settings: dict = None,
            memo: Memo = None, **params) -> dict:
    """A (192, 256, 3) sRGB uint8 source converted by the app's pipeline at these settings (the module's docstring):
    the render, the bitmap, the pairs."""
    graph = graph_for(source, coherence, dbs, dithering, style, settings, **params)
    memo = memo if memo is not None else Memo()
    t0 = time.perf_counter()
    result = evaluate(graph, 'optimise', memo)
    seconds = time.perf_counter() - t0
    halftone = evaluate(graph, 'halftone', memo)
    pair_list = np.array(list(result.palette.iter_idxs_pairs()))
    labels = np.asarray(result.best_attr_indexes).copy()
    return dict(rgb=as_ubyte(result.dithered_result), bitmap=result.dithered_bitmap > 0.5,
                pairs=pair_list[labels], labels=labels, pair_list=pair_list, palette=result.palette.as_ubyte().copy(),
                halftone_rgb=as_ubyte(halftone.dithered_result), target=np.asarray(result.image_rgb, np.float32),
                graph=graph, seconds=seconds)


def _setting(text: str):
    """'node.param=value' -> (name, value): the value as Python literal when it parses, else the string."""
    import ast
    name, _, value = text.partition('=')
    try:
        return name, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return name, value


def main():
    from free import prepare
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('images', nargs='+')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--coherence', nargs='+', type=float, default=[None], help='Select pairs coherence values; the '
                    f'default {ops.NEWEST_PRESET["coherence"]} when not given')
    ap.add_argument('--style', nargs='+', default=['dbs'], choices=list(STYLES))
    ap.add_argument('--set', nargs='*', default=[], metavar='NODE.PARAM=VALUE', help='any other node parameter')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = dict(map(_setting, args.set))
    for path in args.images:
        stem = Path(path).stem
        src = prepare(path)
        Image.fromarray(src).save(args.out / f'{stem}_source.png')
        memo = Memo()
        for style in args.style:
            for coherence in args.coherence:
                r = convert(src, coherence=coherence, style=style, settings=settings, memo=memo)
                c = r['graph']['select'].params.coherence
                name = f'{stem}_{style}_coh{c:g}.png'
                Image.fromarray(r['rgb']).save(args.out / name)
                print(f'{name}: {r["seconds"]:.2f} s, {len(np.unique(r["labels"]))} pairs used', flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
