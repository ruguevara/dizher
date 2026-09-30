"""Test projects (tests/images/NAME/project.json) run through the pipeline as the app would, with the selection
params of a method, a variant or a run; their reference screens and painted cells.

A hand-painted project is its own reference: the painting is meant with the project's Tune and converter settings as
saved, so the reference is the project converted as saved, painted cells included, frozen to cache/reference.scr
(refrozen when project.json changes) and judged as the picture it shows, never attribute by attribute. A project
with a reference.scr of its own (the zxart set) uses that."""
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from mokit.graph import Memo, evaluate
from mokit.project import load_project

from dizher import ops
from dizher.converter.energy import NEWEST

from .scr import read_scr, shown

IMAGES = Path(__file__).parent.parent / 'images'
DEFAULTS = ('metric', 'eye', 'select')      # nodes a run resets, so every image is judged under one setting


def project_graph(name: str, defaults=()):
    """The project's graph in the current pipeline's shape (older projects lack nodes, e.g. Overpaint); the nodes in
    defaults keep the pipeline's defaults instead of the project's values."""
    project = load_project(IMAGES / name)
    graph = ops.make_graph()
    for nid in graph.ids():
        if nid in project.graph and nid not in defaults:
            graph = graph.with_params(nid, project.graph[nid].params)
    return graph


def painted_cells(graph) -> set:
    return {(r, c) for r, c, *_ in graph['overpaint'].params.overrides}


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


def finish(graph, memo, selection, optimise=True):
    """The project's Halftone and Optimise run on a selection, as the app would; optimise False stops at the halftone."""
    halftone = ops.halftone(selection)
    return ops.optimise(halftone, **vars(graph['optimise'].params)) if optimise else halftone


def project_hash(name) -> str:
    return hashlib.sha1((IMAGES / name / 'project.json').read_bytes()).hexdigest()[:12]


def has_reference(name) -> bool:
    """A screen of its own, a frozen one, or painted cells to freeze one from."""
    return (IMAGES / name / 'reference.scr').exists() or (IMAGES / name / 'cache' / 'reference.scr').exists() \
        or bool(painted_cells(project_graph(name)))


def freeze(name, force=False, log=None) -> Path:
    """The project as saved converted whole (painting included) to cache/reference.scr, unless it is there for this
    project.json already; returns the file."""
    cache = IMAGES / name / 'cache'
    scr, meta = cache / 'reference.scr', cache / 'reference.json'
    h = project_hash(name)
    if not force and scr.exists() and meta.exists() and json.loads(meta.read_text()).get('project') == h:
        return scr
    graph = project_graph(name)
    t = time.time()
    conv = evaluate(graph, 'optimise', Memo())
    cache.mkdir(parents=True, exist_ok=True)
    conv.save(str(scr))
    meta.write_text(json.dumps(dict(project=h, painted=len(painted_cells(graph)), frozen=time.strftime('%Y-%m-%d'))))
    if log:
        log(f'{name}: reference frozen, {len(painted_cells(graph))} painted cells, {time.time() - t:.1f} s')
    return scr


def reference_file(name) -> Path:
    """The picture's reference screen: its own reference.scr, else the project frozen (freeze)."""
    own = IMAGES / name / 'reference.scr'
    return own if own.exists() else freeze(name)


def reference(name):
    """The reference's bitmap, its (paper, ink) indexes, the colours each cell shows, and the painted cells."""
    bitmap, ref_idx = read_scr(reference_file(name))
    return bitmap, ref_idx, shown(bitmap, ref_idx), painted_cells(project_graph(name))


def convert_as_saved(name, memo=None, hide_painting=True):
    """The project under its own settings, the painted cells hidden (the conversion the user corrected):
    (graph, selection, final converter, seconds)."""
    graph = project_graph(name)
    if hide_painting:
        graph = graph.with_params('overpaint', replace(graph['overpaint'].params, overrides=()))
    memo = memo if memo is not None else Memo()
    t = time.time()
    conv = evaluate(graph, 'select', memo)
    final = finish(graph, memo, conv)
    return graph, conv, final, time.time() - t


def convert(name, method=NEWEST, memo=None, optimise=True, **params):
    """A project converted by a method with params over its preset: (graph, selection, final converter, seconds)."""
    graph = ops.apply_preset(project_graph(name, DEFAULTS), method)
    memo = memo if memo is not None else Memo()
    t = time.time()
    conv = select(graph, memo, **params)
    final = finish(graph, memo, conv, optimise)
    return graph, conv, final, time.time() - t
