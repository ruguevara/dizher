"""Test projects (tests/images/NAME/project.json) run through the pipeline as the app would, with the selection
params of a method, a variant or a run; their reference screens and painted cells."""
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


def reference_file(name) -> Path:
    return IMAGES / name / 'reference.scr'


def reference(name):
    """The reference's bitmap, its (paper, ink) indexes, the colours each cell shows, and the painted cells."""
    bitmap, ref_idx = read_scr(reference_file(name))
    return bitmap, ref_idx, shown(bitmap, ref_idx), painted_cells(project_graph(name))


def convert(name, method=NEWEST, memo=None, optimise=True, **params):
    """A project converted by a method with params over its preset: (graph, selection, final converter, seconds)."""
    graph = ops.apply_preset(project_graph(name, DEFAULTS), method)
    memo = memo if memo is not None else Memo()
    t = time.time()
    conv = select(graph, memo, **params)
    final = finish(graph, memo, conv, optimise)
    return graph, conv, final, time.time() - t
