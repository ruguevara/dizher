"""A colouring is a label map, one pair per cell; it is judged only as the project renders it: through the project's
own pipeline (its Tune, Target, Halftoner, Metric, Select pairs and Optimise), painted as a whole field. Cells are compared by the colours they show in that render, so paper/ink order and a colour a cell does not
show do not count."""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from mokit.graph import Memo, evaluate
from mokit.project import load_project

from dizher import ops

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / 'tests' / 'images'
DATA = Path(__file__).resolve().parent / 'data'   # renders and pairs, rebuilt by build.py; not in git
SHARED = (('metric', 'chroma'), ('select', 'luma_noise'), ('select', 'chroma_noise'))   # Optimise's own since PLAN step 2


def painted_projects() -> list:
    """Names of the test projects with painted cells."""
    names = []
    for folder in sorted(p.parent for p in IMAGES.glob('*/project.json')):
        if project_graph(folder.name)['overpaint'].params.overrides:
            names.append(folder.name)
    return names


def project_graph(name: str):
    """The project's graph in the current pipeline's shape (older projects lack nodes); a loose picture in
    tests/images, without a project, gets a new project's graph, as the app opens it."""
    graph = ops.make_graph()
    if not (IMAGES / name / 'project.json').exists():
        path, = IMAGES.glob(f'{name}.*')
        return graph.with_params('source', replace(graph['source'].params, path=path))
    project = load_project(IMAGES / name)
    for nid in graph.ids():
        if nid in project.graph:
            graph = graph.with_params(nid, project.graph[nid].params)
    return graph


class Project:
    """One test project: its graph, a memo shared by its renders, the selection and the pair list."""

    def __init__(self, name: str, **graph_params) -> None:
        """graph_params: {node id: {param: value}} changed from the project's own, the selection and energy too."""
        self.name = name
        self.graph = project_graph(name)
        self.graph = self.changed(**graph_params)
        self.memo = Memo()
        self.selection = evaluate(self.graph, 'select', self.memo)   # Select pairs at the project's settings
        self.pairs = np.array(list(self.selection.palette.iter_idxs_pairs()))   # label -> (paper, ink)
        self.shape = self.selection.best_attr_indexes.shape

    def labels(self, overrides=None) -> np.ndarray:
        """(R, C) the label map Overpaint makes of these painted cells (the project's own by default)."""
        if overrides is None:
            overrides = self.graph['overpaint'].params.overrides
        return ops.overpaint(self.selection, tuple(overrides)).best_attr_indexes.copy()

    def render(self, labels: np.ndarray, **graph_params) -> np.ndarray:
        """(H, W, 3) sRGB 0..1: the label map painted as a whole field and rendered as the project exports it.
        graph_params: {node id: {param: value}} changed for this render only (a null pair's halftone origin)."""
        graph = self.changed(**graph_params)
        cells = tuple((r, c, int(p), int(i)) for (r, c), (p, i) in zip(np.ndindex(*self.shape),
                                                                      self.pairs[labels].reshape(-1, 2)))
        graph = graph.with_params('overpaint', replace(graph['overpaint'].params, overrides=cells))
        return evaluate(graph, 'optimise', self.memo).dithered_result.astype(np.float32)

    def changed(self, **graph_params):
        """The project's graph with these node params changed: {node id: {param: value}}. A Metric chroma or Select
        pairs noise weight moves Optimise's too, unless that is given: one setting, as in every round before step 2."""
        graph_params = {nid: dict(params) for nid, params in graph_params.items()}
        for nid, k in SHARED:
            if k in graph_params.get(nid, {}):
                graph_params.setdefault('optimise', {}).setdefault(k, graph_params[nid][k])
        graph = self.graph
        for nid, params in graph_params.items():
            graph = graph.with_params(nid, replace(graph[nid].params, **params))
        return graph

    def convert(self, **graph_params) -> np.ndarray:
        """(H, W, 3) sRGB 0..1: Select pairs' colouring, no cell painted, at these node params, rendered as exported."""
        graph = self.changed(**graph_params, overpaint=dict(overrides=()))
        return evaluate(graph, 'optimise', self.memo).dithered_result.astype(np.float32)

    def target(self) -> np.ndarray:
        """(H, W, 3) sRGB 0..1: the tuned picture the converter aims at."""
        return self.selection.image_rgb.astype(np.float32)

    def energy(self, labels: np.ndarray) -> float:
        """The project's selection energy of a label map (eye-model error, seams, coherence)."""
        return self.selection.energy.energy(labels)


def shown(image: np.ndarray, cell=(8, 8)) -> np.ndarray:
    """(R, C) int: a code for the set of colours each cell shows; equal codes, equal sets."""
    h, w = cell
    q = (image * 255).round().astype(np.int64)
    code = (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]
    R, C = code.shape[0] // h, code.shape[1] // w
    blocks = np.sort(code.reshape(R, h, C, w).transpose(0, 2, 1, 3).reshape(R, C, h * w), axis=-1)
    first = np.concatenate([np.ones((R, C, 1), bool), blocks[..., 1:] != blocks[..., :-1]], -1)
    sets = np.where(first, blocks, -1)   # each colour once, the repeats -1
    sets = np.sort(sets, axis=-1)
    return np.array([[hash(row.tobytes()) for row in r] for r in sets])


def segments(target: np.ndarray, k: int = 32, position: float = 2.0, cell=(8, 8)) -> np.ndarray:
    """(R, C) segment of each cell: k-means of the cells' mean CIELAB and their position, so a segment is a surface
    of the picture. Seeded, so a rebuild gives the same segments."""
    h, w = cell
    R, C = target.shape[0] // h, target.shape[1] // w
    lab = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2Lab).reshape(R, h, C, w, 3).mean((1, 3))
    r, c = np.indices((R, C))
    feats = np.concatenate([lab, position * np.stack([r, c], -1) * 100 / C], -1).reshape(-1, 5).astype(np.float32)
    cv2.setRNGSeed(1)
    _, km, _ = cv2.kmeans(feats, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5,
                          cv2.KMEANS_PP_CENTERS)
    return km.reshape(R, C)


def window(cells: np.ndarray, grow: int = 2, cell=(8, 8)) -> np.ndarray:
    """(H, W) bool pixel mask of the cells grown by grow cells each way: where a local change is judged."""
    k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    grown = cv2.dilate(cells.astype(np.uint8), k) > 0
    return np.repeat(np.repeat(grown, cell[0], 0), cell[1], 1)
