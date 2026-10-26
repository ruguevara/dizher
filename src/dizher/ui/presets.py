"""Conversion presets: the Convert column's settings but the Target (the machine and palette stay the image's) and the
painted cells, as a graph of those nodes alone. Built-in ones here; a user's in `<user config>/dizher/presets/<name>.json`
in the format of project.json, shared across projects. Storage only, no UI."""
import itertools
import os
from dataclasses import replace
from pathlib import Path

from imgui_bundle import hello_imgui

from mokit import project
from mokit.graph import Graph
from mokit.paths import exists, os_path
from mokit.project import Unresolved

from .. import ops

NODES = ('metric', 'eye', 'halftoner', 'select', 'halftone', 'optimise')   # what a preset holds, in pipeline order


def of(graph) -> Graph:
    """The preset of a graph's settings."""
    return Graph(tuple((nid, graph[nid]) for nid in NODES))


def apply(graph, preset) -> Graph:
    """The graph with the preset's settings; a node the preset lacks, or has with another op or unreadable, keeps the
    graph's (a preset from before a stage was added or changed), a matrix under an old name takes its new one."""
    for nid in NODES:
        if nid in preset and preset[nid].op == graph[nid].op and not isinstance(preset[nid].params, Unresolved):
            graph = graph.with_params(nid, preset[nid].params)
    return ops.renamed(graph)


def _with(graph, nid: str, **values) -> Graph:
    return graph.with_params(nid, replace(graph[nid].params, **values))


def _builtin() -> dict:
    default = ops.make_graph()
    exact = _with(_with(_with(default, 'select', coherence=0.0, luma_noise=0.1, surface=0.0),
                        'halftone', dithering=0.0), 'optimise', enabled=False)
    return {
        'Default': (of(default), "The settings a new project starts with: pairs scored on each cell's exact mixture"),
        'Halftoned (0.2.4)': (of(ops.apply_preset(default, ops.LEGACY)),
                              'Pairs scored on one halftone of their mixture, with the values that method was tuned '
                              'with: the conversion of 0.2.4'),
        'Exact screen': (of(exact), "A picture already drawn for the screen, at its size in its colours and two to a "
                                    "cell, kept pixel for pixel: no coherence, dithering or optimiser, and a little "
                                    "luma noise so a dim colour on black stays dim. Leave Tune at its defaults. In the "
                                    "file a cell's ink and paper may swap, and a cell of one colour gets a second of "
                                    "its own"),
    }


BUILTIN = _builtin()   # name -> (preset, what it is for)


def folder() -> Path:
    """Where a user's presets live: beside the app's ui.ini."""
    return Path(hello_imgui.ini_folder_location(hello_imgui.IniFolderType.app_user_config_folder)) / 'dizher' / 'presets'


def _path(folder, name) -> Path:
    return Path(folder) / f'{name}.json'


def names(folder) -> list:
    """The presets' names, by name ignoring case; none when there is no folder."""
    if not exists(folder):
        return []
    return sorted((p.stem for p in Path(os_path(folder)).glob('*.json')), key=lambda n: (n.casefold(), n))


def load(folder, name) -> Graph:
    return project.loads(Path(os_path(_path(folder, name))).read_text())[0]


def save(folder, name, graph) -> None:
    """The graph's settings as preset name, new or updated."""
    Path(os_path(folder)).mkdir(parents=True, exist_ok=True)
    Path(os_path(_path(folder, name))).write_text(project.dumps(of(graph), None))


def rename(folder, old, new) -> None:
    """FileExistsError when a preset has the new name."""
    if exists(_path(folder, new)):
        raise FileExistsError(new)
    os.replace(os_path(_path(folder, old)), os_path(_path(folder, new)))


def delete(folder, name) -> None:
    Path(os_path(_path(folder, name))).unlink()


def free_name(folder, base) -> str:
    """base, or "base 2", "base 3"... when a preset or a built-in one has that name."""
    taken = set(names(folder)) | set(BUILTIN)
    return next(n for i in itertools.count(1) if (n := base if i == 1 else f'{base} {i}') not in taken)
