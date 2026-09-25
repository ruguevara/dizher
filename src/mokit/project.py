"""A `.motool` project folder: `project.json` (graph + view), `cache/` (memo, gitignored), `build/` (exports).

Paths in params are stored relative to the folder. Loading is tolerant: an op whose package is missing keeps its
node (params as an `Unresolved` mapping) and a diagnostic; unknown fields are dropped with a diagnostic.
"""
from __future__ import annotations

import json
import os
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass, MISSING as DC_MISSING
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .graph import Graph, GraphError, Node, Op, _strip_annotated, _type_hints

FORMAT = "mokit-graph/1"
PROJECT_FILE = "project.json"
GITIGNORE = "cache/\n"


@dataclass(frozen=True)
class Unresolved:
    """Params of a node whose op could not be imported; round-trips through save unchanged."""
    items: Tuple[Tuple[str, Any], ...]


@dataclass(frozen=True)
class Project:
    folder: Path
    graph: Graph
    view: dict
    diagnostics: Tuple[str, ...] = ()

    @property
    def cache_dir(self) -> Path:
        return self.folder / "cache"

    @property
    def build_dir(self) -> Path:
        return self.folder / "build"


# ----- values <-> json -----------------------------------------------------------------------------------

def _to_json_value(value, root: Optional[Path]):
    if isinstance(value, Path):
        if root is None:
            return str(value)
        try:
            return os.path.relpath(value, root)
        except ValueError:      # different drive on Windows
            return str(value)
    if isinstance(value, tuple):
        return [_to_json_value(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_value(v, root) for k, v in value.items()}
    return value


def params_to_json(params, root: Optional[Path]) -> Optional[dict]:
    if params is None:
        return None
    if isinstance(params, Unresolved):
        return dict(params.items)
    return {f.name: _to_json_value(getattr(params, f.name), root) for f in fields(params)}


def _coerce(value, tp, root: Optional[Path]):
    tp = _strip_annotated(tp)
    origin = typing.get_origin(tp)
    if origin is typing.Union or (hasattr(typing, "UnionType") and isinstance(tp, getattr(__import__("types"), "UnionType"))):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], root) if len(args) == 1 else value
    if tp is Path:
        if value is None:
            return None
        return Path(value) if os.path.isabs(value) or root is None else Path(os.path.normpath(root / value))
    if origin is tuple:
        args = typing.get_args(tp)
        item = args[0] if args else Any
        return tuple(_coerce(v, item, root) for v in value)
    if tp is bool:
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return bool(value)
    if tp in (int, float, str):
        return tp(value)
    return value


def params_from_json(params_type, data: Optional[dict], root: Optional[Path], diagnostics: List[str], where: str):
    if params_type is None:
        return None
    data = dict(data or {})
    hints = _type_hints(params_type)
    values = {}
    for f in fields(params_type):
        if f.name in data:
            raw = data.pop(f.name)
            try:
                values[f.name] = _coerce(raw, hints.get(f.name, f.type), root)
            except (TypeError, ValueError) as e:
                diagnostics.append(f"{where}: {f.name}={raw!r} does not coerce ({e}); using default")
        elif f.default is DC_MISSING and f.default_factory is DC_MISSING:
            raise GraphError(f"{where}: required field {f.name!r} missing")
    for name in data:
        diagnostics.append(f"{where}: unknown field {name!r} ignored")
    return params_type(**values)


# ----- graph <-> json ------------------------------------------------------------------------------------

def graph_to_json(graph: Graph, view: Optional[dict], root: Optional[Path]) -> dict:
    return {
        "format": FORMAT,
        "nodes": {
            nid: {"op": node.op, "params": params_to_json(node.params, root), "inputs": list(node.inputs)}
            for nid, node in graph.nodes
        },
        "view": dict(view or {}),
    }


def graph_from_json(data: dict, root: Optional[Path]) -> Tuple[Graph, dict, Tuple[str, ...]]:
    if data.get("format") != FORMAT:
        raise GraphError(f"unknown project format {data.get('format')!r}, expected {FORMAT!r}")
    diagnostics: List[str] = []
    nodes = []
    for nid, entry in data.get("nodes", {}).items():
        op_id = entry["op"]
        raw = entry.get("params") or {}
        inputs = tuple(entry.get("inputs", ()))
        try:
            op = Op.resolve(op_id)
        except GraphError as e:
            diagnostics.append(str(e))
            nodes.append((nid, Node(op_id, Unresolved(tuple(sorted(raw.items()))), inputs)))
            continue
        params = params_from_json(op.params_type, raw, root, diagnostics, f"node {nid!r}")
        nodes.append((nid, Node(op_id, params, inputs)))
    return Graph(tuple(nodes)), dict(data.get("view", {})), tuple(diagnostics)


def dumps(graph: Graph, view: Optional[dict], root: Optional[Path] = None) -> str:
    """JSON text; with root=None paths stay absolute (user prefs), with a folder they are stored relative to it."""
    return json.dumps(graph_to_json(graph, view, root), indent=2) + "\n"   # node order is document order


def loads(text: str, root: Optional[Path] = None) -> Tuple[Graph, dict, Tuple[str, ...]]:
    return graph_from_json(json.loads(text), root)


# ----- folder ----------------------------------------------------------------------------------------------

def load_project(folder) -> Project:
    folder = Path(folder).absolute()
    graph, view, diagnostics = loads((folder / PROJECT_FILE).read_text(), folder)
    return Project(folder, graph, view, diagnostics)


def save_project(folder, graph: Graph, view: Optional[dict] = None) -> Path:
    folder = Path(folder).absolute()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / PROJECT_FILE
    path.write_text(dumps(graph, view, folder))
    return path


def create_project(folder, graph: Graph = Graph(), view: Optional[dict] = None) -> Project:
    """Make a project folder with an empty (or given) graph and a .gitignore for cache/."""
    folder = Path(folder).absolute()
    if (folder / PROJECT_FILE).exists():
        raise FileExistsError(folder / PROJECT_FILE)
    save_project(folder, graph, view)
    (folder / ".gitignore").write_text(GITIGNORE)
    return Project(folder, graph, dict(view or {}))
