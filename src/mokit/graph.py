"""Node graph of pure ops with results memoized by key.

A node is an op (import path), a frozen params dataclass and input node ids. Its key is
``(op, op version, params-with-file-digests, *input keys)``: the same key means the same result, on any machine.
Evaluation pulls results through a ``Memo`` (RAM, optionally disk in the value's native file form).
Nothing here knows about screens, images or UIs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import shutil
import importlib
import inspect
import os
import pickle
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Node", "Graph", "GraphError", "Op", "meta", "evaluate", "ready_steps", "next_step", "run_step", "Memo", "Digests", "UNBOUND", "discover_ops",
           "FileType", "FILE_TYPES", "register_file_type", "file_fields", "output_fields", "MISSING", "type_name", "accepts", "mro_lookup"]


class GraphError(Exception):
    """`node` names the node the error belongs to, when there is one (next_step sets it)."""
    def __init__(self, message: str, node: Optional[str] = None) -> None:
        super().__init__(message)
        self.node = node


MISSING = object()


# ----- params metadata -------------------------------------------------------------------------------

def meta(*, min=None, max=None, step=None, choices=(), editor="", label="", help="", output=False) -> dict:
    """`field(metadata=meta(...))`: editor hints for a params field. Keyword-only so typos fail.
    `output=True` marks a Path the op writes: its location, not its content, enters the key."""
    return {k: v for k, v in dict(min=min, max=max, step=step, choices=tuple(choices), editor=editor,
                                  label=label, help=help, output=output).items()
            if v is not None and v is not False and v != "" and v != ()}


def _type_hints(cls) -> dict:
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except Exception:  # unresolved forward references: fall back to raw annotations
        return dict(getattr(cls, "__annotations__", {}))


def _strip_annotated(tp):
    """Annotated[X, ...] -> X; Optional[X] / X | None -> X."""
    if typing.get_origin(tp) is typing.Annotated:
        tp = typing.get_args(tp)[0]
    if typing.get_origin(tp) in (typing.Union, getattr(types, "UnionType", None)):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _strip_annotated(args[0])
    return tp


def _type_args(tp) -> tuple:
    """The alternatives of a Union / X | Y (None dropped), else (tp,)."""
    tp = _strip_annotated(tp)
    if typing.get_origin(tp) in (typing.Union, getattr(types, "UnionType", None)):
        return tuple(a for a in typing.get_args(tp) if a is not type(None))
    return (tp,)

def type_name(tp) -> str:
    """Short display name of a port type: "Frames", "ClusterResult | DataFrame", "any" for untyped."""
    if tp is None or tp is Any or tp is inspect.Parameter.empty:
        return "any"
    if isinstance(tp, str):
        return tp.rsplit(".", 1)[-1]
    return " | ".join(getattr(a, "__name__", str(a)) for a in _type_args(tp))

def accepts(expected, actual) -> bool:
    """Can a value typed `actual` feed a slot typed `expected`? Untyped on either side accepts everything."""
    if type_name(expected) == "any" or type_name(actual) == "any":
        return True
    exp, act = _type_args(expected), _type_args(actual)
    if any(isinstance(t, str) for t in exp + act):     # unresolved forward references: compare by name
        return bool({type_name(e) for e in exp} & {type_name(a) for a in act})
    return any(isinstance(a, type) and isinstance(e, type) and issubclass(a, e) for a in act for e in exp)

def file_fields(params_type) -> Tuple[str, ...]:
    """Names of the fields typed `Path` or `tuple[Path, ...]`: their content, not their path, enters the key.
    Fields marked `meta(output=True)` are excluded (their path string is the key)."""
    if params_type is None or not is_dataclass(params_type):
        return ()
    names = []
    for f in fields(params_type):
        if f.metadata.get("output"):
            continue
        tp = _strip_annotated(_type_hints(params_type).get(f.name, f.type))
        if tp is Path or (typing.get_origin(tp) is tuple and Path in typing.get_args(tp)):
            names.append(f.name)
    return tuple(names)


def output_fields(params_type) -> Tuple[str, ...]:
    """Names of the `meta(output=True)` Path fields: paths the op writes, filled in by the host from its output template."""
    if params_type is None or not is_dataclass(params_type):
        return ()
    names = []
    for f in fields(params_type):
        if not f.metadata.get("output"):
            continue
        tp = _strip_annotated(_type_hints(params_type).get(f.name, f.type))
        if tp is Path or (typing.get_origin(tp) is tuple and Path in typing.get_args(tp)):
            names.append(f.name)
    return tuple(names)


# ----- file digests ------------------------------------------------------------------------------------

def _stat_signature(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, st.st_ino


def _list_dir(path: Path) -> List[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file())


def _digest_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:   # ponytail: whole-file sha1; chunked/partial digest if sources get huge
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Digests:
    """Content digests of files and folders, memoized behind stat signatures (mtime, size, inode).

    `refresh()` re-stats every file seen so far and re-lists every folder seen so far (a file added, removed or
    renamed under a folder changes its listing); the façade calls it about once a second so keys stay cheap in
    between (`digest()` never touches the disk for a file whose last refresh is current).
    """

    def __init__(self) -> None:
        self._files: Dict[Path, Tuple[Any, str]] = {}   # path -> (stat signature, digest)
        self._dirs: Dict[Path, List[Path]] = {}         # folder -> its last listing
        self._dirty = True

    def refresh(self) -> bool:
        """Re-stat every known file, re-list every known folder; returns True when some digest changed."""
        changed = False
        for path, (sig, digest) in list(self._files.items()):
            if _stat_signature(path) != sig:
                del self._files[path]
                changed = True
        for path, entries in list(self._dirs.items()):   # ponytail: one rglob per folder per poll; folder mtimes if listings get slow
            if _list_dir(path) != entries:
                del self._dirs[path]
                changed = True
        self._dirty = True
        return changed

    def digest(self, path) -> str:
        path = Path(path)
        if path.is_dir():
            entries = self._dirs[path] = _list_dir(path)
            h = hashlib.sha1()
            for p in entries:
                h.update(f"{p.relative_to(path)}:{self._file(p)}\n".encode())
            return h.hexdigest()
        return self._file(path)

    def _file(self, path: Path) -> str:
        cached = self._files.get(path)
        if cached is not None:
            return cached[1]
        sig = _stat_signature(path)
        digest = "missing" if sig is None else _digest_file(path)
        self._files[path] = (sig, digest)
        return digest


# ----- nodes and graph ---------------------------------------------------------------------------------

UNBOUND = ""   # an input slot without a link: the op receives None there


@dataclass(frozen=True)
class Node:
    op: str                          # import path "package.module:attr"
    params: Any = None               # frozen dataclass instance declared by the op, or None
    inputs: Tuple[str, ...] = ()     # upstream node ids in the op's positional order; UNBOUND for an empty slot


@dataclass(frozen=True)
class Graph:
    nodes: Tuple[Tuple[str, Node], ...] = ()

    def __getitem__(self, node_id: str) -> Node:
        for nid, node in self.nodes:
            if nid == node_id:
                return node
        raise KeyError(node_id)

    def __contains__(self, node_id: str) -> bool:
        return any(nid == node_id for nid, _ in self.nodes)

    def ids(self) -> Tuple[str, ...]:
        return tuple(nid for nid, _ in self.nodes)

    def with_node(self, node_id: str, node: Node) -> "Graph":
        if node_id in self:
            return Graph(tuple((nid, node if nid == node_id else n) for nid, n in self.nodes))
        return Graph(self.nodes + ((node_id, node),))

    def without(self, node_id: str) -> "Graph":
        """Drop a node; slots that pointed at it become UNBOUND."""
        return Graph(tuple((nid, dataclasses.replace(n, inputs=tuple(UNBOUND if i == node_id else i for i in n.inputs)))
                           for nid, n in self.nodes if nid != node_id))

    def with_params(self, node_id: str, params) -> "Graph":
        return self.with_node(node_id, dataclasses.replace(self[node_id], params=params))

    def with_inputs(self, node_id: str, inputs: Tuple[str, ...]) -> "Graph":
        return self.with_node(node_id, dataclasses.replace(self[node_id], inputs=tuple(inputs)))

    def order(self) -> Tuple[str, ...]:
        """Topological order, deterministic (document order among independent nodes). Cycle -> GraphError."""
        done, out, visiting = set(), [], set()

        def visit(nid, path):
            if nid in done:
                return
            if nid in visiting:
                raise GraphError(f"cycle through {' -> '.join(path + [nid])}")
            if nid not in self:
                raise GraphError(f"unknown node {nid!r} referenced by {path[-1]!r}")
            visiting.add(nid)
            for dep in self[nid].inputs:
                if dep != UNBOUND:
                    visit(dep, path + [nid])
            visiting.discard(nid)
            done.add(nid)
            out.append(nid)

        for nid, _ in self.nodes:
            visit(nid, [])
        return tuple(out)

    def key(self, node_id: str, digests: Optional[Digests] = None, _cache: Optional[dict] = None) -> tuple:
        """(op, op version, params with file digests, *input keys). Pass one `_cache` dict across calls to avoid
        re-walking. The version is "" for an op that does not resolve (its package is not installed)."""
        if _cache is not None and node_id in _cache:
            return _cache[node_id]
        node = self[node_id]
        try:
            version = Op.resolve(node.op).version
        except GraphError:
            version = ""
        digests = digests or Digests()
        params = node.params
        if is_dataclass(params) and not isinstance(params, type):
            files = file_fields(type(params))
            items = []
            for f in fields(params):
                value = getattr(params, f.name)
                if f.name in files:
                    paths = value if isinstance(value, tuple) else (value,)
                    value = tuple(digests.digest(p) for p in paths if p is not None)
                items.append((f.name, value))
            params_key = (type(params).__qualname__, tuple(items))
        else:
            params_key = params
        try:
            hash(params_key)
        except TypeError as e:
            raise GraphError(f"params of {node_id!r} are not hashable: {e}") from e
        inputs = tuple(None if i == UNBOUND else self.key(i, digests, _cache) for i in node.inputs)
        result = (node.op, version, params_key, *inputs)
        if _cache is not None:
            _cache[node_id] = result
        return result


# ----- ops ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Op:
    """A pure function over input results and a params dataclass.

    `fn(*inputs, params=..., progress=...)`; `progress(fraction_or_None, text)` is the cancel point.
    `Op.resolve(import_path)` loads one from any package; a plain annotated function is wrapped by `from_function`.
    """
    name: str
    fn: Callable
    params_type: Optional[type] = None
    result_type: Optional[type] = None
    inputs: Tuple[str, ...] = ()        # input names, for pins
    input_types: Tuple[Any, ...] = ()   # one per input: the annotation, or None when untyped
    disk_cache: bool = False
    version: str = ""                   # enters every key; bump by hand when the op's result changes for equal inputs
    wants_progress: bool = True
    _unpack_params: bool = False        # from_function with keyword params: call fn(**asdict(params))

    def __call__(self, inputs, params=None, progress=None):
        progress = progress or (lambda fraction, text: None)
        kwargs = {}
        if self._unpack_params:
            kwargs.update(dataclasses.asdict(params) if params is not None else {})
        elif self.params_type is not None:
            kwargs["params"] = params if params is not None else self.params_type()
        if self.wants_progress:
            kwargs["progress"] = progress
        return self.fn(*inputs, **kwargs)

    def default_params(self):
        return self.params_type() if self.params_type is not None else None

    def input_type(self, slot: int):
        return self.input_types[slot] if slot < len(self.input_types) else None

    def accepts(self, slot: int, result_type) -> bool:
        """Can an output typed `result_type` be linked into input `slot`?"""
        return accepts(self.input_type(slot), result_type)

    @property
    def short_name(self) -> str:
        return self.name.rsplit(":", 1)[-1]

    _registry: typing.ClassVar[Dict[str, "Op"]] = {}

    @classmethod
    def resolve(cls, op_id: str) -> "Op":
        """Import "package.module:attr". ImportError/AttributeError -> GraphError naming the package."""
        op = cls._registry.get(op_id)
        if op is not None:
            return op
        module_name, _, attr = op_id.partition(":")
        if not attr:
            raise GraphError(f"op id must be 'package.module:attr', got {op_id!r}")
        try:
            obj = getattr(importlib.import_module(module_name), attr)
        except ImportError as e:
            raise GraphError(f"op {op_id!r} needs package {module_name.split('.')[0]!r}: {e}") from e
        except AttributeError as e:
            raise GraphError(f"op {op_id!r}: {e}") from e
        op = obj if isinstance(obj, Op) else cls.from_function(obj, op_id)
        if op.name != op_id:
            op = dataclasses.replace(op, name=op_id)
        cls._registry[op_id] = op
        return op

    @classmethod
    def from_function(cls, fn: Callable, name: Optional[str] = None, *, result_type=MISSING,
                      disk_cache: bool = False, version: str = "") -> "Op":
        """Wrap a plain function. Positional parameters are inputs; a `params` parameter typed with a dataclass is
        the params type; otherwise keyword parameters with defaults become one (Annotated[..., meta()] kept);
        `progress` is passed when accepted; the return annotation is the result type. A function-scanned module
        (mokit.ops) opts one op into the disk memo by setting `fn.disk_cache = True` and `fn.version`."""
        disk_cache = disk_cache or getattr(fn, "disk_cache", False)
        version = version or getattr(fn, "version", "")
        sig = inspect.signature(fn)
        hints = _type_hints(fn)
        inputs, input_types, kw = [], [], []
        params_type, wants_progress, unpack = None, False, False
        for p in sig.parameters.values():
            if p.name == "progress":
                wants_progress = True
            elif p.name == "params":
                params_type = _strip_annotated(hints.get("params", p.annotation))
                if not is_dataclass(params_type):
                    raise GraphError(f"{fn.__name__}: `params` must be typed with a dataclass")
            elif p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                inputs.append(p.name)
                tp = hints.get(p.name, p.annotation)
                input_types.append(None if tp is inspect.Parameter.empty else tp)
            elif p.default is not inspect.Parameter.empty:
                kw.append(p)
        if params_type is None and kw:
            specs = []
            for p in kw:
                tp = hints.get(p.name, p.annotation)
                metadata = {}
                if typing.get_origin(tp) is typing.Annotated:
                    tp, *extras = typing.get_args(tp)
                    for extra in extras:
                        if isinstance(extra, dict):
                            metadata.update(extra)
                specs.append((p.name, tp if tp is not inspect.Parameter.empty else Any,
                              field(default=p.default, metadata=metadata)))
            params_type = dataclasses.make_dataclass(f"{fn.__name__}_params", specs, frozen=True)
            unpack = True
        if result_type is MISSING:
            result_type = hints.get("return")
            if result_type is inspect.Signature.empty:
                result_type = None
        return cls(name=name or f"{fn.__module__}:{fn.__qualname__}", fn=fn, params_type=params_type,
                   result_type=result_type, inputs=tuple(inputs), input_types=tuple(input_types), disk_cache=disk_cache,
                   version=version, wants_progress=wants_progress, _unpack_params=unpack)


def discover_ops() -> Dict[str, Op]:
    """Ops offered by installed packages: every module named in the `mokit.ops` entry-point group, its public
    `Op` instances (`"module:name"` -> Op). A module with no `Op` instances at all falls back to auto-wrapping its
    public annotated functions (mokit.ops style); a module that already wraps its ops explicitly is trusted to have
    excluded its helpers that way, so it is not also function-scanned. Discovery only; `Op.resolve` never needs it."""
    from importlib.metadata import entry_points
    found: Dict[str, Op] = {}
    for ep in entry_points(group="mokit.ops"):
        try:
            module = importlib.import_module(ep.value)
        except ImportError:
            continue
        members = {name: obj for name, obj in vars(module).items() if not name.startswith("_")}
        module_ops = {name: obj for name, obj in members.items() if isinstance(obj, Op)}
        if module_ops:
            for name, obj in module_ops.items():
                found[obj.name] = obj
            continue
        for name, obj in members.items():
            if inspect.isfunction(obj) and obj.__module__ == module.__name__ and "return" in getattr(obj, "__annotations__", {}):
                found[f"{module.__name__}:{name}"] = Op.resolve(f"{module.__name__}:{name}")
    return found


# ----- result file forms and the memo -------------------------------------------------------------------

@dataclass(frozen=True)
class FileType:
    suffix: str                       # ".npy", ".bin"; a folder form ends with "/" and save/load get the folder
    save: Callable[[Any, Path], None]
    load: Callable[[Path], Any]


FILE_TYPES: Dict[type, FileType] = {}


def register_file_type(tp: type, suffix: str, save, load) -> None:
    FILE_TYPES[tp] = FileType(suffix, save, load)


def mro_lookup(table: dict, tp, default=None):
    """The entry of the first class in `tp.__mro__` found in `table`, else `default`: one rule for FILE_TYPES,
    `describe` tables and UI panels keyed by type."""
    for klass in getattr(tp, "__mro__", ()):
        if klass in table:
            return table[klass]
    return default


def file_type_for(tp: type) -> FileType:
    return mro_lookup(FILE_TYPES, tp, _PICKLE)


def _pickle_save(value, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(value, f)


def _pickle_load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


_PICKLE = FileType(".pkl", _pickle_save, _pickle_load)
register_file_type(bytes, ".bin", lambda v, p: p.write_bytes(v), lambda p: p.read_bytes())
register_file_type(str, ".txt", lambda v, p: p.write_text(v), lambda p: p.read_text())
try:
    import numpy as _np
    register_file_type(_np.ndarray, ".npy", lambda v, p: _np.save(p, v), lambda p: _np.load(p))
except ImportError:  # pragma: no cover
    pass


def _stale_pickle(value) -> bool:
    """A pickled dataclass instance whose class has since gained fields lacks them: the key cannot see a class
    change, so the loader checks the shape instead."""
    if not is_dataclass(value) or isinstance(value, type):
        return False
    present = vars(value) if hasattr(value, "__dict__") else None      # a class default would fool hasattr
    return any((f.name not in present) if present is not None else not hasattr(value, f.name)
               for f in fields(type(value)))


def key_hash(key: tuple) -> str:
    return hashlib.sha1(repr(key).encode()).hexdigest()


class Memo:
    """Results by key: RAM always, disk when the op opts in (`Op.disk_cache`), in the value's file form."""

    def __init__(self, dir=None) -> None:
        self._ram: Dict[tuple, Any] = {}
        self.dir = Path(dir) if dir else None

    def __contains__(self, key: tuple) -> bool:
        return key in self._ram

    def get(self, key: tuple, op: Optional[Op] = None, default=MISSING):
        if key in self._ram:
            return self._ram[key]
        if op is not None and op.disk_cache and self.dir is not None and op.result_type is not None:
            path = self.path(key, op.result_type)
            if path.exists():
                value = file_type_for(op.result_type).load(path)
                if _stale_pickle(value):      # written by an older shape of the result class: recompute
                    path.unlink()
                    return default
                self._ram[key] = value
                return value
        return default

    def put(self, key: tuple, value, op: Optional[Op] = None) -> None:
        self._ram[key] = value
        if op is not None and op.disk_cache and self.dir is not None and value is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.path(key, type(value))
            file_type_for(type(value)).save(value, path)

    def path(self, key: tuple, tp: type) -> Path:
        return self.dir / (key_hash(key) + file_type_for(tp).suffix.rstrip("/"))

    def clear(self) -> None:
        """Forget every result: RAM, and on disk the files this memo wrote (a key hash plus a type suffix); anything
        else in the dir stays."""
        self._ram.clear()
        if self.dir is not None and self.dir.is_dir():
            for path in self.dir.iterdir():
                stem = path.name.split(".", 1)[0]
                if len(stem) == 40 and all(c in "0123456789abcdef" for c in stem):
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

    def forget_ram(self, keep=()) -> None:
        keep = set(keep)
        for k in [k for k in self._ram if k not in keep]:    # in place: a worker thread may `put` meanwhile
            del self._ram[k]


def ready_steps(graph: Graph, keys: Dict[str, tuple], memo: "Memo", nodes=None):
    """Every node in topological order (within `nodes` when given) whose key is not in the memo's RAM and whose
    inputs all have results, as `(node_id, op, inputs)`; a node whose input has no result yet is skipped. An
    unbound input slot on the first stale node is a GraphError naming the slot ("input tilemap of cluster is
    unbound"); later unbound nodes wait their turn. Hosts share this rule: `evaluate` loops over `next_step`, an
    interactive runner starts as many ready steps as it has workers."""
    steps = []
    for nid in graph.order():
        if nodes is not None and nid not in nodes:
            continue
        if keys[nid] in memo:
            continue
        node = graph[nid]
        op = Op.resolve(node.op)
        inputs = []
        for slot, src in enumerate(node.inputs):
            if src == UNBOUND:
                if steps:
                    break
                name = op.inputs[slot] if slot < len(op.inputs) else str(slot)
                raise GraphError(f"input {name} of {nid} is unbound", nid)
            value = memo.get(keys[src])
            if value is MISSING:
                break
            inputs.append(value)
        else:
            steps.append((nid, op, inputs))
    return steps


def next_step(graph: Graph, keys: Dict[str, tuple], memo: "Memo", nodes=None):
    """The first of `ready_steps`, or None when every node is done."""
    steps = ready_steps(graph, keys, memo, nodes)
    return steps[0] if steps else None


def run_step(op: Op, key: tuple, inputs, params, memo: "Memo", progress=None):
    """Run one node through the memo: a disk-cached result when the op has one, else the op itself; stored under
    `key`. Plain arguments only, so a host may call it on a worker thread."""
    value = memo.get(key, op)
    if value is MISSING:
        value = op(inputs, params, progress)
        memo.put(key, value, op)
    return value


def evaluate(graph: Graph, node_id: str, memo: Optional[Memo] = None, progress=None,
             digests: Optional[Digests] = None):
    """Pull `node_id` through the memo: every missing ancestor runs once, in topological order."""
    memo = memo if memo is not None else Memo()
    digests = digests or Digests()
    cache: dict = {}
    keys = {nid: graph.key(nid, digests, cache) for nid in graph.order()}
    needed = set()

    def mark(nid):
        if nid not in needed:
            needed.add(nid)
            for dep in graph[nid].inputs:
                if dep != UNBOUND:
                    mark(dep)
    mark(node_id)
    while (step := next_step(graph, keys, memo, needed)) is not None:
        nid, op, inputs = step
        run_step(op, keys[nid], inputs, graph[nid].params, memo, progress)
    return memo.get(keys[node_id])
