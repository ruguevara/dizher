"""Pipeline host, AmaZX's ConverterApp cut down to a fixed linear graph: the mokit Graph of ops.PIPELINE, a memo
of stage results and one worker thread running the next stale stage. No imgui here; window.py polls update()
every frame and draws from it."""
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Optional

from mokit.graph import MISSING, Digests, GraphError, Memo, ready_steps, run_step
from mokit import project
from mokit.project import Unresolved

from .. import ops

DEBOUNCE = 0.3   # s from the last edit to the next start, so a dragged slider does not restart a stage every frame
HISTORY = 200    # undo steps kept
CACHED = 5       # graphs on each side of the current one in the history whose results stay in RAM: undo shows them at once


def project_folder(image) -> Path:
    """The image's project, as a sidecar: the folder beside it named as the image, or "<name> 2", "<name> 3"... when
    that one is taken by anything but this image's project. It may not exist yet."""
    image = Path(image).resolve()
    for n in itertools.count(1):
        folder = image.with_name(image.stem if n == 1 else f'{image.stem} {n}')
        if not folder.exists():
            return folder
        try:
            if Path(project.load_project(folder).graph['source'].params.path).resolve() == image:
                return folder
        except (OSError, ValueError, KeyError, AttributeError, GraphError):   # not a project, or not readable
            pass


class Cancelled(Exception):
    pass


class Job:
    """One running stage; it is also the op's mokit progress, whose calls are the cancel points."""

    def __init__(self, node_id: str, key: tuple) -> None:
        self.node_id, self.key = node_id, key
        self.cancel = threading.Event()
        self.started = time.monotonic()
        self.text = ''
        self.image = None     # latest preview the op sent (progress.report_progress): a Converter.snapshot
        self.future = None

    def __call__(self, fraction, text) -> None:
        if self.cancel.is_set():
            raise Cancelled()
        if text:
            self.text = text

    def preview(self, image) -> None:
        self.image = image


class Pipeline:
    def __init__(self) -> None:
        self.graph = ops.make_graph()
        self.memo = Memo()
        self.digests = Digests()
        self.errors = {}          # node id -> message of its failed run; cleared by the next edit
        self.past, self.future = [], []   # graphs undo and redo go to, the nearest last
        self._held = False        # the last edit came from a widget still held: the next one folds into its step
        self._latest = {}         # node id -> its last finished result, shown while an edit recomputes it
        self.job: Optional[Job] = None
        self.cancelled = False    # a user cancel holds everything until the next edit
        self._deadline = 0.0
        # ponytail: one thread and the GIL; the pair sweeps are Python loops that may stutter the UI, a process pool if so
        self._executor = ThreadPoolExecutor(1)
        self._sync()

    def _sync(self) -> None:
        cache = {}
        self.keys = {nid: self.graph.key(nid, self.digests, cache) for nid in self.graph.order()}
        if self.job is not None and self.job.key != self.keys[self.job.node_id]:
            self.job.cancel.set()   # its result is stale

    # ----- commands --------------------------------------------------------------------------------

    def set_params(self, node_id: str, params, held: bool = False) -> None:
        self.set_graph(self.graph.with_params(node_id, params), held)

    def set_graph(self, graph, held: bool = False) -> None:
        """An edit, one undo step. held: made by a widget still held (a dragged slider, a typed field), so the edits
        until release() fold into one step."""
        if graph != self.graph:
            if not self._held:
                self.past = (self.past + [self.graph])[-HISTORY:]
            self.future.clear()
            self._held = held   # not on a no-op: a press that moves nothing yet must not fold into the last step
        self._apply(graph)

    def release(self) -> None:
        """No widget holds the edit any more: the next one is a new step."""
        self._held = False

    def undo(self) -> None:
        if self.past:
            self.future.append(self.graph)
            self._apply(self.past.pop())

    def redo(self) -> None:
        if self.future:
            self.past.append(self.graph)
            self._apply(self.future.pop())

    def _apply(self, graph) -> None:
        self.graph = graph
        self.errors, self.cancelled = {}, False
        self._deadline = time.monotonic() + DEBOUNCE
        self._sync()

    def open(self, path) -> None:
        """A new document: the image with default params, nothing shown from before, no history."""
        g = ops.make_graph()
        self.restore(g.with_params('source', replace(g['source'].params, path=Path(path))))
        self._latest.clear()

    def restore(self, graph) -> None:
        """The params of a saved graph on the current pipeline: a node it lacks keeps its defaults, one the pipeline
        lacks or has with another op is dropped, so a project from before a stage was added or removed still opens.
        The history starts over."""
        g = ops.make_graph()
        for nid, node in graph.nodes:
            if nid in g and node.op == g[nid].op and not isinstance(node.params, Unresolved):
                g = g.with_params(nid, node.params)
        self.past, self.future, self._held = [], [], False   # a new document
        self._apply(g)

    def cancel(self) -> None:
        if self.job is not None:
            self.job.cancel.set()
            self.cancelled = True

    def close(self) -> None:
        self.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ----- reads -----------------------------------------------------------------------------------

    def result(self, node_id: str):
        value = self.memo.get(self.keys[node_id])
        return None if value is MISSING else value

    def shown(self, node_id: str):
        """The current result, else the last finished one: what to display while the node recomputes."""
        value = self.result(node_id)
        if value is None:
            return self._latest.get(node_id)
        self._latest[node_id] = value   # a memo hit finishes no job: without this an edit would show an older result
        return value

    def status(self, node_id: str) -> str:
        if self.job is not None and self.job.node_id == node_id:
            return 'running'
        if node_id in self.errors:
            return 'error'
        return 'done' if self.keys[node_id] in self.memo else 'stale'

    @property
    def busy(self) -> bool:
        return self.job is not None or time.monotonic() < self._deadline

    # ----- scheduling ------------------------------------------------------------------------------

    def update(self) -> None:
        job = self.job
        if job is not None and job.future.done():
            self.job = None
            if job.key == self.keys[job.node_id]:
                try:
                    self._latest[job.node_id] = job.future.result()
                except Cancelled:
                    pass
                except Exception as e:
                    self.errors[job.node_id] = str(e) if isinstance(e, (ValueError, FileNotFoundError)) else f'{type(e).__name__}: {e}'
            self.memo.forget_ram(keep=self._kept())
        if self.job is None and not self.cancelled and time.monotonic() >= self._deadline:
            self._start_next()

    def _kept(self) -> set:
        """Keys of the current graph's results and of the nearest ones in the history."""
        kept = set(self.keys.values())
        for g in self.past[-CACHED:] + self.future[-CACHED:]:
            cache = {}
            kept.update(g.key(nid, self.digests, cache) for nid in g.order())
        return kept

    def _start_next(self) -> None:
        nodes = [n for n in self.graph.ids() if n not in self.errors]
        try:
            steps = ready_steps(self.graph, self.keys, self.memo, nodes)
        except GraphError as e:
            self.errors[e.node] = str(e)
            return
        steps = [s for s in steps if not (s[0] == 'source' and self.graph['source'].params.path is None)]
        if not steps:
            return
        node_id, op, inputs = steps[0]
        job = self.job = Job(node_id, self.keys[node_id])
        job.future = self._executor.submit(run_step, op, job.key, tuple(inputs), self.graph[node_id].params,
                                           self.memo, job)
