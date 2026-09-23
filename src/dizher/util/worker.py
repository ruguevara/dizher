# -*- coding: utf-8 -*-

from multiprocessing.pool import Pool, AsyncResult
from multiprocessing import TimeoutError
import multiprocessing as mp
import queue
import signal
import threading
import time
import traceback
from contextlib import contextmanager

from typing import Any, Callable, Dict, Tuple, Union


PROGRESS_INTERVAL = 0.01  # s between intermediate results sent to the GUI
_progress_queue = None
_progress_last = 0.0

_local = threading.local()   # .progress: the mokit progress of the op running on this thread (ui/app.py)

@contextmanager
def reporting(progress):
    """Inside a mokit op: report_stage/report_progress go to progress(fraction, text), the host's cancel point,
    and preview images to progress.preview(image) when the host has one."""
    _local.progress, _local.last = progress, 0.0
    try:
        yield
    finally:
        _local.progress = None

def report_progress(render: Callable[[], Any]):
    """Inside a worker task: send render() to the GUI as an intermediate result, at most every
    PROGRESS_INTERVAL. render is only called when due, so it may be expensive. No-op outside a worker."""
    global _progress_last
    now = time.monotonic()
    progress = getattr(_local, 'progress', None)
    if progress is not None:
        progress(None, None)   # every step is a cancel point
        preview = getattr(progress, 'preview', None)
        if preview is not None and now - _local.last >= PROGRESS_INTERVAL:
            preview(render())
            _local.last = time.monotonic()
        return
    if _progress_queue is None or now - _progress_last < PROGRESS_INTERVAL:
        return
    _progress_queue.put(('image', render()))
    _progress_last = time.monotonic()  # the interval runs from the end of render, so a slow render cannot eat the task

def report_stage(text: str):
    """Inside a worker task: name the step now running, for the status bar. No-op outside a worker."""
    progress = getattr(_local, 'progress', None)
    if progress is not None:
        progress(None, text)
    elif _progress_queue is not None:
        _progress_queue.put(('stage', text))

class AbstractWorker:
    def abort(self):
        pass

    def cancel(self, priority: int):
        """Abort the running task if its priority is at most priority: its result is already stale."""
        if self.is_alive() and self._task_priority <= priority:
            self.abort()

    def is_alive(self):
        return False

    def apply(self, priority: int, task: Callable, args: Tuple = (), kwds: Dict = {}, callback: Callable = None,
              error_callback: Callable = None, abort_callback: Callable = None, progress_callback: Callable = None) -> bool:
        raise NotImplementedError()


class SyncWorker(AbstractWorker):
    def apply(self, priority: int, task: Callable, args: Tuple = (), kwds: Dict = {}, callback: Callable = None,
              error_callback: Callable = None, abort_callback: Callable = None, progress_callback: Callable = None) -> bool:
        result = task(*args, **kwds)
        if callback:
            callback(result)
        return True


class SingleAsyncPriorityWorker(AbstractWorker):
    def __init__(self) -> None:
        self._process: Union[mp.Process, None] = None
        self._queue: mp.Queue = mp.Queue()
        self._task_priority: int = 0
        self._abort_callback: Union[Callable, None] = None
        self._callback: Union[Callable, None] = None
        self._progress_callback: Union[Callable, None] = None  # read() runs every GUI tick, before any apply()

    def is_alive(self):
        return self._process is not None and self._process.is_alive()

    def abort(self):
        if self._process is None:
            return
        if self._process.is_alive():
            self._process.terminate()
            if self._process.is_alive():
                self._process.kill()
            if self._abort_callback:
                self._abort_callback(SystemExit)
        self._abort_callback = None
        self._process.join()
        self._process.close()
        self._process = None
        self._queue.close()
        self._queue = mp.Queue()

    def apply(self, priority: int, task: Callable, args: Tuple = (), kwds: Dict = {}, callback: Callable = None,
              error_callback: Callable = None, abort_callback: Callable = None, progress_callback: Callable = None) -> bool:
        # returns True if the worker was applied sucessfully
        # if there is a higher priority task, returns False
        if self.is_alive():
            # previous task is still active
            if self._task_priority > priority:
                return False
        self.abort()

        self._task_priority = priority
        self._abort_callback = abort_callback
        self._callback = callback
        self._progress_callback = progress_callback

        def run(queue, *args, **kwargs):
            global _progress_queue
            _progress_queue = queue  # forked child: only this process sees it
            # Ctrl-C in the terminal reaches the whole process group; the GUI owns it and terminates us.
            # A forked child must not run its own teardown: it inherits the GUI's CoreFoundation state.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                result = task(*args, **kwargs)
                queue.put(('done', result))
            except:
                print("FATAL: worker({0}) exited while multiprocessing".format(str(task)))
                traceback.print_exc()

        self._process = mp.Process(target=run, args=tuple([self._queue]) + args, kwargs=kwds)
        self._process.start()
        return True

    def read(self, timeout: int = 0) -> Any:
        latest = {}
        while self._queue is not None and not self._queue.empty():
            kind, result = self._queue.get(False, timeout / 1000)
            if kind != 'done':
                latest[kind] = result  # only the latest image and stage are worth drawing
                continue
            self._process.join()
            self._process.close()
            self._process = None
            self._queue.close()
            self._queue = mp.Queue()
            if self._callback:
                self._callback(result)
            return result
        if self._progress_callback:
            for kind, value in latest.items():
                self._progress_callback(kind, value)


# class SingleAsyncPriorityWorker(AbstractWorker):
#     def __init__(self) -> None:
#         self._process: Union[mp.Process, None] = None
#         self._parent_conn: Union[mp.connection.Connection, None] = None
#         self._child_conn: Union[mp.connection.Connection, None] = None
#         self._task_priority: int = 0
#         self._abort_callback: Union[Callable, None] = None

#     def is_alive(self):
#         return self._process is not None and self._process.is_alive()

#     def abort(self):
#         if self._process is None:
#             return
#         if self._process.is_alive():
#             self._process.terminate()
#             if self._process.is_alive():
#                 self._process.kill()
#             if self._abort_callback:
#                 self._abort_callback(SystemExit)
#                 self._abort_callback = None
#             self._process.join()
#             self._process.close()
#             self._parent_conn.close()

#     def apply(self, priority: int, task: Callable, args: Tuple = (), kwds: Dict = {}, callback: Callable = None,
#               error_callback: Callable = None, abort_callback: Callable = None, progress_callback: Callable = None) -> bool:
#         # returns True if the worker was applied sucessfully
#         # if there is a higher priority task, returns False
#         if self.is_alive():
#             # previous task is still active
#             if self._task_priority > priority:
#                 return False
#         self.abort()

#         self._task_priority = priority
#         self._abort_callback = abort_callback
#         self._callback = callback

#         def run(conn, *args, **kwargs):
#             try:
#                 print("work started")
#                 result = task(*args, **kwargs)
#                 conn.send(result)
#                 print("got result into pipe")
#             except:
#                 print("FATAL: worker exited while multiprocessing")
#                 traceback.print_exc()

#         self._parent_conn, self._child_conn = mp.Pipe()
#         self._process = mp.Process(target=run, args=tuple([self._child_conn]) + args, kwargs=kwds)
#         self._process.start()
#         return True

#     def read(self, timeout: int = 0) -> Any:
#         if not self._parent_conn or not self._parent_conn.poll():
#             return

#         try:
#             result = self._parent_conn.recv()
#             print("got result from pipe")
#             self._parent_conn.close()
#             self._parent_conn = None
#             self._process.join()
#             self._process.close()
#             self._process = None
#             if self._callback:
#                 self._callback(result)
#             return result
#         except Exception as e:
#             print("FATAL: worker result reading caused an error")
#             traceback.print_exc()
