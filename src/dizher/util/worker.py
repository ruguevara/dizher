# -*- coding: utf-8 -*-

from multiprocessing.pool import Pool, AsyncResult
from multiprocessing import TimeoutError
import multiprocessing as mp
import queue
import traceback

from typing import Any, Callable, Dict, Tuple, Union


class AbstractWorker:
    def abort(self):
        pass

    def apply(self, priority: int, task: Callable, args: Tuple = (), kwds: Dict = {}, callback: Callable = None,
              error_callback: Callable = None, abort_callback: Callable = None) -> bool:
        raise NotImplementedError()


class SyncWorker(AbstractWorker):
    def apply(self, priority: int, task: Callable, args: Tuple = (), kwds: Dict = {}, callback: Callable = None,
              error_callback: Callable = None, abort_callback: Callable = None) -> bool:
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
              error_callback: Callable = None, abort_callback: Callable = None) -> bool:
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

        def run(queue, *args, **kwargs):
            try:
                result = task(*args, **kwargs)
                queue.put(result)
            except:
                print("FATAL: worker({0}) exited while multiprocessing".format(str(task)))
                traceback.print_exc()

        self._process = mp.Process(target=run, args=tuple([self._queue]) + args, kwargs=kwds)
        self._process.start()
        return True

    def read(self, timeout: int = 0) -> Any:
        if self._queue is not None and not self._queue.empty():
            result = self._queue.get(False, timeout / 1000)
            self._process.join()
            self._process.close()
            self._process = None
            self._queue.close()
            self._queue = mp.Queue()
            if self._callback:
                self._callback(result)
            return result


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
#               error_callback: Callable = None, abort_callback: Callable = None) -> bool:
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
