# -*- coding: utf-8 -*-

from multiprocessing.pool import Pool, AsyncResult
from multiprocessing import TimeoutError
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
        self._pool: Pool = Pool(1)
        self._async_result: Union[AsyncResult, None] = None
        self._task_priority: int = 0
        self._abort_callback: Union[Callable, None] = None

    def abort(self):
        if self._async_result is not None and not self._async_result.ready():
            self._pool.terminate()
            self._pool.close()
            del self._pool
            self._pool = Pool(1)
            if self._abort_callback:
                self._abort_callback(SystemExit)
                self._abort_callback = None

    def apply(self, priority: int, task: Callable, args: Tuple = (), kwds: Dict = {}, callback: Callable = None,
              error_callback: Callable = None, abort_callback: Callable = None) -> bool:
        # returns True if the worker was applied sucessfully
        # if there is a higher priority task, returns False
        if self._async_result is not None and not self._async_result.ready():
            # previous task is still active
            if self._task_priority > priority:
                return False
        self.abort()

        self._task_priority = priority
        self._abort_callback = abort_callback
        self._async_result = self._pool.apply_async(task, args, kwds, error_callback=error_callback)
        self._callback = callback
        return True

    def read(self, timeout: int = 0):
        if self._async_result is None:
            return None
        try:
            result = self._async_result.get(timeout / 1000)
            self._async_result = None
            if self._callback:
                self._callback(result)
            return result
        except TimeoutError:
            return None
