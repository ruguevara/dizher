# -*- coding: utf-8 -*-

from multiprocessing.pool import Pool, AsyncResult
from typing import Any, Callable, Dict, Tuple, Union


class SingleAsyncPriorityWorker:
    def __init__(self) -> None:
        self._pool: Pool = Pool(1)
        self._async_result: Union[AsyncResult, None] = None
        self._task_priority: int = 0
        self._abort_callback: Union[Callable, None] = None

    def apply(self, priority: int, task: Callable, args: Tuple = (), kwds: Dict = {}, callback: Callable = None,
              error_callback: Callable = None, abort_callback: Callable = None) -> bool:
        # returns True if this worker was allied sucessfully
        # if there is a higher priority task, returns False
        if self._async_result is not None and not self._async_result.ready():
            # previous task is still active
            if self._task_priority > priority:
                return False
            if not self._async_result.ready():  # must check second time
                # aborting current task
                self._pool.terminate()
                self._pool.close()
                del self._pool
                self._pool: Pool = Pool(1)
                if self._abort_callback:
                    self._abort_callback(SystemExit)
                    self._abort_callback = None
        self._task_priority = priority
        self._abort_callback = abort_callback
        self._async_result = self._pool.apply_async(task, args, kwds, callback, error_callback)
        return True
