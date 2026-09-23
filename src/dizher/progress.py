"""Progress of a running stage: the converter's inner loops name their step and send preview images through here,
and the op (ops.py) routes both to the host's mokit progress, whose calls are also the cancel points."""
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

PROGRESS_INTERVAL = 0.01  # s between preview images
_local = threading.local()   # .progress: the mokit progress of the op running on this thread


@contextmanager
def reporting(progress):
    """Inside a mokit op: report_stage/report_progress go to progress(fraction, text), and preview images to
    progress.preview(image) when the host has one."""
    _local.progress, _local.last = progress, 0.0
    try:
        yield
    finally:
        _local.progress = None


def report_progress(render: Callable[[], Any]):
    """A step of an inner loop: a cancel point, and render() as a preview at most every PROGRESS_INTERVAL.
    render is only called when due, so it may be expensive. No-op outside `reporting`."""
    progress = getattr(_local, 'progress', None)
    if progress is None:
        return
    progress(None, None)
    preview = getattr(progress, 'preview', None)
    if preview is not None and time.monotonic() - _local.last >= PROGRESS_INTERVAL:
        preview(render())
        _local.last = time.monotonic()  # the interval runs from the end of render, so a slow render cannot eat the task


def report_stage(text: str):
    """Name the step now running, for the status bar. No-op outside `reporting`."""
    progress = getattr(_local, 'progress', None)
    if progress is not None:
        progress(None, text)
