"""File paths as the OS takes them past MAX_PATH (260 characters) on Windows, where long paths are off on most
machines: a project beside an image doubles its name (the folder, then build/ and the export in it)."""
import os
import sys


def os_path(filename) -> str:
    """filename as open(), os.stat and PIL take it on any machine: on Windows the \\\\?\\ form, absolute."""
    if sys.platform != 'win32':
        return str(filename)
    filename = os.path.abspath(filename)
    if filename.startswith('\\\\?\\'):
        return filename
    return '\\\\?\\UNC\\' + filename[2:] if filename.startswith('\\\\') else '\\\\?\\' + filename


def exists(path) -> bool:
    return os.path.exists(os_path(path))
