"""Snapshots of a project's graph: `<project>/snapshots/<name>.json` in the format of project.json (paths relative to the
project folder), and optionally `<name>.png`, a picture of the conversion. Storage only, no UI."""
import itertools
import os
import re
from pathlib import Path

import cv2
import numpy as np

from mokit import project
from mokit.paths import exists, os_path

DIR = 'snapshots'


def _path(folder, name, ext='json') -> Path:
    return Path(folder) / DIR / f'{name}.{ext}'


def names(folder) -> list:
    """The snapshot names, newest first by the json's mtime, ties by name; none when there is no snapshots folder."""
    d = Path(folder) / DIR
    if not exists(d):
        return []
    found = [(-os.stat(os_path(p)).st_mtime, p.stem) for p in Path(os_path(d)).glob('*.json')]
    return [name for _, name in sorted(found)]


def load(folder, name):
    """The snapshot's graph, its paths resolved against the project folder."""
    return project.loads(Path(os_path(_path(folder, name))).read_text(), Path(folder))[0]


def save(folder, name, graph, picture=None) -> None:
    """Write the snapshot; picture is (H, W, 3) float RGB in 0..1, and None leaves a picture already saved under the name."""
    Path(os_path(_path(folder, name).parent)).mkdir(parents=True, exist_ok=True)
    Path(os_path(_path(folder, name))).write_text(project.dumps(graph, None, Path(folder)))
    if picture is not None:   # imencode and Python's open, as cv2.imwrite takes no non-ASCII paths on Windows
        ok, data = cv2.imencode('.png', (np.asarray(picture)[..., ::-1] * 255).round().astype(np.uint8))
        assert ok, name
        with open(os_path(_path(folder, name, 'png')), 'wb') as f:
            f.write(bytes(data))


def picture(folder, name):
    """The snapshot's picture as (H, W, 3) float32 RGB in 0..1, or None when it has none or cannot be read."""
    try:
        bgr = cv2.imdecode(np.fromfile(os_path(_path(folder, name, 'png')), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None
    return None if bgr is None else bgr[..., ::-1].astype(np.float32) / 255


def rename(folder, old, new) -> None:
    """Move the json and the picture to another name; FileExistsError when a snapshot has it."""
    if exists(_path(folder, new)):
        raise FileExistsError(new)
    for ext in ('json', 'png'):
        if exists(_path(folder, old, ext)):
            os.replace(os_path(_path(folder, old, ext)), os_path(_path(folder, new, ext)))


def delete(folder, name) -> None:
    """Remove the json and the picture, if it has one."""
    Path(os_path(_path(folder, name))).unlink()
    Path(os_path(_path(folder, name, 'png'))).unlink(missing_ok=True)


def clean(name) -> str:
    """A file name from what a user typed: no separators, wildcards or control characters, no leading dots, and no
    trailing spaces and dots, which Windows drops; empty when nothing is left."""
    return re.sub(r'[/\\:*?"<>|\x00-\x1f]', '-', name.strip()).lstrip('.').rstrip(' .')


def free_name(folder, base) -> str:
    """base, or "base 2", "base 3"... when a snapshot has that name."""
    taken = set(names(folder))
    return next(n for i in itertools.count(1) if (n := base if i == 1 else f'{base} {i}') not in taken)
