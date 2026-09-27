"""A source image with a long path, as the window opens, converts, autosaves, exports and reopens it, without the imgui
loop; the save dialog gives its default, as when the user presses Save. Run: pytest tests/test_long_source.py (on
Windows with long paths off, as on most machines, the case it is for)."""
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from mokit.paths import exists, os_path
from dizher.converter.dither import Ordered

IMAGE = Path(__file__).parent / 'images' / 'goldhill-256.png'


def long_paths(tmp: Path):
    """(case, image): the image's path past MAX_PATH (260) in deep folders; its name at 255 characters, the most a
    name takes on Windows and in bytes on Linux (ASCII here), in a short folder."""
    deep = tmp.resolve() / ('d' * 100) / ('e' * 100)   # on Windows the temp folder may come as a short name, RUNNER~1
    yield 'deep folders', deep / 'sunset.png'
    yield 'name of 255', tmp.resolve() / ('n' * 251 + '.png')


def place(image: Path) -> None:
    Path(os_path(image.parent)).mkdir(parents=True, exist_ok=True)
    with open(os_path(image), 'wb') as f:
        f.write(IMAGE.read_bytes())


def converted(ui, timeout=120):
    """The first conversion, the window's scheduling as its frames call it."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and not ui.app.errors and ui.result is None:
        ui.app.update()
        time.sleep(0.02)
    assert not ui.app.errors, ui.app.errors
    assert ui.result is not None, 'no conversion in time'


@pytest.mark.parametrize('case', ['deep folders', 'name of 255'])
def test_long_source(case, tmp_path, monkeypatch):
    from mokit import project
    from dizher.ui import window
    image = dict(long_paths(tmp_path))[case]
    place(image)
    assert len(str(image)) > 255
    dialog = []
    monkeypatch.setattr(window, 'save_dialog', lambda title, folder, name: dialog.append(window.dialog_default(folder, name)) or dialog[-1])

    ui = window.Window()
    ui.autosave = True
    try:
        ui._open_image(image)
        folder = ui.project
        assert folder is not None and exists(folder / project.PROJECT_FILE), f'no project at {folder}'
        ui.app.set_params('halftoner', replace(ui.app.graph['halftoner'].params, halftoner=Ordered.label))   # fast
        ui.app.set_params('optimise', replace(ui.app.graph['optimise'].params, enabled=False))
        ui.app._deadline = 0   # no debounce: nothing is dragged
        converted(ui)
        ui._save_project()   # the autosave after an edit
        assert ui.saved == ui.app.graph and 'project' not in ui.app.errors, ui.app.errors
        for ext in ('.png', '.scr'):
            ui._save(ext)
            assert 'save' not in ui.app.errors, ui.app.errors
            saved = Path(dialog[-1])   # on Windows the build folder by its 8.3 short name, to fit MAX_PATH
            assert saved.suffix == ext and exists(saved), dialog[-1]
            assert saved.parent != Path() and os.path.samefile(os_path(saved.parent), os_path(folder / 'build')), dialog[-1]
            if sys.platform == 'win32':
                assert len(dialog[-1]) < window.MAX_PATH
        assert ui.recent and ui.recent[0] == image.resolve()
    finally:
        ui.app.close()

    again = window.Window(image)   # the project beside the image, found again
    try:
        assert again.project == folder and not again.app.errors, (again.project, again.app.errors)
        assert again.app.graph['source'].params.path == image
        assert again.app.graph['halftoner'].params.halftoner == Ordered.label
        converted(again)
    finally:
        again.app.close()
