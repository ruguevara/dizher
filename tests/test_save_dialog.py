"""The Windows save dialog opens under a long image name. Run: pytest tests/test_save_dialog.py (on Windows)."""
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='the Windows save dialog')

SAVE = 'import sys; from imgui_bundle import portable_file_dialogs as pfd; print(repr(pfd.save_file("Save", sys.argv[1]).result()))'


def test_long_name_opens_the_dialog(tmp_path):
    """Into a project's build folder, which doubles the image's name: past MAX_PATH the dialog gave "" at once."""
    from dizher.ui.window import MAX_PATH, dialog_default
    stem = ' '.join(['очень длинное имя'] * 7)   # no trailing space, which Windows drops from a folder's name
    build = tmp_path / stem / 'build'
    build.mkdir(parents=True)
    assert len(str(build / stem)) >= MAX_PATH
    default = dialog_default(str(build), stem + '.scr')
    assert len(default) < MAX_PATH and default.startswith(str(build)) and default.endswith('.scr')
    with pytest.raises(subprocess.TimeoutExpired):   # open, waiting for the user
        subprocess.run([sys.executable, '-c', SAVE, default], capture_output=True, timeout=10)
