"""The session the window keeps in its user prefs, without the imgui loop: the project on disk wins unless the session
has unsaved edits. Run: pytest tests/test_session.py"""
import shutil
from dataclasses import replace
from pathlib import Path

from mokit import project

IMAGE = Path(__file__).parent / 'images' / 'goldhill.png'


def restarted(window):
    """A new window from the prefs the last one saved at quit."""
    ui = window.Window()
    ui._load_prefs()
    return ui


def test_session(tmp_path, monkeypatch):
    from dizher.ui import window
    prefs = {}
    monkeypatch.setattr(window.hello_imgui, 'save_user_pref', prefs.__setitem__)
    monkeypatch.setattr(window.hello_imgui, 'load_user_pref', lambda k: prefs.get(k, ''))
    image = tmp_path / 'sunset.png'
    shutil.copy(IMAGE, image)

    ui = window.Window()
    ui.autosave = True
    ui._open_image(image)
    ui.app.set_params('optimise', replace(ui.app.graph['optimise'].params, enabled=False))
    ui._save_project()   # the frame's autosave
    ui._save_prefs()
    ui.app.close()
    # edited outside the app while it is closed: git checkout, a script
    graph = project.load_project(ui.project).graph
    painted = ((0, 0, 1, 2),)
    project.save_project(ui.project, graph.with_params('overpaint', replace(graph['overpaint'].params, overrides=painted)))

    ui = restarted(window)
    assert ui.app.graph['overpaint'].params.overrides == painted, 'the saved session hid the project on disk'
    assert ui.app.graph == ui.saved

    ui.autosave = False
    ui.app.set_params('overpaint', replace(ui.app.graph['overpaint'].params, overrides=()))   # unsaved
    ui._save_prefs()
    ui.app.close()
    ui = restarted(window)
    assert ui.app.graph['overpaint'].params.overrides == (), 'the unsaved edit was lost'
    assert ui.app.graph != ui.saved
    ui.app.close()
