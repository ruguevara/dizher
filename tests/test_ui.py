"""UI tests under the imgui test engine, ToolZX's pattern: a real window, simulated mouse, tests in TESTS order on
one shared Window. Run: .venv/bin/python tests/test_ui.py (exit code 0 when all pass)."""
import sys
from dataclasses import replace
from pathlib import Path

from imgui_bundle import imgui

from dizher import tone
from dizher.converter.dither import Ordered
from dizher.ui.window import REDO, UNDO, Window

IMAGE = Path(__file__).parent / 'images' / 'lena.png'
ui = Window(IMAGE)
ui.app.set_params('halftoner', replace(ui.app.graph['halftoner'].params, halftoner=Ordered.label))  # fast
ui.app.set_params('optimise', replace(ui.app.graph['optimise'].params, enabled=False))


def params(nid):
    return ui.app.graph[nid].params


def reset(nid):
    ui.app.set_params(nid, type(params(nid))())


def wait(ctx, cond, what, frames=3000):
    for _ in range(frames):
        if cond():
            return
        ctx.yield_()
    assert cond(), f'timed out waiting for {what}'


def drag(ctx, a: imgui.ImVec2, b: imgui.ImVec2):
    """ToolZX's drag: press, move with real frames between, release (sleep_short is a no-op at fast speed)."""
    ctx.mouse_move_to_pos(a)
    ctx.mouse_down(0)
    ctx.yield_(2)
    ctx.mouse_move_to_pos(b)
    ctx.yield_(2)
    ctx.mouse_up(0)
    ctx.yield_(2)


def rect(ctx, window, item):
    ctx.set_ref(window)
    return ctx.item_info(item).rect_full


def test_conversion_lands(ctx):
    wait(ctx, lambda: ui.app.result('optimise') is not None, 'the first conversion')
    assert not ui.app.errors, ui.app.errors


def test_slider_drag_moves_param(ctx):
    r = rect(ctx, '//Tune', '**/contrast/contrast')   # the slider, under params_editor's push_id; '**/contrast' is the header
    y = (r.min.y + r.max.y) / 2
    steps = len(ui.app.past)
    drag(ctx, imgui.ImVec2((r.min.x + r.max.x) / 2, y), imgui.ImVec2((r.min.x + r.max.x) / 2 + 60, y))
    moved = params('contrast').contrast
    assert moved > 0, f'contrast slider did not move: {moved}'
    ctx.yield_(5)
    assert params('contrast').contrast == moved, 'the slider fell back'
    assert len(ui.app.past) == steps + 1, 'a drag is one undo step'
    ctx.key_press(UNDO)
    ctx.yield_(2)
    assert params('contrast').contrast == 0.0, 'undo'
    ctx.key_press(REDO)
    ctx.yield_(2)
    assert params('contrast').contrast == moved, 'redo'
    reset('contrast')


def test_levels_handles_drag(ctx):
    wait(ctx, lambda: ui.app.result('light') is not None, 'the levels input')
    r = rect(ctx, '//Tune', '**/##in')
    pad, y = (r.max.y - r.min.y) / 4, r.max.y - 3        # on the triangles, under the gradient
    drag(ctx, imgui.ImVec2(r.min.x + pad, y), imgui.ImVec2(r.min.x + pad + 50, y))
    p = params('levels')
    assert p.in_black > 0 and p.gamma == 1.0, f'black handle: {p}'
    drag(ctx, imgui.ImVec2(r.max.x - pad, y), imgui.ImVec2(r.max.x - pad - 50, y))
    p = params('levels')
    assert p.in_white < 255 and p.gamma == 1.0, f'white handle: {p}'
    mid = r.min.x + pad + (p.in_black + p.in_white) / 2 / 255 * (r.max.x - r.min.x - 2 * pad)
    drag(ctx, imgui.ImVec2(mid, y), imgui.ImVec2(mid + 30, y))
    assert params('levels').gamma < 1.0, f'midtone handle right must darken: {params("levels")}'
    reset('levels')


def test_levels_auto(ctx):
    ctx.set_ref('//Tune')
    ctx.item_click('**/Auto')
    ctx.yield_(2)
    p = params('levels')
    assert (p.in_black, p.in_white) == tone.auto_levels(ui.app.result('light')), p
    reset('levels')



def test_history_panel(ctx):
    ui.app.set_params('contrast', replace(params('contrast'), contrast=10.0))
    ui.app.set_params('contrast', replace(params('contrast'), contrast=20.0))
    ctx.yield_(2)
    ctx.set_ref('//History')
    ctx.item_click(f'**/Contrast: contrast 10##{len(ui.app.past) - 1}')
    ctx.yield_(2)
    assert params('contrast').contrast == 10.0 and len(ui.app.future) == 1, params('contrast')
    reset('contrast')


def test_autosave(ctx):
    import json, tempfile
    folder = ui.project
    assert folder == IMAGE.with_name('lena') and not folder.exists() and not ui.autosave   # tests never write it
    with tempfile.TemporaryDirectory() as tmp:
        ui.project, ui.autosave = Path(tmp) / 'lena', True
        ui.app.set_params('contrast', replace(params('contrast'), contrast=15.0))
        ctx.yield_(2)
        stored = json.loads((ui.project / 'project.json').read_text())['nodes']['contrast']['params']
        assert stored['contrast'] == 15.0 and ui.saved == ui.app.graph, stored
        ui.project, ui.autosave = folder, False
    reset('contrast')


def test_unsaved_dialog(ctx):
    """Opening another image over unsaved edits asks first; with nothing unsaved it does not."""
    opened = []
    ui.saved = ui.app.graph
    ui._close(lambda: opened.append(1))
    assert opened == [1]
    ui.app.set_params('contrast', replace(params('contrast'), contrast=15.0))
    for button, result in (('Cancel', [1]), ("Don't save", [1, 1])):
        ui._close(lambda: opened.append(1))
        ctx.yield_(2)
        ctx.item_click(f'//Unsaved changes/{button}')
        ctx.yield_(2)
        assert opened == result and ui._after_close is None, (button, opened)
    reset('contrast')


def test_recent_images(ctx):
    assert ui.recent == [IMAGE.resolve()]              # the image given at start
    images = sorted(IMAGE.parent.glob('*.*'))
    for image in images * 2:                           # more than fit, each twice
        ui._open_image(image)
    assert ui.recent == [p.resolve() for p in images[::-1]][:20]   # the latest first, no repeats
    ctx.menu_click('//##MainMenuBar/File/Open recent/lena.png##' + str(ui.recent.index(IMAGE.resolve())))
    ctx.yield_(2)
    assert ui._source() == IMAGE.resolve() and ui.recent[0] == IMAGE.resolve()
    ui.app.set_params('halftoner', replace(params('halftoner'), halftoner=Ordered.label))
    ui.app.set_params('optimise', replace(params('optimise'), enabled=False))

def test_block_collapses_and_expands(ctx):
    ctx.set_ref('//Tune')
    ctx.item_click('**/###framing')
    ctx.yield_(2)
    assert ui.expanded['framing'] is False
    ctx.item_click('**/###framing')
    ctx.yield_(2)
    assert ui.expanded['framing'] is True


def test_block_reset(ctx):
    ui.app.set_params('light', replace(params('light'), exposure=0.37))
    ctx.yield_(2)
    ctx.set_ref('//Tune')
    ctx.item_click('**/Reset##light')
    ctx.yield_(2)
    assert params('light') == type(params('light'))(), params('light')


def test_views_and_grid(ctx):
    wait(ctx, lambda: ui.app.result('optimise') is not None, 'a conversion to view')
    ctx.set_ref('//Preview')
    for view in ('Bitmap', 'Attrs', 'Projected', 'Eye', 'Error', 'Energy', 'Seams'):
        ctx.item_click(f'**/{view}')
        ctx.yield_(2)
        assert ui.view == view and ui._converted().shape == (192, 256, 3), view
    ctx.item_click('**/grid/#')
    ctx.yield_(2)
    assert ui.grid


def test_mode_switch_refits_previews(ctx):
    """Spectrum at zoom 2, then C64 at zoom 1 in the narrow test window: immvision kept the tuned preview's
    zoom 2 matrix, showing its top-left quarter."""
    from imgui_bundle import immvision
    from dizher import ops
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'the Spectrum conversion')
    ctx.yield_(2)
    zx = ui.images['tuned'].image_display_size
    ui.app.set_params('target', replace(params('target'), mode=ops.c64.HIRES.name))
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'the C64 conversion')
    ctx.yield_(2)
    c64 = ui.images['tuned'].image_display_size
    assert zx[0] // 256 > c64[0] // 320, ('the zoom must drop to test this', zx, c64)
    for key in ('tuned', 'converted'):
        p = ui.images[key]
        assert p.zoom_pan_matrix == immvision.make_zoom_pan_matrix_full_view((320, 200), p.image_display_size), \
            (key, p.zoom_pan_matrix)
    reset('target')
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'the Spectrum conversion again')


def test_hover_inspector(ctx):
    from imgui_bundle.imgui.test_engine import CaptureFlags_
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'a conversion to inspect')
    ui.view = 'Screen'
    r = rect(ctx, '//Preview', '**/Screen')   # the display-only images have no item id: the first is under the bar
    ctx.mouse_move_to_pos(imgui.ImVec2(r.min.x + 100, r.max.y + 100))
    ctx.yield_(4)
    tooltip = imgui.internal.find_window_by_name('##Tooltip_00')
    assert tooltip is not None and tooltip.active and tooltip.size.y > 300, 'no inspector tooltip'
    ctx.capture_set_filename('/tmp/dizher_inspect.png')
    ctx.capture_screenshot(CaptureFlags_.hide_mouse_cursor.value)
    ctx.mouse_move_to_pos(imgui.ImVec2(r.min.x - 50, r.min.y - 50))


def test_capture_layout(ctx):   # keep last: a picture of the default layout for review
    from imgui_bundle.imgui.test_engine import CaptureFlags_
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'the conversion to settle')
    ctx.set_ref('//Tune')
    ctx.item_info('**/##in')
    ctx.yield_(4)
    ctx.capture_set_filename('/tmp/dizher_ui.png')
    ctx.capture_screenshot(CaptureFlags_.hide_mouse_cursor.value)


TESTS = [
    ('ui', 'conversion_lands', test_conversion_lands),
    ('ui', 'slider_drag_moves_param', test_slider_drag_moves_param),
    ('ui', 'levels_handles_drag', test_levels_handles_drag),
    ('ui', 'levels_auto', test_levels_auto),
    ('ui', 'history_panel', test_history_panel),
    ('ui', 'autosave', test_autosave),
    ('ui', 'unsaved_dialog', test_unsaved_dialog),
    ('ui', 'recent_images', test_recent_images),
    ('ui', 'block_collapses_and_expands', test_block_collapses_and_expands),
    ('ui', 'block_reset', test_block_reset),
    ('ui', 'views_and_grid', test_views_and_grid),
    ('ui', 'mode_switch_refits_previews', test_mode_switch_refits_previews),
    ('ui', 'hover_inspector', test_hover_inspector),
    ('ui', 'capture_layout', test_capture_layout),
]

if __name__ == '__main__':
    sys.exit(0 if ui.run_tests(TESTS) else 1)
