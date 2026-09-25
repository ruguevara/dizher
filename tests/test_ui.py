"""UI tests under the imgui test engine, ToolZX's pattern: a real window, simulated mouse, tests in TESTS order on
one shared Window. Run: .venv/bin/python tests/test_ui.py (exit code 0 when all pass)."""
import sys
from dataclasses import replace
from pathlib import Path

from imgui_bundle import imgui

from dizher import tone
from dizher.converter.dither import Ordered
from dizher.ui.window import REDO, UNDO, Window

IMAGE = Path(__file__).parent / 'images' / 'goldhill-256.png'
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
    assert folder == IMAGE.with_name('goldhill-256') and not folder.exists() and not ui.autosave   # tests never write it
    with tempfile.TemporaryDirectory() as tmp:
        ui.project, ui.autosave = Path(tmp) / 'goldhill-256', True
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
    ctx.menu_click('//##MainMenuBar/File/Open recent/goldhill-256.png##' + str(ui.recent.index(IMAGE.resolve())))
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


def pick(ctx, combo, item):
    """combo_click wants a popup it does not find behind widgets.combo: open it, click the item in the popup."""
    ctx.item_click(combo)
    ctx.yield_(2)
    ctx.item_click(f'//##Combo_00/{item}')


def test_custom_palette(ctx):
    """A toggled colour makes the palette Custom; a named subset turns its colours on and carries over a mode switch."""
    from dizher import ops
    ctx.set_ref('//Convert')
    ctx.item_click('target/##colour1')
    ctx.yield_(2)
    assert params('target').colours == (0, *range(2, 16))
    assert ops.MODES[params('target').mode].palette.subset_name(params('target').colours) == 'Custom'
    pick(ctx, 'target/palette', 'Grayscale')
    ctx.yield_(2)
    assert params('target').colours == (0, 7, 8, 15), params('target')
    pick(ctx, 'target/mode', ops.c64.HIRES.name)
    ctx.yield_(2)
    assert params('target') == replace(params('target'), mode=ops.c64.HIRES.name, colours=(0, 1, 11, 12, 15)), params('target')
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


def test_cell_popup(ctx):
    """Right-click a cell: a row of the pair table paints it, a click in the zoom moves to a neighbour, a right click
    on a swatch paints that one's paper, Auto gives it back."""
    from imgui_bundle.imgui.test_engine import CaptureFlags_
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'a conversion to overpaint')
    ui.view = 'Screen'
    r = rect(ctx, '//Preview', '**/Screen')
    ctx.mouse_move_to_pos(imgui.ImVec2(r.min.x + 100, r.max.y + 100))
    ctx.yield_(2)
    ctx.mouse_click(1)
    ctx.yield_(2)
    row, col = ui._cell
    labels = ui.result.best_attr_indexes
    order = ui.result.energy.cell_candidates(labels, row, col).sum(1).argsort()
    other = int(order[order != labels[row, col]][0])   # a listed pair not chosen
    ctx.set_ref('//$FOCUSED')
    pair = list(ui.result.palette.iter_idxs_pairs())[other]
    ctx.item_click(f'**/###pair{other}')
    ctx.yield_(2)
    assert params("overpaint").overrides == ((row, col, *pair),), (params("overpaint"), row, col, pair)
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'the overpainted conversion')
    assert ui.result.best_attr_indexes[row, col] == other
    ctx.capture_set_filename('/tmp/dizher_cell_popup.png')
    ctx.capture_screenshot(CaptureFlags_.hide_mouse_cursor.value)
    popup = ctx.get_window_by_ref('//$FOCUSED')
    pad = imgui.get_style().window_padding
    ctx.mouse_move_to_pos(imgui.ImVec2(popup.pos.x + pad.x + 5, popup.pos.y + pad.y + 5))   # the zoom's top-left cell
    ctx.mouse_click(0)
    ctx.yield_(2)
    R, C = labels.shape
    corner = max(0, min(row - 1, R - 3)), max(0, min(col - 1, C - 3))
    assert ui._cell == corner, (ui._cell, corner)
    ctx.set_ref(f'//{popup.name}')
    ctx.item_click('##colour2', imgui.MouseButton_.right)
    ctx.yield_(2)
    assert (*corner, 2, -1) in params('overpaint').overrides, params('overpaint')
    ctx.item_click('**/Auto')
    ctx.yield_(2)
    assert params("overpaint").overrides == ((row, col, *pair),), (params("overpaint"), row, col, pair)
    ctx.key_press(imgui.Key.escape)
    ctx.yield_(2)
    reset('overpaint')
    ctx.mouse_move_to_pos(imgui.ImVec2(r.min.x - 50, r.min.y - 50))


def test_paint(ctx):
    """Paint mode: a left click on a swatch picks the ink, a right click the paper, either turns it on; a left drag
    over the preview paints the cells it crosses, one undo step; a right click picks a cell's colours up; Auto as
    both erases; Esc ends it; Clear drops every cell."""
    from imgui_bundle.imgui.test_engine import CaptureFlags_
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'a conversion to paint')
    brush = ui.editors['overpaint']
    ctx.set_ref('//Convert')
    ctx.scroll_to_item('overpaint/Paint', imgui.internal.Axis.y)   # item_click's own scroll misses it by a pixel
    ctx.item_click('overpaint/##colour2')   # picking a colour turns Paint on
    ctx.item_click('overpaint/##colour5', imgui.MouseButton_.right)
    ctx.yield_(2)
    assert brush.on and (brush.ink, brush.paper) == (2, 5), vars(brush)
    r = rect(ctx, '//Preview', '**/Screen')
    a, b = imgui.ImVec2(r.min.x + 100, r.max.y + 100), imgui.ImVec2(r.min.x + 160, r.max.y + 100)
    steps = len(ui.app.past)
    ctx.mouse_move_to_pos(a)
    ctx.mouse_down(0)
    for t in range(1, 7):   # frame by frame: every cell on the way
        ctx.mouse_move_to_pos(imgui.ImVec2(a.x + (b.x - a.x) * t / 6, a.y))
    ctx.capture_set_filename('/tmp/dizher_paint.png')
    ctx.capture_screenshot(CaptureFlags_.none.value)
    ctx.mouse_up(0)
    ctx.yield_(2)
    overrides = params('overpaint').overrides
    assert len(overrides) >= 2 and all(o[2:] == (5, 2) for o in overrides), overrides
    assert len(ui.app.past) == steps + 1, 'a stroke is one undo step'
    wait(ctx, lambda: ui.app.result('overpaint') is not None, 'the painted cells')
    ctx.mouse_move_to_pos(imgui.ImVec2(a.x, a.y + 40))   # the eyedropper, on a cell not painted
    ctx.mouse_click(1)
    ctx.yield_(2)
    shown = ui._colours(ui.app.result('overpaint'), *next(o[:2] for o in overrides))   # a painted one reads as painted
    assert shown == (5, 2), shown
    picked = (brush.paper, brush.ink)
    assert picked != (5, 2) and params('overpaint').overrides == overrides, (picked, params('overpaint'))
    ctx.set_ref('//Convert')
    ctx.item_click('overpaint/##colour-2')                             # Auto ink and paper: the eraser
    ctx.item_click('overpaint/##colour-2', imgui.MouseButton_.right)
    ctx.mouse_move_to_pos(a)
    ctx.mouse_down(0)
    for t in range(1, 7):
        ctx.mouse_move_to_pos(imgui.ImVec2(a.x + (b.x - a.x) * t / 6, a.y))
    ctx.mouse_up(0)
    ctx.yield_(2)
    assert params('overpaint').overrides == (), params('overpaint')
    ctx.key_press(UNDO)
    ctx.yield_(2)
    assert params('overpaint').overrides == overrides
    ctx.key_press(imgui.Key.escape)
    ctx.yield_(2)
    assert not brush.on
    ctx.set_ref('//Convert')
    ctx.item_click('overpaint/Clear')
    ctx.yield_(2)
    assert params('overpaint').overrides == ()
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
    ('ui', 'custom_palette', test_custom_palette),
    ('ui', 'hover_inspector', test_hover_inspector),
    ('ui', 'cell_popup', test_cell_popup),
    ('ui', 'paint', test_paint),
    ('ui', 'capture_layout', test_capture_layout),
]

if __name__ == '__main__':
    sys.exit(0 if ui.run_tests(TESTS) else 1)
