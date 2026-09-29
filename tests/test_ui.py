"""UI tests under the imgui test engine, ToolZX's pattern: a real window, simulated mouse, tests in TESTS order on
one shared Window. Run: .venv/bin/python tests/test_ui.py (exit code 0 when all pass)."""
import sys
from dataclasses import replace
from pathlib import Path

from imgui_bundle import imgui

from dizher import tone
from dizher.converter.dither import Ordered
from dizher.ui.levels import CROSS, pick_label
from dizher.ui.window import REDO, UNDO, Window

IMAGE = Path(__file__).parent / 'images' / 'goldhill.png'
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



def test_levels_channel(ctx):
    """R shows the red channel: its black handle moves the red levels only, the composite stays."""
    editor = ui.editors['levels']
    ctx.set_ref('//Tune')
    ctx.item_click('**/R##channel')
    ctx.yield_(2)
    assert editor.channel == 1
    r = rect(ctx, '//Tune', '**/##in')
    pad, y = (r.max.y - r.min.y) / 4, r.max.y - 3
    drag(ctx, imgui.ImVec2(r.min.x + pad, y), imgui.ImVec2(r.min.x + pad + 50, y))
    p = params('levels')
    assert p.channels[0][0] > 0 and p.channels[1:] == type(p)().channels[1:] and p.in_black == 0, p
    ctx.item_click('**/RGB##channel')
    ctx.yield_(2)
    assert editor.channel == 0


def test_tone_mode_switch(ctx):
    """Curves brings the levels along as curves; a click on the graph adds a point; Levels keeps its own values."""
    levels = params('levels')
    ctx.set_ref('//Tune')
    ctx.item_click('**/Curves')
    ctx.yield_(2)
    p = params('levels')
    assert p.mode == 'Curves' and p.curves == tone.levels_to_curves(
        (p.in_black, p.in_white, p.gamma, p.out_black, p.out_white), p.channels), p
    assert p.curves[1] != tone.IDENTITY, 'the red levels became a red curve'
    ctx.item_click('**/G##channel')
    r = rect(ctx, '//Tune', '**/##curve')
    ctx.mouse_move_to_pos(imgui.ImVec2(r.min.x + (r.max.x - r.min.x) * 0.3, r.min.y + (r.max.y - r.min.y) * 0.3))
    ctx.mouse_click(0)
    ctx.yield_(2)
    assert len(tone.points(params('levels').curves[2])) == 3, params('levels').curves
    ctx.item_click('**/Levels')
    ctx.yield_(2)
    assert params('levels') == replace(levels, curves=params('levels').curves), 'Levels keeps its own values'
    ctx.item_click('**/Curves')   # the curves are edited now: it asks, and Keep the curves keeps them
    ctx.yield_(2)
    ctx.item_click('//$FOCUSED/Keep the curves')
    ctx.yield_(2)
    assert params('levels').mode == 'Curves' and len(tone.points(params('levels').curves[2])) == 3
    ctx.item_click('**/RGB##channel')
    reset('levels')


def preview_point(ctx, dx, dy):
    r = rect(ctx, '//Preview', '**/Screen')   # the tuned image is under the view bar
    return imgui.ImVec2(r.min.x + dx, r.max.y + dy)


def test_eyedroppers(ctx):
    """The grey target is picked from the palette; the armed grey eyedropper takes clicks in the preview, off Paint
    mode: in Levels it sets the channels' gamma so the sample lands on the target, in Curves every click adds a point
    group, listed with a delete button; Esc disarms."""
    from imgui_bundle.imgui.test_engine import CaptureFlags_, TestRef
    wait(ctx, lambda: not ui.app.busy and ui.app.result('light') is not None, 'the tone input')
    editor, brush = ui.editors['levels'], ui.editors['overpaint']
    ctx.set_ref('//Tune')
    # a **/ wildcard never ends on a label that is only ##id: the swatch is found beside its eyedropper
    ctx.set_ref(TestRef(ctx.item_info('**/' + pick_label('grey')).parent_id))
    ctx.item_click('##target grey')
    ctx.yield_(2)
    ctx.item_click('//$FOCUSED/##colour5')   # non-bright cyan
    ctx.yield_(2)
    assert params('levels').targets == (-1, 5, -1), params('levels')
    brush.on = True
    ctx.yield_(2)
    ctx.set_ref('//Tune')
    ctx.item_click('**/' + pick_label('grey'))
    ctx.yield_(2)
    assert editor.armed == 'grey' and not brush.on, 'arming ends Paint mode'
    ctx.mouse_move_to_pos(preview_point(ctx, 60, 60))
    ctx.yield_(2)
    ctx.capture_set_filename('/tmp/dizher_eyedropper.png')
    ctx.capture_screenshot(CaptureFlags_.none.value)
    ctx.mouse_click(0)
    ctx.yield_(2)
    p = params('levels')
    s = editor.last['Levels', 'grey']
    got, cyan = editor.result(p, s), ui._palette().as_float()[5]
    assert p.channels != type(p)().channels and abs(got[1:] - cyan[1:]).max() <= 1 / 255, (p.channels, got, cyan)
    assert not params('overpaint').overrides, 'the click painted'
    ctx.set_ref('//Tune')
    ctx.set_ref(TestRef(ctx.item_info('**/' + pick_label('grey')).parent_id))
    ctx.item_click(f'{CROSS}##clear grey')   # puts the gammas back; the next click sets them again
    ctx.yield_(2)
    assert params('levels').channels == type(p)().channels and ('Levels', 'grey') not in editor.last
    ctx.mouse_move_to_pos(preview_point(ctx, 60, 60))
    ctx.mouse_click(0)
    ctx.yield_(2)
    assert params('levels') == p
    ctx.set_ref('//Tune')
    ctx.item_click('**/Curves')   # the curves are as new: the levels come along without asking
    ctx.yield_(2)
    assert params('levels').mode == 'Curves' and params('levels').curves[1] != tone.IDENTITY and editor.armed == 'grey'
    for dx in (40, 120):
        ctx.mouse_move_to_pos(preview_point(ctx, dx, 60))
        ctx.mouse_click(0)
        ctx.yield_(2)
    assert len(params('levels').picks) == 2, params('levels').picks
    ctx.set_ref('//Tune')
    ctx.set_ref(TestRef(ctx.item_info('**/' + pick_label('grey')).parent_id))
    ctx.item_click(f'{CROSS}##drop0')
    ctx.yield_(2)
    assert len(params('levels').picks) == 1
    ctx.key_press(imgui.Key.escape)
    ctx.yield_(2)
    assert editor.armed is None
    ctx.capture_set_filename('/tmp/dizher_curves.png')
    ctx.capture_screenshot(CaptureFlags_.hide_mouse_cursor.value)
    ctx.mouse_move_to_pos(preview_point(ctx, -50, -500))
    reset('levels')
def test_edit_keeps_the_live_snapshot(ctx):
    """An edit while the optimiser runs cancels it; till the next stage sends a snapshot the last one stays on screen,
    not the older finished conversion."""
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'the conversion to settle')
    before = ui.result
    ui.app.set_params('optimise', replace(params('optimise'), enabled=True))
    running = lambda: ui.app.job is not None and ui.app.job.node_id == 'optimise' and ui.app.job.image is not None
    wait(ctx, running, 'an optimiser snapshot')
    ctx.yield_()
    snapshot = ui._live()
    assert snapshot is not None
    ui.app.set_params('overpaint', replace(params('overpaint'), overrides=((0, 0, 1, 7),)))
    for _ in range(3000):
        ctx.yield_()
        live = ui._live()
        shown = live if live is not None else ui.result   # what _converted draws
        assert shown is not None and shown is not before, 'the preview fell back to the older conversion'
        if shown is not snapshot:   # the next stage's snapshot, or its finished conversion
            break
    else:
        raise AssertionError('no newer conversion came')
    reset('overpaint')
    ui.app.set_params('optimise', replace(params('optimise'), enabled=False))
    wait(ctx, lambda: not ui.app.busy, 'the pipeline to settle')


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
    assert folder == IMAGE.with_name('goldhill') and not folder.exists() and not ui.autosave   # tests never write it
    with tempfile.TemporaryDirectory() as tmp:
        ui.project, ui.autosave = Path(tmp) / 'goldhill', True
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
    ctx.menu_click('//##MainMenuBar/File/Open recent/goldhill.png##' + str(ui.recent.index(IMAGE.resolve())))
    ctx.yield_(2)
    assert ui._source() == IMAGE.resolve() and ui.recent[0] == IMAGE.resolve()
    ui.app.set_params('halftoner', replace(params('halftoner'), halftoner=Ordered.label))
    ui.app.set_params('optimise', replace(params('optimise'), enabled=False))

def test_about(ctx):
    """Help > About Dizher opens the dialog with the version; Cool closes it."""
    from imgui_bundle.imgui.test_engine import CaptureFlags_
    ctx.menu_click('//##MainMenuBar/Help/About Dizher…')
    ctx.yield_(2)
    assert ctx.item_exists('//About Dizher/Cool'), 'no About dialog'
    ctx.capture_set_filename('/tmp/dizher_about.png')
    ctx.capture_screenshot(CaptureFlags_.hide_mouse_cursor.value)
    ctx.item_click('//About Dizher/Cool')
    ctx.yield_(2)
    assert not ctx.item_exists('//About Dizher/Cool'), 'About still open'

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


def test_method_switch_brings_its_values(ctx):
    """Picking another selection method sets the Metric and Select pairs values it was tuned with, as one undo step."""
    from dizher.converter.energy import METHODS
    before = ui.app.graph
    ctx.set_ref('//Convert')
    pick(ctx, 'metric/method', 'Halftoned')
    ctx.yield_(2)
    preset = METHODS['Halftoned'].preset
    assert params('metric').method == 'Halftoned' and params('metric').chroma == preset['chroma'], params('metric')
    assert (params('select').coherence, params('select').chroma_noise) == (preset['coherence'], preset['chroma_noise'])
    ui.app.undo()
    ctx.yield_(2)
    assert ui.app.graph == before
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'the conversion again')


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
    both erases; Esc ends it; Clear drops every cell; Fix paints every cell with the colours it shows; Hide
    shows the conversion without them, keeping them, and Paint shows them again."""
    from imgui_bundle.imgui.test_engine import CaptureFlags_
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'a conversion to paint')
    brush = ui.editors['overpaint']
    ctx.set_ref('//Convert')
    ctx.scroll_to_item('overpaint/Paint', imgui.internal.Axis.y)   # item_click's own scroll misses it by a pixel
    ctx.item_click('overpaint/##colour2')   # picking a colour turns Paint on
    ctx.item_click('overpaint/##colour5', imgui.MouseButton_.right)
    ctx.yield_(2)
    assert brush.on and (brush.ink, brush.paper) == (2, 5), vars(brush)
    ctx.item_click('overpaint/##colour13', imgui.MouseButton_.right)   # a bright paper brightens the ink
    ctx.yield_(2)
    assert (brush.ink, brush.paper) == (10, 13), vars(brush)
    ctx.item_click('overpaint/##colour5', imgui.MouseButton_.right)
    ctx.yield_(2)
    assert (brush.ink, brush.paper) == (2, 5), vars(brush)
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
    ctx.set_ref('//Convert')
    ctx.item_click('overpaint/##colour-3')   # B1 as ink and paper: the cells keep their colours, made bright
    ctx.yield_(2)
    assert (brush.ink, brush.paper) == (-3, -3), vars(brush)
    wait(ctx, lambda: not ui.app.busy and ui.app.result('overpaint') is not None, 'the painted cells again')
    ctx.mouse_move_to_pos(a)
    ctx.mouse_down(0)
    for t in range(1, 7):
        ctx.mouse_move_to_pos(imgui.ImVec2(a.x + (b.x - a.x) * t / 6, a.y))
    ctx.mouse_up(0)
    ctx.yield_(2)
    assert all(o[2:] == (13, 10) for o in params('overpaint').overrides), params('overpaint')
    ctx.key_press(UNDO)
    ctx.yield_(2)
    ctx.key_press(imgui.Key.escape)
    ctx.yield_(2)
    assert not brush.on
    ctx.set_ref('//Convert')
    ctx.item_click('overpaint/Clear')
    ctx.yield_(2)
    assert params('overpaint').overrides == ()
    ui.app.set_params('overpaint', replace(params('overpaint'), overrides=((2, 3, -1, 2), (4, 5, 7, 1))))   # Auto paper
    ctx.item_click('overpaint/Paint')
    ctx.item_click('overpaint/Hide')
    ctx.yield_(2)
    assert ui.app.unpainted and not brush.on and len(params('overpaint').overrides) == 2, 'Hide keeps the cells'
    ctx.item_click('overpaint/Paint')
    ctx.yield_(2)
    assert brush.on and not ui.app.unpainted, 'painting shows the painted cells'
    ctx.item_click('overpaint/Paint')
    ctx.yield_(2)
    steps = len(ui.app.past)
    ctx.item_click('overpaint/Fix')
    ctx.yield_(2)
    fixed = params('overpaint').overrides
    cell = {o[:2]: o[2:] for o in fixed}
    assert len(fixed) == 24 * 32 and all(-1 not in o for o in fixed), 'every cell painted'
    assert cell[2, 3][1] == 2 and cell[4, 5] == (7, 1) and len(ui.app.past) == steps + 1, 'Fix is one undo step'
    assert ui.editors['overpaint'].fixed(fixed, ui.app.shown('select')) == fixed, 'nothing left to fix: Fix disabled'
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
    ('ui', 'levels_channel', test_levels_channel),
    ('ui', 'tone_mode_switch', test_tone_mode_switch),
    ('ui', 'eyedroppers', test_eyedroppers),
    ('ui', 'edit_keeps_the_live_snapshot', test_edit_keeps_the_live_snapshot),
    ('ui', 'history_panel', test_history_panel),
    ('ui', 'autosave', test_autosave),
    ('ui', 'unsaved_dialog', test_unsaved_dialog),
    ('ui', 'recent_images', test_recent_images),
    ('ui', 'about', test_about),
    ('ui', 'block_collapses_and_expands', test_block_collapses_and_expands),
    ('ui', 'block_reset', test_block_reset),
    ('ui', 'views_and_grid', test_views_and_grid),
    ('ui', 'mode_switch_refits_previews', test_mode_switch_refits_previews),
    ('ui', 'custom_palette', test_custom_palette),
    ('ui', 'method_switch_brings_its_values', test_method_switch_brings_its_values),
    ('ui', 'hover_inspector', test_hover_inspector),
    ('ui', 'cell_popup', test_cell_popup),
    ('ui', 'paint', test_paint),
    ('ui', 'capture_layout', test_capture_layout),
]

if __name__ == '__main__':
    sys.exit(0 if ui.run_tests(TESTS) else 1)
