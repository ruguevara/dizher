"""The pictures in docs/images: the app driven under the imgui test engine as tests/test_ui.py drives it, on a fresh
project of a copy of a test image (nothing in tests/ is written), at twice the 96 dpi size, each shot cropped to its
panel. A page shows them at half their width: sharp on a Retina screen, an exact 2:1 on others.
Run: python docs/screenshots.py; on Linux without a display:
xvfb-run -a -s '-screen 0 4200x2600x24' python docs/screenshots.py"""
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from imgui_bundle import imgui
from imgui_bundle.imgui.test_engine import CaptureFlags_, TestRef
from PIL import Image

from dizher import ops
from dizher.ui import gallery
from dizher.ui.levels import pick_label
from dizher.ui.window import Window

DOCS = Path(__file__).parent
IMAGES = DOCS.parent / 'tests' / 'images'
IMAGE = IMAGES / 'anubis.png'
OUT = DOCS / 'images'
SCALE = 2              # screen px per 96 dpi px
SIZE = (2000, 1150)    # the window at 96 dpi: Convert wide enough for its labels
MARGIN = 6             # 96 dpi px around a panel
GALLERY_ROW = 5        # gallery images a row
SECTIONS = 2           # the pattern combo's sections shown

tmp = Path(tempfile.mkdtemp())
shutil.copy(IMAGE, tmp)
ui = Window(tmp / IMAGE.name)
ui.expanded = {nid: False for nid, *_ in ops.TUNE + ops.CONVERT} | {'export': False}   # each shot opens its blocks
gallery.ZOOM = 0.5     # gallery image px per 96 dpi px: a screen px each, small on the page
COLUMNS = {'//Tune': [nid for nid, *_ in ops.TUNE[1:]], '//Convert': [nid for nid, *_ in ops.CONVERT] + ['Export']}
_runner = ui.runner_params


def runner_params(persist=True):
    p = _runner(persist)
    p.app_window_params.window_geometry.size = SIZE
    p.dpi_aware_params.dpi_window_size_factor = SCALE
    return p


ui.runner_params = runner_params


def params(nid):
    return ui.app.graph[nid].params


def wait(ctx, cond, what, frames=30000):
    for _ in range(frames):
        if cond():
            return
        ctx.yield_()
    assert cond(), f'timed out waiting for {what}'


def settled(ctx):
    wait(ctx, lambda: not ui.app.busy and ui.app.result('optimise') is not None, 'the conversion')
    ctx.yield_(4)


def header(nid):
    return f'**/###{nid}' if nid != 'Export' else '**/Export'


def block(ctx, window, *nids):
    """The rect of blocks next to each other in a column: the first's header to the header after the last."""
    ctx.set_ref(window)
    column = COLUMNS[window]
    top = ctx.item_info(header(nids[0])).rect_full
    bottom = ctx.item_info(header(column[column.index(nids[-1]) + 1])).rect_full
    return top.min.x, top.min.y, top.max.x, bottom.min.y - imgui.get_style().item_spacing.y


def window(name):
    w = imgui.internal.find_window_by_name(name)
    return w.pos.x, w.pos.y, w.pos.x + w.size.x, w.pos.y + w.size.y


def shot(ctx, name, *rects, margin=MARGIN, hover=False):
    """The screen cropped to the rects, margin 96 dpi px around them, as docs/images/<name>.png. The pointer goes to
    the status bar first, so no tooltip shows, unless the shot is of what hovering shows."""
    if not hover:
        view = imgui.get_main_viewport()
        ctx.mouse_move_to_pos(imgui.ImVec2(view.pos.x + view.size.x - 10, view.pos.y + view.size.y - 10))
        ctx.yield_(4)
    full = tmp / f'{name}-screen.png'
    ctx.capture_set_filename(str(full))
    ctx.capture_screenshot(CaptureFlags_.hide_mouse_cursor.value)
    m = margin * SCALE
    x0, y0 = round(min(r[0] for r in rects) - m), round(min(r[1] for r in rects) - m)
    x1, y1 = round(max(r[2] for r in rects) + m), round(max(r[3] for r in rects) + m)
    x1, y1 = x1 + (x1 - x0) % 2, y1 + (y1 - y0) % 2   # even: a page halves it exactly
    Image.open(full).convert('RGB').crop((x0, y0, x1, y1)).save(OUT / f'{name}.png', optimize=True)
    print(f'{name}.png: width="{(x1 - x0) // 2}"')   # the width to show it at


def opened(ctx, window, *nids):
    ctx.set_ref(window)
    for nid in nids:
        ctx.item_open(header(nid))
    ctx.yield_(2)


def closed(ctx, window, *nids):
    ctx.set_ref(window)
    for nid in nids:
        ctx.item_close(header(nid))
    ctx.yield_(2)


def preview(ctx, dx, dy):
    """A point on the tuned image, dx, dy 96 dpi px from its top-left: it is under the view bar."""
    ctx.set_ref('//Preview')
    bar = ctx.item_info('**/Screen').rect_full
    return imgui.ImVec2(bar.min.x + dx * SCALE, bar.max.y + imgui.get_style().item_spacing.y + dy * SCALE)


def metric_method(ctx):
    settled(ctx)
    opened(ctx, '//Convert', 'metric')
    ctx.item_click('metric/method')
    ctx.yield_(4)
    shot(ctx, 'metric-method', block(ctx, '//Convert', 'metric'), window('##Combo_00'))
    ctx.popup_close_all()   # a combo takes no Esc: keyboard nav is off
    closed(ctx, '//Convert', 'metric')


def select_pairs(ctx):
    ui.app.set_params('select', replace(params('select'), surface=20.0))
    settled(ctx)
    opened(ctx, '//Convert', 'select')
    shot(ctx, 'select-pairs', block(ctx, '//Convert', 'select'))
    closed(ctx, '//Convert', 'select')
    ui.app.set_params('select', replace(params('select'), surface=0.0))


def halftone_optimise(ctx):
    settled(ctx)
    opened(ctx, '//Convert', 'halftone', 'optimise')
    shot(ctx, 'halftone-optimise', block(ctx, '//Convert', 'halftone', 'optimise'))
    closed(ctx, '//Convert', 'halftone', 'optimise')


def pattern_combo(ctx):
    settled(ctx)
    opened(ctx, '//Convert', 'halftoner')
    ctx.item_click('halftoner/matrix')
    ctx.yield_(4)
    x0, y0, x1, _ = window('##Combo_00')
    last = list(gallery.MATRIX_GROUPS.values())[SECTIONS - 1][-1]
    y1 = ctx.item_info(f'//##Combo_00/##{last}').rect_full.max.y + imgui.get_style().item_spacing.y
    shot(ctx, 'pattern-combo', (x0, y0, x1, y1), margin=0)
    ctx.popup_close_all()   # a combo takes no Esc: keyboard nav is off
    closed(ctx, '//Convert', 'halftoner')


def pattern_gallery(ctx):
    """The first section, GALLERY_ROW images a row, cropped under the last row."""
    settled(ctx)
    picker, first = ui.editors['halftoner'].picker, next(iter(gallery.MATRIX_GROUPS))
    opened(ctx, '//Convert', 'halftoner')
    ctx.item_click('halftoner/Gallery…')
    ctx.yield_(2)
    ctx.set_ref('//$FOCUSED')
    ctx.item_click(f'**/{first}')
    s = imgui.get_style()
    frame = 2 * (s.window_padding.x + s.popup_border_size) + s.scrollbar_size
    width = frame + GALLERY_ROW * 256 * SCALE * gallery.ZOOM + (GALLERY_ROW - 1) * s.item_spacing.x + 1
    ctx.window_resize('//$FOCUSED', imgui.ImVec2(width, ctx.get_window_by_ref('//$FOCUSED').size.y))
    wait(ctx, lambda: not picker.thumbs.pending and set(gallery.MATRIX_GROUPS[first]) <= set(picker.thumbs.images),
         'the section halftoned')
    ctx.yield_(4)
    modal, images = ctx.get_window_by_ref('//$FOCUSED'), ctx.window_info('//$FOCUSED/images').window
    shot(ctx, 'gallery', (modal.pos.x, modal.pos.y, modal.pos.x + modal.size.x,
                          images.pos.y + images.content_size.y + s.window_padding.y), margin=0)
    ctx.key_press(imgui.Key.escape)
    closed(ctx, '//Convert', 'halftoner')


def levels_curves(ctx):
    settled(ctx)
    opened(ctx, '//Tune', 'levels')
    ctx.item_click('**/Curves')
    editor = ui.editors['levels']
    for (dx, dy), colour in (((60, 150), 6), ((200, 120), 2)):   # orange onto yellow, brown onto red
        ctx.set_ref('//Tune')
        ctx.set_ref(TestRef(ctx.item_info('**/' + pick_label('grey')).parent_id))   # the swatch is beside it
        ctx.item_click('##target grey')
        ctx.yield_(2)
        ctx.item_click(f'//$FOCUSED/##colour{colour}')
        ctx.yield_(2)
        if editor.armed != 'grey':
            ctx.item_click(pick_label('grey'))
        ctx.mouse_move_to_pos(preview(ctx, dx, dy))
        ctx.mouse_click(0)
        ctx.yield_(2)
    ctx.key_press(imgui.Key.escape)
    settled(ctx)
    shot(ctx, 'levels-curves', block(ctx, '//Tune', 'levels'))
    closed(ctx, '//Tune', 'levels')
    ui.app.set_params('levels', type(params('levels'))())


def overpaint(ctx):
    painted = json.loads((IMAGES / 'anubis' / 'project.json').read_text())['nodes']['overpaint']['params']['overrides']
    ui.app.set_params('overpaint', replace(params('overpaint'), overrides=tuple(map(tuple, painted))))
    settled(ctx)
    opened(ctx, '//Convert', 'overpaint')
    ctx.item_click('overpaint/##colour6')                              # yellow ink, non-bright
    ctx.item_click('overpaint/##colour10', imgui.MouseButton_.right)   # red paper: both bright
    ctx.yield_(2)
    shot(ctx, 'overpaint', block(ctx, '//Convert', 'overpaint'))
    ctx.key_press(imgui.Key.escape)
    closed(ctx, '//Convert', 'overpaint')


def cell_popup(ctx):
    settled(ctx)
    ctx.mouse_move_to_pos(preview(ctx, 150, 100))
    wait(ctx, lambda: ui._tip is not None, 'the inspector')
    ctx.mouse_click(1)
    ctx.yield_(4)
    popup = ctx.get_window_by_ref('//$FOCUSED')
    shot(ctx, 'cell-popup', (popup.pos.x, popup.pos.y, popup.pos.x + popup.size.x, popup.pos.y + popup.size.y),
         margin=0)
    ctx.key_press(imgui.Key.escape)
    ui.app.set_params('overpaint', type(params('overpaint'))())


def save_snapshot(ctx, name):
    settled(ctx)
    ctx.item_click('//Snapshots/Save snapshot')
    ctx.yield_(2)
    ctx.key_chars_replace_enter(name)
    ctx.yield_(2)


def snapshots(ctx):
    save_snapshot(ctx, 'Exact mixture')
    ui.app.set_graph(ops.apply_preset(ui.app.graph, 'Halftoned'))
    save_snapshot(ctx, 'Halftoned')
    ui.app.set_graph(ops.apply_preset(ui.app.graph, 'Exact mixture'))
    ui.app.set_params('halftone', replace(params('halftone'), dithering=0.5, checker=True))
    save_snapshot(ctx, 'Checker')
    ctx.set_ref('//Snapshots')
    row = ctx.item_info('**/###Exact mixture').rect_full
    ctx.mouse_move_to_pos(imgui.ImVec2(row.min.x + 20 * SCALE, (row.min.y + row.max.y) / 2))   # the tooltip in the dock
    wait(ctx, lambda: (t := imgui.internal.find_window_by_name('##Tooltip_00')) is not None and t.active, 'the tooltip')
    ctx.yield_(4)
    x0, y0, x1, _ = window('Snapshots')
    shot(ctx, 'snapshots', (x0, y0, x1, window('##Tooltip_00')[3]), margin=0, hover=True)


TESTS = [('docs', fn.__name__, fn) for fn in (metric_method, select_pairs, halftone_optimise, pattern_combo,
                                              pattern_gallery, levels_curves, overpaint, cell_popup, snapshots)]

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    ok = ui.run_tests(TESTS)
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)
