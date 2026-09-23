"""UI tests under the imgui test engine, ToolZX's pattern: a real window, simulated mouse, tests in TESTS order on
one shared Window. Run: .venv/bin/python tests/test_ui.py (exit code 0 when all pass)."""
import sys
from dataclasses import replace
from pathlib import Path

from imgui_bundle import imgui

from dizher import tone
from dizher.converter.dither import OrderedBayer
from dizher.ui.window import Window

IMAGE = Path(__file__).parent / 'images' / 'lena.png'
ui = Window(IMAGE)
ui.app.set_params('halftone', replace(ui.app.graph['halftone'].params, halftoner=OrderedBayer.label))  # fast


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
    wait(ctx, lambda: ui.app.result('halftone') is not None, 'the first conversion')
    assert not ui.app.errors, ui.app.errors


def test_slider_drag_moves_param(ctx):
    r = rect(ctx, '//Tune', '**/contrast/contrast')   # the slider, under params_editor's push_id; '**/contrast' is the header
    y = (r.min.y + r.max.y) / 2
    drag(ctx, imgui.ImVec2((r.min.x + r.max.x) / 2, y), imgui.ImVec2((r.min.x + r.max.x) / 2 + 60, y))
    moved = params('contrast').contrast
    assert moved > 0, f'contrast slider did not move: {moved}'
    ctx.yield_(5)
    assert params('contrast').contrast == moved, 'the slider fell back'
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


def test_block_collapses_and_expands(ctx):
    ctx.set_ref('//Tune')
    ctx.item_click('**/###crop')
    ctx.yield_(2)
    assert ui.expanded['crop'] is False
    ctx.item_click('**/###crop')
    ctx.yield_(2)
    assert ui.expanded['crop'] is True


def test_block_reset(ctx):
    ui.app.set_params('light', replace(params('light'), exposure=0.37))
    ctx.yield_(2)
    ctx.set_ref('//Tune')
    ctx.item_click('**/Reset##light')
    ctx.yield_(2)
    assert params('light') == type(params('light'))(), params('light')


def test_capture_layout(ctx):   # keep last: a picture of the default layout for review
    from imgui_bundle.imgui.test_engine import CaptureFlags_
    wait(ctx, lambda: not ui.app.busy and ui.app.result('halftone') is not None, 'the conversion to settle')
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
    ('ui', 'block_collapses_and_expands', test_block_collapses_and_expands),
    ('ui', 'block_reset', test_block_reset),
    ('ui', 'capture_layout', test_capture_layout),
]

if __name__ == '__main__':
    sys.exit(0 if ui.run_tests(TESTS) else 1)
