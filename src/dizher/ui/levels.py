"""Photoshop-style editor of the levels node: the input histogram with the black, midtone and white handles under
it, the output range strip with its two handles, the numbers and Auto (Reset is on the block header). The
midtone handle sits where the curve crosses mid grey, so moving the black or white point carries it along at
the same gamma, as in Photoshop."""
import math
from dataclasses import replace

from imgui_bundle import em_size, imgui

from mokit.ui import style

from .. import tone

HIST_HEIGHT = 5.0     # em
STRIP_HEIGHT = 0.7    # em, the gradient under the histogram
HANDLE = 0.45         # em, half the width of a handle triangle
NUMBER_WIDTH = 3.5    # em
MIN_GAP = 2           # in_white - in_black, as in Photoshop


def midtone(b: int, w: int, gamma: float) -> float:
    """Input value that the curve maps to mid grey."""
    return b + (w - b) * 0.5 ** gamma


def gamma_at(x: float, b: int, w: int) -> float:
    t = min(max((x - b) / (w - b), 1e-3), 1 - 1e-3)
    return round(min(max(math.log(t) / math.log(0.5), 0.1), 9.99), 2)


def grey(v: float, a: float = 1.0) -> int:
    return style.u32(imgui.ImVec4(v, v, v, a))


class LevelsEditor:
    def __init__(self) -> None:
        self._drag = None          # handle name while the mouse holds one
        self._hist = (None, None)  # (input array, its histogram): recomputed only when the input changes

    def draw(self, params, picture, on_change, id: str) -> None:
        set_ = lambda **kw: on_change(replace(params, **kw))
        width = imgui.get_content_region_avail().x
        imgui.push_id(id)
        self._histogram(picture, width)
        b, w, g = params.in_black, params.in_white, params.gamma
        moved = self._strip('in', width, {'in_black': (b, 0.0), 'midtone': (midtone(b, w, g), 0.5), 'in_white': (w, 1.0)})
        if moved:
            name, x = moved
            if name == 'in_black':
                set_(in_black=min(round(x), w - MIN_GAP))
            elif name == 'in_white':
                set_(in_white=max(round(x), b + MIN_GAP))
            else:
                set_(gamma=gamma_at(x, b, w))
        self._numbers((('in_black', b, 0, w - MIN_GAP), ('gamma', g, 0.1, 9.99), ('in_white', w, b + MIN_GAP, 255)), width, set_)
        imgui.text('Output levels')
        ob, ow = params.out_black, params.out_white
        moved = self._strip('out', width, {'out_black': (ob, 0.0), 'out_white': (ow, 1.0)})
        if moved:
            set_(**{moved[0]: round(moved[1])})
        self._numbers((('out_black', ob, 0, 255), ('out_white', ow, 0, 255)), width, set_)
        imgui.begin_disabled(picture is None)
        if imgui.button('Auto'):
            in_black, in_white = tone.auto_levels(picture)
            set_(in_black=in_black, in_white=in_white)
        imgui.end_disabled()
        imgui.pop_id()

    def _histogram(self, picture, width: float) -> None:
        pos, height = imgui.get_cursor_screen_pos(), em_size(HIST_HEIGHT)
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(pos, imgui.ImVec2(pos.x + width, pos.y + height), grey(0.1))
        if picture is not None:
            if self._hist[0] is not picture:
                self._hist = (picture, tone.histogram(picture))
            counts = self._hist[1]
            top = max(counts[1:255].max(), 1)   # the end bins are often clipped spikes; they may overflow
            bin_w = width / 256
            for i, n in enumerate(counts):
                if n:
                    h = min(n / top, 1.0) * height
                    draw.add_rect_filled(imgui.ImVec2(pos.x + i * bin_w, pos.y + height - h),
                                         imgui.ImVec2(pos.x + (i + 1) * bin_w, pos.y + height), grey(0.75))
        imgui.dummy(imgui.ImVec2(width, height))

    def _strip(self, id: str, width: float, handles: dict):
        """Gradient strip with triangle handles under it; the handle the mouse drags and its new value (0..255),
        or None. handles: name -> (value, fill grey)."""
        pos, strip, r = imgui.get_cursor_screen_pos(), em_size(STRIP_HEIGHT), em_size(HANDLE)
        pad = r   # handles at 0 and 255 must not stick out of the column
        x_of = lambda v: pos.x + pad + v / 255 * (width - 2 * pad)
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled_multi_color(imgui.ImVec2(pos.x + pad, pos.y), imgui.ImVec2(pos.x + width - pad, pos.y + strip),
                                         grey(0), grey(1), grey(1), grey(0))
        for value, fill in handles.values():
            x, y = x_of(value), pos.y + strip
            points = imgui.ImVec2(x, y), imgui.ImVec2(x + r, y + 2 * r), imgui.ImVec2(x - r, y + 2 * r)
            draw.add_triangle_filled(*points, grey(fill))
            draw.add_triangle(*points, grey(0.5), 1.0)
        imgui.invisible_button(f'##{id}', imgui.ImVec2(width, strip + 2 * r))
        mouse = imgui.get_io().mouse_pos.x
        value = min(max((mouse - pos.x - pad) / (width - 2 * pad) * 255, 0), 255)
        if imgui.is_item_activated():
            self._drag = min(handles, key=lambda n: abs(x_of(handles[n][0]) - mouse))
        if not imgui.is_item_active() or self._drag not in handles:
            return None
        return self._drag, value

    def _numbers(self, specs, width: float, set_) -> None:
        """One field per value, spread over the width: left, (centre,) right."""
        field = em_size(NUMBER_WIDTH)
        start = imgui.get_cursor_pos_x()
        for i, (name, value, lo, hi) in enumerate(specs):
            if i:
                imgui.same_line(start + (width - field) * i / (len(specs) - 1))
            imgui.set_next_item_width(field)
            if isinstance(value, float):
                changed, new = imgui.input_float(f'##{name}', value, format='%.2f')
            else:
                changed, new = imgui.input_int(f'##{name}', value, step=0)
            if changed:
                set_(**{name: min(max(new, lo), hi)})
