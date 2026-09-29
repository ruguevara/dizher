"""Photoshop-style editor of the tone node (ops.levels): Levels or Curves, a channel selector (RGB, the composite, or R,
G, B), and three eyedroppers whose targets are palette colours.

Levels: the channel's input histogram with the black, midtone and white handles under it, the output range strip with
its two handles, the numbers and Auto (Reset is on the block header). The midtone handle sits where the curve crosses
mid grey, so moving the black or white point carries it along at the same gamma, as in Photoshop. Curves: the curve
over the channel's histogram; a click adds a point, a drag moves one between its neighbours, dragging it off the graph
or a right click removes it. A channel's histogram is of its input: the picture after the composite.

The eyedroppers: a toggle each (Esc or a second click disarms; arming ends Paint mode), then a left click in the preview
samples the node's input there (Window._eyedrop) and the node maps it onto the target: black and white set each
channel's end points, grey its gamma (Levels) or adds a point group (Curves), listed with its sample and target and a
delete button. The swatch before an eyedropper picks its target from the mode's palette, or Auto: the palette's black,
grey (the grey nearest mid lightness) or white, following the mode."""
import math
from dataclasses import replace

import numpy as np
from imgui_bundle import em_size, imgui
from imgui_bundle import icons_fontawesome_6

from mokit.ui import style, widgets
from mokit.ui.style import Palette

from .. import tone

HIST_HEIGHT = 5.0     # em
STRIP_HEIGHT = 0.7    # em, the gradient under the histogram
HANDLE = 0.45         # em, half the width of a handle triangle, and of a curve point
NUMBER_WIDTH = 3.5    # em
EYEDROPPER = icons_fontawesome_6.ICON_FA_EYE_DROPPER
MIN_GAP = tone.MIN_GAP
OFF = 1.5             # em a dragged curve point goes beyond the graph to be removed
CHANNELS = ('RGB', 'R', 'G', 'B')
INK = (imgui.ImVec4(0.85, 0.85, 0.85, 1), imgui.ImVec4(1, 0.35, 0.35, 1), imgui.ImVec4(0.35, 0.9, 0.35, 1),
       imgui.ImVec4(0.4, 0.55, 1, 1))   # each channel's curve
LEVELS = ('in_black', 'in_white', 'gamma', 'out_black', 'out_white')


def midtone(b: float, w: float, gamma: float) -> float:
    """Input value that the curve maps to mid grey."""
    return b + (w - b) * 0.5 ** gamma


def gamma_at(x: float, b: float, w: float) -> float:
    t = min(max((x - b) / (w - b), 1e-3), 1 - 1e-3)
    return round(min(max(math.log(t) / math.log(0.5), 0.1), 9.99), 2)


def grey(v: float, a: float = 1.0) -> int:
    return style.u32(imgui.ImVec4(v, v, v, a))


def composite(params) -> tuple:
    return tuple(getattr(params, n) for n in LEVELS)


def swatch(id: str, rgb, tip: str = '', size: float = 1.0) -> bool:
    side = imgui.get_frame_height() * size
    clicked = imgui.color_button(id, imgui.ImVec4(*map(float, np.clip(rgb, 0, 1)), 1.0),
                                 imgui.ColorEditFlags_.no_tooltip.value, imgui.ImVec2(side, side))
    if tip:
        imgui.set_item_tooltip(tip)
    return clicked


def rgb255(rgb) -> str:
    return '(%d, %d, %d)' % tuple(np.round(np.asarray(rgb) * 255))


PICKERS = {'black': 'Black point eyedropper', 'grey': 'Mid point eyedropper', 'white': 'White point eyedropper'}


def pick_label(role: str) -> str:
    """An eyedropper's toggle: the icon, the role in the id (the target swatch before it shows the colour)."""
    return f'{EYEDROPPER}##pick {role}'


def flow(label: str, first: bool = False) -> None:
    """Before a small button: on the same line when it fits in the block, else on the next (a narrow dock)."""
    if not first and widgets.fits_on_line(imgui.calc_text_size(label.split('##')[0]).x + 2 * imgui.get_style().frame_padding.x):
        imgui.same_line(0, 0)


class LevelsEditor:
    def __init__(self, palette=None, grid=None) -> None:
        """palette(): the target mode's palette, the eyedroppers' targets; grid: window.palette_grid."""
        self.palette, self.grid = palette, grid
        self.channel = 0           # the one shown: 0 the composite, 1..3 R, G, B
        self.armed = None          # the eyedropper ('black', 'grey', 'white') a click in the preview goes to
        self.last = {}             # (mode, role) -> the colour it last sampled, 0..1: where it landed is shown
        self._drag = None          # levels handle name while the mouse holds one
        self._point = None         # a dragged curve point: (channel, curves and picks before the drag, its input)
        self._spot = None          # where the dragged point is now, None while it is off the graph
        self._hist = (None, {})    # (input array, {(channel, composite): histogram}): recomputed on a new input
        self._roles = (None, None)   # (palette, its palette_roles)
        self._ask = False          # the Levels -> Curves question is to open

    # ----- targets and picks, also for the window --------------------------------------------------------

    def target(self, params, role: str):
        """(palette index, rgb 0..1, auto) of an eyedropper's target."""
        palette = self.palette()
        if self._roles[0] is not palette:
            self._roles = (palette, tone.palette_roles(palette.as_float()))
        k = tone.ROLES.index(role)
        i = params.targets[k]
        auto = not 0 <= i < len(palette)
        i = self._roles[1][k] if auto else i
        return i, palette.as_float()[i], auto

    def pick(self, params, picture, y: int, x: int, on_change) -> None:
        """The armed eyedropper's click at pixel (y, x) of picture, the node's input."""
        role, s = self.armed, tone.sample(picture, y, x)
        i, t, _ = self.target(params, role)
        self.last[params.mode, role] = s
        if params.mode == 'Levels':
            on_change(replace(params, channels=tone.levels_pick(composite(params), params.channels, role, s, t)))
        elif role == 'grey':
            curves, picks = tone.curves_grey(params.curves, params.picks, s, t, i)
            on_change(replace(params, curves=curves, picks=picks))
        else:
            curves, picks = tone.curves_end(params.curves, params.picks, role, s, t)
            on_change(replace(params, curves=curves, picks=picks))

    def result(self, params, rgb) -> np.ndarray:
        """A colour (0..1) through the node as it is set."""
        rgb = np.asarray(rgb, np.float32)[None, None]
        if params.mode == 'Curves':
            return tone.curves(rgb, params.curves)[0, 0]
        return tone.all_levels(rgb, composite(params), params.channels)[0, 0]

    # ----- drawing ---------------------------------------------------------------------------------

    def draw(self, params, picture, on_change, id: str) -> None:
        width = imgui.get_content_region_avail().x
        imgui.push_id(id)
        self._mode(params, on_change)
        for c, name in enumerate(CHANNELS):
            flow(name)
            if widgets.toggle_button(f'{name}##channel', self.channel == c):
                self.channel = c
            imgui.set_item_tooltip('The composite, the channels after it' if not c else f'The {name} channel, after the composite')
        if params.mode == 'Levels':
            self._levels(params, picture, on_change, width)
        else:
            self._curves(params, picture, on_change, width)
        self._eyedroppers(params, on_change)
        self._results(params, on_change)
        imgui.pop_id()

    def _mode(self, params, on_change) -> None:
        """Levels | Curves. To Curves the levels come along as curves, unless the curves are already edited: then it
        asks."""
        for m in ('Levels', 'Curves'):
            flow(m, m == 'Levels')
            if widgets.toggle_button(m, params.mode == m) and params.mode != m:
                neutral = composite(params) == tone.NEUTRAL and all(tuple(c) == tone.NEUTRAL for c in params.channels)
                edited = params.picks or any(tuple(c) != tone.IDENTITY for c in params.curves)
                if m == 'Levels' or neutral:
                    on_change(replace(params, mode=m))
                elif not edited:
                    on_change(self._converted(params))
                elif params.curves != tone.levels_to_curves(composite(params), params.channels):
                    self._ask = True
                else:
                    on_change(replace(params, mode=m))
            imgui.set_item_tooltip('Levels per channel; its values stay when Curves is on' if m == 'Levels' else
                                   'Curves per channel; switching from Levels brings the levels along as curves')
        if self._ask:
            imgui.open_popup('curves from levels')
            self._ask = False
        if imgui.begin_popup_modal('curves from levels', None, imgui.WindowFlags_.always_auto_resize.value)[0]:
            imgui.text('The curves are edited. Replace them with the levels?')
            if imgui.button('Replace with levels'):
                on_change(self._converted(params))
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Keep the curves'):
                on_change(replace(params, mode='Curves'))
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel') or imgui.is_key_pressed(imgui.Key.escape):
                imgui.close_current_popup()
            imgui.end_popup()

    @staticmethod
    def _converted(params):
        return replace(params, mode='Curves', curves=tone.levels_to_curves(composite(params), params.channels), picks=())

    def _input(self, params, picture):
        """What the shown channel's handles act on: the picture, or one channel of it after the composite."""
        if picture is None or not self.channel:
            return picture
        if params.mode == 'Curves':
            return tone.apply_lut(picture[..., self.channel - 1], tone.curve_lut(params.curves[0]))
        return tone.levels(picture[..., self.channel - 1], *composite(params))

    def _histogram_of(self, params, picture):
        if picture is None:
            return None
        if self._hist[0] is not picture:
            self._hist = (picture, {})
        key = (self.channel, params.mode, params.curves[0] if params.mode == 'Curves' else composite(params))
        if key not in self._hist[1]:
            self._hist[1][key] = tone.histogram(self._input(params, picture))
        return self._hist[1][key]

    # ----- Levels ----------------------------------------------------------------------------------

    def _levels(self, params, picture, on_change, width: float) -> None:
        c = self.channel
        values = dict(zip(LEVELS, composite(params) if not c else params.channels[c - 1]))

        def set_(**kw):
            if not c:
                return on_change(replace(params, **kw))
            channels = list(params.channels)
            channels[c - 1] = tuple(float(kw.get(n, values[n])) for n in LEVELS)
            on_change(replace(params, channels=tuple(channels)))

        self._histogram(self._histogram_of(params, picture), width, INK[c])
        b, w, g = values['in_black'], values['in_white'], values['gamma']
        moved = self._strip('in', width, {'in_black': (b, 0.0), 'midtone': (midtone(b, w, g), 0.5), 'in_white': (w, 1.0)})
        if moved:
            name, x = moved
            if name == 'in_black':
                set_(in_black=min(round(x), w - MIN_GAP))
            elif name == 'in_white':
                set_(in_white=max(round(x), b + MIN_GAP))
            else:
                set_(gamma=gamma_at(x, b, w))
        fmt = '%.1f' if c else None
        self._numbers((('in_black', b, 0, w - MIN_GAP, fmt), ('gamma', g, 0.1, 9.99, '%.2f'),
                       ('in_white', w, b + MIN_GAP, 255, fmt)), width, set_)
        imgui.text('Output levels')
        ob, ow = values['out_black'], values['out_white']
        moved = self._strip('out', width, {'out_black': (ob, 0.0), 'out_white': (ow, 1.0)})
        if moved:
            set_(**{moved[0]: round(moved[1])})
        self._numbers((('out_black', ob, 0, 255, fmt), ('out_white', ow, 0, 255, fmt)), width, set_)
        imgui.begin_disabled(picture is None)
        if imgui.button('Auto'):
            in_black, in_white = tone.auto_levels(self._input(params, picture))
            set_(in_black=in_black, in_white=in_white)
        imgui.end_disabled()
        imgui.set_item_tooltip("Black and white points clipping 0.1% of the shown channel's values at each end")

    def _histogram(self, counts, width: float, ink=None) -> None:
        pos, height, pad = imgui.get_cursor_screen_pos(), em_size(HIST_HEIGHT), em_size(HANDLE)
        # bin v centred over the handle at v in _strip, so a handle can be set under a chosen bar
        bin_w = (width - 2 * pad) / 255
        x_of = lambda v: pos.x + pad + v * bin_w
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(imgui.ImVec2(x_of(-0.5), pos.y), imgui.ImVec2(x_of(255.5), pos.y + height), grey(0.1))
        if counts is not None:
            top = max(counts[1:255].max(), 1)   # the end bins are often clipped spikes; they may overflow
            col = style.u32(imgui.ImVec4(ink.x, ink.y, ink.z, 0.8)) if ink is not None and self.channel else grey(0.75)
            for i, n in enumerate(counts):
                if n:
                    h = min(n / top, 1.0) * height
                    draw.add_rect_filled(imgui.ImVec2(x_of(i - 0.5), pos.y + height - h),
                                         imgui.ImVec2(x_of(i + 0.5), pos.y + height), col)
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
        """One field per value, spread over the width: left, (centre,) right. A spec's format None is a whole number."""
        field = em_size(NUMBER_WIDTH)
        start = imgui.get_cursor_pos_x()
        for i, (name, value, lo, hi, fmt) in enumerate(specs):
            if i:
                imgui.same_line(start + (width - field) * i / (len(specs) - 1))
            imgui.set_next_item_width(field)
            if fmt:
                changed, new = imgui.input_float(f'##{name}', float(value), format=fmt)
            else:
                changed, new = imgui.input_int(f'##{name}', value, step=0)
            if changed:
                set_(**{name: min(max(new, lo), hi)})

    # ----- Curves ----------------------------------------------------------------------------------

    def _curves(self, params, picture, on_change, width: float) -> None:
        """The graph, width square: the histogram, quarter grid and diagonal, the other channels' curves faint on the
        composite, the shown curve and its points."""
        c, pad = self.channel, em_size(HANDLE)
        pos = imgui.get_cursor_screen_pos()
        side = width - 2 * pad
        X = lambda v: pos.x + pad + v / 255 * side
        Y = lambda v: pos.y + pad + (1 - v / 255) * side
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(imgui.ImVec2(X(0), Y(255)), imgui.ImVec2(X(255), Y(0)), grey(0.1))
        counts = self._histogram_of(params, picture)
        if counts is not None:
            top = max(counts[1:255].max(), 1)
            for i, n in enumerate(counts):
                if n:
                    draw.add_rect_filled(imgui.ImVec2(X(i - 0.5), Y(0) - min(n / top, 1.0) * side * 0.6),
                                         imgui.ImVec2(X(i + 0.5), Y(0)), grey(0.3))
        for q in (64, 128, 192):
            draw.add_line(imgui.ImVec2(X(q), Y(0)), imgui.ImVec2(X(q), Y(255)), grey(0.3, 0.6))
            draw.add_line(imgui.ImVec2(X(0), Y(q)), imgui.ImVec2(X(255), Y(q)), grey(0.3, 0.6))
        draw.add_line(imgui.ImVec2(X(0), Y(0)), imgui.ImVec2(X(255), Y(255)), grey(0.4, 0.8))
        shown = [(k, 0.5) for k in (1, 2, 3) if not c and tuple(params.curves[k]) != tone.IDENTITY] + [(c, 1.0)]
        for k, alpha in shown:
            lut = tone.curve_lut(params.curves[k]) * 255
            col = style.u32(imgui.ImVec4(INK[k].x, INK[k].y, INK[k].z, alpha))
            for i in range(255):
                draw.add_line(imgui.ImVec2(X(i), Y(lut[i])), imgui.ImVec2(X(i + 1), Y(lut[i + 1])), col, 1.5 if alpha == 1 else 1.0)
        pts = tone.points(params.curves[c])
        imgui.invisible_button('##curve', imgui.ImVec2(width, width))
        m = imgui.get_mouse_pos()
        mx, my = (min(max((m.x - X(0)) / side * 255, 0), 255), min(max((Y(0) - m.y) / side * 255, 0), 255))
        near = [j for j, (x, y) in enumerate(pts) if abs(X(x) - m.x) <= 2 * pad and abs(Y(y) - m.y) <= 2 * pad]
        hovered = min(near, key=lambda j: abs(X(pts[j][0]) - m.x)) if near and imgui.is_item_hovered() else None
        if imgui.is_item_activated():
            if hovered is not None:
                self._point, self._spot = (c, params.curves, params.picks, pts[hovered][0]), tuple(pts[hovered])
            elif np.all(np.abs(pts[:, 0] - round(mx)) >= MIN_GAP):   # a new point on the curve, dragged from there
                x = float(round(mx))
                y = float(np.interp(x, np.arange(256), tone.curve_lut(params.curves[c]) * 255))
                curves, picks = tone.curve_edit(params.curves, params.picks, c, -1.0, (x, round(y)))
                self._point, self._spot = (c, curves, picks, x), (x, round(y))
                on_change(replace(params, curves=curves, picks=picks))
        if imgui.is_item_active() and self._point is not None and self._point[0] == c:
            self._move(params, on_change, (m.x < X(0) - em_size(OFF) or m.x > X(255) + em_size(OFF) or
                                           m.y < Y(255) - em_size(OFF) or m.y > Y(0) + em_size(OFF)), mx, my)
        elif not imgui.is_item_active():
            self._point = None
        if hovered is not None and imgui.is_item_clicked(imgui.MouseButton_.right) and len(pts) > 2:
            curves, picks = tone.curve_edit(params.curves, params.picks, c, pts[hovered][0], None)
            on_change(replace(params, curves=curves, picks=picks))
        spot = self._spot if self._point is not None else tuple(pts[hovered]) if hovered is not None else None
        for x, y in tone.points(params.curves[c]):
            a, b = imgui.ImVec2(X(x) - pad * 0.7, Y(y) - pad * 0.7), imgui.ImVec2(X(x) + pad * 0.7, Y(y) + pad * 0.7)
            draw.add_rect_filled(a, b, grey(1.0 if spot is not None and x == spot[0] else 0.1))
            draw.add_rect(a, b, style.u32(INK[c]))
        widgets.hint('Input %.0f  Output %.0f' % spot if spot else
                     'Click adds a point; drag it off or right-click to remove')

    def _move(self, params, on_change, off: bool, mx: float, my: float) -> None:
        """The dragged point to (mx, my) between its neighbours, or removed while off the graph (back on, it returns)."""
        c, curves, picks, x0 = self._point
        pts = tone.points(curves[c])
        j = int(np.flatnonzero(pts[:, 0] == x0)[0])
        if off and len(pts) > 2:
            new, self._spot = tone.curve_edit(curves, picks, c, x0, None), None
        else:
            lo = pts[j - 1, 0] + 1 if j > 0 else 0
            hi = pts[j + 1, 0] - 1 if j + 1 < len(pts) else 255
            self._spot = (float(min(max(round(mx), lo), hi)), float(round(my)))
            new = tone.curve_edit(curves, picks, c, x0, self._spot)
        if new != (params.curves, params.picks):
            on_change(replace(params, curves=new[0], picks=new[1]))

    # ----- eyedroppers -----------------------------------------------------------------------------

    def _eyedroppers(self, params, on_change) -> None:
        """Per role: its target swatch (a click opens the palette) and its toggle."""
        if self.palette is None:
            return
        palette = self.palette()
        inner = imgui.get_style().item_inner_spacing.x
        for k, role in enumerate(tone.ROLES):
            if k and widgets.fits_on_line(imgui.get_frame_height() + inner + imgui.calc_text_size(EYEDROPPER).x
                                          + 2 * imgui.get_style().frame_padding.x):
                imgui.same_line()
            i, rgb, auto = self.target(params, role)
            if swatch(f'##target {role}', rgb, f"Target: {'Auto, ' if auto else ''}{palette.name(i)} {rgb255(rgb)}; "
                                                 f"a click picks another"):
                imgui.open_popup(f'target {role}')
            imgui.same_line(0, inner)
            if widgets.toggle_button(pick_label(role), self.armed == role):
                self.armed = None if self.armed == role else role
            imgui.set_item_tooltip(f'{PICKERS[role]}, to {palette.name(i)}: click the picture where it should be that colour; '
                                   + {'black': 'sets the black point', 'white': 'sets the white point',
                                      'grey': 'sets the gamma' if params.mode == 'Levels' else 'adds a point per channel'}[role]
                                   + '. Esc ends')
            if imgui.begin_popup(f'target {role}'):
                auto_i = tone.palette_roles(palette.as_float())[k]
                if imgui.selectable(f'Auto {role}: {palette.name(auto_i)}', auto)[0]:
                    self._set_target(params, on_change, k, -1)
                if self.grid is not None:
                    def click(j, button):
                        self._set_target(params, on_change, k, j)
                        imgui.close_current_popup()
                    self.grid(palette, click, lambda j: palette.name(j), {} if auto else {i: '✓'})
                imgui.end_popup()
        if self.armed:
            widgets.hint(f'{PICKERS[self.armed]}: click the preview to sample; Esc ends')

    @staticmethod
    def _set_target(params, on_change, k: int, i: int) -> None:
        targets = list(params.targets)
        targets[k] = i
        on_change(replace(params, targets=tuple(targets)))

    def _results(self, params, on_change) -> None:
        """Levels: where the last sample of each eyedropper lands against its target. Curves: the same for black and
        white, and the grey point groups: sample -> target, the landing when it misses, notes, a delete button."""
        rows = [(role, self.last[params.mode, role]) for role in tone.ROLES
                if (params.mode, role) in self.last and not (params.mode == 'Curves' and role == 'grey')]
        for role, s in rows:
            _, t, _ = self.target(params, role)
            self._landing(f'{role}', s, t, self.result(params, s), role.capitalize())
        if params.mode != 'Curves' or not params.picks:
            return
        palette = self.palette() if self.palette is not None else None
        for k, p in enumerate(params.picks):
            s, t = np.asarray(p[:3]) / 255, np.asarray(p[3:6]) / 255
            index = int(p[6])
            name = palette.name(index) if palette is not None and 0 <= index < len(palette) else rgb255(t)
            notes = []
            conflict = tone.pick_conflicts(params.curves, p)
            if conflict:
                notes.append('conflict in ' + ''.join(n for b, n in zip(range(3), 'RGB') if conflict >> b & 1))
            if int(p[tone.PICK_REPLACED]):
                notes.append('replaced ' + ''.join(n for b, n in zip(range(3), 'RGB') if int(p[tone.PICK_REPLACED]) >> b & 1))
            gone = [n for x, n in zip(p[tone.PICK_X], 'RGB') if x < 0]
            if gone:
                notes.append('lost ' + ''.join(gone))
            if imgui.small_button(f'x##drop{k}'):
                curves, picks = tone.curves_drop(params.curves, params.picks, k)
                on_change(replace(params, curves=curves, picks=picks))
            imgui.set_item_tooltip('Remove this grey point from the curves')
            imgui.same_line()
            self._landing(f'pick{k}', s, t, self.result(params, s), name, notes)

    def _landing(self, id: str, s, t, got, label: str, notes=()) -> None:
        """sample -> target swatches, the landing between them when it misses by more than 1/255."""
        swatch(f'##{id} sample', s, f'Sampled {rgb255(s)}', 0.8)
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.text('->')
        miss = np.abs(np.asarray(got) - t).max() > 1 / 255 + 1e-6
        if miss:
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            swatch(f'##{id} got', got, f'Lands on {rgb255(got)}: the target cannot be reached', 0.8)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            imgui.text('/')
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        swatch(f'##{id} target', t, f'Target {rgb255(t)}', 0.8)
        text = ', '.join((label, *notes))
        if widgets.fits_on_line(imgui.calc_text_size(text).x):
            imgui.same_line()
        if miss or notes:
            with style.text_color(Palette.warn):
                imgui.text_wrapped(text)
        else:
            imgui.text_wrapped(text)
