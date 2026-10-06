"""The Halftoner block's pattern picker, for Ordered's matrix and Error diffusion's kernel: a combo in sections
(matrices.GROUPS, kernels.GROUPS), each entry after a grey ramp dithered by it, and a Gallery button beside it opening
a modal of the current conversion halftoned by every pattern of a section, a tab per section; a click picks one.

A gallery image is Halftone's result with another pattern: the pairs selected with the current one kept, Halftone's
Dithering, Checker and chroma as set, no Optimise. Ordered takes a millisecond an image, error diffusion about a
second, so a worker thread fills the shown tab in while it is open."""
import threading
from dataclasses import replace
from functools import lru_cache

import numpy as np
from imgui_bundle import em_size, hello_imgui, imgui, immvision

from mokit.ui import style, widgets
from mokit.ui.style import Palette

from .. import ops
from ..converter.dither import ErrorDiffusion, Ordered
from ..halftoning.error_distribution.kernels import GROUPS as KERNEL_GROUPS
from ..halftoning.ordered.matrices import GROUPS as MATRIX_GROUPS

PICKERS = {Ordered.label: ('matrix', MATRIX_GROUPS), ErrorDiffusion.label: ('kernel', KERNEL_GROUPS)}   # halftoner -> its param, sections
EMPTY = {'Custom': 'Patterns drawn in the pattern editor, to come'}   # what an empty section says
ZOOM = 2        # gallery image px per screen px on a 96 dpi display
RAMP = 4        # icon width / height


def halftoner(label: str, name: str, origin=(0, 0)):
    """The halftoner `label` with its pattern param set to `name`."""
    return ops.HALFTONERS[label](**{PICKERS[label][0]: name}, origin=origin)


def ramp(label: str, name: str, height: int) -> np.ndarray:
    """(height, RAMP * height) bool: a left-to-right ramp from paper to ink dithered by the pattern, ink True."""
    levels = np.tile(np.linspace(0, 1, RAMP * height, dtype=np.float32), (height, 1))
    return halftoner(label, name).threshold(levels)


def halftoned(conv, ditherer) -> np.ndarray:
    """Halftone's result (ops.halftone) with another halftoner: conv's pairs, Dithering, Checker and weights."""
    c = conv.copy(ditherer=ditherer)
    c.halftone()
    return c.dithered_result


class Thumbnails:
    """The gallery's images, halftoned on a worker thread one at a time, in the order last asked for. A new source (a
    finished conversion, another halftoner or origin) drops them all."""

    def __init__(self) -> None:
        self.images = {}       # name -> (H, W, 3) uint8
        self._source = None    # (the halftone Converter, halftoner label, origin) the images are of
        self._wanted = []      # names to halftone, the next first
        self._busy = None      # the name the worker is on
        self._lock = threading.Condition()
        threading.Thread(target=self._work, daemon=True).start()

    def want(self, conv, label: str, origin, names) -> None:
        """Halftone the names of these not done yet, before any asked for earlier; the others asked for are dropped."""
        with self._lock:
            source = self._source
            if source is None or source[0] is not conv or source[1:] != (label, origin):
                self._source, self.images = (conv, label, origin), {}
            self._wanted = [n for n in names if n not in self.images and n != self._busy]
            self._lock.notify()

    def stop(self) -> None:
        """Nothing more to halftone; the images done stay."""
        with self._lock:
            self._wanted = []

    @property
    def pending(self) -> bool:
        return bool(self._wanted) or self._busy is not None

    def _work(self) -> None:
        while True:
            with self._lock:
                while not self._wanted:
                    self._lock.wait()
                name = self._busy = self._wanted.pop(0)
                source = self._source
            conv, label, origin = source
            try:
                image = (np.clip(halftoned(conv, halftoner(label, name, origin)), 0, 1) * 255).round().astype(np.uint8)
            except Exception:   # a pattern that fails shows as missing, it must not stop the worker
                image = None
            with self._lock:
                self._busy = None
                if self._source is source and image is not None:
                    self.images[name] = image


class Textures:
    """GL textures of images, made on the UI thread when first drawn, each pixel uploaded as a block of the screen's
    physical pixels it covers, so it stays sharp at any display scale."""

    def __init__(self) -> None:
        self._made = {}   # key -> (the image, its GlTexture)

    def ref(self, key, image: np.ndarray, scale: int) -> imgui.ImTextureRef:
        hit = self._made.get(key)
        if hit is None or hit[0] is not image:
            k = max(1, round(scale * imgui.get_io().display_framebuffer_scale.x))
            pixels = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2)
            pixels = np.ascontiguousarray(np.repeat(np.repeat(pixels, k, axis=0), k, axis=1))
            hit = self._made[key] = (image, immvision.GlTexture(pixels, False))
        return imgui.ImTextureRef(hit[1].texture_id)

    def keep(self, keys) -> None:
        """Free the textures of all but these keys."""
        if set(self._made) - set(keys):
            self._made = {k: v for k, v in self._made.items() if k in keys}


@lru_cache(maxsize=None)
def ramp_pixels(label: str, name: str, height: int) -> np.ndarray:
    """ramp as (height, width) uint8, ink white and paper black as the Bitmap view shows them."""
    return ramp(label, name, height).astype(np.uint8) * 255


def section_of(groups: dict, name: str):
    return next((g for g, names in groups.items() if name in names), next(iter(groups)))


class PatternPicker:
    """The combo and the gallery for the halftoner's pattern param; draw it inside the block's id."""

    def __init__(self, app) -> None:
        self.app = app
        self.thumbs = Thumbnails()
        self.icons = Textures()    # (halftoner, pattern) -> its ramp
        self.images = Textures()   # pattern -> its gallery image, the shown tab's only
        self.tab = None        # the gallery's section shown; None: the current pattern's, on the next opening
        self._open = False     # the Gallery button was pressed: the modal opens this frame
        self.visible = False   # the modal was drawn last frame

    def draw(self, params, on_change) -> None:
        label = params.halftoner
        field, groups = PICKERS[label]
        current = getattr(params, field)
        pick = lambda name: on_change(replace(params, **{field: name}))
        px = max(1, round(hello_imgui.dpi_window_size_factor()))   # icon px per screen px, as a 96 dpi screen shows it
        h = max(4, int(imgui.get_text_line_height() / px))
        icon = imgui.ImVec2(RAMP * h * px, h * px)
        style_ = imgui.get_style()
        spacing, pad = style_.item_spacing.x, style_.frame_padding.x

        # wide enough for the icon and the longest name, as far as the row leaves room for the label and the button
        longest = max(imgui.calc_text_size(n).x for names in groups.values() for n in names)
        needed = icon.x + spacing + longest + 2 * pad + imgui.get_frame_height()
        room = (imgui.get_content_region_avail().x - imgui.calc_text_size(field).x
                - imgui.calc_text_size('Gallery…').x - 2 * pad - 2 * spacing)
        imgui.set_next_item_width(max(em_size(style.FIELD_WIDTH), min(needed, room)))
        flags = imgui.ComboFlags_.height_largest.value | imgui.internal.ComboFlagsPrivate_.custom_preview.value
        if imgui.begin_combo(field, '', flags):   # a custom preview takes no text
            for group, names in groups.items():
                imgui.separator_text(group)
                if not names:
                    imgui.text_disabled(EMPTY.get(group, 'None'))
                for name in names:
                    x = imgui.get_cursor_pos_x()
                    if imgui.selectable(f'##{name}', name == current, 0, imgui.ImVec2(0, icon.y))[0]:
                        pick(name)
                    if name == current and imgui.is_window_appearing():
                        imgui.set_scroll_here_y()
                    imgui.same_line(x)
                    imgui.image(self.icons.ref((label, name), ramp_pixels(label, name, h), px), icon)
                    imgui.same_line()
                    imgui.text(name)
            imgui.end_combo()
        if imgui.internal.begin_combo_preview():   # the current one's icon and name in the closed combo
            at = imgui.get_cursor_screen_pos()   # same_line does not lay out in a combo preview: placed by hand
            imgui.get_window_draw_list().add_image(self.icons.ref((label, current), ramp_pixels(label, current, h), px),
                                                   at, imgui.ImVec2(at.x + icon.x, at.y + icon.y))
            imgui.set_cursor_screen_pos(imgui.ImVec2(at.x + icon.x + spacing, at.y))
            imgui.text(current)
            imgui.internal.end_combo_preview()
        imgui.same_line()
        if imgui.button('Gallery…'):
            self._open, self.tab = True, None
        imgui.set_item_tooltip(f'Every {field} of a section on the conversion, to pick one')
        self._gallery(label, field, groups, current, pick, params)

    def _gallery(self, label, field, groups, current, pick, params) -> None:
        if self._open:
            imgui.open_popup('gallery')
            self._open = False
            viewport = imgui.get_main_viewport()
            imgui.set_next_window_size(imgui.ImVec2(viewport.work_size.x * 0.9, viewport.work_size.y * 0.9))
            imgui.set_next_window_pos(viewport.get_center(), imgui.Cond_.always.value, imgui.ImVec2(0.5, 0.5))
        self.visible = imgui.begin_popup_modal(f'{label} gallery###gallery', True,
                                               imgui.WindowFlags_.no_saved_settings.value)[0]
        if not self.visible:
            self.thumbs.stop()
            self.images.keep(())
            return
        if imgui.is_key_pressed(imgui.Key.escape):
            imgui.close_current_popup()
        conv = self.app.shown('halftone')
        if self.tab is None:
            self.tab = section_of(groups, current)
            select = self.tab
        else:
            select = None
        if imgui.begin_tab_bar('sections'):
            for group in groups:
                flags = imgui.TabItemFlags_.set_selected.value if group == select else 0
                if imgui.begin_tab_item(group, None, flags)[0]:
                    self.tab = group
                    imgui.end_tab_item()
            imgui.end_tab_bar()
        names = groups[self.tab]
        origin = (params.noise_y, params.noise_x)
        footer = imgui.get_frame_height_with_spacing()
        imgui.begin_child('images', imgui.ImVec2(0, -footer))
        if conv is None:
            widgets.hint('The gallery shows once the conversion has run through Halftone')
        elif not names:
            widgets.hint(EMPTY.get(self.tab, 'None'))
        else:
            self.thumbs.want(conv, label, origin, names)
            picked = self._grid(names, current, conv.dithered_result.shape[:2])
            if picked is not None:
                pick(picked)
                imgui.close_current_popup()
        imgui.end_child()
        if imgui.button('Close'):
            imgui.close_current_popup()
        if self.thumbs.pending:
            imgui.same_line()
            imgui.text_disabled('Halftoning…')
        imgui.end_popup()

    def _grid(self, names, current, shape):
        """The images in rows as many as fit, each with its name under it; the name clicked, else None."""
        zoom = max(1, round(ZOOM * hello_imgui.dpi_window_size_factor()))   # whole screen px per image px
        size = imgui.ImVec2(shape[1] * zoom, shape[0] * zoom)
        style_ = imgui.get_style()
        pad, gap = style_.frame_padding, style_.item_spacing
        cell = size.x + 2 * pad.x
        per_row = max(1, int((imgui.get_content_region_avail().x + gap.x) // (cell + gap.x)))
        picked, shown = None, set()
        for i, name in enumerate(names):
            if i % per_row:
                imgui.same_line()
            imgui.begin_group()
            image = self.thumbs.images.get(name)
            if image is not None and image.shape[:2] == tuple(shape):
                shown.add(name)
                if imgui.image_button(name, self.images.ref(name, image, zoom), size):
                    picked = name
            else:   # not halftoned yet: a button of the same size, so the grid does not jump
                imgui.button(f'…##{name}', imgui.ImVec2(cell, size.y + 2 * pad.y))
            if name == current:
                lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                imgui.get_window_draw_list().add_rect(lo, hi, style.u32(Palette.hovered), 0, 3)
            imgui.text(name)
            imgui.end_group()
        self.images.keep(shown)
        return picked
