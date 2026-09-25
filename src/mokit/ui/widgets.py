"""
Reusable components: small functions that draw one thing and report what the user did.

Vocabulary (use these instead of raw `imgui.text` / `text_disabled` / `set_next_item_width` in the tab and panel
files; containers such as rows, panes and splitters live in layout.py):
  section(title)         gap + header: one per group of related controls in a tab
  header(title)          titled separator without the gap (top of a panel or child)
  gap()                  vertical breathing room between groups
  hint(text)             muted, wrapped helper text under a control
  input_int / input_text / combo / path_input   width-managed inputs that call `on_change(value)`
  toggle_button(label, active) -> clicked
"""
import subprocess
import sys
from contextlib import contextmanager
from typing import Callable, Iterable

from imgui_bundle import imgui, immvision, em_size, em_to_vec2
from imgui_bundle import portable_file_dialogs as pfd

from . import style
from .style import Palette


# ----- structure -----------------------------------------------------------------------------------

def section(title: str) -> None:
    gap()
    header(title)


def header(title: str) -> None:
    imgui.separator_text(title)


def gap() -> None:
    imgui.dummy(em_to_vec2(0, style.GAP))


def hint(text: str, wrap: bool = True) -> None:
    with style.muted():
        (imgui.text_wrapped if wrap else imgui.text)(text)


def fits_on_line(width: float) -> bool:
    """True when an item `width` px wide fits to the right of the last item (flow layout: then same_line())."""
    right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
    return imgui.get_item_rect_max().x + imgui.get_style().item_spacing.x + width <= right


def checkbox_width(label: str) -> float:
    return imgui.get_frame_height() + imgui.get_style().item_inner_spacing.x + imgui.calc_text_size(label).x


def status_dot(col: imgui.ImVec4) -> None:
    """Filled circle the height of a text line (the default font has no ● glyph)."""
    pos, h = imgui.get_cursor_screen_pos(), imgui.get_text_line_height()
    r, (dx, dy) = em_size(style.STATUS_DOT["radius"]), (em_size(v) for v in style.STATUS_DOT["offset"])
    imgui.get_window_draw_list().add_circle_filled(imgui.ImVec2(pos.x + r + dx, pos.y + h / 2 + dy), r, style.u32(col))
    imgui.dummy(imgui.ImVec2(2 * r, h))


# ----- inputs --------------------------------------------------------------------------------------

def input_int(label: str, value: int, on_change: Callable[[int], None], width_em: float = style.FIELD_WIDTH,
              step: int = 0) -> None:
    """With a step, -/+ buttons; Ctrl-click on them steps 10 times as far."""
    imgui.set_next_item_width(em_size(width_em))
    changed, new = imgui.input_int(label, value, step=step, step_fast=10 * step)
    if changed:
        on_change(new)


def input_text(label: str, value: str, on_change: Callable[[str], None], width_em: float = style.FIELD_WIDTH) -> None:
    imgui.set_next_item_width(em_size(width_em))
    changed, new = imgui.input_text(label, value)
    if changed:
        on_change(new)


def combo(label: str, current: str, choices: Iterable[str], on_change: Callable[[str], None],
          width_em: float = style.FIELD_WIDTH) -> None:
    imgui.set_next_item_width(em_size(width_em))
    if imgui.begin_combo(label, current, imgui.ComboFlags_.height_largest):   # the popup shows as many rows as fit the screen
        for key in choices:
            if imgui.selectable(key, key == current)[0]:
                on_change(key)
        imgui.end_combo()


def toggle_button(label: str, active: bool) -> bool:
    """Small button drawn as selected when `active`; returns True when clicked."""
    if not active:
        return imgui.small_button(label)
    with style.selected():
        return imgui.small_button(label)


def delta_cell(value, prev_value) -> None:
    """Δ column of the results table: green when smaller, red when larger, '—' otherwise."""
    delta = None if prev_value is None else value - prev_value
    if not delta:
        imgui.text("—")
    elif delta < 0:
        imgui.text_colored(Palette.ok, str(delta))
    else:
        imgui.text_colored(Palette.error, f"+{delta}")


# ----- file pickers --------------------------------------------------------------------------------

def native_pick(kind: str, title: str, default: str) -> str:
    """Native folder/file picker; returns "" when cancelled. kind: "folder" | "file"."""
    if sys.platform == "darwin":
        # pfd runs osascript in a background process whose dialog does not get focus; run it
        # ourselves with an `activate` first so the picker comes up in front.
        script = (f'activate\nPOSIX path of (choose {kind} with prompt "{title}" '
                  f'default location POSIX file "{default}")')
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return result.stdout.strip().rstrip("/") if result.returncode == 0 else ""
    if kind == "folder":
        return pfd.select_folder(title, default).result()
    chosen = pfd.open_file(title, default).result()
    return chosen[0] if chosen else ""


def browse(kind: str, title: str, default: str) -> str:
    """Browse button after the previous widget; returns the pick or "" (cancelled / not clicked)."""
    imgui.same_line()
    if imgui.button(f"Browse...##{title}"):
        return native_pick(kind, title, default)
    return ""


def path_input(id: str, value: str, on_change: Callable[[str], None], kind: str, title: str, default: str,
               width_em: float = 0) -> bool:
    """
    Text input plus a Browse button on the same line. `width_em=0` fills the line. `on_change` gets the
    typed or picked path; returns True when a path was picked with the button.
    """
    button = imgui.calc_text_size("Browse...").x + 2 * imgui.get_style().frame_padding.x
    imgui.set_next_item_width(em_size(width_em) if width_em else -(button + imgui.get_style().item_spacing.x))
    changed, new = imgui.input_text(id, value)
    if changed:
        on_change(new)
    if picked := browse(kind, title, default):
        on_change(picked)
        return True
    return False


# ----- images --------------------------------------------------------------------------------------

def image_params(cache: dict, key: str, size, image_size=None) -> immvision.ImageParams:
    """Per-image immvision params (nearest-neighbour, display only), kept in `cache` across frames. With the image's
    (w, h) the view is pinned to the whole image every frame: immvision can keep a stale zoom after the image
    or display size changes, and a display-only image has no zoom of its own to lose."""
    params = cache.get(key)
    if params is None:
        params = cache[key] = immvision.factor_image_params_display_only()
        params.interpolation_mode = immvision.ImageInterpolationMode.nearest
        params.refresh_image = True
    params.image_display_size = size
    if image_size is not None:
        params.zoom_pan_matrix = immvision.make_zoom_pan_matrix_full_view(image_size, size)
    return params
