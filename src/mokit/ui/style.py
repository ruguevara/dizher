"""
Look and feel of every mokit-based tool, in one place. An app re-exports this module from its own
`ui/style.py` and adds its layout constants there; designers edit these two files and nothing else.

Three layers, in CSS terms:

- **Style sheet** — `THEME` (a hello_imgui theme) plus `METRICS` (paddings, spacing, rounding, in em).
  `install(runner_params)` applies them (and FONT) once at start-up. Live preview: menu *Settings > Style editor...*,
  then copy the values you like back here.
- **Nodes** — `NODES` / `NODE_COLORS` / `LINKS` / `PORTS` / `NODE_LAYOUT` / `NODE_TITLE` / `STATUS_DOT`: everything the node editor canvas looks like.
- **Palette** — the semantic colours widgets and overlays use (`Palette.ok`, `Palette.error`, ...). Nothing outside
  this file spells out an RGB value.
- **Classes** — context managers for local overrides: `with muted(): imgui.text("hint")`,
  `with selected(): imgui.small_button("x2")`. Add a new one here rather than pushing style vars in a widget file.
"""
from contextlib import contextmanager

from imgui_bundle import imgui, hello_imgui, em_size, em_to_vec2

THEME = hello_imgui.ImGuiTheme_.so_dark_accent_blue
FONT, FONT_SIZE = "fonts/Roboto/Roboto-Regular.ttf", 16.0   # imgui_bundle asset; hello_imgui scales it for DPI
FONT_BOLD = "fonts/Roboto/Roboto-Bold.ttf"
FONTS: dict = {}     # "regular" / "bold" -> ImFont, filled at start-up by `install`; `font("bold")` to push one

# ImGuiStyle metrics, in em (1 em = font height). Applied after the theme, so they win.
METRICS = {
    "window_padding": (0.75, 0.6),
    "frame_padding": (0.5, 0.3),
    "item_spacing": (0.5, 0.4),
    "item_inner_spacing": (0.4, 0.3),
    "cell_padding": (0.4, 0.2),
    "indent_spacing": 1.2,
    "scrollbar_size": 0.7,
    "grab_min_size": 0.7,
    "frame_rounding": 0.2,
    "child_rounding": 0.3,
    "popup_rounding": 0.3,
    "tab_rounding": 0.25,
    "grab_rounding": 0.2,
    "separator_text_padding": (1.0, 0.3),
    "separator_text_border_size": 0.1,
    "child_border_size": 0.08,
    "frame_border_size": 0.0,
}

# Colour overrides on top of THEME (ImGuiCol name -> RGBA). The theme's separators are almost invisible.
COLORS = {
    "separator": (1, 1, 1, 0.18),
}

# Node editor (mokit.ui.nodes). NODES are imgui-node-editor style vars, NODE_COLORS its colours (names as in
# `ed.StyleVar` / `ed.StyleColor`); LINKS the wires, PORTS and NODE_LAYOUT our own port disks and auto-placement. All lengths in em.
NODES = {
    "node_padding": (0.5, 0.35, 0.5, 0.35),   # left, top, right, bottom: title/name margin inside the node
    "node_rounding": 0.6,
    "node_border_width": 0.1,
    "hovered_node_border_width": 0.2,
    "selected_node_border_width": 0.2,
    "pin_rounding": 0.25,
    "pin_border_width": 0.0,
    "link_strength": 6.0,        # bezier tension of links
    "grid_size": (2.0, 2.0),
}
NODE_COLORS = {
    "bg": (0.24, 0.24, 0.27, 0.8),
    "grid": (0.47, 0.47, 0.47, 0.16),
    "node_bg": (0.125, 0.125, 0.125, 0.8),
    "node_border": (1, 1, 1, 0.38),
    "hov_node_border": (0.2, 0.7, 1, 1),
    "sel_node_border": (1, 0.7, 0.2, 1),
    "hov_link_border": (0.2, 0.7, 1, 1),
    "sel_link_border": (1, 0.7, 0.2, 1),
}
LINKS = {"color": (1, 1, 1, 0.8), "thickness": 0.12}
PORTS = {
    "radius": 0.25,       # port disk, centred on the node edge
    "label_gap": 0,     # disk to port name
    "column_gap": 1.5,    # minimum room between the input names and the output name
    "color": (1, 1, 1, 1),
}
NODE_LAYOUT = {"column_gap": 12.5, "row_gap": 4.5, "margin": 1.0}   # auto-placement of new nodes (canvas units at zoom 1)
NODE_TITLE = {
    "font": "bold",          # key of FONTS
    "size": 1.0,             # em
    "color": (1, 1, 1, 1),
    "head_gap": 0.4,         # room between the app's head widget (a status dot) and the title
    "gap_below": 0.0,        # extra room between the title and the node body
}
STATUS_DOT = {"radius": 0.22, "offset": (0.0, 0.0)}   # `widgets.status_dot`: disk radius and nudge from the line centre

# layout constants, in em

# layout constants, in em
GAP = 0.6            # vertical gap between groups (`widgets.gap()`)
FIELD_WIDTH = 10     # numeric inputs and combos
DIALOG_WIDTH = 40    # text inputs in dialogs
PROGRESS_WIDTH = 8   # status-bar progress bar


class Palette:
    """Semantic colours. Text: ok / warn / error / muted. Overlays: in_frame / hovered / attr_mark / grid."""
    ok = imgui.ImVec4(0.4, 0.9, 0.4, 1)
    warn = imgui.ImVec4(1, 0.7, 0.2, 1)
    error = imgui.ImVec4(1, 0.3, 0.3, 1)
    muted = imgui.ImVec4(0.6, 0.6, 0.6, 1)
    in_frame = imgui.ImVec4(0.3, 0.5, 1, 1)      # tile / cell used in the current frame (blue)
    hovered = imgui.ImVec4(1, 0.6, 0.1, 1)       # hovered tile (orange)
    attr_mark = imgui.ImVec4(1, 0.75, 0.2, 1)    # changed-attr droplet (amber)
    grid = imgui.ImVec4(1, 1, 1, 0.25)           # 8x8 cell grid


def u32(col: imgui.ImVec4) -> int:
    """Palette colour as the packed value draw lists take."""
    return imgui.color_convert_float4_to_u32(col)


# ----- style sheet ---------------------------------------------------------------------------------

def install(params: hello_imgui.RunnerParams) -> None:
    """
    Wire the theme into the runner: THEME is the first-run default, then hello_imgui restores the View > Theme
    pick from its ini. Metrics and colour overrides are applied by `sync()` each frame.
    """
    params.imgui_window_params.tweaked_theme.theme = THEME
    params.imgui_window_params.remember_theme = True
    params.callbacks.load_additional_fonts = _load_fonts


def _load_fonts() -> None:
    FONTS["regular"] = hello_imgui.load_font(FONT, FONT_SIZE)
    FONTS["bold"] = hello_imgui.load_font(FONT_BOLD, FONT_SIZE)


@contextmanager
def font(key: str, size_em: float = 1.0):
    """`with font("bold", 1.2): imgui.text(...)`. Unknown key or fonts not loaded yet: the current font."""
    imgui.push_font(FONTS.get(key), em_size(size_em))
    try:
        yield
    finally:
        imgui.pop_font()


def sync() -> None:
    """
    Call once per frame before drawing. hello_imgui re-applies the theme on its first frame and whenever the
    user picks another one in View > Theme, which resets ImGuiStyle; one sentinel compare detects that and
    re-applies METRICS and COLORS.
    """
    style = imgui.get_style()
    if style.separator_text_border_size == em_size(METRICS["separator_text_border_size"]):
        return
    for name, value in METRICS.items():
        setattr(style, name, em_to_vec2(*value) if isinstance(value, tuple) else em_size(value))
    for name, rgba in COLORS.items():
        style.set_color_(getattr(imgui.Col_, name).value, imgui.ImVec4(*rgba))


# ----- classes (local overrides) -------------------------------------------------------------------

@contextmanager
def text_color(col: imgui.ImVec4):
    imgui.push_style_color(imgui.Col_.text.value, col)
    try:
        yield
    finally:
        imgui.pop_style_color()


def muted():
    """Secondary text: hints, stale results, captions."""
    return text_color(Palette.muted)


@contextmanager
def selected():
    """A button drawn in its pressed colour (toggle buttons in the 'on' state)."""
    active = imgui.get_style_color_vec4(imgui.Col_.button_active)
    imgui.push_style_color(imgui.Col_.button.value, active)
    imgui.push_style_color(imgui.Col_.button_hovered.value, active)
    try:
        yield
    finally:
        imgui.pop_style_color(2)


@contextmanager
def bar_color(col: imgui.ImVec4):
    """Progress-bar fill colour (budget gauge over budget → Palette.error)."""
    imgui.push_style_color(imgui.Col_.plot_histogram.value, col)
    try:
        yield
    finally:
        imgui.pop_style_color()
