"""imgui_bundle window over the pipeline host, in AmaZX's layout: the Tune dock on the left and the Convert dock
on the right, each with one collapsible block per stage in pipeline order (mokit's params editor inside, or a
custom one), the Preview dock between them with the tuned image and the conversion, and the History dock under
Convert's with every undo step. A block header shows its
stage's state: plain when done, tinted while it runs or after it failed, muted while waiting to run. hello_imgui's
ini keeps the dock layout, its user prefs which blocks are open and the session: the project folder and the params,
unsaved edits included, restored on the next start when no path is given.

A project is AmaZX's mokit folder: project.json holds every node's params, the source path relative to the folder;
exports default to its build/. Every image has one, as a sidecar: opening an image opens the project beside it named
as the image (app.project_folder), or starts it there. Autosave (on by default) writes it after every edit; off, Save
project does, without asking where."""
import json
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import asdict, field, fields, make_dataclass, replace
from pathlib import Path

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, immvision
from imgui_bundle import portable_file_dialogs as pfd

from mokit import project
from mokit.ui import style, widgets
from mokit.ui.params import params_editor
from mokit.graph import GraphError, Op
from mokit.ui.style import Palette

from .. import __version__, ops
from .app import Pipeline, project_folder
from .levels import LevelsEditor
from . import views

HEADER_TINT = dict(running=Palette.warn, error=Palette.error)   # header background of a running or failed stage
LABELS = {nid: label for nid, label, _, _ in ops.PIPELINE}
VIEWS = {'Screen': 'The conversion as the machine shows it', 'Bitmap': 'Ink pixels white, paper black',
         'Attrs': "Each cell's paper with a disc of its ink"}   # ToolZX's screen views
DEBUG = {   # view -> (tooltip, the stage it needs, its image from that stage's Converter or a running stage's snapshot)
    'Projected': ("What the halftoner aims at: each pixel moved to the nearest mix of its cell's pair",
                  'select', lambda c: c.projected_target()),
    'Unoptimised': ("The halftoner's result, what the optimiser started from",   # a Select pairs snapshot has none yet
                    'halftone', lambda c: c.dithered_result if c.halftoned is None else np.where(c.halftoned[..., None] > 0, c.best_ink, c.best_paper)),
    'Eye': ('Both images as the eye model sees them: what the energy compares',
            'optimise', lambda c: c.eye_view(c.dithered_result)),
    'Error': ('Result minus target through the eye model, around grey: lighter/darker is luma error, '
              'the tint is the colour the result adds', 'optimise', views.error_view),
    'Energy': ("Each cell's energy: red its own error, green the eye-model seams, blue coherence",
               'select', views.energy_view),
    'Seams': ('Cell seams the original has an edge across, bright: there a pair change costs no coherence',
              'select', views.seam_view)}
GRID = imgui.ImVec4(0.5, 0.5, 0.5, 0.6)   # grey reads over black and white alike
INSPECT_CELLS, INSPECT_ZOOM, INSPECT_PAIRS = 3, 10, 8   # the hover tooltip: cells a side, its zoom, pairs listed
UNDO, REDO = imgui.Key.mod_ctrl | imgui.Key.z, imgui.Key.mod_ctrl | imgui.Key.mod_shift | imgui.Key.z   # Cmd on macOS


@lru_cache(maxsize=1024)
def change(before, after) -> str:
    """A history step's label: the params that differ between two graphs, with their new values."""
    def show(v):
        return f'{v:.3g}' if isinstance(v, float) else v.name if isinstance(v, Path) else str(v)
    parts = []
    for nid, node in after.nodes:
        old, new = before[nid].params, node.params
        if old != new:
            diff = [f'{f.name} {show(getattr(new, f.name))}' for f in fields(new) if getattr(new, f.name) != getattr(old, f.name)]
            parts.append(f"{LABELS[nid]}: {', '.join(diff)}")
    return '; '.join(parts)

def cell_icon(cell, size) -> np.ndarray:
    """ToolZX's attrs view: a disc of ink in every cell, (H, W) bool."""
    h, w = cell
    y, x = np.mgrid[:h, :w]
    disc = (x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2 <= (min(h, w) * 3 / 8) ** 2
    return np.tile(disc, (size[0] // h, size[1] // w))


def save_dialog(title: str, folder: str, name: str) -> str:
    """Native save picker; "" when cancelled. On macOS through osascript, as mokit's native_pick, to get focus."""
    if sys.platform == 'darwin':
        script = (f'activate\nPOSIX path of (choose file name with prompt "{title}" '
                  f'default location POSIX file "{folder}" default name "{name}")')
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ''
    return pfd.save_file(title, str(Path(folder) / name)).result()


@contextmanager
def header_style(status: str):
    tint = HEADER_TINT.get(status)
    cols = []
    if tint is not None:
        for col, alpha in ((imgui.Col_.header, 0.45), (imgui.Col_.header_hovered, 0.6), (imgui.Col_.header_active, 0.75)):
            cols.append((col, imgui.ImVec4(tint.x, tint.y, tint.z, alpha)))
    elif status == 'stale':
        cols.append((imgui.Col_.text, Palette.muted))
    for col, value in cols:
        imgui.push_style_color(col.value, value)
    try:
        yield
    finally:
        imgui.pop_style_color(len(cols))


def as_ubyte(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


class HalftoneEditor:
    """The halftoner combo plus only the params that method uses (Ditherer.controls), as a narrowed params view."""

    def draw(self, params, picture, on_change, id: str) -> None:
        names = ('halftoner',) + ops.HALFTONERS[params.halftoner].controls
        view = make_dataclass('halftone', [(f.name, f.type, field(default=f.default, metadata=f.metadata))
                                           for f in fields(params) if f.name in names], frozen=True)
        params_editor(view(**{n: getattr(params, n) for n in names}),
                      lambda v: on_change(replace(params, **asdict(v))), id=id, help='tooltip')

class Window:
    def __init__(self, path=None) -> None:
        self.app = Pipeline()
        self.project = None                # the open image's project folder, written or not yet
        self.saved = self.app.graph        # the graph as last saved or opened: another one is unsaved
        self.autosave = False              # the user pref, on by default, comes with the prefs: tests never write
        self.restore_session = path is None
        if path and (Path(path) / project.PROJECT_FILE).exists():
            self._open_project(path)
        elif path:
            self._open_image(path)
        self.images = {}       # immvision params per preview
        self.expanded = {}     # node id -> block open; imgui keeps no header state in its ini
        self.editors = {'levels': LevelsEditor(), 'halftoner': HalftoneEditor()}   # node id -> custom params editor
        self.view, self.grid = 'Screen', False     # the conversion's view and the cell grid; not persisted
        self._debug = {}       # image key -> (the Converter it came from, the image)
        self._steps = 0        # history length last frame: the History list follows a new step
        self._after_close = None   # what waits for the unsaved changes dialog: opening another image

    def runner_params(self, persist: bool = True) -> hello_imgui.RunnerParams:
        immvision.use_rgb_color_order()
        p = hello_imgui.RunnerParams()
        p.app_window_params.window_title = 'Dizher'
        p.app_window_params.window_geometry.size = (1600, 1000)
        p.app_window_params.restore_previous_geometry = True
        p.imgui_window_params.show_menu_bar = True
        p.imgui_window_params.show_menu_app = False    # File, Edit, View, About drawn in _menus
        p.imgui_window_params.show_menu_view = False
        p.imgui_window_params.show_status_bar = True
        p.imgui_window_params.show_status_fps = False
        p.imgui_window_params.remember_status_bar_settings = False
        p.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
        p.ini_filename = 'dizher/ui.ini'
        p.fps_idling.fps_idle = 10.0
        p.callbacks.show_gui = self._frame
        p.callbacks.show_menus = self._menus
        p.callbacks.show_status = self._status
        p.callbacks.post_init = self._load_prefs
        p.callbacks.before_exit = self._save_prefs
        style.install(p)
        p.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        p.docking_params.docking_splits = [
            hello_imgui.DockingSplit('MainDockSpace', 'TuneSpace', imgui.Dir.left, 0.22),
            hello_imgui.DockingSplit('MainDockSpace', 'ConvertSpace', imgui.Dir.right, 0.28),
            hello_imgui.DockingSplit('ConvertSpace', 'HistorySpace', imgui.Dir.down, 0.25)]
        p.docking_params.dockable_windows = [
            hello_imgui.DockableWindow('Tune', 'TuneSpace', lambda: self._column(ops.TUNE[1:])),   # the source: File menu
            hello_imgui.DockableWindow('Convert', 'ConvertSpace', lambda: (self._column(ops.CONVERT), self._export())),
            hello_imgui.DockableWindow('Preview', 'MainDockSpace', self._preview),
            hello_imgui.DockableWindow('History', 'HistorySpace', self._history)]
        if not persist:   # tests: the default layout in an ini of their own, the user's stays untouched
            p.app_window_params.window_geometry.size = (1100, 1000)   # narrow: C64 fits at a smaller zoom than ZX
            p.app_window_params.restore_previous_geometry = False
            p.ini_folder_type = hello_imgui.IniFolderType.temp_folder
            p.docking_params.layout_condition = hello_imgui.DockingLayoutCondition.application_start
            p.callbacks.post_init = p.callbacks.before_exit = lambda: None
        return p

    def run_tests(self, tests) -> bool:
        """ToolZX's harness: (category, name, fn(ctx)) run in order under the imgui test engine in the real window,
        on the default layout; exits once all have run. True when every test passed."""
        from imgui_bundle.imgui import test_engine as te
        engine, summary = [], []
        params = self.runner_params(persist=False)
        params.use_imgui_test_engine = True

        def register():
            engine.append(hello_imgui.get_imgui_test_engine())
            io = te.get_io(engine[0])
            io.config_run_speed, io.config_stop_on_error, io.config_log_to_tty = te.TestRunSpeed.fast, False, True
            for category, name, fn in tests:
                t = te.register_test(engine[0], category, name)
                t.test_func = fn
                te.queue_test(engine[0], t)

        def post_new_frame():
            if not engine or not te.is_test_queue_empty(engine[0]) or summary:
                return
            s = te.TestEngineResultSummary()
            te.get_result_summary(engine[0], s)
            if s.count_tested >= len(tests):
                summary.append(s)
                hello_imgui.get_runner_params().app_shall_exit = True

        params.callbacks.register_tests = register
        params.callbacks.post_new_frame = post_new_frame
        self.run(params)
        if not summary:
            print('test engine never completed')
            return False
        print(f'{summary[0].count_success}/{summary[0].count_tested} tests passed')
        return summary[0].count_success == summary[0].count_tested

    def run(self, params=None) -> None:
        # the render loop swallows KeyboardInterrupt: Ctrl-C asks for a normal exit, so the ini is saved
        signal.signal(signal.SIGINT, lambda *_: setattr(hello_imgui.get_runner_params(), 'app_shall_exit', True))
        try:
            immapp.run(params or self.runner_params())
        finally:
            self.app.close()

    def _frame(self) -> None:
        style.sync()
        if not imgui.is_any_item_active():
            self.app.release()
        # global routing: a focused text field keeps its own Cmd+Z
        if imgui.shortcut(UNDO, imgui.InputFlags_.route_global.value):
            self.app.undo()
        if imgui.shortcut(REDO, imgui.InputFlags_.route_global.value):
            self.app.redo()
        if self.autosave and self.project and self.app.graph != self.saved and not imgui.is_any_item_active():
            self._save_project()   # once a drag is let go, not every frame of it
        self._unsaved_dialog()
        self.app.update()
        hello_imgui.get_runner_params().fps_idling.enable_idling = not self.app.busy

    # ----- docks -----------------------------------------------------------------------------------

    def _column(self, nodes) -> None:
        app = self.app
        for nid, label, op_id, inputs in nodes:
            status, params = app.status(nid), app.graph[nid].params
            title = f'{label} · {app.job.text}' if status == 'running' and app.job.text else label
            x0, width = imgui.get_cursor_pos_x(), imgui.get_content_region_avail().x
            imgui.set_next_item_open(self.expanded.get(nid, True), imgui.Cond_.once.value)
            imgui.set_next_item_allow_overlap()   # the Reset button sits on the header
            with header_style(status):
                is_open = self.expanded[nid] = imgui.collapsing_header(f'{title}###{nid}')
            imgui.set_item_tooltip(app.errors[nid] if status == 'error' else Op.resolve(op_id).fn.__doc__ or '')
            if params is not None:
                self._reset_button(nid, params, x0 + width)
            if status == 'error':
                with style.text_color(Palette.error):
                    imgui.text_wrapped(app.errors[nid])
            if not is_open:
                continue
            on_change = lambda p, n=nid: app.set_params(n, p, held=imgui.is_any_item_active())
            if nid in self.editors:
                self.editors[nid].draw(params, app.shown(inputs[0]) if inputs else None, on_change, id=nid)
            elif params is not None:
                params_editor(params, on_change, id=nid, help='tooltip')
            if nid == 'halftoner' and 'noise_x' in ops.HALFTONERS[params.halftoner].controls:
                if imgui.button('Random'):
                    x, y = np.random.randint(ops.BLUE_NOISE_RESOLUTION, size=2)
                    self.app.set_params(nid, replace(params, noise_x=int(x), noise_y=int(y)))
                imgui.set_item_tooltip('Restart from a random tile origin')
            widgets.gap()

    def _history(self) -> None:
        """Every step, the newest first, the current one selected and the undone ones muted; a click undoes or redoes
        to it."""
        app = self.app
        steps = app.past + [app.graph] + app.future[::-1]
        here = len(app.past)
        for i in reversed(range(len(steps))):
            label = change(steps[i - 1], steps[i]) if i else 'Start'
            with style.text_color(Palette.muted if i > here else imgui.get_style_color_vec4(imgui.Col_.text)):
                clicked = imgui.selectable(f'{label}##{i}', i == here)[0]
            imgui.set_item_tooltip(label)
            if i == here and len(steps) != self._steps:
                imgui.set_scroll_here_y()   # follow a new step
            if clicked:
                for _ in range(here - i):
                    app.undo()
                for _ in range(i - here):
                    app.redo()
                break
        self._steps = len(steps)

    def _export(self) -> None:
        """The block under Convert's: not a graph node, it saves the last finished result."""
        imgui.set_next_item_open(self.expanded.get('export', True), imgui.Cond_.once.value)
        self.expanded['export'] = imgui.collapsing_header('Export')
        if self.expanded['export']:
            result = self.result
            native = result.mode.file_type[1].lstrip('*') if result is not None and result.mode.file_type else None
            imgui.begin_disabled(result is None)
            if imgui.button('Save PNG…'):
                self._save('.png')
            imgui.end_disabled()
            imgui.same_line()
            imgui.begin_disabled(native is None)
            if imgui.button(f'Save {(native or ".scr")[1:].upper()}…'):
                self._save(native)
            imgui.end_disabled()

    def _reset_button(self, nid: str, params, right: float) -> None:
        """At the right end of the block header; disabled while every param is at its default."""
        default = type(params)()
        style_ = imgui.get_style()
        imgui.same_line(right - imgui.calc_text_size('Reset').x - 3 * style_.frame_padding.x)
        imgui.begin_disabled(params == default)
        if imgui.small_button(f'Reset##{nid}'):
            self.app.set_params(nid, default)
        imgui.end_disabled()

    def _view_bar(self) -> None:
        for view, tip in [*VIEWS.items(), *((v, d[0]) for v, d in DEBUG.items())]:
            if widgets.toggle_button(view, self.view == view):
                self.view = view
            imgui.set_item_tooltip(tip)
            imgui.same_line(0, 0)
        imgui.same_line()
        imgui.push_id('grid')   # the bare '#' label would read as an id marker
        if widgets.toggle_button('#', self.grid):
            self.grid = not self.grid
        imgui.set_item_tooltip('Cell grid over both previews')
        imgui.pop_id()

    def _live(self):
        """The running pair selection's or optimiser's latest snapshot."""
        job = self.app.job
        return job.image if job is not None and job.node_id in ('select', 'optimise') and not job.cancel.is_set() else None

    def _converted(self):
        """The view of the running stage's live snapshot, else of the last finished result."""
        live = self._live()
        if self.view in DEBUG:
            _, stage, fn = DEBUG[self.view]
            return self._debug_image(self.view, live if live is not None else self.app.shown(stage), fn)
        c = live if live is not None else self.result
        if c is None:
            return None
        if self.view == 'Screen':
            return c.dithered_result
        if self.view == 'Bitmap':
            return np.repeat(c.dithered_bitmap[..., None], 3, axis=2)
        return np.where(cell_icon(c.cell, c.size)[..., None], c.best_ink, c.best_paper)

    def _debug_image(self, key: str, c, fn):
        """fn of a Converter, computed once per Converter."""
        if c is None:
            return None
        hit = self._debug.get(key)
        if hit is None or hit[0] is not c:
            hit = self._debug[key] = (c, fn(c))
        return hit[1]

    def _cell_grid(self, cell, zoom: int) -> None:
        """Cell boundaries over the image just drawn."""
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        x0, y0, x1, y1 = lo.x, lo.y, hi.x, hi.y
        draw, col = imgui.get_window_draw_list(), imgui.get_color_u32(GRID)
        for x in np.arange(x0, x1 + 1, cell[1] * zoom):
            draw.add_line(imgui.ImVec2(x, y0), imgui.ImVec2(x, y1), col)
        for y in np.arange(y0, y1 + 1, cell[0] * zoom):
            draw.add_line(imgui.ImVec2(x0, y), imgui.ImVec2(x1, y), col)

    def _preview(self) -> None:
        self._view_bar()
        tuned = (self._debug_image('eye target', self.app.shown('prepare'), lambda c: c.eye_view(c.image_rgb)) if self.view == 'Eye'
                 else self.app.shown(ops.TUNED))
        converted = self._converted()
        shown = [(key, image) for key, image in (('tuned', tuned), ('converted', converted)) if image is not None]
        if not shown:
            return
        # the largest whole zoom that fits both side by side, or stacked when the dock is tall and narrow
        avail, spacing = imgui.get_content_region_avail(), imgui.get_style().item_spacing
        h, w = max(i.shape[0] for _, i in shown), max(i.shape[1] for _, i in shown)
        side = min((avail.x - spacing.x) / (2 * w), avail.y / h)
        stacked = min(avail.x / w, (avail.y - spacing.y) / (2 * h))
        zoom = max(1, int(max(side, stacked)))
        cell, hovered = ops.MODES[self.app.graph['target'].params.mode].cell, None
        for key, image in shown:
            ih, iw = image.shape[:2]
            immvision.image(f'##{key}', as_ubyte(image), widgets.image_params(self.images, key, (iw * zoom, ih * zoom), (iw, ih)))
            if imgui.is_item_hovered():
                m, lo = imgui.get_mouse_pos(), imgui.get_item_rect_min()
                hovered = min(int(m.y - lo.y) // zoom, ih - 1), min(int(m.x - lo.x) // zoom, iw - 1)
            if self.grid:
                self._cell_grid(cell, zoom)
            if side >= stacked:
                imgui.same_line()
        if hovered is not None and all(i.shape == shown[0][1].shape for _, i in shown):   # not mid mode switch
            self._inspect(hovered, cell, shown)

    def _inspect(self, pixel, cell, shown) -> None:
        """The hover tooltip: the cells around the one under the cursor zoomed in every preview, that cell outlined,
        and the pairs the selection energy scores best there, the other cells' pairs fixed."""
        (h, w), (H, W), z = cell, shown[0][1].shape[:2], INSPECT_ZOOM
        r, c = pixel[0] // h, pixel[1] // w
        # the block of cells centred on the hovered one, moved inside at the image's edges
        r0, c0 = (max(0, min(i - INSPECT_CELLS // 2, n - INSPECT_CELLS)) for i, n in ((r, H // h), (c, W // w)))
        size = INSPECT_CELLS * w, INSPECT_CELLS * h
        imgui.begin_tooltip()
        for key, image in shown:
            crop = image[r0 * h:r0 * h + size[1], c0 * w:c0 * w + size[0]]
            immvision.image(f'##inspect {key}', as_ubyte(crop),
                            widgets.image_params(self.images, f'inspect {key}', (size[0] * z, size[1] * z), size))
            self._cell_grid(cell, z)
            lo = imgui.get_item_rect_min()
            a = imgui.ImVec2(lo.x + (c - c0) * w * z, lo.y + (r - r0) * h * z)
            imgui.get_window_draw_list().add_rect(a, imgui.ImVec2(a.x + w * z, a.y + h * z),
                                                  imgui.get_color_u32(Palette.warn), thickness=2)
            imgui.same_line()
        imgui.new_line()
        self._candidates(r, c)
        imgui.end_tooltip()

    def _candidates(self, r: int, c: int) -> None:
        """The inspector's table: the best pairs at block (r, c) by SelectionEnergy.cell_candidates, and the chosen
        one (>) when it is not among them: selection stops after a sweep limit, a live snapshot mid-sweep."""
        conv = self._live()
        conv = conv if conv is not None else self.result
        labels = None if conv is None else conv.best_attr_indexes
        if labels is None or r >= labels.shape[0] or c >= labels.shape[1]:   # nothing selected, or another mode's
            return
        costs = conv.energy.cell_candidates(labels, r, c)
        total, chosen = costs.sum(1), labels[r, c]
        order = list(np.argsort(total)[:INSPECT_PAIRS])
        if chosen not in order:
            order.append(chosen)
        idx = list(conv.palette.iter_idxs_pairs())
        imgui.text(f'Cell {c}, {r}: pair scores with the neighbours as they are')
        if not imgui.begin_table('pairs', 6, imgui.TableFlags_.row_bg.value | imgui.TableFlags_.sizing_fixed_fit.value):
            return
        for name in ('', 'paper/ink', 'total', 'own', 'seams', 'coherence'):
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        swatch = imgui.ImVec2(imgui.get_text_line_height(), imgui.get_text_line_height())
        for p in order:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text('>' if p == chosen else '')
            imgui.table_next_column()
            for k, rgb in enumerate(conv.color_pairs[p]):
                imgui.color_button(f'##{p}.{k}', imgui.ImVec4(*map(float, rgb), 1.0), imgui.ColorEditFlags_.no_tooltip.value, swatch)
                imgui.same_line()
            imgui.text('%d/%d' % idx[p])
            for v in (total[p], *costs[p]):
                imgui.table_next_column()
                imgui.text(f'{v:.4f}')
        imgui.end_table()

    def _menus(self) -> None:
        if imgui.begin_menu('File'):
            if imgui.menu_item_simple('Open image…'):
                self._open()
            if imgui.menu_item_simple('Save project', enabled=self.project is not None and self.app.graph != self.saved):
                self._save_project()
            self.autosave = imgui.menu_item('Autosave project', '', self.autosave)[1]
            imgui.set_item_tooltip('Save the project after every edit')
            imgui.separator()
            if imgui.menu_item_simple('Save conversion…', enabled=self.result is not None):
                self._save()
            imgui.separator()
            if imgui.menu_item_simple('Quit'):
                hello_imgui.get_runner_params().app_shall_exit = True
            imgui.end_menu()
        if imgui.begin_menu('Edit'):
            if imgui.menu_item_simple('Undo', 'Cmd+Z' if sys.platform == 'darwin' else 'Ctrl+Z', enabled=bool(self.app.past)):
                self.app.undo()
            if imgui.menu_item_simple('Redo', 'Shift+Cmd+Z' if sys.platform == 'darwin' else 'Ctrl+Shift+Z',
                                      enabled=bool(self.app.future)):
                self.app.redo()
            imgui.end_menu()
        hello_imgui.show_view_menu(hello_imgui.get_runner_params())
        if imgui.begin_menu('About'):
            imgui.text(f'Dizher {__version__}')
            imgui.text('Images to 8-bit screens through an eye model')
            imgui.end_menu()

    def _status(self) -> None:
        imgui.text(f"{self.project.name if self.project else 'untitled'}{' *' if self.app.graph != self.saved else ''}")
        imgui.set_item_tooltip(str(self.project) if self.project else 'Open an image from the File menu')
        imgui.same_line()
        job = self.app.job
        if job is not None:
            imgui.text(f'{LABELS[job.node_id]}: {job.text} · {time.monotonic() - job.started:.1f} s')
            imgui.same_line()
            if imgui.small_button('Cancel'):
                self.app.cancel()
        elif self.app.errors:
            nid, message = next(iter(self.app.errors.items()))
            with style.text_color(Palette.error):
                imgui.text(f'{LABELS.get(nid, nid)}: {message}')
        elif self.app.cancelled:
            imgui.text('cancelled')
        else:
            imgui.text('pending' if self.app.busy else 'ready')

    # ----- files -----------------------------------------------------------------------------------

    @property
    def result(self):
        """The conversion on screen and in Save: the last one finished."""
        return self.app.shown('optimise')

    def _source(self):
        return self.app.graph['source'].params.path

    def _open(self) -> None:
        source = self._source()
        path = widgets.native_pick('file', 'Open image', str(source.parent if source else Path.home()))
        if path:
            self._close(lambda: self._open_image(path))

    def _close(self, then) -> None:
        """then() once the project is closed: at once when it is saved, else after the unsaved changes dialog.
        Quitting does not ask: the session keeps the unsaved edits for the next start."""
        if self.project and self.app.graph != self.saved:
            self._after_close = then
        else:
            then()

    def _unsaved_dialog(self) -> None:
        if self._after_close is None:
            return
        imgui.open_popup('Unsaved changes')   # imgui keeps it open when asked again
        if not imgui.begin_popup_modal('Unsaved changes', None, imgui.WindowFlags_.always_auto_resize.value)[0]:
            return
        imgui.text(f'Save the changes to {self.project.name}?')
        then, choice = self._after_close, None
        for label in ('Save', "Don't save", 'Cancel'):
            if imgui.button(label):
                choice = label
            imgui.same_line()
        imgui.new_line()
        if choice == 'Save':
            self._save_project()
        if choice:
            self._after_close = None
            imgui.close_current_popup()
        imgui.end_popup()
        if choice == "Don't save" or choice == 'Save' and self.app.graph == self.saved:   # not when the save failed
            then()

    def _open_image(self, path) -> None:
        """Its project when it has one, else a new one beside it, written by the first save."""
        folder = project_folder(path)
        if (folder / project.PROJECT_FILE).exists():
            self._open_project(folder)
        else:
            self.app.open(path)
            self.project, self.saved = folder, None

    def _save(self, ext: str = None) -> None:
        """ext picks the format (Converter.save); by default the mode's screen file, else PNG."""
        source, mode = self._source(), self.result.mode
        ext = ext or (mode.file_type[1].lstrip('*') if mode.file_type else '.png')
        folder = source.parent if source else Path.home()
        if self.project and self.project.exists():
            folder = self.project / 'build'
            folder.mkdir(exist_ok=True)
        path = save_dialog('Save conversion', str(folder), (source.stem if source else 'conversion') + ext)
        if path:
            self.result.save(path)

    def _open_project(self, folder) -> None:
        try:
            p = project.load_project(folder)
        except (OSError, ValueError, GraphError) as e:   # json's decode error is a ValueError
            self.app.errors['project'] = f'cannot open {folder}: {e}'
            return
        self.app.restore(p.graph)
        self.project, self.saved = p.folder, self.app.graph

    def _save_project(self) -> None:
        graph = self.app.graph
        try:
            if (self.project / project.PROJECT_FILE).exists():
                project.save_project(self.project, graph)
            else:
                project.create_project(self.project, graph)
        except OSError as e:
            self.app.errors['project'] = f'cannot save {self.project}: {e}; autosave is off'
            self.autosave = False   # not a retry every frame
            return
        self.saved = graph

    def _load_prefs(self) -> None:
        self.expanded.update(json.loads(hello_imgui.load_user_pref('expanded') or '{}'))
        self.autosave = json.loads(hello_imgui.load_user_pref('autosave') or 'true')
        if not self.restore_session:
            return
        try:
            session = json.loads(hello_imgui.load_user_pref('session') or '{}')
            graph = project.graph_from_json(session['graph'], None)[0] if 'graph' in session else None
        except (ValueError, KeyError, TypeError, GraphError):
            return   # a session from an incompatible version: start blank
        folder = Path(session['project']) if session.get('project') else None
        if folder and (folder / project.PROJECT_FILE).exists():
            self._open_project(folder)
        elif folder:
            self.project, self.saved = folder, None   # never saved
        if graph is not None:
            self.app.restore(graph)   # the unsaved edits over the project

    def _save_prefs(self) -> None:
        hello_imgui.save_user_pref('expanded', json.dumps(self.expanded))
        hello_imgui.save_user_pref('autosave', json.dumps(self.autosave))
        hello_imgui.save_user_pref('session', json.dumps({
            'project': str(self.project) if self.project else None,
            'graph': project.graph_to_json(self.app.graph, None, None)}))
