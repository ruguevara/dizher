"""imgui_bundle window over the pipeline host, in AmaZX's layout: the Tune dock on the left and the Convert dock
on the right, each with one collapsible block per stage in pipeline order (mokit's params editor inside, or a
custom one), and the Preview dock between them with the tuned image and the conversion. A block header shows its
stage's state: plain when done, tinted while it runs or after it failed, muted while waiting to run. hello_imgui's
ini keeps the dock layout, its user prefs which blocks are open."""
import json
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, immvision
from imgui_bundle import portable_file_dialogs as pfd

from mokit.ui import style, widgets
from mokit.ui.params import params_editor
from mokit.graph import Op
from mokit.ui.style import Palette

from .. import ops
from .app import Pipeline
from .levels import LevelsEditor
from . import views

HEADER_TINT = dict(running=Palette.warn, error=Palette.error)   # header background of a running or failed stage
NO_RESET = {'source'}   # resetting would drop the image
LABELS = {nid: label for nid, label, _, _ in ops.PIPELINE}
VIEWS = {'Screen': 'The conversion as the machine shows it', 'Bitmap': 'Ink pixels white, paper black',
         'Attrs': "Each cell's paper with a disc of its ink"}   # ToolZX's screen views
DEBUG = {   # view -> (tooltip, the stage it needs, its image from that stage's Converter or a running stage's snapshot)
    'Projected': ("What the halftoner aims at: each pixel moved to the nearest mix of its cell's pair",
                  'select', lambda c: c.projected_target()),
    'Eye': ('Both images as the eye model sees them: what the energy compares',
            'halftone', lambda c: c.eye_view(c.dithered_result)),
    'Error': ('Result minus target through the eye model, around grey: lighter/darker is luma error, '
              'the tint is the colour the result adds', 'halftone', views.error_view),
    'Energy': ("Each cell's energy: red its own error, green the eye-model seams, blue coherence",
               'select', views.energy_view),
    'Seams': ('Cell seams the original has an edge across, bright: there a pair change costs no coherence',
              'select', views.seam_view)}
GRID = imgui.ImVec4(0.5, 0.5, 0.5, 0.6)   # grey reads over black and white alike


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


class Window:
    def __init__(self, path=None) -> None:
        self.app = Pipeline()
        if path:
            self.app.open(path)
        self.images = {}       # immvision params per preview
        self.expanded = {}     # node id -> block open; imgui keeps no header state in its ini
        self.editors = {'levels': LevelsEditor()}   # node id -> custom params editor
        self.view, self.grid = 'Screen', False     # the conversion's view and the cell grid; not persisted
        self._debug = {}       # image key -> (the Converter it came from, the image)

    def runner_params(self, persist: bool = True) -> hello_imgui.RunnerParams:
        immvision.use_rgb_color_order()
        p = hello_imgui.RunnerParams()
        p.app_window_params.window_title = 'Dizher'
        p.app_window_params.window_geometry.size = (1600, 1000)
        p.app_window_params.restore_previous_geometry = True
        p.imgui_window_params.show_menu_bar = True
        p.imgui_window_params.show_status_bar = True
        p.imgui_window_params.show_status_fps = False
        p.imgui_window_params.remember_status_bar_settings = False
        p.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
        p.ini_filename = 'dizher/ui.ini'
        p.fps_idling.fps_idle = 10.0
        p.callbacks.show_gui = self._frame
        p.callbacks.show_menus = self._menus
        p.callbacks.show_status = self._status
        p.callbacks.post_init = lambda: self.expanded.update(json.loads(hello_imgui.load_user_pref('expanded') or '{}'))
        p.callbacks.before_exit = lambda: hello_imgui.save_user_pref('expanded', json.dumps(self.expanded))
        style.install(p)
        p.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        p.docking_params.docking_splits = [
            hello_imgui.DockingSplit('MainDockSpace', 'TuneSpace', imgui.Dir.left, 0.22),
            hello_imgui.DockingSplit('MainDockSpace', 'ConvertSpace', imgui.Dir.right, 0.28)]
        p.docking_params.dockable_windows = [
            hello_imgui.DockableWindow('Tune', 'TuneSpace', lambda: self._column(ops.TUNE)),
            hello_imgui.DockableWindow('Convert', 'ConvertSpace', lambda: self._column(ops.CONVERT)),
            hello_imgui.DockableWindow('Preview', 'MainDockSpace', self._preview)]
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
            if params is not None and nid not in NO_RESET:
                self._reset_button(nid, params, x0 + width)
            if status == 'error':
                with style.text_color(Palette.error):
                    imgui.text_wrapped(app.errors[nid])
            if not is_open:
                continue
            on_change = lambda p, n=nid: app.set_params(n, p)
            if nid in self.editors:
                self.editors[nid].draw(params, app.shown(inputs[0]), on_change, id=nid)
            elif params is not None:
                params_editor(params, on_change, id=nid, help='tooltip')
            if nid == 'prepare':
                if imgui.button('Random'):
                    x, y = np.random.randint(ops.BLUE_NOISE_RESOLUTION, size=2)
                    self.app.set_params(nid, replace(params, noise_x=int(x), noise_y=int(y)))
                imgui.set_item_tooltip('Restart from a random blue-noise origin')
            if nid == 'halftone':
                imgui.begin_disabled(self.result is None)
                if imgui.button('Save…'):
                    self._save()
                imgui.end_disabled()
            widgets.gap()

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

    def _converted(self):
        """The view of the running stage's live snapshot, else of the last finished result."""
        job = self.app.job
        live = job.image if job is not None and job.node_id in ('select', 'halftone') else None
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
        for key, image in shown:
            ih, iw = image.shape[:2]
            immvision.image(f'##{key}', as_ubyte(image), widgets.image_params(self.images, key, (iw * zoom, ih * zoom), (iw, ih)))
            if self.grid:
                self._cell_grid(ops.MODES[self.app.graph['target'].params.mode].cell, zoom)
            if side >= stacked:
                imgui.same_line()

    def _menus(self) -> None:
        if imgui.begin_menu('File'):
            if imgui.menu_item_simple('Open image…'):
                self._open()
            if imgui.menu_item_simple('Save conversion…', enabled=self.result is not None):
                self._save()
            imgui.end_menu()

    def _status(self) -> None:
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
        return self.app.shown('halftone')

    def _source(self):
        return self.app.graph['source'].params.path

    def _open(self) -> None:
        source = self._source()
        path = widgets.native_pick('file', 'Open image', str(source.parent if source else Path.home()))
        if path:
            self.app.open(path)

    def _save(self) -> None:
        source, mode = self._source(), self.result.mode
        ext = mode.file_type[1].lstrip('*') if mode.file_type else '.png'
        path = save_dialog('Save conversion', str(source.parent if source else Path.home()),
                           (source.stem if source else 'conversion') + ext)
        if path:
            self.result.save(path)
