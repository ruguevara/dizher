"""Run: .venv/bin/python tests/test_gui_updates.py (or pytest)."""
import copy
import multiprocessing as mp
from unittest.mock import patch

import numpy as np

from dizher.converter.converter import Converter
from dizher.converter.dither import OrderedBayer, Stohastic
from dizher.gui.app import DizherApp
from dizher.gui.state import DizherState, apply_eye_model, apply_metric_weights
from dizher.platforms import Mode
from dizher.platforms.zxspectrum import STANDARD
from dizher.util.worker import SingleAsyncPriorityWorker


class Worker:
    def __init__(self):
        self.tasks = []

    def apply(self, priority, task, args, kwds, callback=None, **_):
        self.tasks.append((task, copy.deepcopy(args), copy.deepcopy(kwds), callback))

    def complete(self, index):
        task, args, kwds, callback = self.tasks[index]
        callback(task(*args, **kwds))


class Pane:
    def update(self, *_, **__):
        pass


class Window:
    def __getitem__(self, _):
        return Pane()


def app():
    state = DizherState()
    mode = Mode('small', (16, 16), (8, 8), STANDARD.palette)
    state.converter = Converter(state.metric_weights, mode)
    state.current_dithering = OrderedBayer
    result = DizherApp.__new__(DizherApp)
    result._state = state
    result.worker = Worker()
    result._converter_revision = result._tuner_revision = 0
    result._conversion_image = None
    result.converted = []
    result.update_converted_image = result.converted.append
    result.update_tuned_image = lambda image: setattr(result, 'tuned', image)
    result.debug_log = lambda *_: None
    return result


def fresh_converter(converter, image):
    fresh = Converter(dict(converter.energy.weights), converter.mode,
                      converter.gamma, converter.luma_alpha, converter.luma_scale,
                      converter.chroma_alpha, converter.chroma_scale, converter.coherence,
                      converter.luma_noise, converter.chroma_noise, converter.structure)
    fresh.set_palette(fresh.palette.with_subset(converter.palette.subset))
    fresh.set_image(image, type(converter.ditherer)())
    fresh.calc_best_on_metrics()
    return fresh


def assert_fresh_conversion(converter, image):
    fresh = fresh_converter(converter, image)
    np.testing.assert_allclose(converter.dithered_result, fresh.dithered_result)
    for group in converter.energy.D:
        np.testing.assert_allclose(converter.energy.D[group], fresh.energy.D[group])


def test_converter_changes_restart_from_the_latest_parent_state():
    gui = app()
    image = np.full((16, 16, 3), .5, dtype=np.float32)
    gui.convert_image(image)
    gui.handle_converter_param('chroma_noise', apply_metric_weights, .3)
    gui.worker.complete(0)
    assert gui.converted == []
    gui.worker.complete(1)
    assert len(gui.converted) == 1 and gui._state.converter.chroma_noise == .3

    gui.handle_palette('Mono')
    gui.handle_converter_param('chroma_noise', apply_metric_weights, .4)
    gui.worker.complete(2)
    assert len(gui.converted) == 1
    gui.worker.complete(3)

    converter = gui._state.converter
    assert converter.palette.subset == 'Mono' and converter.chroma_noise == .4

    gui.handle_converter_param('luma_scale', apply_eye_model, .7)
    gui.handle_halftone('halftone-stohastic')
    gui.handle_converter_param('structure', apply_eye_model, .2)
    gui.worker.complete(4)
    assert len(gui.converted) == 2
    gui.worker.complete(6)

    converter = gui._state.converter
    assert converter.luma_scale == .7 and converter.structure == .2
    assert isinstance(converter.ditherer, Stohastic)
    assert_fresh_conversion(converter, image)


def test_tuner_restarts_from_root_after_an_invalidated_downstream_input():
    gui = app()
    tuner = gui._state.tuner
    tuner.input = np.full((8, 8, 3), .5, dtype=np.float32)
    filter = tuner.filters[1]
    gui.handle_slider(filter, 'gain', .3)
    gui.handle_slider(filter, 'gain', .8)

    gui.worker.complete(0)
    assert not hasattr(gui, 'tuned')
    gui.worker.complete(2)

    np.testing.assert_allclose(gui.tuned, gui._state.tuner())
    assert filter.params.gain == .8


def test_mode_change_discards_an_old_tuner_result():
    gui = app()
    gui.window = Window()
    gui._state.tuner.input = np.full((8, 8, 3), .5, dtype=np.float32)
    gui.handle_slider(gui._state.tuner.filters[1], 'gain', .3)
    gui.handle_mode('C64 hires')

    gui.worker.complete(0)
    assert not hasattr(gui, 'tuned')
    gui.worker.complete(2)
    assert gui.tuned is gui._state.tuner.output
    assert gui.tuned.shape == (200, 320, 3)


def test_metric_rebuilds_candidates_and_eye_uses_the_requested_halftoner():
    gui = app()
    converter = gui._state.converter
    image = np.random.default_rng(4).random((16, 16, 3), dtype=np.float32)
    converter.set_image(image, OrderedBayer())
    converter.energy.update(Luma=3.)
    halftoner = OrderedBayer()
    apply_metric_weights(converter, halftoner)
    assert_fresh_conversion(converter, image)

    apply_eye_model(converter, halftoner)
    assert converter.ditherer is halftoner


def test_completed_worker_cannot_supply_a_previous_result():
    # The GUI entry point uses fork; its worker target is deliberately local to apply().
    context = mp.get_context('fork')
    worker = SingleAsyncPriorityWorker()
    try:
        worker._process = context.Process(target=worker._queue.put, args=('old',))
        worker._process.start()
        worker._process.join(timeout=5)
        assert not worker.is_alive()
        with patch.object(mp, 'Process', context.Process):
            worker.apply(1, int, args=('2',))
        assert worker._queue.get(timeout=5) == 2
    finally:
        worker.abort()
        worker._queue.close()


if __name__ == '__main__':
    test_converter_changes_restart_from_the_latest_parent_state()
    test_tuner_restarts_from_root_after_an_invalidated_downstream_input()
    test_mode_change_discards_an_old_tuner_result()
    test_metric_rebuilds_candidates_and_eye_uses_the_requested_halftoner()
    test_completed_worker_cannot_supply_a_previous_result()
    print('ok')
