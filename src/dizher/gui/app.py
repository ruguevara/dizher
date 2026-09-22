#! python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, List, Tuple, Type
import os
import sys
from enum import Enum, IntEnum
from multiprocessing import current_process
from functools import partial

import FreeSimpleGUI as sg
from skimage import img_as_ubyte, img_as_float
import cv2
import numpy as np
import tkinter as tk

from ..tuner import Tuner
from ..tuner.filters import Filter
from ..converter.colors import gray2rgb
from ..converter.dither import EDStucki, Ditherer, OrderedBayer, Stohastic, DBS
from .. import __version__
from ..util.worker import SingleAsyncPriorityWorker
from .state import DizherState, convert_image, dither, apply_filter, apply_metric_weights, apply_eye_model


class BGTask(IntEnum):
    NONE = 0
    HALFTONER = 1
    CONVERTER = 2
    TUNER = 3


class ImagePane(sg.Image):
    def __init__(self, zoom: int, dims: Tuple[int, int], key=None):
        self.scale = zoom  # not .zoom: FreeSimpleGUI Image.__init__ overwrites it
        super().__init__(key=key, size=(dims[0] * zoom, dims[1] * zoom), background_color='black')

    def makePhotoImage(self, image: np.ndarray, scale: int=1):
        if image is None:
            return
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        image = img_as_ubyte(image)
        # turn our ndarray into a bytesarray of PPM image by adding a simple header:
        # this header is good for RGB. for monochrome, use P5 (look for PPM docs)
        ppm = ('P6 %d %d 255 ' % (image.shape[1], image.shape[0])).encode('ascii') + image.tobytes()
        return ppm

    def update(self, image: np.ndarray):
        data=self.makePhotoImage(image, self.scale)
        super().update(data=data)


def HalftoneButtons(ditherers: List[Type[Ditherer]]):
    buttons = []
    for i, ditherer in enumerate(ditherers):
        name = ditherer.label.lower().replace(' ', '-')
        buttons.append(sg.Radio(ditherer.label, 'halftone', enable_events=True,
                                key=f'halftone-{name}', default=i==0))
    return buttons


class DizherApp:
    title = 'Dizher'
    version = __version__
    title_text = '{:s} the 8-bit Graphics Converter — version {:s}'.format(title, version)
    psg_theme = 'Dark Blue 3'

    def __init__(self):
        sg.theme(self.psg_theme)
        self._state = DizherState()
        self.worker = SingleAsyncPriorityWorker()
        self.dispatcher = {}
        self.slider_defaults = {}

        zoom = self._state.params.zoom
        dims = (self._state.converter.size[1], self._state.converter.size[0])
        self.window = sg.Window(self.title_text, [
                [sg.Menu([
                    ['&File', ['&Open Image', '&Save', 'E&xit',]],
                    ['&Optimize', ['&Brightness',]],
                    ['Help', '&About {:s}...'.format(self.title)],],
                    font=(None, 13)
                )], [
                    sg.Column(
                        [[
                            sg.Text(f'{self.title}', font=(None, 20), text_color="black", auto_size_text=True),
                            # sg.Text(f'v.{self.version}', font=(None, 11), text_color="black", auto_size_text=True),
                            ]] +
                        # [[sg.Sizer(20, 20)]] +
                        self.tuner_sliders(),
                        vertical_alignment="top"
                    ),
                    sg.Column([
                        [sg.Button('Open Image', key = 'open-image')],
                        [ImagePane(key='image-original', dims=dims, zoom=zoom)]
                    ]),
                    sg.Column([
                        [
                        ] +
                            HalftoneButtons(self._state.dither_classes)
                        + [
                            sg.VerticalSeparator(pad=None),
                            sg.Button('Save', key = 'save-conversion'),
                        ],
                        [ImagePane(key='image-conversion', dims=dims, zoom=zoom)]
                    ]),
                    sg.Column(
                        self.metric_sliders() + self.eye_sliders(),
                        vertical_alignment="top"
                    ),
                ],
            ],
            finalize=True,
            return_keyboard_events=True,
        )
        self.window.disable_debugger()
        self.window.bind('<Control-o>', 'open-image')

    def bind(self, event: str, handler: Callable):
        self.dispatcher[event] = handler

    def dispatch(self, event: str, values: Dict[str, Any]) -> bool:
        handler = self.dispatcher.get(event)
        if handler:
            handler(values.get(event))
            return True
        return False

    def label_slider(self, param_name: str, key: str, default: float, range: Tuple[float, float],
                     resolution:float = 0.1, pad: Tuple[int, int]=(5, 7)):
        hpad, vpad = pad
        self.slider_defaults[key] = default
        return [
            [sg.Text(param_name.capitalize(), font=(None, 10), size=(15, 1), pad=(hpad, (vpad, 0)))],
            [sg.Slider(range=range, default_value=default, resolution=resolution,
                    orientation='h', size=(20, 10), pad=(hpad, (0, vpad)), enable_events=True,
                    font=(None, 10), key=key)],
        ]

    def label_slider_filter(self, filter: Filter, param_name: str, key: str, pad: Tuple[int, int]=(5, 7)):
        hpad, vpad = pad
        default = filter.params.get_default(param_name)
        range = filter.params.get_range(param_name)
        return self.label_slider(param_name, key, default, range, pad=pad)

    def handle_slider(self, filter: Filter, param: str, value: Any):
        def callback(result):
            new_filter, output = result
            filter.copy_from(new_filter)
            self._state.tuner.output = output
            self.update_tuned_image(output)

        self.debug_log("Slider {}={}", param, value)
        filter.update(**{param: value})
        if filter.input is None:  # no image loaded yet, params are kept for when it is
            return
        self.update_async(BGTask.TUNER, apply_filter, (filter,), {}, callback=callback)

    def handle_metric_weight(self, param: str, value: Any):
        def callback(result):
            self._state.converter = result
            self.debug_log("handle_metric_weight callback")
            self.update_converted_image(self._state.converter.dithered_result)

        self.debug_log("Slider {}={}", param, value)
        self._state.converter.energy.update(**{param: value})
        self._state.converter.invalidate_result()
        self.update_async(BGTask.CONVERTER, apply_metric_weights,
            (self._state.converter, self._state.current_dithering()),
            callback=callback)

    def handle_converter_param(self, attr: str, task: Callable, value: Any):
        def callback(result):
            self._state.converter = result
            self.update_converted_image(self._state.converter.dithered_result)

        converter = self._state.converter
        setattr(converter, attr, value)
        if converter.image_rgb is None:
            return
        converter.invalidate_result()
        self.update_async(BGTask.CONVERTER, task, (converter, self._state.current_dithering()), callback=callback)

    def converter_param_sliders(self, key: str, specs):
        sliders, keys = [], []
        converter = self._state.converter
        for label, attr, task, range, res in specs:
            event = f"slider-{key}-{attr}"
            sliders.extend(self.label_slider(label, event, getattr(converter, attr), range, resolution=res))
            self.bind(event, partial(self.handle_converter_param, attr, task))
            keys.append(event)
        return sliders + [self.reset_button(f'reset-{key}', keys)]

    def eye_sliders(self):
        return self.converter_param_sliders('eye', (
            ('Luma alpha', 'luma_alpha', apply_eye_model, (0.5, 2.0), 0.05),
            ('Luma blur px', 'luma_scale', apply_eye_model, (0.3, 3.0), 0.1),
            ('Chroma alpha', 'chroma_alpha', apply_eye_model, (0.5, 2.0), 0.05),
            ('Chroma blur px', 'chroma_scale', apply_eye_model, (0.3, 8.0), 0.1),
        ))

    def reset_sliders(self, keys: List[str]):
        # Downstream first: the last dispatched handler wins in the worker, and for the tuner
        # that must be the head of the chain, pickled after every downstream param is already reset.
        for key in reversed(keys):
            default = self.slider_defaults[key]
            self.window[key].update(value=default)
            self.dispatcher[key](default)

    def reset_button(self, key: str, slider_keys: List[str]):
        self.bind(key, lambda _: self.reset_sliders(slider_keys))
        return [sg.Button('Reset', key=key, font=(None, 10), pad=(5, 7))]

    def tuner_sliders(self):
        sliders, keys = [], []
        for filter in self._state.tuner.filters:
            for param in filter.Params.defaults.keys():
                event = f"slider-{filter.__class__.__name__}-{param.lower()}"
                sliders.extend(self.label_slider_filter(filter, param, event))
                self.bind(event, partial(self.handle_slider, filter, param))
                keys.append(event)
        return sliders + [self.reset_button('reset-tuner', keys)]

    def metric_sliders(self):
        sliders, keys = [], []
        for label, weight in self._state.converter.energy.weights.items():
            event = f"slider-metric-weight-{label}"
            sliders.extend(self.label_slider(label.capitalize(), event, weight, (0., 4.0), resolution=0.05))
            self.bind(event, partial(self.handle_metric_weight, label))
            keys.append(event)
        event = 'slider-metric-coherence'
        sliders.extend(self.label_slider('Coherence', event, self._state.converter.coherence, (0., 8.0), resolution=0.1))
        self.bind(event, partial(self.handle_converter_param, 'coherence', apply_metric_weights))
        keys.append(event)
        return sliders + [self.reset_button('reset-metrics', keys)]

    def update_async(self, priority, task, args=(), kwds={}, callback: Callable = None):
        task_descr = "{} ({})".format(task.__name__, priority.name)
        def error_callback(value):
            self.debug_log("error {}, value={}", task_descr, value)

        def abort_callback(value):
            self.debug_log("aborted {}, value={}", task_descr, value)

        self.debug_log("update_async {}", task_descr)
        self.worker.apply(priority, task, args, kwds, callback=callback,
                          error_callback=error_callback, abort_callback=abort_callback)

    def update_tuned_image(self, image):
        self.debug_log('called update_tuned_image from {}', current_process())
        self.window['image-original'].update(image)
        self.window.refresh()
        self.convert_image(image)

    def update_converted_image(self, image):
        self.debug_log('called update_converted_image')
        self.window['image-conversion'].update(image)
        self.window.refresh()

    def convert_image(self, image):
        self.debug_log("called convert_image")

        def callback(converter):
            self._state.converter = converter
            self.debug_log("convert_image callback")
            self.update_converted_image(self._state.converter.dithered_result)

        self._state.converter.invalidate()
        self.update_async(BGTask.CONVERTER, convert_image,
            (self._state.converter, self._state.current_dithering(), image),
            callback=callback)

    def open_image(self, filename):
        if not filename:
            return
        image = self._state.tuner.load_image(filename)
        self.update_tuned_image(image)

    def save_conversion(self, filename):
        if not filename:
            return
        # image = self._state.tuner.load_image(filename)
        # self.update_tuned_image(image)

    def handle_halftone(self, event):
        if event == 'halftone-dbs':
            self._state.current_dithering = DBS
        elif event == 'halftone-stohastic':
            self._state.current_dithering = Stohastic
        elif event == 'halftone-ordered-bayer':
            self._state.current_dithering = OrderedBayer
        elif event == 'halftone-ed-stucki':
            self._state.current_dithering = EDStucki
        else:
            self.not_so_fast(event)
            return

        def callback(converter):
            self._state.converter = converter
            self.debug_log("handle_halftone callback")
            self.update_converted_image(self._state.converter.dithered_result)

        self.update_async(BGTask.HALFTONER, dither,
                          (self._state.converter, self._state.current_dithering()), callback=callback)

    def popup_error(self, *args, custom_text='Okay :-(', **kwargs):
        sg.popup(*args, custom_text=custom_text, **kwargs)

    def not_so_fast(self, function=None):
        func_text = 'Function "{:s}"'.format(function) if function else 'This function'
        self.popup_error(
            '{:s} is not implemented yet.'.format(func_text),
            title='Wow-wow! Not so fast, kid!',
        )

    def debug_log(self, message, *args, **kwargs):
        if self._state.params.debug:
            print(message.format(*args, **kwargs))
            # sg.eprint(message.format(*args, **kwargs))

    def event_loop(self):
        while True:
            event, values = self.window.read(self._state.params.timeout)
            self.window.refresh()
            try:
                if event == sg.TIMEOUT_KEY:
                    async_result = self.worker.read(self._state.params.timeout)
                    if async_result is not None:
                        self.debug_log("async worker event {}", type(async_result))
                    continue

                self.debug_log(str(event))
                if self.dispatch(event, values):
                    continue
                if event in ('Exit', 'Cancel', 'Quit', sg.WIN_CLOSED):
                    break
                elif event == 'open-image' or event == "o":
                    filename = sg.popup_get_file(
                        'Image for conversion', no_window=True,
                        file_types = (('Image Files', '*.png *.jpeg *.jpg *.bmp'),),
                    )
                    self.open_image(filename)
                elif event.startswith('halftone-'):
                    self.handle_halftone(event)
                elif event == 'save-conversion':
                    filename = sg.popup_get_file(
                        'Save converted image', no_window=True,
                        file_types = (('Image Files', '*.png *.jpeg *.jpg *.bmp'),),
                    )
                    self.save_conversion(filename)
                # else:
                #     self.not_so_fast("{}".format(event))
            except Exception as e:
                if self._state.params.debug:
                    raise
                else:
                    self.popup_error("Error: {} {}".format(type(e), str(e)))
        self.window.close()
