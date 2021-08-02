#! python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, List, Tuple, Type
import os
import sys
from enum import Enum, IntEnum
from multiprocessing import current_process
from functools import partial

import PySimpleGUI as sg
from skimage import img_as_ubyte, img_as_float
import cv2
import numpy as np
import tkinter as tk

from ..tuner import Tuner
from ..tuner.filters import Filter
from ..converter.colors import gray2rgb
from ..converter.dither import EDStucki, Ditherer, OrderedBayer, Stohastic
from ..converter.zxconverter import ConversionMetric
from .. import __version__
from ..util.worker import SingleAsyncPriorityWorker
from .state import DizherState, convert_image, optimize_brightness, dither, apply_filter, apply_metric_weights


class BGTask(IntEnum):
    NONE = 0
    HALFTONER = 1
    CONVERTER = 2
    TUNER = 3


class ImagePane(sg.Image):
    def __init__(self, zoom: int, dims: Tuple[int, int], key=None):
        self.zoom = zoom
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
        data=self.makePhotoImage(image, self.zoom)
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
                            sg.Button('Optimize brightness', key = 'optimize_brightness'),
                            sg.VerticalSeparator(pad=None),
                        ] +
                            HalftoneButtons(self._state.dither_classes)
                        + [
                            sg.VerticalSeparator(pad=None),
                            sg.Button('Save', key = 'save-conversion'),
                        ],
                        [ImagePane(key='image-conversion', dims=dims, zoom=zoom)]
                    ]),
                    sg.Column(
                        self.metric_sliders(),
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
            handler(values[event])
            return True
        return False

    def label_slider(self, param_name: str, key: str, default: float, range: Tuple[float, float],
                     resolution:float = 0.1, pad: Tuple[int, int]=(5, 7)):
        hpad, vpad = pad
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
        self.update_async(BGTask.TUNER, apply_filter, (filter,), {}, callback=callback)

    def handle_metric_weight(self, param: str, value: Any):
        def callback(result):
            self._state.converter = result
            self.debug_log("handle_metric_weight callback")
            self.update_converted_image(self._state.converter.dithered_result)

        self.debug_log("Slider {}={}", param, value)
        tuner = self._state.converter.metric_tuner
        tuner.update(**{param: value})
        self._state.converter.invalidate_result()
        self.update_async(BGTask.CONVERTER, apply_metric_weights,
            (self._state.converter, self._state.current_dithering()),
            callback=callback)

    def tuner_sliders(self):
        sliders = []
        for filter in self._state.tuner.filters:
            for param in filter.Params.defaults.keys():
                event = f"slider-{filter.__class__.__name__}-{param.lower()}"
                sliders.extend(self.label_slider_filter(filter, param, event))
                self.bind(event, partial(self.handle_slider, filter, param))
        return sliders

    def metric_sliders(self):
        sliders = []
        tuner = self._state.converter.metric_tuner
        for label, weight in tuner.weights.items():
            event = f"slider-metric-weight-{label}"
            sliders.extend(self.label_slider(label.capitalize(), event, weight, (0., 1.0), resolution=0.01))
            self.bind(event, partial(self.handle_metric_weight, label))
        return sliders

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

    def optimize_brightness(self):
        def callback(converter):
            self._state.converter = converter
            self.debug_log("optimize_brightness callback")
            self.update_converted_image(self._state.converter.dithered_result)

        self._state.converter.invalidate_result()
        self.update_async(BGTask.CONVERTER, optimize_brightness,
                          (self._state.converter, self._state.current_dithering()), callback=callback)

    def open_image(self, filename):
        if not filename:
            return
        image = self._state.tuner.load_image(filename)
        self.update_tuned_image(image)

    def handle_halftone(self, event):
        if event == 'halftone-stohastic':
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
                elif event in ('optimize_brightness', 'Brightness') :
                    self.optimize_brightness()
                elif event.startswith('halftone-'):
                    self.handle_halftone(event)
                # else:
                #     self.not_so_fast("{}".format(event))
            except Exception as e:
                if self._state.params.debug:
                    raise
                else:
                    self.popup_error("Error: {} {}".format(type(e), str(e)))
        self.window.close()
