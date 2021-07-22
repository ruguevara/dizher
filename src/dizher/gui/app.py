#! python
# -*- coding: utf-8 -*-

import threading
from tkinter.constants import S
from typing import Callable, Tuple, Type
import PySimpleGUI as sg

import os
import time
from io import BytesIO
import sys
import threading
from enum import Enum, IntEnum
from multiprocessing import current_process

sys.path.append(os.path.abspath(os.path.join(os.path.basename(__file__), '..')))

from skimage import img_as_ubyte, img_as_float
import cv2
import numpy as np
from PIL import Image, ImageTk

from ..tuner import Tuner
from ..tuner.filters import ExposureFilter
from ..tuner.reshaper import ReshaperFilter
from ..converter.zxconverter import Converter, LumaMetric, ChromaMetric, SmoothnessMetric
from ..converter.colors import gray2rgb
from ..converter.dither import EDStucki, Ditherer, OrderedBayer, Stohastic
from .. import __version__
from ..util.worker import SingleAsyncPriorityWorker
from .state import DizherState, convert_image, optimize_brightness, dither, exposure_update


class BGTask(IntEnum):
    NONE = 0
    HALFTONER = 1
    CONVERTER = 2
    TUNER = 3


def asPhotoImage(image: np.ndarray, scale: int=1):
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return ImageTk.PhotoImage(Image.fromarray(img_as_ubyte(image)))


class ImagePane(sg.Image):
    def __init__(self, zoom: int, dims: Tuple[int, int], key=None):
        self.zoom = zoom
        super().__init__(key=key, size=(dims[0] * zoom, dims[1] * zoom), background_color='black')

    def update(self, image: np.ndarray):
        super().update(data=asPhotoImage(image, self.zoom))


def LabelSlider(label, filter, param_name, pad=(5, 7)):
    hpad, vpad = pad
    default = filter.get_defaults().get(param_name)
    range = filter.get_ranges().get(param_name)
    return [
        [sg.Text(label.capitalize(), font=(None, 10), size=(15, 1), pad=(hpad, (vpad, 0)))],
        [sg.Slider(range=range, default_value=default, resolution=0.1,
                   orientation='h', size=(20, 10), pad=(hpad, (0, vpad)), enable_events=True,
                   font=(None, 10), key="slider-{:s}".format(label.lower()))],
    ]


class DizherApp:
    title = 'Dizher'
    version = __version__
    title_text = '{:s} the 8-bit Graphics Converter — version {:s}'.format(title, version)
    psg_theme = 'Dark Blue 3'

    def __init__(self):
        sg.theme(self.psg_theme)
        self._state = DizherState()

        zoom = self._state.params.zoom
        dims = (self._state.converter.size[1], self._state.converter.size[0])
        self.window = sg.Window(self.title_text, [
                [sg.Menu([
                    ['&File', ['&Open Image', '&Save', 'E&xit',]],
                    ['&Optimize', ['&Brightness',]],
                    ['Help', '&About {:s}...'.format(self.title)],],
                    font=(None, 13)
                )], [
                    sg.Button('Open Image', key = 'open-image'),
                    sg.Button('Optimize brightness', key = 'optimize_brightness'),
                    sg.VerticalSeparator(pad=None),

                    sg.Button('Stohastic', key = 'halftone-noise'),
                    sg.Button('Ordered Bayer', key = 'halftone-ordered'),
                    sg.Button('ED Stucki', key = 'halftone-ed'),

                    sg.VerticalSeparator(pad=None),
                    sg.Button('Save', key = 'save-conversion'),
                ], [
                    sg.Column(
                        LabelSlider('Exposure', self._state.exposure, 'value') +
                        # LabelSlider("Gamma",      range=(-100, 100), default_value=0) +
                        # LabelSlider("Saturation", range=(-100, 100), default_value=0) +
                        # LabelSlider("Vibe",       range=(-100, 100), default_value=0) +
                        [],
                        vertical_alignment="top"
                    ),
                    ImagePane(key='image-original', dims=dims, zoom=zoom),
                    ImagePane(key='image-conversion', dims=dims, zoom=zoom),
                    sg.Column([
                        # [sg.Slider(range=(-100, 100), default_value=0, label="luma")],
                        # [sg.Slider(range=(-100, 100), default_value=0, label="chroma")],
                        # [sg.Slider(range=(-100, 100), default_value=0, label="1")],
                        # [sg.Slider(range=(-100, 100), default_value=0, label="2")],
                        # [sg.Sizer(20, zoom * dims[1] - 100)],
                    ]),
                ],
            ],
            finalize=True,
            return_keyboard_events=True,
        )
        self.worker = SingleAsyncPriorityWorker()

    def update_async(self, priority, event, task, args=(), kwds={}, callback: Callable = None):
        task_descr = "{} ({}) -> {}".format(task.__name__, priority.name, event)
        def wrap_callback(value):
            if callback:
                callback(value)
            self.debug_log("sending {}", task_descr)
            self.window.write_event_value(event, None)

        def error_callback(value):
            self.debug_log("error {}, value={}", task_descr, value)

        def abort_callback(value):
            self.debug_log("aborted {}, value={}", task_descr, value)

        self.debug_log("update_async {}", task_descr)
        self.worker.apply(priority, task, args, kwds,
                          callback=wrap_callback, error_callback=error_callback, abort_callback=abort_callback)

    def update_tuned_image(self, image):
        self.window['image-original'].update(image)
        self.convert_image(image)

    def update_converted_image(self, image):
        self.window['image-conversion'].update(image)

    def convert_image(self, image):
        # TODO can this events and callbacks be simplified somehow?
        def callback(converter):
            self._state.converter = converter

        self._state.converter.invalidate()
        self.update_async(BGTask.CONVERTER, 'update_converted_image', convert_image,
                          (self._state.converter, self._state.current_dithering(), image, self._state.metric_weights),
                           callback=callback)

    def optimize_brightness(self):
        def callback(converter):
            self._state.converter = converter

        self._state.converter.invalidate_result()
        self.update_async(BGTask.CONVERTER, 'update_converted_image', optimize_brightness,
                          (self._state.converter, self._state.current_dithering()), callback=callback)

    def open_image(self, filename):
        if not filename:
            return
        image = self._state.tuner.load_image(filename)
        self.update_tuned_image(image)

    def handle_sliders(self, event, values):
        if not event.startswith("slider-"):
            return
        key = event[len("slider-"):]
        value = values[event]
        if key == "exposure":
            def callback(image):
                self._state.tuner.result = image

            self.debug_log("Slider {}", value)
            self._state.exposure.update(value=value)
            self.update_async(BGTask.TUNER, 'update_tuned_image',
                              exposure_update, (self._state.exposure,), {}, callback=callback)

    def handle_halftone(self, event):
        if event == 'halftone-noise':
            self._state.current_dithering = Stohastic
        elif event == 'halftone-ordered':
            self._state.current_dithering = OrderedBayer
        elif event == 'halftone-ed':
            self._state.current_dithering = EDStucki
        else:
            self.not_so_fast(event)
            return

        def callback(converter):
            self._state.converter = converter

        self.update_async(BGTask.HALFTONER, 'update_converted_image', dither,
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
            event, values = self.window.read()
            self.debug_log(str(event))
            try:
                if event in (None, 'Exit', 'Cancel'):
                    break
                elif event == 'open-image' or event == "o":
                    filename = sg.popup_get_file(
                        'Image for conversion', no_window=True,
                        file_types = (('Image Files', '*.png *.jpeg *.jpg *.bmp'),),
                    )
                    self.open_image(filename)
                elif event in ('optimize_brightness', 'Brightness') :
                    self.optimize_brightness()
                elif event == 'update_tuned_image':
                    self.update_tuned_image(self._state.tuner.result)
                elif event == 'update_converted_image':
                    self.update_converted_image(self._state.converter.dithered_result)
                elif event.startswith('halftone-'):
                    self.handle_halftone(event)
                elif event.startswith('slider-'):
                    self.handle_sliders(event, values)
                # else:
                #     self.not_so_fast("{}".format(event))
            except Exception as e:
                raise
                # self.popup_error("Error: {} {}".format(type(e), str(e)))
