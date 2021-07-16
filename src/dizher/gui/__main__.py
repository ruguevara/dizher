#! python
# -*- coding: utf-8 -*-

from tkinter.constants import S
from typing import Tuple, Type
import PySimpleGUI as sg

import os
from io import BytesIO
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.basename(__file__), '..')))

from skimage import img_as_ubyte, img_as_float
import cv2
import numpy as np
from PIL import Image, ImageTk

from ..converter.reshaper import Reshaper
from ..converter.zxconverter import Converter, LumaMetric, ChromaMetric, SmoothnessMetric
from ..converter.colors import gray2rgb
from ..converter.dither import EDStucki, Ditherer, OrderedBayer, Stohastic
from .. import __version__


class Params:
    zoom: int = 2

def asPhotoImage(image: np.ndarray, scale: int=1):
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return ImageTk.PhotoImage(Image.fromarray(img_as_ubyte(image)))


class ImagePane(sg.Image):
    def __init__(self, zoom: int, dims: Tuple[int, int], key=None):
        self.zoom = zoom
        super().__init__(key=key, size=(dims[0] * zoom, dims[1] * zoom), background_color='black')

    def update(self, image: np.ndarray):
        super().update(data=asPhotoImage(image, self.zoom))


class DizherApp:
    title = 'Dizher'
    version = __version__
    title_text = '{:s} the 8-bit Graphics Converter — version {:s}'.format(title, version)
    dither_classes = [
        Stohastic,
        EDStucki,
        OrderedBayer,
    ]
    current_dithering = Stohastic
    metric_classes = [
        LumaMetric,
        ChromaMetric,
        SmoothnessMetric
    ]
    metric_weights = [0.6, 0.4, 0.002]
    psg_theme = 'Dark Blue 3'

    def __init__(self):
        sg.theme(self.psg_theme)
        self.params = Params()
        self.converter = Converter(self.metric_classes)
        zoom = self.params.zoom
        dims = (self.converter.size[1], self.converter.size[0])
        self.window = sg.Window(self.title_text, [
                [sg.Menu([
                    ['File', ['Open Image', 'Save', 'Exit',]],
                    ['Optimize', ['Brightness',]],
                    ['Help', 'About {:s}...'.format(self.title)],],
                    font=(None, 13)
                )], [
                    sg.Button('Open Image', key = 'open-image'),
                    sg.Button('Optimize brightness', key = 'optimize-brightness'),
                    sg.VerticalSeparator(pad=None),

                    sg.Button('Stohastic', key = 'halftone-noise'),
                    sg.Button('Ordered Bayer', key = 'halftone-ordered'),
                    sg.Button('ED Stucki', key = 'halftone-ed'),

                    sg.VerticalSeparator(pad=None),
                    sg.Button('Save', key = 'save-conversion'),
                ], [
                    ImagePane(key='image-original', dims=dims, zoom=zoom),
                    ImagePane(key='image-conversion', dims=dims, zoom=zoom),
                ],
            ],
            finalize=True
        )

    def open_image(self, filename):
        if not filename:
            return
        image = Reshaper(filename, self.converter.size)()
        self.window['image-original'].update(image)
        self.convert_image(image)

    def convert_image(self, image):
        self.converter.set_image(image)
        self.converter.calc_best_on_metrics(self.metric_weights)
        self.apply_dither()

    def optimize_brightness(self):
        self.converter.optimize_brights()
        self.apply_dither()

    def handle_halftone(self, event):
        if event == 'halftone-noise':
            self.set_dither(Stohastic)
        elif event == 'halftone-ordered':
            self.set_dither(OrderedBayer)
        elif event == 'halftone-ed':
            self.set_dither(EDStucki)
        else:
            self.not_so_fast(event)
            return
        self.apply_dither()

    def set_dither(self, dither_class: Type[Ditherer]):
        self.current_dithering = dither_class

    def apply_dither(self):
        # TODO maybe move current DitherMethod to converter and make "conversion stages" feature
        self.converter.dither(self.current_dithering())
        self.show_conversion()

    def show_conversion(self):
        self.window['image-conversion'].update(self.converter.dithered_result)

    def popup_error(self, *args, custom_text='Okay :-(', **kwargs):
        sg.popup(*args, custom_text=custom_text, **kwargs)

    def not_so_fast(self, function=None):
        func_text = 'Function "{:s}"'.format(function) if function else 'This function'
        self.popup_error(
            '{:s} is not implemented yet.'.format(func_text),
            title='Wow-wow! Not so fast, kid!',
        )

    def event_loop(self):
        while True:
            event, values = self.window.read()
            try:
                if event in (None, 'Exit', 'Cancel'):
                    break
                elif event == 'open-image':
                    filename = sg.popup_get_file(
                        'Image for conversion', no_window=True,
                        file_types = (('Image Files', '*.png *.jpeg *.jpg *.bmp'),),
                    )
                    self.open_image(filename)
                elif event in ('optimize-brightness', 'Brightness') :
                    self.optimize_brightness()
                elif event.startswith('halftone-'):
                    self.handle_halftone(event)
                else:
                    self.not_so_fast(event)
            except Exception as e:
                self.popup_error(str(e))

def main():
    app = DizherApp()
    initial_fname = sys.argv[1] if len(sys.argv) > 1 else None
    app.open_image(initial_fname)
    app.event_loop()


if __name__ == "__main__":
    main()
