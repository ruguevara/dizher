# -*- coding: utf-8 -*-

from typing import List, Type

import numpy as np

from ..tuner import Tuner
from ..tuner.filters import ExposureFilter, ContrastFilter, GainFilter, ColorBalanceFilter, Filter
from ..tuner.vibe import VibeSatFilter
from ..tuner.reshaper import ReshaperFilter
from ..converter.converter import Converter
from ..platforms import zxspectrum, c64
from ..converter.dither import DBS, EDStucki, Ditherer, OrderedBayer, Stohastic

class Params:
    zoom: int = 2
    # debug: bool = True
    debug: bool = False
    timeout: int = 10


class DizherState:
    dither_classes: List[Type[Ditherer]] = [
        DBS,
        OrderedBayer,
        EDStucki,
        Stohastic,
    ]
    current_dithering: Type[Ditherer] = dither_classes[0]
    metric_weights = {'Luma': 1.0, 'Chroma': 1.0}
    modes = [zxspectrum.STANDARD, c64.HIRES]

    def __init__(self) -> None:
        self.params = Params()
        self.converter = Converter(self.metric_weights, self.modes[0])
        self.reshaper = ReshaperFilter(self.converter.size)
        self.tuner = Tuner([
            self.reshaper,
            GainFilter(),
            ExposureFilter(),
            ContrastFilter(),
            VibeSatFilter(),
            ColorBalanceFilter(),
        ])

    def set_mode(self, name: str):
        mode = next(m for m in self.modes if m.name == name)
        self.converter = self.converter.with_mode(mode)
        self.reshaper.height, self.reshaper.width = mode.size

    # def convert_image(self, image):
    #     self.converter.set_image(image)
    #     self.converter.calc_best_on_metrics(self.metric_weights)
    #     # print("from the state, in separate process, self._state.converter.best_levels= {}".format(self.converter.best_levels.shape))
    #     return self.dither()

    # def optimize_brightness(self):
    #     self.converter.optimize_brights()
    #     return self.dither()

    # def dither(self):
    #     # TODO maybe move current DitherMethod to converter as a subfilter
    #     return self.converter.dither(self.current_dithering())


def convert_image(converter: Converter, halftoner: Ditherer, image: np.ndarray):
    converter.set_image(image, halftoner)
    converter.calc_best_on_metrics()
    return converter

def apply_filter(filter: Filter):
    assert filter.input is not None
    output = filter.apply(filter.input)
    return (filter, output)
