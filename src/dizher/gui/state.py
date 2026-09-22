# -*- coding: utf-8 -*-

from typing import Dict, List, Sequence, Type

import numpy as np

from ..tuner import Tuner
from ..tuner.filters import ExposureFilter, ContrastFilter, GainFilter, ColorBalanceFilter, Filter
from ..tuner.vibe import VibeSatFilter
from ..tuner.reshaper import ReshaperFilter
from ..converter.zxconverter import Converter
from ..converter.dither import Ditherer
from ..converter.colors import gray2rgb
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

    def __init__(self) -> None:
        self.params = Params()
        self.converter = Converter(self.metric_weights)
        self.tuner = Tuner([
            ReshaperFilter(self.converter.size),
            GainFilter(),
            ExposureFilter(),
            ContrastFilter(),
            VibeSatFilter(),
            ColorBalanceFilter(),
        ])

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

def apply_metric_weights(converter: Converter, halftoner: Ditherer):
    converter.calc_best_on_metrics()
    dither(converter, halftoner)
    return converter

def apply_eye_model(converter: Converter, halftoner: Ditherer):
    converter.energy.calc()
    converter.calc_best_on_metrics()
    return converter

def dither(converter: Converter, halftoner: Ditherer):
    # TODO maybe move current DitherMethod to converter as a subfilter
    converter.dither(halftoner)
    return converter

def apply_filter(filter: Filter):
    assert filter.input is not None
    output = filter.apply(filter.input)
    return (filter, output)
