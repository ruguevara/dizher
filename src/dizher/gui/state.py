# -*- coding: utf-8 -*-

from typing import Dict, List, Sequence, Type

import numpy as np

from ..tuner import Tuner
from ..tuner.filters import ExposureFilter
from ..tuner.reshaper import ReshaperFilter
from ..converter.zxconverter import Converter, ConversionMetric, LumaMetric, ChromaMetric, SmoothnessMetric
from ..converter.dither import Ditherer
from ..converter.colors import gray2rgb
from ..converter.dither import EDStucki, Ditherer, OrderedBayer, Stohastic

class Params:
    zoom: int = 2
    debug: bool = False


class DizherState:
    dither_classes: List[Type[Ditherer]] = [
        Stohastic,
        EDStucki,
        OrderedBayer,
    ]
    current_dithering: Type[Ditherer] = Stohastic
    metric_classes: List[Type[ConversionMetric]] = [
        LumaMetric,
        ChromaMetric,
        SmoothnessMetric
    ]
    metric_weights = [0.6, 0.4, 0.002]

    def __init__(self) -> None:
        self.params = Params()
        self.converter = Converter(self.metric_classes)
        reshaper = ReshaperFilter(self.converter.size)
        self.exposure = ExposureFilter()
        self.tuner = Tuner([
            reshaper,
            self.exposure,
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


def dither(converter: Converter, halftoner: Ditherer):
    # TODO maybe move current DitherMethod to converter as a subfilter
    converter.dither(halftoner)
    return converter

def convert_image(converter: Converter, halftoner: Ditherer, image: np.ndarray, metric_weights: Sequence):
    converter.set_image(image)
    converter.calc_best_on_metrics(metric_weights)
    dither(converter, halftoner)
    return converter

def optimize_brightness(converter: Converter, halftoner: Ditherer):
    converter.optimize_brights()
    dither(converter, halftoner)
    return converter

def exposure_update(exposure: ExposureFilter):
    image = exposure.apply(exposure.image)
    return image
