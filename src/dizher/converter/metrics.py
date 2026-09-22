# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple, Type, Union

import numpy as np
import cv2

from ..converter.colors import convert_color, lrgb2luminance
from .utils import reshape_by_charblock
from .eye import eye_blur


# Eye model (see eye.py): luminance is judged at near pixel resolution, chroma at a coarser one,
# in the spirit of S-CIELAB (Zhang & Wandell 1996). Blurs act in linear light, where the eye integrates dither.
# Defaults for Converter(eye_alpha, luma_scale, chroma_scale); the GUI exposes all three as sliders.
EYE_ALPHA = 0.95
LUMA_SCALE = 1.0
CHROMA_SCALE = 2.0

class ConversionMetric:
    label = 'You can not get label of an abstract base ConversionMetric class'

    def __init__(self, converter):
        self.converter = converter

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class LumaMetric(ConversionMetric):
    label = 'Luma'

    def __call__(self, **kwargs):
        # residual dither ripple after the blur is the dither-noise penalty
        c = self.converter
        blur = lambda a: eye_blur(a.astype(np.float32), c.luma_scale, c.eye_alpha)
        gamma = c.gamma
        image_luma = blur(c.image_luma) ** (1/gamma)
        reconstruct_luma = np.stack([
            blur(lrgb2luminance(realized ** gamma)) ** (1/gamma)
            for realized in self.converter.realized
        ])
        return np.abs(reconstruct_luma - image_luma)

class ChromaMetric(ConversionMetric):
    label = 'Chroma'

    def __call__(self, **kwargs):
        c = self.converter
        gamma = c.gamma
        to_luv = lambda lrgb: convert_color(eye_blur(lrgb, c.chroma_scale, c.eye_alpha).clip(0, 1) ** (1/gamma), 'RGB', 'LUV')
        image_luv = to_luv(self.converter.image_lrgb.astype(np.float32))
        reconstruct_luv = np.stack([to_luv(realized ** gamma) for realized in self.converter.realized])

        diff_u = (image_luv[..., 1] - reconstruct_luv[..., 1]) / 180
        diff_v = (image_luv[..., 2] - reconstruct_luv[..., 2]) / 180
        return np.sqrt(diff_u ** 2 + diff_v ** 2)

class DitherMetric(ConversionMetric):
    label = 'Ditherness'

    def __call__(self, **kwargs):
        # слабые уровни — мало точек для градиента: плохо. Точек 50/50 — хорошо, точек почти 0/100 — хорошо
        purity = (1 - np.abs((self.converter.levels - 0.5) * 2))
        purity = (1 - np.abs((purity - 0.5) * 2))
        purity[purity < 1/64] = 0
        return purity


class SmoothnessMetric(ConversionMetric):
    label = 'Smoothness'

    def __call__(self, **kwargs):
        color_pair_luma = lrgb2luminance(self.converter.color_pairs ** self.converter.gamma) ** (1/self.converter.gamma)
        luma_dist = np.abs(color_pair_luma[:, 1] - color_pair_luma[:, 0])
        height, width = self.converter.size
        luma_dist = luma_dist[:, np.newaxis].repeat(height * width, axis=1).reshape(-1, height, width)
        return luma_dist



class MetricTuner:
    def __init__(self, converter, metric_classes: Sequence[Type[ConversionMetric]], defaults: Sequence[float]) -> None:
        self.converter = converter
        self.metrics = OrderedDict((metric_class.label, metric_class(converter)) for metric_class in metric_classes)
        self.weights = OrderedDict((cls.label, default) for cls, default in zip(metric_classes, defaults))
        self.invalidate()

    def invalidate(self) -> None:
        self.metric_arrays = OrderedDict()

    def calc_metrics(self) -> None:
        for label, metric in self.metrics.items():
            self.metric_arrays[label] = metric()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.weights:
                raise ValueError("Can not set weight for unexistent metric {}".format(k))
            self.weights[k] = v

    def apply(self) -> None:
        assert len(self.weights) == len(self.metric_arrays), "{} != {}".format(len(self.weights), len(self.metric_arrays))
        integral_errors = np.sum([
            metric * weight
            # TODO calc metrics on arrays already reshaped by charblocks
            for weight, metric in zip(self.weights.values(), self.metric_arrays.values())
        ], axis=0)

        integral_errors = reshape_by_charblock(integral_errors)
        mse_by_combs_and_blocks = ((integral_errors * 255) ** 2).sum(axis=(3, 4)) / 64
        self.converter.set_best_conversion(mse_by_combs_and_blocks.argmin(0))
