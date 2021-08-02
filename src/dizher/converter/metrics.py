# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple, Type, Union

import numpy as np
import cv2

from ..converter.colors import convert_color, lrgb2luminance
from .utils import reshape_by_charblock


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
        # returns error in range 0..1
        gamma = self.converter.gamma
        image_luma = self.converter.image_luma ** (1/gamma)
        reconstruct_luma = lrgb2luminance(self.converter.recolorized ** gamma) ** (1/gamma)
        return np.abs(reconstruct_luma - image_luma)


class ChromaMetric(ConversionMetric):
    label = 'Chroma'

    def __call__(self, blur_size=3, **kwargs):
        # TODO make params adjustable
        # returns error in range 0..1
        image_rgb = cv2.GaussianBlur(self.converter.image_rgb, ksize=(blur_size, blur_size), sigmaX=0)
        image_luv = convert_color(image_rgb, 'RGB', 'LUV')

        reconstruct_luv = np.empty_like(self.converter.recolorized)
        for i, recolorized in enumerate(self.converter.recolorized):
            reconstruct_rgb = cv2.GaussianBlur(recolorized, ksize=(blur_size, blur_size), sigmaX=0)
            reconstruct_luv[i] = convert_color(reconstruct_rgb.astype(np.float32), 'RGB', 'LUV')

        diff_u = (image_luv[..., 1] - reconstruct_luv[..., 1]) / 180
        diff_v = (image_luv[..., 2] - reconstruct_luv[..., 2]) / 180
        # TODO можно перевести LUV в LHS и учитывать расстояния по H и S с разными весами
        result = np.sqrt(diff_u ** 2 + diff_v ** 2)
        return result


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
        color_pair_luma = lrgb2luminance(self.converter.color_pairs) ** (1/self.converter.gamma)
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
