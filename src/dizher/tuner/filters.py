# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Tuple, Type, Union

import numpy as np
import cv2
from skimage import img_as_float

from ..converter.colors import rgb2lab, lab2rgb
from .params import Parameter, ParamSet
from .base_filter import Filter


class ExposureFilter(Filter):
    class Params(Filter.Params):
        exposure = Parameter(default=0., range=(-4., 4.))

    params: ExposureFilter.Params = Params()

    def __call__(self) -> Union[np.ndarray, None]:
        if self.input is not None:
            gamma = np.exp(-self.params.exposure)
            # NOTE Guess we do not need LUT for that, as we work with float32 images
            #      But if we work with uint16 unstead, LUT would be helpful
            return self.input ** gamma


class ContrastFilter(Filter):
    class Params(Filter.Params):
        contrast = Parameter(default=0., range=(-4., 4.))

    params: ContrastFilter.Params = Params()

    def __call__(self) -> Union[np.ndarray, None]:
        if self.input is None:
            return

        image_lab = rgb2lab(self.input.astype(np.float32))
        X = image_lab[..., 0] / 100

        if self.params.contrast > 0:
            gain = np.exp(self.params.contrast)
            a = (np.exp(gain/2) + 1) / (np.exp(gain/2) - 1)
            Y = (1 / (1 + np.exp(-(X-0.5) * gain)) - 0.5) * a + 0.5
        else:
            a = 1 - self.params.contrast / 5
            Y = (X - 0.5) / (a * a) + 0.5
        Y = np.clip(Y, 0., 1.)
        image_lab[..., 0] = Y * 100
        return lab2rgb(image_lab.astype(np.float32))

class GainFilter(Filter):
    class Params(Filter.Params):
        gain = Parameter(default=0., range=(-2., 2.))

    params: GainFilter.Params = Params()

    def __call__(self) -> Union[np.ndarray, None]:
        if self.input is not None:
            return np.clip(self.input * np.exp(self.params.gain), 0., 1.)

class ColorBalanceFilter(Filter):
    # Lab a*/b* offsets: a* runs green(-) -> red(+), b* runs blue(-) -> yellow(+); slider unit = 10 Lab units
    class Params(Filter.Params):
        green_red = Parameter(default=0., range=(-5., 5.))
        blue_yellow = Parameter(default=0., range=(-5., 5.))

    params: ColorBalanceFilter.Params = Params()

    def __call__(self) -> Union[np.ndarray, None]:
        if self.input is None:
            return
        image_lab = rgb2lab(self.input.astype(np.float32))
        image_lab[..., 1] += 10 * self.params.green_red
        image_lab[..., 2] += 10 * self.params.blue_yellow
        return np.clip(lab2rgb(image_lab), 0., 1.)
