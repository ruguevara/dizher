# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Type, Union
import numpy as np

from ..converter.colors import convert_color, rgb2lab, lab2rgb
from .params import Parameter
from .base_filter import Filter


def vibe_transform(x: np.ndarray, power: float) -> np.ndarray:
    power = np.clip(power, 0.001, 100)
    return x ** (1/power)

def saturation_transform(x: np.ndarray, power: float) -> np.ndarray:
    return (x * power)

def vibe_hsv(image: np.ndarray, transform: Callable) -> np.ndarray:
    result = image.copy()
    VS = result[..., 1] * result[..., 2]
    VS = transform(VS).clip(0, 1)
    V = result[..., 2]
    result[..., 1] = np.divide(VS, V, out=np.zeros_like(VS), where=V > 0)  # black has no saturation
    return result

def vibe_sat_fx(image_rgb: np.ndarray, transform: Callable) -> np.ndarray:
    image_hsv = convert_color(image_rgb, 'RGB', 'HSV')
    result_hsv = vibe_hsv(image_hsv, transform=transform)
    result_rgb = convert_color(result_hsv, 'HSV', 'RGB')

    # restore luma
    result_lab = convert_color(result_rgb, 'RGB', 'LAB')
    luma = convert_color(image_rgb, 'RGB', 'LAB')[..., 0]
    result_lab[..., 0] = luma
    result = convert_color(result_lab, 'LAB', 'RGB')
    return result


class VibeSatFilter(Filter):
    class Params(Filter.Params):
        vibe = Parameter(default=0., range=(-3., 3.))
        saturation = Parameter(default=0., range=(-3., 3.))

    params: VibeSatFilter.Params = Params()

    def __call__(self) -> Union[np.ndarray, None]:
        if self.input is None:
            return

        def transform(x: np.ndarray) -> np.ndarray:
            vibe = np.exp(self.params.vibe)
            saturation = np.exp(self.params.saturation)
            return saturation_transform(vibe_transform(x, vibe), saturation)

        return vibe_sat_fx(self.input, transform)
