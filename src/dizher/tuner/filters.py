# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Any, Dict, Tuple, Type, Union
from collections import namedtuple

import numpy as np
import cv2
from skimage import img_as_float

from ..converter.colors import convert_color
from .params import Parameter, ParamSet


class Filter:

    class Params(ParamSet):
        pass

    # There are only two reasons to apply filter and change its result:
    # 1. Updated params of this filter — .update(params)
    # 2. Loaded new image or updated params of the filter up in the chain — .apply(image)

    def __init__(self) -> None:
        self.params = self.Params()
        self.chain_filter = None
        self.image: Union[np.ndarray, Any] = None  # @type: Union[np.ndarray, None]

    def insert(self, chain_filter) -> None:
        chain_filter.link_to(self.chain_filter)
        self.link_to(chain_filter)

    def link_to(self, chain_filter) -> None:
        self.chain_filter = chain_filter

    def load_image(self, filename: str) -> Union[np.ndarray, None]:
        image = cv2.imread(filename)
        if image is None:
            raise RuntimeError('Image file "{:s}" not found'.format(filename))
        image = convert_color(image, 'BGR', 'RGB')
        return self.apply(image)

    def apply(self, image: np.ndarray) -> Union[np.ndarray, None]:
        if image is None:
            return
        self.image = img_as_float(image).astype(np.float32)
        image = self()
        if self.chain_filter is not None:
            image = self.chain_filter.apply(image)
        return image

    def invalidate(self) -> None:
        if self.chain_filter is not None:
            self.chain_filter.image = None
            self.chain_filter.invalidate()

    def update(self, **params) -> Union[np.ndarray, None]:
        # NOTE .update and .apply methods were splitted,
        #      because in multiprocesses environment we first update, then fork and apply
        self.params.update(**params)
        self.invalidate()

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class ExposureFilter(Filter):
    class Params(Filter.Params):
        exposure = Parameter(default=0., range=(-4., 4.))

    params = Params()

    def __call__(self) -> Union[np.ndarray, None]:
        if self.image is not None:
            gamma = np.exp(-self.params.exposure)
            # NOTE Guess we do not need LUT for that, as we work with float32 images
            #      But if we work with uint16 unstead, LUT would be helpful
            return self.image ** gamma


class SaturationVibeFilter(Filter):
    class Params(Filter.Params):
        saturation = Parameter(default=0., range=(-4., 4.))
        vibe = Parameter(default=0., range=(-4., 4.))

    params = Params()

    def __call__(self) -> Union[np.ndarray, None]:
        if self.image is not None:
            gamma = np.exp(-self.params.saturation)
            # NOTE Guess we do not need LUT for that, as we work with float32 images
            #      But if we work with uint16 unstead, LUT would be helpful
            return self.image ** gamma
