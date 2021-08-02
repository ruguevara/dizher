# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Tuple, Type, Union

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
        self.params: Filter.Params = self.Params()
        self.chain_filter: Union[Filter, None] = None
        self.input: Union[np.ndarray, None] = None

    def insert(self, filter) -> None:
        filter.link_to(self.chain_filter)
        self.link_to(filter)

    def link_to(self, filter) -> None:
        self.chain_filter = filter

    def load_image(self, filename: str) -> Union[np.ndarray, None]:
        image = cv2.imread(filename)
        if image is None:
            raise RuntimeError('Image file "{:s}" not found'.format(filename))
        image = img_as_float(convert_color(image, 'BGR', 'RGB'))
        return self.apply(image)

    def apply(self, input: np.ndarray) -> Union[np.ndarray, None]:
        if input is None:
            return
        self.input = input
        output = self()
        if self.chain_filter is not None:
            self.chain_filter.input = output
            output = self.chain_filter.apply(self.chain_filter.input)
        return output

    def invalidate(self) -> None:
        if self.chain_filter is not None:
            self.chain_filter.input = None
            self.chain_filter.invalidate()

    def update(self, **params) -> Union[np.ndarray, None]:
        # NOTE .update and .apply methods were split
        #      because in multiprocess environment we first update, then fork and apply
        self.params.update(**params)
        self.invalidate()

    def copy_from(self, other: Filter):
        if (type(self) == type(other) and self.chain_filter and other.chain_filter):
            self.chain_filter.input = other.chain_filter.input
            self.chain_filter.copy_from(other.chain_filter)

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()
