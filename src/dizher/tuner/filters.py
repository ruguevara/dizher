# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Any, Dict, Type, Union
from collections import namedtuple

import numpy as np
import cv2
from skimage import img_as_float

from dizher.converter.colors import convert_color


class Filter:
    label = 'Abstract Filter'

    # There are only two reasons to apply filter and change its result:
    # 1. Loaded new image or updated params of the filter up in the chain — .apply(image)
    # 2. Updated params of this filter — .update(params)

    # TODO filter params as descriptors-properties

    def __init__(self) -> None:
        self.chain_filter = None
        self.image: Union[np.ndarray, Any] = None  # @type: Union[np.ndarray, None]
        self.params = {}

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
        self.params = params
        self.invalidate()

    def update_and_apply(self, **params) -> Union[np.ndarray, None]:
        self.update(**params)
        if self.image is not None:
            return self.apply(self.image)

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class Parameter:
    def __get__(self, obj, objtype=None):
        return 1


class ExposureFilter(Filter):
    label = 'exposure'
    # TODO for params we need something like ProtoBuffers, serializable and introspectionable
    # TODO we need to construct forms and controls for params, like for models in MVC

    # class Params:
        # value = Parameter()
        # value = Parameter(default=0.0, range=(-3., 3.))

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        return {
            'value': 0.0
        }

    @classmethod
    def get_ranges(cls) -> Dict[str, Any]:
        return {
            'value': (-4.0, 4.0)
        }

    def __init__(self):
        super().__init__()
        self.params = self.get_defaults()

    def __call__(self) -> Union[np.ndarray, None]:
        if self.image is not None:
            gamma = np.exp(-self.params['value'])
            # TODO use cv2.LUT, make LUTFilter class for this
            return self.image ** gamma
