# -*- coding: utf-8 -*-

from typing import Sequence, Union
import numpy as np
import cv2
from skimage import img_as_ubyte, img_as_float

from .filters import Filter
from ..converter.colors import convert_color


class ReshaperFilter(Filter):
    def __init__(self, size: Sequence[int], interpolation: int = cv2.INTER_AREA):
        super().__init__()
        self.height, self.width = size
        self.interpolation = interpolation

    def resize(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        factor_h, factor_w = self.height / h, self.width / w
        max_factor = max(factor_h, factor_w)
        return cv2.resize(self.image, (0, 0), fx = max_factor, fy = max_factor, interpolation = self.interpolation)

    def crop(self, image: np.ndarray) -> np.ndarray:
        new_h, new_w = image.shape[:2]
        crop_h = (new_h - self.height) // 2
        crop_w = (new_w - self.width) // 2
        return image[crop_h:crop_h + self.height, crop_w:crop_w + self.width]

    def __call__(self) -> Union[np.ndarray, None]:
        if self.image is not None:
            return self.crop(self.resize(self.image))
