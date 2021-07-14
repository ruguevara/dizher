# -*- coding: utf-8 -*-

from typing import Sequence
import numpy as np
import cv2
from skimage import img_as_ubyte, img_as_float

from .colors import convert_color


class Reshaper:
    def __init__(self, filename: str, size: Sequence[int], interpolation: int = cv2.INTER_AREA):
        self.height, self.width = size
        image = cv2.imread(filename)
        if image is None:
            raise RuntimeError('Image file "{:s}" not found'.format(filename))
        self.image = convert_color(image, 'BGR', 'RGB')
        self.interpolation = interpolation

    def crop(self):
        new_h, new_w = self.image.shape[:2]
        crop_h = (new_h - self.height) // 2
        crop_w = (new_w - self.width) // 2
        self.image = self.image[crop_h:crop_h + self.height, crop_w:crop_w + self.width].clip(0, 1)

    def __call__(self) -> np.ndarray:
        image = img_as_float(self.image)
        h, w = image.shape[:2]
        factor_h, factor_w = self.height / h, self.width / w
        max_factor = max(factor_h, factor_w)
        self.image = cv2.resize(image, (0, 0), fx = max_factor, fy = max_factor, interpolation = self.interpolation)
        self.crop()
        return self.image
