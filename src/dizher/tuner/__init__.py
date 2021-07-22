# -*- coding: utf-8 -*-

from typing import Sequence

import numpy as np
import cv2
from skimage import img_as_float

from ..converter.colors import convert_color
from .filters import Filter


class Tuner(Filter):
    # Container pattern and Chain of Responsibility pattern
    # TODO export effect values in serialized form
    # API consist of
    #   tuner.load_image/set_image — load new image, and zero out all effects, return image
    #   TODO use inotify and if image has changed — load and apply all effects without zeroing them out
    #   filter.update() — change params of one effect in chain — update it and all effects down the chain
    #                     return resulting image

    def __init__(self, filters: Sequence[Filter]) -> None:
        super().__init__()
        self.filters = filters
        assert len(self.filters) != 0
        # make linked list of filters
        prev_filter = self.filters[0]
        for filter in self.filters[1:]:
            prev_filter.link_to(filter)
            prev_filter = filter
        self.result = None

    def __call__(self):
        if self.image is not None and self.filters:
            self.result = self.filters[0].apply(self.image.copy())
            return self.result
