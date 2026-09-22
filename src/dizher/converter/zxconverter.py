from typing import Sequence, Tuple, Type

import numpy as np
import cv2
from skimage import img_as_float

from .palette import Palette, ZXPalette
from .colors import convert_color, lrgb2luminance, gray2rgb
from .dither import Ditherer, Stohastic
from .ssim import greedy_ssim_optimize
from .metrics import ConversionMetric, MetricTuner, EYE_ALPHA, LUMA_SCALE, CHROMA_SCALE
from .utils import attrs2rgb, apply_attrs

class Converter:
    def __init__(self,
            metric_classes: Sequence[Type[ConversionMetric]],
            default_weights: Sequence[float],
            size: Tuple[int, int] = (192, 256),
            palette: Palette = ZXPalette(),
            gamma: float = 2.2,
            eye_alpha: float = EYE_ALPHA,
            luma_scale: float = LUMA_SCALE,
            chroma_scale: float = CHROMA_SCALE,
    ):
        assert isinstance(palette, Palette)
        assert len(size) == 2
        self.size = size
        self.palette = palette
        self.gamma = gamma
        self.eye_alpha = eye_alpha  # eye model, see eye.py: kernel shape and blur scales in pixels
        self.luma_scale = luma_scale
        self.chroma_scale = chroma_scale
        self.metric_tuner = MetricTuner(self, metric_classes, default_weights)
        self.color_pairs = self.palette.color_pairs()
        self.ditherer = None
        self.invalidate()

    def invalidate(self):
        self.image_rgb = None
        self.image_lrgb = None
        self.image_luma = None
        self.levels = None
        self.bitmaps = None
        self.realized = None
        self.best_attr_indexes = None
        self.metric_tuner.invalidate()
        self.invalidate_result()

    def invalidate_result(self):
        self.best_paper = None
        self.best_ink = None
        self.dithered_bitmap = None
        self.dithered_result = None

    def load_image(self, filename: str, ditherer: Ditherer):
        self.set_image(convert_color(cv2.imread(filename), 'BGR', 'RGB'), ditherer)

    def preprocess_image(self, image_rgb: np.ndarray) -> np.ndarray:
        image_rgb = img_as_float(image_rgb).astype(np.float32)
        assert image_rgb.shape[:2] == self.size, "Wrong image size!"

        if image_rgb.shape[2] == 1:
            image_rgb = gray2rgb(image_rgb)
        assert image_rgb.shape[2] == 3
        return image_rgb

    def set_image(self, image_rgb: np.ndarray, ditherer: Ditherer) -> None:
        self.image_rgb = self.preprocess_image(image_rgb)
        self.image_lrgb = self.image_rgb ** self.gamma
        self.image_luma = lrgb2luminance(self.image_lrgb)
        self.levels = self.fit_duocolors()
        # candidates are scored on a blue-noise dither: cheap for all pairs, right noise amplitude for choosing them
        self.bitmaps = Stohastic().threshold(self.levels)
        paper = self.color_pairs[:, 0, np.newaxis, np.newaxis, :]
        ink = self.color_pairs[:, 1, np.newaxis, np.newaxis, :]
        self.realized = np.where(self.bitmaps[..., np.newaxis], ink, paper).astype(np.float32)
        self.metric_tuner.calc_metrics()
        self.ditherer = ditherer

    def fit_duocolors(self) -> np.ndarray:
        assert self.image_rgb is not None
        levels = np.empty((len(self.color_pairs),) + self.image_rgb.shape[:-1])
        for i, (c1, c2) in enumerate(self.palette.iter_color_pairs()):
            levels[i] = self.fit_duocolor(c1, c2)
        return levels

    def fit_duocolor(self, c1, c2) -> np.ndarray:
        """Amount of c2 in a linear-light mix of c1 and c2 that matches the image luminance."""
        c1_lrgb = np.asarray(c1, dtype=np.float32) ** self.gamma
        c2_lrgb = np.asarray(c2, dtype=np.float32) ** self.gamma
        c1_luminance, c2_luminance = lrgb2luminance(c1_lrgb), lrgb2luminance(c2_lrgb)
        c_lum_range = c2_luminance - c1_luminance
        if c_lum_range == 0:
            return np.zeros_like(self.image_luma)
        return ((self.image_luma - c1_luminance) / c_lum_range).clip(0, 1)

    def best_paper_ink(self, attr_indexes):
        best_paper_i = self.color_pairs[attr_indexes][..., 0, :]
        best_ink_i = self.color_pairs[attr_indexes][..., 1, :]
        best_paper = attrs2rgb(best_paper_i)
        best_ink   = attrs2rgb(best_ink_i)
        return best_paper, best_ink

    def calc_best_on_metrics(self):
        self.metric_tuner.apply()

    def set_best_conversion(self, attr_indexes):
        self.best_attr_indexes = attr_indexes
        self.best_paper, self.best_ink = self.best_paper_ink(self.best_attr_indexes)
        self.halftone()

    def halftone(self):
        """Run the chosen halftoner once on the final composite, quantising each pixel to its block's paper or ink."""
        paper_luma = lrgb2luminance(self.best_paper ** self.gamma)
        ink_luma = lrgb2luminance(self.best_ink ** self.gamma)
        self.dithered_bitmap = self.ditherer(self.image_luma, paper_luma, ink_luma, scale=self.luma_scale, alpha=self.eye_alpha).astype(np.float32)
        self.dithered_result = apply_attrs(self.dithered_bitmap, self.best_paper, self.best_ink)

    def dither(self, ditherer: Ditherer) -> np.ndarray:
        if self.best_attr_indexes is None:
            self.ditherer = ditherer
            self.calc_best_on_metrics()
        elif type(ditherer) is not type(self.ditherer):
            self.ditherer = ditherer
            self.halftone()
        return self.dithered_result

    def optimize_brights(self):
        greedy_ssim_optimize(self)
        self.set_best_conversion(self.best_attr_indexes)
