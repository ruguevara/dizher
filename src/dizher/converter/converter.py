import copy
from typing import Dict

import numpy as np
import cv2
from skimage import img_as_float

from .palette import Palette
from .colors import convert_color, lrgb2luminance, gray2rgb
from .dither import Ditherer, Stohastic, duo_levels
from .eye import LUMA_ALPHA, LUMA_SCALE, CHROMA_ALPHA, CHROMA_SCALE, eye_kernel
from .energy import SelectionEnergy, pair_dissimilarity, LRGB2OPP

class Converter:
    def __init__(self,
            weights: Dict[str, float],
            mode,   # platforms.Mode: screen size, attribute cell size, palette, native file encoder
            gamma: float = 2.2,
            luma_alpha: float = LUMA_ALPHA,
            luma_scale: float = LUMA_SCALE,
            chroma_alpha: float = CHROMA_ALPHA,
            chroma_scale: float = CHROMA_SCALE,
            coherence: float = 2.0,
            luma_noise: float = 0.0,
            chroma_noise: float = 0.05,
            structure: float = 0.06,
    ):
        self.mode = mode
        self.size = mode.size
        self.cell = mode.cell
        self.gamma = gamma
        self.luma_alpha = luma_alpha  # eye model, see eye.py: kernel shape and blur scales in pixels
        self.luma_scale = luma_scale
        self.chroma_alpha = chroma_alpha
        self.chroma_scale = chroma_scale
        self.luma_noise = luma_noise      # weights of the unblurred error: dot noise the low-pass eye model would miss, see energy.py
        self.chroma_noise = chroma_noise
        self.coherence = coherence  # cost of a pair change between neighbours where the original is smooth, see energy.py
        self.structure = structure  # weight of the contrast-weighted SSIM term in the DBS halftoner, see halftoning/dbs.py
        self.energy = SelectionEnergy(self, weights)
        self.image_rgb = None
        self.ditherer = None
        self.set_palette(mode.palette)

    def with_mode(self, mode) -> 'Converter':
        """Same parameters and halftoner on another mode; the image is dropped, it has the old size."""
        c = copy.copy(self)
        c.energy = SelectionEnergy(c, self.energy.weights)
        c.mode, c.size, c.cell = mode, mode.size, mode.cell
        c.image_rgb = None
        c.set_palette(mode.palette)
        return c

    def set_palette(self, palette: Palette):
        """Swap the attribute pair set; redoes the whole conversion if an image is loaded."""
        assert isinstance(palette, Palette)
        self.palette = palette
        self.color_pairs = palette.color_pairs()
        self.pair_dissimilarity = pair_dissimilarity(self.color_pairs)
        image, ditherer = self.image_rgb, self.ditherer
        self.invalidate()
        if image is not None:
            self.set_image(image, ditherer)
            self.calc_best_on_metrics()

    def invalidate(self):
        self.image_rgb = None
        self.image_lrgb = None
        self.image_luma = None
        self.levels = None
        self.bitmaps = None
        self.realized = None
        self.best_attr_indexes = None
        self.energy.invalidate()
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
        image_rgb = self.preprocess_image(image_rgb)
        self.invalidate()
        self.image_rgb = image_rgb
        self.image_lrgb = self.image_rgb ** self.gamma
        self.image_luma = lrgb2luminance(self.image_lrgb)
        self.levels = self.fit_duocolors()
        # candidates are scored on a blue-noise dither: cheap for all pairs, right noise amplitude for choosing them
        self.bitmaps = Stohastic().threshold(self.levels)
        paper = self.color_pairs[:, 0, np.newaxis, np.newaxis, :]
        ink = self.color_pairs[:, 1, np.newaxis, np.newaxis, :]
        self.realized = np.where(self.bitmaps[..., np.newaxis], ink, paper).astype(np.float32)
        self.energy.calc()
        self.ditherer = ditherer

    def fit_duocolors(self) -> np.ndarray:
        assert self.image_rgb is not None
        levels = np.empty((len(self.color_pairs),) + self.image_rgb.shape[:-1])
        for i, (c1, c2) in enumerate(self.palette.iter_color_pairs()):
            levels[i] = self.fit_duocolor(c1, c2)
        return levels

    def fit_duocolor(self, c1, c2) -> np.ndarray:
        """Closest mixture in the weighted colour space used to score candidates."""
        paper = self.opponent(np.asarray(c1, dtype=np.float32) ** self.gamma)
        ink = self.opponent(np.asarray(c2, dtype=np.float32) ** self.gamma)
        return duo_levels(self.opponent(self.image_lrgb), paper, ink)

    def opponent(self, linear_rgb):
        w = self.energy.weights
        return (linear_rgb @ LRGB2OPP.T) * np.sqrt(np.array([w['Luma'], w['Chroma'], w['Chroma']], dtype=np.float32))

    def eye_kernels(self):
        # ponytail: bounded support keeps all interactions inside adjacent cells; wider support
        # requires a different label optimiser, not dropping terms from its squared-error energy.
        radius = min(self.cell) // 2
        luma = eye_kernel(self.luma_scale, self.luma_alpha, max_radius=radius)
        chroma = eye_kernel(self.chroma_scale, self.chroma_alpha, max_radius=radius)
        return luma, chroma, chroma

    def expand_cells(self, per_cell: np.ndarray) -> np.ndarray:
        """(R, C, ...) one value per cell -> (H, W, ...) one value per pixel."""
        h, w = self.cell
        return np.repeat(np.repeat(per_cell, h, axis=0), w, axis=1)

    def best_paper_ink(self, attr_indexes):
        pairs = self.color_pairs[attr_indexes]                 # (R, C, 2, 3)
        return self.expand_cells(pairs[..., 0, :]), self.expand_cells(pairs[..., 1, :])

    def calc_best_on_metrics(self):
        self.energy.apply()

    def set_best_conversion(self, attr_indexes):
        self.best_attr_indexes = attr_indexes
        self.best_paper, self.best_ink = self.best_paper_ink(self.best_attr_indexes)
        self.halftone()

    def halftone(self):
        """Run the chosen halftoner once on the final composite, quantising each pixel to its block's paper or ink."""
        paper = self.opponent(self.best_paper ** self.gamma)
        ink = self.opponent(self.best_ink ** self.gamma)
        self.dithered_bitmap = self.ditherer(self.opponent(self.image_lrgb), paper, ink,
            scale=self.luma_scale, alpha=self.luma_alpha, structure=self.structure,
            kernels=self.eye_kernels(), noise=(self.luma_noise, self.chroma_noise, self.chroma_noise)).astype(np.float32)
        self.dithered_result = np.where(self.dithered_bitmap[..., np.newaxis], self.best_ink, self.best_paper)

    def save(self, filename: str) -> None:
        """The mode's native extension writes its screen file; any other extension writes the composite through OpenCV."""
        assert self.dithered_result is not None, "Nothing converted yet"
        if self.mode.file_type and filename.lower().endswith(self.mode.file_type[1].lstrip('*')):
            idx_pairs = np.array(list(self.palette.iter_idxs_pairs()))[self.best_attr_indexes]
            with open(filename, 'wb') as f:
                f.write(self.mode.encode(self.dithered_bitmap > 0.5, idx_pairs))
        else:
            assert cv2.imwrite(filename, convert_color((self.dithered_result * 255).round().astype(np.uint8), 'RGB', 'BGR')), filename

    def dither(self, ditherer: Ditherer) -> np.ndarray:
        if self.best_attr_indexes is None:
            self.ditherer = ditherer
            self.calc_best_on_metrics()
        elif type(ditherer) is not type(self.ditherer) or self.dithered_result is None:
            # a result may have been invalidated (parameter change) without a rerun yet: rebuild it from the labels
            self.ditherer = ditherer
            self.set_best_conversion(self.best_attr_indexes)
        return self.dithered_result
