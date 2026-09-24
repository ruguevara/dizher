import copy
from typing import Dict

import numpy as np
import cv2
from skimage import img_as_float

from .palette import Palette
from .colors import convert_color, lrgb2luminance, gray2rgb
from .dither import Ditherer, Stohastic, duo_levels
from ..halftoning.dbs import dbs_duo
from .eye import LUMA_ALPHA, LUMA_SCALE, CHROMA_ALPHA, CHROMA_SCALE, eye_kernel
from .energy import SelectionEnergy, pair_dissimilarity, lightness_gain, LRGB2OPP, EDGE_SIGMA
from ..progress import report_progress, report_stage

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
            edge: float = EDGE_SIGMA,
            luma_noise: float = 0.0,
            chroma_noise: float = 0.05,
            structure: float = 0.06,
            ditherer: Ditherer = None,   # halftones the pair candidates and, after selection, the result
            flare: float = 0.1,
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
        self.edge = edge            # step of the original across a seam that counts as a real edge, see energy.py
        self.coherence = coherence  # cost of a pair change between neighbours where the original is smooth, see energy.py
        self.structure = structure  # weight of the contrast-weighted SSIM term in the DBS optimiser, see halftoning/dbs.py
        self.ditherer = ditherer or Stohastic()
        self.flare = flare          # flattens the per-pixel lightness gain of the error, see energy.lightness_gain
        self.energy = SelectionEnergy(self, weights)
        self.image_rgb = None
        self.set_palette(mode.palette)

    def copy(self, **attrs) -> 'Converter':
        """Shallow copy for a later pipeline stage (ops.py): arrays are shared read-only, the energy is rebound to the copy."""
        c = copy.copy(self)
        c.energy = copy.copy(self.energy)
        c.energy.converter = c
        for name, value in attrs.items():
            assert hasattr(c, name), name
            setattr(c, name, value)
        return c

    def set_palette(self, palette: Palette):
        """Swap the attribute pair set; drops the image, whose setup was for the old pairs."""
        assert isinstance(palette, Palette)
        self.palette = palette
        self.color_pairs = palette.color_pairs()
        self.pair_dissimilarity = pair_dissimilarity(self.color_pairs)
        self.invalidate()

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
        self.halftoned = None       # the halftoner's bitmap, kept when optimise() replaces dithered_bitmap
        self.dithered_bitmap = None
        self.dithered_result = None

    def preprocess_image(self, image_rgb: np.ndarray) -> np.ndarray:
        image_rgb = img_as_float(image_rgb).astype(np.float32)
        assert image_rgb.shape[:2] == self.size, "Wrong image size!"

        if image_rgb.shape[2] == 1:
            image_rgb = gray2rgb(image_rgb)
        assert image_rgb.shape[2] == 3
        return image_rgb

    def set_image(self, image_rgb: np.ndarray) -> None:
        """The ~1 s setup: every pair fitted per pixel, its candidate halftoned, the selection energy."""
        image_rgb = self.preprocess_image(image_rgb)
        self.invalidate()
        self.image_rgb = image_rgb
        self.image_lrgb = self.image_rgb ** self.gamma
        self.image_luma = lrgb2luminance(self.image_lrgb)
        self.gain = lightness_gain(self.image_luma, self.flare)
        report_stage(f'fitting {len(self.color_pairs)} pairs')
        self.levels = self.fit_duocolors()
        # candidates are scored as the halftoner would paint them: the pairs are chosen for the dots they will get
        self.bitmaps = self.ditherer.threshold(self.levels)
        paper = self.color_pairs[:, 0, np.newaxis, np.newaxis, :]
        ink = self.color_pairs[:, 1, np.newaxis, np.newaxis, :]
        self.realized = np.where(self.bitmaps[..., np.newaxis], ink, paper).astype(np.float32)
        self.energy.calc()

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

    def render_labels(self, labels):
        """Composite of the blue-noise candidates for a labelling: what pair selection scores."""
        idx = self.expand_cells(labels)
        return self.realized[(idx,) + tuple(np.indices(idx.shape))]

    def eye_opponent(self, rgb):
        """What the selection energy compares: each opponent channel blurred with its eye kernel, unweighted."""
        opp = (rgb.astype(np.float32) ** self.gamma) @ LRGB2OPP.T
        return np.stack([cv2.filter2D(np.ascontiguousarray(opp[..., k]), -1, h, borderType=cv2.BORDER_REFLECT_101)
                         for k, h in enumerate(self.eye_kernels())], axis=-1)

    def eye_view(self, rgb):
        """eye_opponent back in sRGB."""
        return (self.eye_opponent(rgb) @ np.linalg.inv(LRGB2OPP).T).clip(0, 1) ** (1 / self.gamma)

    def set_labels(self, attr_indexes):
        self.best_attr_indexes = attr_indexes
        self.best_paper, self.best_ink = self.best_paper_ink(self.best_attr_indexes)

    def snapshot(self, labels=None, bitmap=None) -> 'Converter':
        """A running stage's state so far, for live previews: pair selection's labels (their blue-noise composite
        standing in for the halftone) or the halftoner's bitmap. Copied, the stage keeps mutating its arrays."""
        c = self.copy()
        if labels is not None:
            c.set_labels(labels.copy())
            idx = c.expand_cells(c.best_attr_indexes)
            bitmap = self.bitmaps[(idx,) + tuple(np.indices(idx.shape))]
        c.set_bitmap(np.array(bitmap, dtype=np.float32))
        return c

    def halftone(self):
        """Run the chosen halftoner once on the final composite, quantising each pixel to its block's paper or ink."""
        paper, ink = self._duo()
        report_stage(self.ditherer.label)
        self.set_bitmap(self.ditherer(self.halftone_target(paper, ink), paper, ink))
        self.halftoned = self.dithered_bitmap

    def optimise(self):
        """Direct binary search from the halftone bitmap under the eye model (halftoning/dbs.py); the start
        stays in halftoned. Target and colours are scaled by the lightness gain, so it minimises the selection's
        metric (its SSIM term then compares gained luma)."""
        paper, ink = self._duo()
        g = self.gain
        report_stage('DBS')
        self.set_bitmap(dbs_duo(self.halftone_target(paper, ink) * g, paper * g, ink * g, init=self.dithered_bitmap,
            scale=self.luma_scale, alpha=self.luma_alpha, structure=self.structure,
            kernels=self.eye_kernels(), noise=(self.luma_noise, self.chroma_noise, self.chroma_noise),
            on_step=lambda b: report_progress(lambda: self.snapshot(bitmap=b))))

    def _duo(self):
        return self.opponent(self.best_paper ** self.gamma), self.opponent(self.best_ink ** self.gamma)

    def set_bitmap(self, bitmap):
        self.dithered_bitmap = np.asarray(bitmap, dtype=np.float32)
        self.dithered_result = np.where(self.dithered_bitmap[..., np.newaxis] > 0, self.best_ink, self.best_paper)

    def halftone_target(self, paper, ink):
        """The halftoner minimises blurred error over the whole image, so the part of a cell's target its
        pair cannot reach would be cancelled by the neighbour's ink along the seam: a line of white dots on
        the attribute grid, plainest in smooth backgrounds. Each pixel is given the reachable projection
        of its target instead, so a cell's residual is zero-mean and there is nothing for the neighbour to
        cancel. The pair optimiser already owns the unreachable part (energy.py)."""
        target = self.opponent(self.image_lrgb)
        return paper + duo_levels(target, paper, ink)[..., None] * (ink - paper)

    def projected_target(self):
        """halftone_target in sRGB, for display: the opponent map is linear, so the same mix in linear RGB."""
        paper, ink = self.best_paper ** self.gamma, self.best_ink ** self.gamma
        t = duo_levels(self.opponent(self.image_lrgb), self.opponent(paper), self.opponent(ink))[..., None]
        return (paper + t * (ink - paper)) ** (1 / self.gamma)

    def save(self, filename: str) -> None:
        """The mode's native extension writes its screen file; any other extension writes the composite through OpenCV."""
        assert self.dithered_result is not None, "Nothing converted yet"
        if self.mode.file_type and filename.lower().endswith(self.mode.file_type[1].lstrip('*')):
            idx_pairs = np.array(list(self.palette.iter_idxs_pairs()))[self.best_attr_indexes]
            with open(filename, 'wb') as f:
                f.write(self.mode.encode(self.dithered_bitmap > 0.5, idx_pairs))
        else:
            assert cv2.imwrite(filename, convert_color((self.dithered_result * 255).round().astype(np.uint8), 'RGB', 'BGR')), filename

    def dither(self, ditherer: Ditherer, optimise: bool = False) -> np.ndarray:
        """Pair selection (once per image), the halftoner and optionally DBS from its result, in one call for
        tests and scripts; the UI runs them as separate stages (ops.py)."""
        if self.best_attr_indexes is None:
            self.set_labels(self.energy.apply())
        self.ditherer = ditherer
        self.halftone()
        if optimise:
            self.optimise()
        return self.dithered_result
