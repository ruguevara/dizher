"""Dizher's conversion stages as mokit ops, one graph node per stage in PIPELINE order.

A node's mokit key covers its params and everything upstream, so an edit reruns only the stages after it:
the halftoner reruns Halftone, coherence reruns Select pairs and Halftone, the eye model or the metric rerun
from Prepare (the ~1 s candidate and selection-energy setup). Converter results are shallow copies sharing
the upstream arrays, which no stage mutates.
"""
from dataclasses import dataclass, replace
from typing import Annotated

import cv2
import numpy as np
from skimage import img_as_float

from mokit.graph import Graph, Node, Op, meta
from mokit.types import Image

from .converter import eye as eye_model
from .converter.converter import Converter
from .converter.dither import DBS, EDStucki, OrderedBayer, Stohastic
from .platforms import Mode, c64, zxspectrum
from . import tone
from .util.worker import reporting

MODES = {m.name: m for m in (zxspectrum.STANDARD, c64.HIRES)}
SUBSETS = tuple(dict.fromkeys(s for m in MODES.values() for s in m.palette.SUBSETS))
HALFTONERS = {cls.label: cls for cls in (DBS, OrderedBayer, EDStucki, Stohastic)}


@dataclass(frozen=True)
class Metric:
    luma: float
    chroma: float


@dataclass(frozen=True)
class Eye:
    luma_alpha: float
    luma_scale: float
    chroma_alpha: float
    chroma_scale: float


def target(mode: Annotated[str, meta(choices=tuple(MODES))] = zxspectrum.STANDARD.name,
           palette: Annotated[str, meta(choices=SUBSETS)] = 'All colours') -> Mode:
    """Screen mode, its palette narrowed to the pairs a cell may use."""
    m = MODES[mode]
    if palette not in m.palette.SUBSETS:
        raise ValueError(f"{mode} has no palette subset {palette!r}")
    return replace(m, palette=m.palette.with_subset(palette))


def crop(frames: Image, target: Mode,
         anchor_x: Annotated[float, meta(min=0.0, max=1.0, help="0 left, 1 right")] = 0.5,
         anchor_y: Annotated[float, meta(min=0.0, max=1.0, help="0 top, 1 bottom")] = 0.5) -> np.ndarray:
    """First frame scaled to cover the screen (area averaging), cropped at the anchor: float RGB in 0..1."""
    rgb = img_as_float(frames.rgb[0]).astype(np.float32)
    h, w = target.size
    f = max(h / rgb.shape[0], w / rgb.shape[1])
    size = max(w, round(rgb.shape[1] * f)), max(h, round(rgb.shape[0] * f))
    scaled = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    y0, x0 = round((scaled.shape[0] - h) * anchor_y), round((scaled.shape[1] - w) * anchor_x)
    return scaled[y0:y0 + h, x0:x0 + w]


def light(picture: np.ndarray,
          exposure: Annotated[float, meta(min=-4.0, max=4.0, help="stops")] = 0.0,
          temperature: Annotated[float, meta(min=-100.0, max=100.0, help="+ warmer")] = 0.0,
          tint: Annotated[float, meta(min=-100.0, max=100.0, help="+ magenta")] = 0.0) -> np.ndarray:
    """Exposure and white balance in linear light."""
    return tone.light(picture, exposure, temperature, tint)


def levels(picture: np.ndarray,
           in_black: Annotated[int, meta(min=0, max=254)] = 0,
           in_white: Annotated[int, meta(min=1, max=255)] = 255,
           gamma: Annotated[float, meta(min=0.1, max=9.99)] = 1.0,
           out_black: Annotated[int, meta(min=0, max=255)] = 0,
           out_white: Annotated[int, meta(min=0, max=255)] = 255) -> np.ndarray:
    """Photoshop Levels; the UI draws it with ui/levels.py."""
    return tone.levels(picture, in_black, in_white, gamma, out_black, out_white)


def contrast(picture: np.ndarray, contrast: Annotated[float, meta(min=-100.0, max=100.0)] = 0.0) -> np.ndarray:
    """S-curve on lightness around mid grey."""
    return tone.contrast(picture, contrast)


def color(picture: np.ndarray,
          vibrance: Annotated[float, meta(min=-100.0, max=100.0, help="mostly muted colours")] = 0.0,
          saturation: Annotated[float, meta(min=-100.0, max=100.0)] = 0.0) -> np.ndarray:
    """Chroma in CIELAB."""
    return tone.color(picture, vibrance, saturation)


def metric(luma: Annotated[float, meta(min=0.0, max=4.0)] = 1.0,
           chroma: Annotated[float, meta(min=0.0, max=4.0)] = 1.0) -> Metric:
    """Weights of the luma and chroma error in the eye-model energy."""
    return Metric(luma, chroma)


def eye(luma_alpha: Annotated[float, meta(min=0.5, max=2.0)] = eye_model.LUMA_ALPHA,
        # the kernel radius is capped at half a cell (4 px): beyond ~1.9 px at alpha 2 the blur changes nothing
        luma_scale: Annotated[float, meta(min=0.3, max=1.9, label="luma blur px")] = eye_model.LUMA_SCALE,
        chroma_alpha: Annotated[float, meta(min=0.5, max=2.0)] = eye_model.CHROMA_ALPHA,
        chroma_scale: Annotated[float, meta(min=0.3, max=1.9, label="chroma blur px")] = eye_model.CHROMA_SCALE) -> Eye:
    """Alpha-stable blur of each channel group, see converter/eye.py."""
    return Eye(luma_alpha, luma_scale, chroma_alpha, chroma_scale)


def prepare(picture: np.ndarray, target: Mode, metric: Metric, eye: Eye, progress=None) -> Converter:
    """Every pair fitted per pixel, blue-noise candidates and the selection energy."""
    c = Converter({'Luma': metric.luma, 'Chroma': metric.chroma}, target, luma_alpha=eye.luma_alpha,
                  luma_scale=eye.luma_scale, chroma_alpha=eye.chroma_alpha, chroma_scale=eye.chroma_scale)
    with reporting(progress):
        c.set_image(picture, None)
    return c


def select_pairs(prepared: Converter,
                 coherence: Annotated[float, meta(min=0.0, max=8.0)] = 2.0,
                 luma_noise: Annotated[float, meta(min=0.0, max=0.5)] = 0.0,
                 chroma_noise: Annotated[float, meta(min=0.0, max=0.5)] = 0.05,
                 progress=None) -> Converter:
    """One (paper, ink) pair per cell; the noise weights also reach the DBS halftoner."""
    c = prepared.copy(coherence=coherence, luma_noise=luma_noise, chroma_noise=chroma_noise)
    with reporting(progress):
        c.set_labels(c.energy.apply())
    return c


def halftone(selection: Converter,
             halftoner: Annotated[str, meta(choices=tuple(HALFTONERS))] = DBS.label,
             structure: Annotated[float, meta(min=0.0, max=0.5, help="DBS only")] = 0.06,
             progress=None) -> Converter:
    """Each pixel quantised to its cell's paper or ink."""
    c = selection.copy(ditherer=HALFTONERS[halftoner](), structure=structure)
    with reporting(progress):
        c.halftone()
    return c


TUNE = (   # (node id, block label, op, inputs): the left column's blocks, top to bottom
    ('source', 'Source', 'mokit.ops:load_media', ()),
    ('crop', 'Crop', 'dizher.ops:crop', ('source', 'target')),
    ('light', 'Light', 'dizher.ops:light', ('crop',)),
    ('levels', 'Levels', 'dizher.ops:levels', ('light',)),
    ('contrast', 'Contrast', 'dizher.ops:contrast', ('levels',)),
    ('color', 'Color', 'dizher.ops:color', ('contrast',)),
)
CONVERT = (   # the right column's
    ('target', 'Target', 'dizher.ops:target', ()),
    ('metric', 'Metric', 'dizher.ops:metric', ()),
    ('eye', 'Eye model', 'dizher.ops:eye', ()),
    ('prepare', 'Prepare', 'dizher.ops:prepare', ('color', 'target', 'metric', 'eye')),
    ('select', 'Select pairs', 'dizher.ops:select_pairs', ('prepare',)),
    ('halftone', 'Halftone', 'dizher.ops:halftone', ('select',)),
)
PIPELINE = TUNE + CONVERT
TUNED = TUNE[-1][0]   # the node whose result the converter takes


def make_graph() -> Graph:
    return Graph(tuple((nid, Node(op, Op.resolve(op).default_params(), inputs)) for nid, _, op, inputs in PIPELINE))
