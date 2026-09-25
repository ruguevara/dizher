"""Dizher's conversion stages as mokit ops, one graph node per stage in PIPELINE order.

A node's mokit key covers its params and everything upstream, so an edit reruns only the stages after it:
the structure weight reruns Optimise, a painted cell from Overpaint, coherence from Select pairs, the halftoner,
the eye model or the metric from Prepare (the ~1 s candidate and selection-energy setup). Converter results are
shallow copies sharing the upstream arrays, which no stage mutates.
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
from .converter.energy import EDGE_SIGMA
from .converter.dither import Ditherer, ErrorDiffusion, Ordered, Stohastic
from .halftoning.noise.noise import BLUE_NOISE_RESOLUTION
from .halftoning.ordered.matrices import MATRICES
from .halftoning.error_distribution.kernels import KERNELS
from .platforms import Mode, c64, zxspectrum
from . import tone
from .progress import reporting

MODES = {m.name: m for m in (zxspectrum.STANDARD, c64.HIRES)}
HALFTONERS = {cls.label: cls for cls in (Stohastic, Ordered, ErrorDiffusion)}


@dataclass(frozen=True)
class Metric:
    chroma: float   # weight of the chroma error against luma's 1
    flare: float    # flattens the lightness gain of the error, see converter/energy.py


@dataclass(frozen=True)
class Eye:
    luma_alpha: float
    luma_scale: float
    chroma_alpha: float
    chroma_scale: float


def target(mode: Annotated[str, meta(choices=tuple(MODES))] = zxspectrum.STANDARD.name,
           colours: Annotated[tuple[int, ...], meta(help="palette indexes a cell may pair")] = tuple(range(16))) -> Mode:
    """Screen mode, its palette narrowed to the colours a cell may use; the UI draws it with TargetEditor."""
    m = MODES[mode]
    if not colours:
        raise ValueError("no colours enabled")
    if not set(colours) <= set(range(len(m.palette))):
        raise ValueError(f"{mode} has {len(m.palette)} colours, not {max(colours) + 1}")
    return replace(m, palette=m.palette.with_colours(colours))


PX = meta(min=-512, max=512, step=1)


def framing(frames: Image, target: Mode,
            fit: Annotated[str, meta(choices=('Fill', 'Fit'), help="Fill covers the screen, Fit shows the whole image")] = 'Fill',
            scale: Annotated[float, meta(min=0.25, max=4.0, help="1 is the fit's size")] = 1.0,
            rotation: Annotated[float, meta(min=-45.0, max=45.0, help="degrees, + counter-clockwise")] = 0.0,
            shift_x: Annotated[int, meta(**PX, help="px, + right")] = 0,
            shift_y: Annotated[int, meta(**PX, help="px, + down")] = 0,
            left: Annotated[int, meta(**PX, help="px the left edge moves out")] = 0,
            top: Annotated[int, meta(**PX, help="px the top edge moves out")] = 0,
            right: Annotated[int, meta(**PX, help="px the right edge moves out")] = 0,
            bottom: Annotated[int, meta(**PX, help="px the bottom edge moves out")] = 0) -> np.ndarray:
    """First frame placed on the screen: scaled to fill or fit it and centred, shifted, each edge pulled out or in
    by whole pixels to land the composition on the cell grid, rotated about the placed centre. Outside is black.
    Float RGB in 0..1."""
    rgb = img_as_float(frames.rgb[0]).astype(np.float32)
    H, W = target.size
    h, w = rgb.shape[:2]
    f = (max if fit == 'Fill' else min)(H / h, W / w) * scale
    sw, sh = round(w * f), round(h * f)
    # the placed rectangle, on whole pixels so that without rotation the warp is a plain copy
    x0, y0 = (W - sw) // 2 + shift_x - left, (H - sh) // 2 + shift_y - top
    rw, rh = sw + left + right, sh + top + bottom
    if rw < 1 or rh < 1:
        raise ValueError(f"the edges leave a {rw}x{rh} px image")
    scaled = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA if rw * rh < w * h else cv2.INTER_LINEAR)
    a = np.radians(rotation)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])   # y down: + turns counter-clockwise
    c = (np.array([rw, rh]) - 1) / 2   # the rectangle's centre in its own pixel coordinates
    t = np.array([x0, y0]) + c - rot @ c
    return cv2.warpAffine(scaled, np.hstack([rot, t[:, None]]), (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


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


def local_tone(picture: np.ndarray,
               local_contrast: Annotated[float, meta(min=-100.0, max=100.0, help="+ takes off the broad lighting, "
                                                     "so contrast goes to detail; - adds it")] = 0.0,
               shadows: Annotated[float, meta(min=-100.0, max=100.0, help="+ lifts dark regions")] = 0.0,
               highlights: Annotated[float, meta(min=-100.0, max=100.0, help="- recovers bright regions")] = 0.0,
               clarity: Annotated[float, meta(min=-100.0, max=100.0, help="contrast of everything smaller than the "
                                              "broad lighting")] = 0.0) -> np.ndarray:
    """Local tone mapping: the broad lighting (an edge-preserving blur of lightness) compressed and curved,
    the detail on it scaled."""
    return tone.local_tone(picture, local_contrast, shadows, highlights, clarity)


def contrast(picture: np.ndarray, contrast: Annotated[float, meta(min=-100.0, max=100.0)] = 0.0) -> np.ndarray:
    """S-curve on lightness around mid grey."""
    return tone.contrast(picture, contrast)


def color(picture: np.ndarray,
          vibrance: Annotated[float, meta(min=-100.0, max=100.0, help="mostly muted colours")] = 0.0,
          saturation: Annotated[float, meta(min=-100.0, max=100.0)] = 0.0) -> np.ndarray:
    """Chroma in CIELAB."""
    return tone.color(picture, vibrance, saturation)


def detail(picture: np.ndarray,
           texture: Annotated[float, meta(min=-100.0, max=100.0, help="contrast of fine texture")] = 0.0,
           sharpen: Annotated[float, meta(min=0.0, max=300.0, help="% of the fine detail added back")] = 0.0,
           radius: Annotated[float, meta(min=0.3, max=3.0, help="px, the size of the detail sharpen boosts")] = 1.0) -> np.ndarray:
    """Texture and unsharp mask on lightness at the screen's size, last so nothing after it softens the edges."""
    return tone.detail(picture, texture, sharpen, radius)


def metric(chroma: Annotated[float, meta(min=0.0, max=4.0, help="weight of chroma error; luma error weighs 1")] = 1.0,
           flare: Annotated[float, meta(min=0.0, max=1.0, help="stray light on the screen, in units of white: 0 weighs "
                                        "errors as CIELAB lightness does, ~7x more in black than in mid grey; "
                                        "higher flattens that towards plain linear light")] = 0.1) -> Metric:
    """Balance of chroma against luma error in the eye-model energy, and how much more an error counts in the
    shadows. One chroma weight: scaling both would only duplicate coherence (the seam cost has no weight), shift
    the edge threshold and the DBS structure term."""
    return Metric(chroma, flare)


def eye(luma_alpha: Annotated[float, meta(min=0.5, max=2.0)] = eye_model.LUMA_ALPHA,
        # the kernel radius is capped at half a cell (4 px): beyond ~1.9 px at alpha 2 the blur changes nothing
        luma_scale: Annotated[float, meta(min=0.3, max=1.9, label="luma blur px")] = eye_model.LUMA_SCALE,
        chroma_alpha: Annotated[float, meta(min=0.5, max=2.0)] = eye_model.CHROMA_ALPHA,
        chroma_scale: Annotated[float, meta(min=0.3, max=1.9, label="chroma blur px")] = eye_model.CHROMA_SCALE) -> Eye:
    """Alpha-stable blur of each channel group, see converter/eye.py."""
    return Eye(luma_alpha, luma_scale, chroma_alpha, chroma_scale)


NOISE = meta(min=0, max=BLUE_NOISE_RESOLUTION - 1, help="px the tile is rolled: another start for pair selection "
                                                        "and DBS, which settle in local optima")

def halftoner(halftoner: Annotated[str, meta(choices=tuple(HALFTONERS))] = Ordered.label,
              matrix: Annotated[str, meta(choices=tuple(MATRICES))] = 'Void dispersed dots',
              kernel: Annotated[str, meta(choices=tuple(KERNELS))] = 'Shiau-Fan 3',
              noise_x: Annotated[int, NOISE] = 0, noise_y: Annotated[int, NOISE] = 0) -> Ditherer:
    """The method that paints the pair candidates and then the result (each cell's paper or ink per pixel);
    each reads its own params (Ditherer.controls): Ordered the threshold matrix, Error diffusion the kernel,
    the tiled ones their origin."""
    return HALFTONERS[halftoner](matrix=matrix, kernel=kernel, origin=(noise_y, noise_x))


def prepare(picture: np.ndarray, target: Mode, metric: Metric, eye: Eye, halftoner: Ditherer, progress=None) -> Converter:
    """Every pair fitted per pixel and halftoned into a candidate, and the selection energy."""
    c = Converter({'Luma': 1.0, 'Chroma': metric.chroma}, target, luma_alpha=eye.luma_alpha,
                  luma_scale=eye.luma_scale, chroma_alpha=eye.chroma_alpha, chroma_scale=eye.chroma_scale,
                  ditherer=halftoner, flare=metric.flare)
    with reporting(progress):
        c.set_image(picture)
    return c


def select_pairs(prepared: Converter,
                 coherence: Annotated[float, meta(min=0.0, max=8.0)] = 2.0,
                 edge: Annotated[float, meta(min=0.02, max=0.4, help="step of the original across a seam that "
                                             "counts as an edge, where a pair change costs no coherence")] = EDGE_SIGMA,
                 luma_noise: Annotated[float, meta(min=0.0, max=0.5)] = 0.0,
                 chroma_noise: Annotated[float, meta(min=0.0, max=0.5)] = 0.05,
                 progress=None) -> Converter:
    """One (paper, ink) pair per cell; the noise weights also reach the DBS optimiser."""
    c = prepared.copy(coherence=coherence, edge=edge, luma_noise=luma_noise, chroma_noise=chroma_noise)
    with reporting(progress):
        c.set_labels(c.energy.apply())
    return c


def overpaint(selection: Converter,
              overrides: Annotated[tuple[tuple[int, int, int, int], ...],
                                   meta(help="cells painted by hand: (row, column, paper, ink), palette indexes, "
                                             "-1 keeping the selection's colour")] = ()
              ) -> Converter:
    """Cells painted by hand over the selection (the UI's Paint mode and right-click popup); the neighbours keep their
    pairs. A paper and ink the Target palette does not pair become the nearest pair it does, the painted colours
    kept first: on the Spectrum a bright ink over a dim paper brightens the paper. A cell off the screen is skipped."""
    if not overrides:
        return selection
    labels = selection.best_attr_indexes.copy()
    pairs = np.array(list(selection.palette.iter_idxs_pairs()))   # (P, 2) paper, ink
    rgb = selection.palette.as_float()
    R, C = labels.shape
    for r, c, *paint in overrides:
        if r >= R or c >= C:
            continue
        want = [p if p >= 0 else a for p, a in zip(paint, pairs[labels[r, c]])]
        weight = [PAINTED if p >= 0 else 1 for p in paint]
        far = lambda i, j: sum(w * np.linalg.norm(rgb[pairs[:, k]] - rgb[x], axis=-1)
                               for k, x, w in zip((i, j), want, weight))
        labels[r, c] = np.minimum(far(0, 1), far(1, 0)).argmin()   # a pair is unordered: paper may be the brighter
    painted = selection.copy()
    painted.set_labels(labels)
    return painted


PAINTED = 1e3   # a painted colour's distance against a kept one's: the nearest pair keeps the painted colours first


def halftone(selection: Converter, progress=None) -> Converter:
    """Each pixel quantised to its cell's paper or ink by the Halftoner: the start of the optimiser, or the result
    when it is off."""
    c = selection.copy()
    with reporting(progress):
        c.halftone()
    return c


def optimise(halftone: Converter,
             enabled: Annotated[bool, meta(help="Direct binary search from the halftone")] = True,
             structure: Annotated[float, meta(min=0.0, max=0.5, help="weight of the SSIM term")] = 0.06,
             progress=None) -> Converter:
    """Every pixel toggled or swapped with a neighbour while the eye-model error drops (halftoning/dbs.py),
    from the halftone as the start. Off passes the halftone through."""
    if not enabled:
        return halftone
    c = halftone.copy(structure=structure)
    with reporting(progress):
        c.optimise()
    return c


TUNE = (   # (node id, block label, op, inputs): the left column's blocks, top to bottom
    ('source', 'Source', 'mokit.ops:load_media', ()),
    ('framing', 'Framing', 'dizher.ops:framing', ('source', 'target')),
    ('light', 'Light', 'dizher.ops:light', ('framing',)),
    ('levels', 'Levels', 'dizher.ops:levels', ('light',)),
    ('local', 'Local tone', 'dizher.ops:local_tone', ('levels',)),
    ('contrast', 'Contrast', 'dizher.ops:contrast', ('local',)),
    ('color', 'Color', 'dizher.ops:color', ('contrast',)),
    ('detail', 'Detail', 'dizher.ops:detail', ('color',)),
)
CONVERT = (   # the right column's
    ('target', 'Target', 'dizher.ops:target', ()),
    ('metric', 'Metric', 'dizher.ops:metric', ()),
    ('eye', 'Eye model', 'dizher.ops:eye', ()),
    ('halftoner', 'Halftoner', 'dizher.ops:halftoner', ()),
    ('prepare', 'Prepare', 'dizher.ops:prepare', ('detail', 'target', 'metric', 'eye', 'halftoner')),
    ('select', 'Select pairs', 'dizher.ops:select_pairs', ('prepare',)),
    ('overpaint', 'Overpaint', 'dizher.ops:overpaint', ('select',)),
    ('halftone', 'Halftone', 'dizher.ops:halftone', ('overpaint',)),
    ('optimise', 'Optimise', 'dizher.ops:optimise', ('halftone',)),
)
PIPELINE = TUNE + CONVERT
TUNED = TUNE[-1][0]   # the node whose result the converter takes


def make_graph() -> Graph:
    return Graph(tuple((nid, Node(op, Op.resolve(op).default_params(), inputs)) for nid, _, op, inputs in PIPELINE))
