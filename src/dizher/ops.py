"""Dizher's conversion stages as mokit ops, one graph node per stage in PIPELINE order.

A node's mokit key covers its params and everything upstream, so an edit reruns only the stages after it:
the structure and noise weights rerun Optimise, the dots' chroma from Halftone, a painted cell from Overpaint,
coherence from Select pairs, the halftoner, the eye model or the metric from Prepare (the ~1 s candidate and
selection-energy setup). Converter results are shallow copies sharing the upstream arrays, which no stage mutates.
"""
from dataclasses import dataclass, replace
from typing import Annotated

import cv2
import numpy as np
from skimage import img_as_float

from mokit.graph import Graph, Like, Node, Op, meta
from mokit.types import Image

from .converter import eye as eye_model
from .converter.converter import Converter
from .converter.energy import METHODS, NEWEST, LEGACY
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
    method: str = NEWEST    # how a pair is scored on a block, see converter/energy.METHODS


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


TONE_MODES = ('Levels', 'Curves')


def levels(picture: np.ndarray,
           mode: Annotated[str, meta(choices=TONE_MODES, help="the one applied; both keep their values")] = 'Levels',
           in_black: Annotated[int, meta(min=0, max=254)] = 0,
           in_white: Annotated[int, meta(min=1, max=255)] = 255,
           gamma: Annotated[float, meta(min=0.1, max=9.99)] = 1.0,
           out_black: Annotated[int, meta(min=0, max=255)] = 0,
           out_white: Annotated[int, meta(min=0, max=255)] = 255,
           channels: Annotated[tuple[tuple[float, ...], ...],
                               meta(help="R, G, B levels after the composite, each (in_black, in_white, gamma, "
                                         "out_black, out_white)")] = (tuple(map(float, tone.NEUTRAL)),) * 3,
           curves: Annotated[tuple[tuple[float, ...], ...],
                             meta(help="RGB, R, G, B curves as flat (x, y) points in 0..255")] = (tone.IDENTITY,) * 4,
           picks: Annotated[tuple[tuple[float, ...], ...], meta(help="the Curves grey eyedropper's point groups, "
                                                                     "see tone.PICK_X")] = (),
           targets: Annotated[tuple[int, ...], meta(help="the black, grey and white eyedroppers' palette indexes, "
                                                        "-1 for the mode's own")] = (-1, -1, -1)) -> np.ndarray:
    """Photoshop Levels or Curves, per channel, with palette-targeted eyedroppers: the composite (RGB) first, then
    each channel's on its result. The in_black..out_white fields are the composite levels, as before channels came.
    The UI draws it with ui/levels.py."""
    if mode == 'Curves':
        return tone.curves(picture, curves)
    return tone.all_levels(picture, (in_black, in_white, gamma, out_black, out_white), channels)


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


NEWEST_PRESET = METHODS[NEWEST].preset   # new projects' Metric, Select pairs, Halftone, Optimise and Eye values


def metric(method: Annotated[str, meta(choices=tuple(METHODS), legacy=LEGACY,
                                       help="how a pair is scored on a cell: Exact mixture, its mixture and a cost "
                                            "for dots of clashing hues; Halftoned, one halftone of it (0.2.4). "
                                            "Picking one sets the Metric, Select pairs, Halftone, Optimise and Eye "
                                            "values it was tuned with")] = NEWEST,
           chroma: Annotated[float, meta(min=0.0, max=4.0, help="weight of chroma error; luma error weighs 1")]
               = NEWEST_PRESET['chroma'],
           flare: Annotated[float, meta(min=0.0, max=1.0, help="stray light on the screen, in units of white: 0 weighs "
                                        "errors as CIELAB lightness does, ~7x more in black than in mid grey; "
                                        "higher flattens that towards plain linear light")] = NEWEST_PRESET['flare']
           ) -> Metric:
    """The selection method, the balance of chroma against luma error in the eye-model energy, and how much more
    an error counts in the shadows. One chroma weight: scaling both would only duplicate coherence (the seam cost
    has no weight) and shift the edge threshold. The dots have weights of their own (halftone, optimise)."""
    return Metric(chroma, flare, method)


def apply_preset(graph: Graph, method: str) -> Graph:
    """The graph with the selection method and the Metric, Select pairs, Halftone, Optimise and Eye values it was
    tuned with."""
    preset = dict(METHODS[method].preset, method=method)
    for nid in ('metric', 'select', 'halftone', 'optimise', 'eye'):
        params = graph[nid].params
        graph = graph.with_params(nid, replace(params, **{k: v for k, v in preset.items() if hasattr(params, k)}))
    return graph


def eye(luma_alpha: Annotated[float, meta(min=0.5, max=2.0)] = eye_model.LUMA_ALPHA,
        # the kernel radius is capped at half a cell (4 px): beyond ~1.9 px at alpha 2 the blur changes nothing
        luma_scale: Annotated[float, meta(min=0.3, max=1.9, label="luma blur px")] = NEWEST_PRESET['luma_scale'],
        chroma_alpha: Annotated[float, meta(min=0.5, max=2.0)] = eye_model.CHROMA_ALPHA,
        chroma_scale: Annotated[float, meta(min=0.3, max=1.9, label="chroma blur px")]
            = NEWEST_PRESET['chroma_scale']) -> Eye:
    """Alpha-stable blur of each channel group, see converter/eye.py; the blurs are the selection method's (a preset
    sets them)."""
    return Eye(luma_alpha, luma_scale, chroma_alpha, chroma_scale)


NOISE = meta(min=0, max=BLUE_NOISE_RESOLUTION - 1, help="px the tile is rolled: another start for DBS (and for pair "
                                                        "selection by the Halftoned method), which settle in local optima")

def halftoner(halftoner: Annotated[str, meta(choices=tuple(HALFTONERS))] = Ordered.label,
              matrix: Annotated[str, meta(choices=tuple(MATRICES))] = 'Void dispersed dots',
              kernel: Annotated[str, meta(choices=tuple(KERNELS))] = 'Shiau-Fan 3',
              noise_x: Annotated[int, NOISE] = 0, noise_y: Annotated[int, NOISE] = 0) -> Ditherer:
    """The method that paints the result (each cell's paper or ink per pixel) and the pair candidates (the live
    preview, and what the Halftoned selection method scores); each reads its own params (Ditherer.controls): Ordered
    the threshold matrix, Error diffusion the kernel, the tiled ones their origin."""
    return HALFTONERS[halftoner](matrix=matrix, kernel=kernel, origin=(noise_y, noise_x))


def prepare(picture: np.ndarray, target: Mode, metric: Metric, eye: Eye, halftoner: Ditherer, progress=None) -> Converter:
    """Every pair fitted per pixel and halftoned into a candidate, and the selection energy of the chosen method."""
    c = Converter({'Luma': 1.0, 'Chroma': metric.chroma}, target, luma_alpha=eye.luma_alpha,
                  luma_scale=eye.luma_scale, chroma_alpha=eye.chroma_alpha, chroma_scale=eye.chroma_scale,
                  ditherer=halftoner, flare=metric.flare, method=metric.method)
    with reporting(progress):
        c.set_image(picture)
    return c


def select_pairs(prepared: Converter,
                 coherence: Annotated[float, meta(min=0.0, max=8.0)] = NEWEST_PRESET['coherence'],
                 edge: Annotated[float, meta(min=0.02, max=0.4, help="step of the original across a seam that "
                                             "counts as an edge, where a pair change costs no coherence")]
                     = NEWEST_PRESET['edge'],
                 luma_noise: Annotated[float, meta(min=0.0, max=0.5, help="cost of dot contrast in lightness")]
                     = NEWEST_PRESET['luma_noise'],
                 chroma_noise: Annotated[float, meta(min=0.0, max=0.5, help="cost of dots of clashing hues, "
                                                     "blue on yellow most, black or white dots none")]
                     = NEWEST_PRESET['chroma_noise'],
                 surface: Annotated[float, meta(min=0.0, max=40.0, label="surface dE",
                                                help="cells within this many dE of their surface's mean colour take "
                                                     "one pair, the surface's cheapest: a face or a sky in one clean "
                                                     "pair instead of a patchwork of two; small accents may go. 0 off")]
                     = 0.0,
                 progress=None) -> Converter:
    """One (paper, ink) pair per cell."""
    c = prepared.copy(coherence=coherence, edge=edge, luma_noise=luma_noise, chroma_noise=chroma_noise,
                      surface=surface)
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
    paint = painted_field(overrides, labels.shape)
    rows, cols = np.nonzero((paint >= 0).any(-1))
    pairs = np.array(list(selection.palette.iter_idxs_pairs()))   # (P, 2) paper, ink
    rgb = selection.palette.as_float()
    paint = paint[rows, cols]                                      # (N, 2) painted cells, -1 keeping the selection's
    want = np.where(paint >= 0, paint, pairs[labels[rows, cols]])
    weight = np.where(paint >= 0, PAINTED, 1).astype(np.float32)   # float32 as the distances, so ties stay ties
    to = [np.linalg.norm(rgb[:, None] - rgb[pairs[:, k]][None], axis=-1) for k in (0, 1)]   # colour -> paper, ink
    far = lambda i, j: weight[:, :1] * to[i][want[:, 0]] + weight[:, 1:] * to[j][want[:, 1]]   # (N, P)
    labels[rows, cols] = np.minimum(far(0, 1), far(1, 0)).argmin(-1)   # a pair is unordered: paper may be the brighter
    painted = selection.copy()
    painted.set_labels(labels)
    return painted


PAINTED = 1e3   # a painted colour's distance against a kept one's: the nearest pair keeps the painted colours first


def painted_field(overrides: tuple, shape) -> np.ndarray:
    """(R, C, 2) every cell's painted (paper, ink), -1 where it keeps the selection's; cells off the screen dropped."""
    field = np.full((*shape, 2), -1, dtype=np.int64)
    o = np.array(overrides, dtype=np.int64).reshape(-1, 4)
    o = o[(o[:, 0] < shape[0]) & (o[:, 1] < shape[1])]
    field[o[:, 0], o[:, 1]] = o[:, 2:]
    return field


def shown_pairs(conv: Converter, painted: np.ndarray) -> np.ndarray:
    """(R, C, 2) every cell's (paper, ink) as conv shows it, a painted colour (painted_field) in its painted role,
    matched by colour as Overpaint matches it, so either black reads as the painted one; the pair is unordered, the
    darker colour its paper where nothing is painted."""
    pairs = np.array(list(conv.palette.iter_idxs_pairs()))[conv.best_attr_indexes]   # (R, C, 2) the darker first
    rgb = conv.palette.as_float()
    off = lambda shown: sum(np.where(painted[..., k] >= 0, np.linalg.norm(rgb[shown[..., k]] - rgb[painted[..., k]],
                                                                         axis=-1), 0) for k in (0, 1))
    return np.where((off(pairs[..., ::-1]) < off(pairs))[..., None], pairs[..., ::-1], pairs)


def fix_overrides(selection: Converter, overrides: tuple) -> tuple:
    """Every cell of the screen painted with the colours it shows, so the whole field no longer follows Select pairs
    and Overpaint's result stays the same: an unpainted cell takes the pair Select pairs gave it, a -1 (Auto) colour
    the one Select pairs gave, a pair the palette cannot show the one Overpaint shows for it. A painted colour shown as
    itself keeps its index (either black), so a fixed field fixes to itself. Cells off the screen are kept as
    painted."""
    shape = selection.best_attr_indexes.shape
    painted = painted_field(overrides, shape)
    shown = shown_pairs(overpaint(selection, overrides), painted)
    rgb = selection.palette.as_ubyte()
    field = np.where((painted >= 0) & (rgb[painted] == rgb[shown]).all(-1), painted, shown)
    off = [o for o in overrides if o[0] >= shape[0] or o[1] >= shape[1]]
    return tuple(sorted([(r, c, int(p), int(i)) for (r, c), (p, i) in zip(np.ndindex(*shape), field.reshape(-1, 2))]
                        + off))


def halftone(selection: Converter,
             chroma: Annotated[float, meta(min=0.0, max=4.0, legacy=Like('metric', 'chroma'),
                                           help="weight of chroma error in the dots, the optimiser's too; luma "
                                                "error weighs 1. Metric's chroma weighs the pairs")]
                 = NEWEST_PRESET['chroma'],
             dithering: Annotated[float, meta(min=0.0, max=1.0, help="share of the lightness range between a cell's "
                                              "paper and ink that is mixed with dots, the rest solid: 0 thresholds each "
                                              "pixel to the nearer colour, 1 dithers every tone")] = 1.0,
             progress=None) -> Converter:
    """Each pixel quantised to its cell's paper or ink by the Halftoner: the start of the optimiser, or the result
    when it is off. The chroma weight is the dots' own, so they can be tuned without moving the pairs: it places each
    pixel's target on its pair's mixtures here and weighs the optimiser's error. Dithering also sets the tones the
    optimiser aims at."""
    c = selection.copy(dithering=dithering)
    c.energy.update(Chroma=chroma)
    with reporting(progress):
        c.halftone()
    return c


def optimise(halftone: Converter,
             enabled: Annotated[bool, meta(help="Direct binary search from the halftone")] = True,
             structure: Annotated[float, meta(min=0.0, max=0.5, help="weight of the SSIM term")] = 0.06,
             luma_noise: Annotated[float, meta(min=0.0, max=0.5, legacy=Like('select', 'luma_noise'),
                                               help="cost of dot contrast in lightness")] = NEWEST_PRESET['luma_noise'],
             chroma_noise: Annotated[float, meta(min=0.0, max=0.5, legacy=Like('select', 'chroma_noise'),
                                                 help="cost of dots of clashing hues, blue on yellow most, black or "
                                                      "white dots none")] = NEWEST_PRESET['chroma_noise'],
             progress=None) -> Converter:
    """Every pixel toggled or swapped with a neighbour while the eye-model error drops (halftoning/dbs.py),
    from the halftone as the start, weighing chroma as Halftone does. Its noise weights are its own, so the dots can
    be tuned without moving the pairs; a preset sets them and Halftone's chroma as Metric's and Select pairs', a
    project saved before they were apart takes those. Off passes the halftone through."""
    if not enabled:
        return halftone
    c = halftone.copy(structure=structure, luma_noise=luma_noise, chroma_noise=chroma_noise)
    with reporting(progress):
        c.optimise()
    return c


TUNE = (   # (node id, block label, op, inputs): the left column's blocks, top to bottom
    ('source', 'Source', 'mokit.ops:load_media', ()),
    ('framing', 'Framing', 'dizher.ops:framing', ('source', 'target')),
    ('light', 'Light', 'dizher.ops:light', ('framing',)),
    ('levels', 'Levels & Curves', 'dizher.ops:levels', ('light',)),
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
