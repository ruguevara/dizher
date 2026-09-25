"""Generic graph ops: plain annotated functions, wrapped by `Op.from_function` when resolved as "mokit.ops:<name>".

Keyword parameters with defaults are the node's params (Annotated[..., meta()] carries editor hints); positional
parameters are inputs. Listed by the `mokit.ops` entry point for the "Add node" palette. Port types: `docs/types.md`.
"""
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Optional, Tuple

import numpy as np

from . import image, media as media_mod
from .graph import meta
from .types import Image, Timing, parse_timecode
from .zx import fmf, img2scr
from .zx.screen import Patch, Screen
from .zx.tiles import Tileset, TileMap, quantize as quantize_tilemap


def load_media(path: Annotated[Optional[Path], meta(editor="file")] = None, progress=None) -> Image:
    """An image, gif/apng, video file, or a folder of images / .scr screens."""
    if path is None:
        raise ValueError("no media path")
    if progress:
        progress(None, f"reading {Path(path).name}")
    return media_mod.read_image(path)


def _frame_range(count: int, in_frame: int, out_frame: int) -> slice:
    if out_frame == -1:
        out_frame = count - 1
    if not 0 <= in_frame <= out_frame < count:
        raise ValueError(f"frame range {in_frame}..{out_frame} does not fit {count} frames")
    return slice(in_frame, out_frame + 1)


def crop_rgb(frames: Image, width: Annotated[int, meta(min=8, max=2048, step=8)] = 256,
         height: Annotated[int, meta(min=8, max=2048, step=8)] = 192,
         mode: Annotated[str, meta(choices=("cover", "contain", "stretch"))] = "cover",
         anchor_x: Annotated[float, meta(min=0.0, max=1.0)] = 0.5,
         anchor_y: Annotated[float, meta(min=0.0, max=1.0)] = 0.5,
         in_frame: Annotated[int, meta(min=0)] = 0,
         out_frame: Annotated[int, meta(min=-1, help="inclusive last frame; -1 = end")] = -1) -> Image:
    """Resize and crop an inclusive, zero-based frame range."""
    span = _frame_range(len(frames), in_frame, out_frame)
    rgba = image.crop_resize(frames.rgba[span], width, height, mode, (anchor_x, anchor_y))
    return Image(rgba, Timing(frames.timing.durations_ms[span]))


def tune(frames: Image, exposure: Annotated[float, meta(min=-4.0, max=4.0)] = 0.0,
         contrast: Annotated[float, meta(min=-1.0, max=1.0)] = 0.0,
         brightness: Annotated[float, meta(min=-1.0, max=1.0)] = 0.0,
         vibrance: Annotated[float, meta(min=-1.0, max=1.0)] = 0.0,
         saturation: Annotated[float, meta(min=0.0, max=3.0)] = 1.0) -> Image:
    rgb = np.ascontiguousarray(frames.rgb)
    if exposure:
        rgb = image.exposure(rgb, exposure)
    if contrast:
        rgb = image.contrast(rgb, contrast)
    if brightness:
        rgb = image.brightness(rgb, brightness)
    if vibrance or saturation != 1.0:
        rgb = image.vibrance(rgb, vibrance, saturation)
    return frames.with_rgb(rgb)


def levels(frames: Image, in_min: Annotated[int, meta(min=0, max=255)] = 0,
           in_max: Annotated[int, meta(min=0, max=255)] = 255,
           midpoint: Annotated[float, meta(min=0.0, max=1.0)] = 0.5,
           out_min: Annotated[int, meta(min=0, max=255)] = 0,
           out_max: Annotated[int, meta(min=0, max=255)] = 255) -> Image:
    """Piecewise-linear RGB levels; input midpoint maps to the middle of the output range."""
    return frames.with_rgb(image.levels(frames.rgb, in_min, in_max, midpoint, out_min, out_max))


def curves(frames: Image, points: Tuple[Tuple[float, float], ...] = (),
           channel: Annotated[str, meta(choices=("rgb", "r", "g", "b"))] = "rgb") -> Image:
    rgb = image.curves(np.ascontiguousarray(frames.rgb), points, channel)
    return frames.with_rgb(rgb)


def convert_dither(frames: Image, luma: Annotated[float, meta(min=0.0, max=4.0)] = 1.0,
               chroma: Annotated[float, meta(min=0.0, max=4.0)] = 1.0,
               smoothness: Annotated[float, meta(min=0.0, max=4.0)] = 0.5,
               dither: Annotated[str, meta(choices=("bayer", "stucki", "blue_noise"))] = "bayer",
               progress=None) -> Screen:
    """Attribute fit (weighted luma/chroma/smoothness) + dithering: 256x192 frames -> ZX screens."""
    rgb = np.ascontiguousarray(frames.rgb)
    fits = []
    for i in range(len(rgb)):        # one frame at a time: fit_attrs holds a (pairs, N, 192, 256) float64 array
        if progress:
            progress(i / len(rgb), f"fitting attributes {i + 1}/{len(rgb)}")
        fits.append(img2scr.fit_attrs(rgb[i:i + 1], (luma, chroma, smoothness)))
    fit = img2scr.AttrFit(*(np.concatenate([getattr(f, name) for f in fits])
                            for name in ("levels", "ink", "paper", "bright")))
    if progress:
        progress(None, f"dithering ({dither})")
    return img2scr.to_screen(img2scr.dither(fit.levels, dither), fit, frames.timing)


convert_dither.disk_cache, convert_dither.version = True, "1"   # ~0.6 s per frame; bump on an img2scr change


def convert_bw_dither(frames: Image,
                      dither: Annotated[str, meta(choices=("bayer", "stucki", "blue_noise"))] = "bayer",
                      progress=None) -> Screen:
    """Grayscale dithering with bright white ink on black paper: 256x192 frames -> ZX screens."""
    if (frames.height, frames.width) != (192, 256):
        raise ValueError(f"expected 256x192 frames, got {frames.rgb.shape}")
    if progress:
        progress(None, f"dithering ({dither})")
    pixels = img2scr.dither(frames.gray.astype(np.float32) / 255.0, dither)
    return Screen(pixels, np.full((len(frames), 24, 32), 0x47, dtype=np.uint8), frames.timing)


def convert_exact(frames: Image) -> Screen:
    """Pre-quantised pixel art (at most two ZX colours per 8x8 block) -> ZX screens, byte-exact with png2scr."""
    return img2scr.quantize_exact(np.ascontiguousarray(frames.rgb), frames.timing)


def crop_cells(patch: Patch, row: Annotated[int, meta(min=0, max=23)] = 0,
               col: Annotated[int, meta(min=0, max=31)] = 0,
               rows: Annotated[int, meta(min=1, max=24)] = 24,
               cols: Annotated[int, meta(min=1, max=32)] = 32,
               in_frame: Annotated[int, meta(min=0)] = 0,
               out_frame: Annotated[int, meta(min=-1, help="inclusive last frame; -1 = end")] = -1) -> Patch:
    """Cell-aligned window of a Patch (a Screen is one): `rows` x `cols` cells from (`row`, `col`)."""
    span = _frame_range(len(patch), in_frame, out_frame)
    patch = replace(patch, pixels=patch.pixels[span], attrs=patch.attrs[span],
                    timing=Timing(patch.timing.durations_ms[span]),
                    mask=None if patch.mask is None else patch.mask[span])
    return patch.crop(row, col, rows, cols)


def merge_repeats(patch: Patch) -> Patch:
    """Merge each run of identical consecutive frames into one frame whose duration is the run's total; unknown
    durations count as the encoders' default (25 fps). A Screen stays a Screen (dataclasses.replace)."""
    if len(patch) == 0:
        return patch
    pixels, attrs, mask = patch.pixels, patch.attrs, patch.mask
    same = (pixels[1:] == pixels[:-1]).all(axis=(1, 2)) & (attrs[1:] == attrs[:-1]).all(axis=(1, 2))
    if mask is not None:
        same &= (mask[1:] == mask[:-1]).all(axis=(1, 2))
    starts = np.flatnonzero(np.concatenate([[True], ~same]))
    durations = np.asarray(patch.timing.durations_ms, float) if patch.timing.known else np.full(len(patch), 1000 / 25)
    sums = np.add.reduceat(durations, starts)
    return replace(patch, pixels=pixels[starts], attrs=attrs[starts],
                   mask=(mask[starts] if mask is not None else None),
                   timing=Timing(tuple(float(x) for x in sums)))


def tiles_identity(patch: Patch) -> TileMap:
    """Identity tilemap of any Patch: every distinct 8x8 cell becomes a tile, the empty cell is tile 0."""
    return TileMap.from_patch(patch)


def tiles_quantize(tilemap: TileMap, tileset: Tileset) -> TileMap:
    """Map every cell of `tilemap` onto the nearest tile of `tileset` (Hamming distance, ties to lowest index)."""
    return quantize_tilemap(tilemap, tileset)


def load_tileset(path: Annotated[Optional[Path], meta(editor="file")] = None) -> Tileset:
    """Load a tileset (8 bytes per tile) from a file."""
    if path is None:
        raise ValueError("no tileset path")
    return Tileset.load(path)


def load_fmf(path: Annotated[Optional[Path], meta(editor="file")] = None,
             start: Annotated[str, meta(help="first frame: ff, ss:ff or mm:ss:ff at 50 fps")] = "0:00",
             end: Annotated[str, meta(help="frame after the last one, same notation; empty = the end of the recording")] = "",
             progress=None) -> Screen:
    """A Fuse Movie File (.fmf, FuseX File > Movie > Record): the screen memory of the recorded frames [start, end),
    20 ms per emulator frame. No emulator needed."""
    if path is None:
        raise ValueError("no movie path")
    first, last = parse_timecode(start), parse_timecode(end) if end else None
    if last is not None and not first < last:
        raise ValueError(f"empty frame range {start}..{end}")
    return fmf.read_fmf(path, first, last, progress)
