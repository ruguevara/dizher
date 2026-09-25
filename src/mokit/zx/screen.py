"""ZX Spectrum screen (.scr) data: unpacked pixels and attribute planes.

Canonical copy for every Python tool in the ecosystem (AmaZX, ToolZX, dizher).
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from mokit.types import Timing

# byte offset of pixel line y (column 0) in the interleaved 6144-byte bitmap
_LINE_OFFSET = (lambda y: ((y & 0xC0) << 5) | ((y & 0x07) << 8) | ((y & 0x38) << 2))(np.arange(192))


def parse_scr_pixels_data(data: np.ndarray) -> np.ndarray:
    """Unpack the 6144-byte interleaved bitmap into a (192, 256) bool array."""
    assert data.shape == (6144,), data.shape
    return np.unpackbits(data[_LINE_OFFSET[:, np.newaxis] + np.arange(32)], axis=1).astype(bool)


def parse_scr_attr_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split 768 attribute bytes into ink (24,32,3) RGB bools, paper, bright (24,32), flash."""
    assert data.shape == (768,), data.shape
    unpacked = np.unpackbits(data).reshape(24, 32, 8)
    flash  = unpacked[..., 0].astype(bool)
    bright = unpacked[..., 1].astype(bool)
    paper  = unpacked[..., [3, 2, 4]]
    ink    = unpacked[..., [6, 5, 7]]
    return ink, paper, bright, flash


def encode_scr_attrs(ink: np.ndarray, paper: np.ndarray, bright: np.ndarray, flash: np.ndarray) -> np.ndarray:
    """
    Encode separate ink, paper, bright and flash arrays into attr bytes.
    ink/paper may be integer colour indices (shape == bright.shape, encoded as B|(R<<1)|(G<<2))
    or RGB boolean arrays (bright.shape + (3,)).
    Returns (*leading, H*W) uint8 attr bytes.
    """
    if ink.ndim == bright.ndim + 1:  # RGB boolean arrays -> integer indices
        ink   = ink[..., 2] | (ink[..., 0] << 1) | (ink[..., 1] << 2)
        paper = paper[..., 2] | (paper[..., 0] << 1) | (paper[..., 1] << 2)
    assert ink.shape == paper.shape == bright.shape == flash.shape
    attr = (flash.astype(np.uint8) << 7) | (bright.astype(np.uint8) << 6) \
         | ((paper.astype(np.uint8) & 7) << 3) | (ink.astype(np.uint8) & 7)
    return attr.reshape(*bright.shape[:-2], -1)


def parse_scr(data: np.ndarray) -> 'ZXScreen':
    """Parse 6912 bytes of screen memory into a ZXScreen."""
    ink, paper, bright, flash = parse_scr_attr_data(data[6144:])
    return ZXScreen(parse_scr_pixels_data(data[:6144]), ink, paper, bright, flash)


def load_scr(filename) -> 'ZXScreen':
    return ZXScreen.from_data(np.fromfile(filename, dtype=np.uint8))


class ZXScreen:
    """ZX Spectrum screen in separate pixels and attr colour planes."""
    _block881 = np.ones((8, 8, 1), dtype=np.uint8)

    def __init__(self, pixels, attr_ink, attr_paper, attr_bright, attr_flash=None):
        assert len(pixels.shape) == 2, pixels.shape
        assert len(attr_ink.shape) == 3, attr_ink.shape
        assert len(attr_paper.shape) == 3, attr_paper.shape
        assert len(attr_bright.shape) == 2, attr_bright.shape
        assert attr_ink.shape[:2] == attr_paper.shape[:2], (attr_ink.shape, attr_paper.shape)
        assert attr_ink.shape[:2] == attr_bright.shape, (attr_ink.shape, attr_bright.shape)
        if attr_flash is not None:
            assert len(attr_flash.shape) == 2, attr_flash.shape
            assert attr_ink.shape[:2] == attr_flash.shape, (attr_ink.shape, attr_flash.shape)
        self.pixels = pixels
        self.attr_ink = attr_ink
        self.attr_paper = attr_paper
        self.attr_bright = attr_bright
        self.attr_flash = attr_flash if attr_flash is not None else np.zeros_like(attr_bright)

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def attrs_shape(self):
        return self.attr_ink.shape[:2]

    def crop(self, top: int, left: int, height: int, width: int) -> None:
        self.pixels = self.pixels[top:top + height, left:left + width]
        rows, cols = slice(top // 8, top // 8 + height // 8), slice(left // 8, left // 8 + width // 8)
        self.attr_ink = self.attr_ink[rows, cols]
        self.attr_paper = self.attr_paper[rows, cols]
        self.attr_bright = self.attr_bright[rows, cols]

    @classmethod
    def empty(cls, height=192, width=256):
        rows, cols = height // 8, width // 8
        return cls(
            pixels=np.zeros((height, width), dtype=bool),
            attr_ink=np.zeros((rows, cols, 3), dtype=bool),
            attr_paper=np.zeros((rows, cols, 3), dtype=bool),
            attr_bright=np.zeros((rows, cols), dtype=bool),
            attr_flash=np.zeros((rows, cols), dtype=bool),
        )

    @classmethod
    def empty_like(cls, screen):
        return cls.empty(*screen.pixels.shape)

    @classmethod
    def from_data(cls, data):
        return parse_scr(data)

    def get_attrs(self):
        rows, cols = self.attr_paper.shape[:2]
        return np.packbits(np.concatenate([
            self.attr_flash [..., np.newaxis],
            self.attr_bright[..., np.newaxis],
            self.attr_paper [..., [1, 0, 2]],
            self.attr_ink   [..., [1, 0, 2]],
        ], axis=2)).reshape(rows, cols)

    def rgb_ink_paper(self):
        bright_level = np.where(self.attr_bright[..., np.newaxis], 255, 205).astype(np.uint8)
        ink = np.kron(self.attr_ink * bright_level, self._block881).astype(np.uint8)
        paper = np.kron(self.attr_paper * bright_level, self._block881).astype(np.uint8)
        return ink, paper

    _attr_icon = np.array([   # 8x8 ink droplet drawn in ink on paper (attrs display mode)
        "........",
        "........",
        "...##...",
        "..####..",
        "..####..",
        "...##...",
        "........",
        "........",
    ]).view('U1').reshape(8, 8) == '#'

    def rgb(self, mode='screen'):
        """RGB image; mode 'screen' (as displayed), 'bitmap' (pixels, white on black) or 'attrs' (ink droplet on paper)."""
        if mode == 'bitmap':
            return np.where(self.pixels[..., np.newaxis], np.uint8(255), np.uint8(0)).repeat(3, axis=2)
        ink, paper = self.rgb_ink_paper()
        if mode == 'attrs':
            rows, cols = self.attrs_shape
            return np.where(np.tile(self._attr_icon, (rows, cols))[..., np.newaxis], ink, paper)
        return np.where(self.pixels[..., np.newaxis], ink, paper)

    def char(self, row, col):
        return self.pixels[row * 8: row * 8 + 8, col * 8: col * 8 + 8]

    def copy(self):
        return self.__class__(self.pixels, self.attr_ink, self.attr_paper, self.attr_bright, self.attr_flash)

    def to_bytes(self) -> bytes:
        """6912-byte .scr form: interleaved bitmap then attrs. Inverse of `parse_scr`."""
        bitmap = np.zeros(6144, dtype=np.uint8)
        bitmap[_LINE_OFFSET[:, np.newaxis] + np.arange(32)] = np.packbits(self.pixels, axis=1)
        return bitmap.tobytes() + self.get_attrs().tobytes()


def _natural_key(name: str) -> tuple:
    """Numbers inside a file name compare numerically (frame2 < frame10)."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


class ZXPalette:
    """The 16 ZX Spectrum colours as RGB uint8, ported from dizher `converter/palette.py`.

    `rgb` is (16, 3): index 0..7 are the normal-brightness colours, 8..15 the bright ones; within each
    half the colour index is `(green << 2) | (red << 1) | blue` (same convention as `ZXScreen`'s
    ink/paper integer indices and `encode_scr_attrs`). `not_bright_level`/`bright_level` are the two
    channel levels (205/255, matching `rgb_ink_paper`).
    """
    not_bright_level = 205
    bright_level = 255

    def __init__(self):
        rgb = np.empty((2, 8, 3), dtype=np.uint8)
        i = 0
        for g in range(2):
            for r in range(2):
                for b in range(2):
                    rgb[0, i] = (r * self.not_bright_level, g * self.not_bright_level, b * self.not_bright_level)
                    rgb[1, i] = (r * self.bright_level, g * self.bright_level, b * self.bright_level)
                    i += 1
        self.rgb = rgb.reshape(-1, 3)


def unpack_attrs(attrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(..., R, C) attr bytes -> ink (..., R, C, 3) RGB bools, paper, bright (..., R, C), flash. Any size."""
    bits = np.unpackbits(np.asarray(attrs, dtype=np.uint8)[..., np.newaxis], axis=-1)
    return bits[..., [6, 5, 7]].astype(bool), bits[..., [3, 2, 4]].astype(bool), \
        bits[..., 1].astype(bool), bits[..., 0].astype(bool)


@dataclass(frozen=True, eq=False)
class Patch:
    """A cell-aligned piece of ZX screen as a sequence: pixels (N, R*8, C*8) bool, attrs (N, R, C) packed attr
    bytes, optional mask (N, R*8, C*8) bool (a sprite is a Patch with a mask), and a `Timing`.
    `Screen` is the fullscreen special case. See `docs/types.md`."""
    pixels: np.ndarray
    attrs: np.ndarray
    timing: Timing
    mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        pixels, attrs = np.asarray(self.pixels, dtype=bool), np.asarray(self.attrs, dtype=np.uint8)
        if pixels.ndim != 3 or attrs.ndim != 3 or pixels.shape != (attrs.shape[0], attrs.shape[1] * 8, attrs.shape[2] * 8):
            raise ValueError(f"{type(self).__name__}: pixels {pixels.shape} do not match attrs {attrs.shape} cells")
        if self.mask is not None and np.asarray(self.mask).shape != pixels.shape:
            raise ValueError(f"{type(self).__name__}: mask {np.asarray(self.mask).shape} != pixels {pixels.shape}")
        if len(self.timing) != len(pixels):
            raise ValueError(f"{type(self).__name__}: {len(pixels)} frames but timing has {len(self.timing)}")
        object.__setattr__(self, "pixels", pixels)
        object.__setattr__(self, "attrs", attrs)
        if self.mask is not None:
            object.__setattr__(self, "mask", np.asarray(self.mask, dtype=bool))

    def __len__(self) -> int:
        return len(self.pixels)

    @property
    def rows(self) -> int:
        return self.attrs.shape[1]

    @property
    def cols(self) -> int:
        return self.attrs.shape[2]

    def screen(self, i: int) -> ZXScreen:
        ink, paper, bright, flash = unpack_attrs(self.attrs[i])
        return ZXScreen(self.pixels[i], ink, paper, bright, flash)

    def rgb(self, i: int, mode: str = 'screen') -> np.ndarray:
        return self.screen(i).rgb(mode)

    def with_timing(self, timing: Timing) -> 'Patch':
        return dataclasses.replace(self, timing=timing)

    def crop(self, row: int, col: int, rows: int, cols: int) -> 'Patch':
        """The cell-aligned window (`row`, `col`) of `rows` x `cols` cells as a Patch (a Screen stays a Screen only
        when nothing is cut). Out-of-range windows are a ValueError."""
        if row < 0 or col < 0 or rows < 1 or cols < 1 or row + rows > self.rows or col + cols > self.cols:
            raise ValueError(f"crop {rows}x{cols} at ({row}, {col}) does not fit {self.rows}x{self.cols} cells")
        if (row, col, rows, cols) == (0, 0, self.rows, self.cols):
            return self
        px = slice(row * 8, (row + rows) * 8), slice(col * 8, (col + cols) * 8)
        return Patch(self.pixels[:, px[0], px[1]], self.attrs[:, row:row + rows, col:col + cols], self.timing,
                     None if self.mask is None else self.mask[:, px[0], px[1]])


@dataclass(frozen=True, eq=False)
class Screen(Patch):
    """Fullscreen 24x32 cells, no mask: the ZX display and the `.scr` file form (folder of frame_NNN.scr)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attrs.shape[1:] != (24, 32) or self.mask is not None:
            raise ValueError(f"Screen must be 24x32 cells without a mask, got {self.attrs.shape[1:]}")

    def to_bytes(self, i: int) -> bytes:
        return self.screen(i).to_bytes()

    @classmethod
    def from_screens(cls, screens: List[ZXScreen], timing: Optional[Timing] = None) -> 'Screen':
        pixels = np.stack([s.pixels for s in screens])
        attrs = np.stack([s.get_attrs() for s in screens])
        return cls(pixels, attrs, timing if timing is not None else Timing.unknown(len(screens)))

    @classmethod
    def from_bytes_list(cls, data: List[bytes], timing: Optional[Timing] = None) -> 'Screen':
        return cls.from_screens([parse_scr(np.frombuffer(d, dtype=np.uint8)) for d in data], timing)

    @classmethod
    def load_folder(cls, folder) -> 'Screen':
        paths = sorted(Path(folder).glob("*.scr"), key=lambda p: _natural_key(p.name))
        if not paths:
            raise FileNotFoundError(f"no .scr files in {folder}")
        return cls.from_bytes_list([p.read_bytes() for p in paths])

    def save_folder(self, folder, template: str = "frame_{:03d}.scr") -> List[Path]:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(len(self)):
            path = folder / template.format(i)
            path.write_bytes(self.to_bytes(i))
            paths.append(path)
        return paths
