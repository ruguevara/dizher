"""Fuse Movie File (.fmf, FuseX's File > Movie > Record): the screen memory of every recorded frame as a `Screen`.

Format (fuse `movie.c`): a 16-byte header `FMF_V1`, endianness, `U`/`Z` (zlib) compression, frame rate, screen
type, timing code, sound format, frequency, stereo, `\\n`; then a chunk stream (one zlib stream when `Z`):
`N` (new frame: rate, screen type, timing code) followed by that frame's `$` slices (x, y, w, h of a 40x240 cell
display including the border, then two RLE planes: bitmap bytes, attribute bytes), `S` (sound, skipped) and `X`
(end). RLE: a byte written twice is followed by a count and stands for count + 2 copies; the byte right after a
count never starts a pair. Every frame is `rate` emulator frames at 50 Hz; Timex hi-res/hi-colour screens are refused.
"""
from pathlib import Path
from typing import Optional
import zlib

import numpy as np

from mokit.types import Timing
from mokit.zx.screen import Screen

SUFFIX = ".fmf"
TICK_MS = 20.0
_LINES, _COLS = 240, 40      # the display with the border, in lines x 8-pixel cells
_TOP, _LEFT = 24, 4          # where the 256x192 screen sits in it
_STANDARD = b"$X"            # screen types with one attribute byte per cell


def _rle(data: bytes, pos: int, n: int):
    """`n` decoded bytes from `data[pos:]` and the position after them."""
    out = bytearray()
    prev = -1
    while len(out) < n:
        b = data[pos]
        pos += 1
        if b == prev:
            out += bytes((b,)) * (data[pos] + 1)
            pos += 1
            prev = -1
        else:
            out.append(b)
            prev = b
    if len(out) != n:
        raise ValueError("fmf: RLE run past the slice")
    return bytes(out), pos


def read_fmf(path, start: int = 0, end: Optional[int] = None, progress=None) -> Screen:
    """The frames of a .fmf whose first tick (a 50 Hz emulator frame) lies in [start, end) as a `Screen`; `end`
    None is the end of the recording. A recording cut off before its `X` keeps the frames completed."""
    raw = Path(path).read_bytes()
    if raw[:6] != b"FMF_V1" or raw[7] not in b"UZ" or raw[9] not in _STANDARD:
        raise ValueError(f"{path}: not a standard-screen Fuse Movie File")
    data = zlib.decompressobj().decompress(raw[16:]) if raw[7] == ord("Z") else raw[16:]
    bitmap = np.zeros((_LINES, _COLS), np.uint8)
    attrs = np.zeros((_LINES, _COLS), np.uint8)
    frames, ticks, rate, tick = [], [], None, 0   # tick: where the frame being read starts, in emulator frames
    pos, size = 0, len(data)

    def close_frame():
        nonlocal tick
        if tick >= start:
            frames.append((bitmap[_TOP:_TOP + 192, _LEFT:_LEFT + 32].copy(),
                           attrs[_TOP:_TOP + 192:8, _LEFT:_LEFT + 32].copy()))
            ticks.append(rate)
        tick += rate
        if progress and len(ticks) % 100 == 0:
            progress(pos / size, f"reading frame {tick}")

    try:
        while pos < size:
            kind = data[pos]
            if kind == ord("N"):
                if rate is not None:
                    close_frame()
                    if end is not None and tick >= end:
                        break
                rate = data[pos + 1]
                if data[pos + 2] not in _STANDARD:
                    raise ValueError(f"{path}: Timex hi-res/hi-colour frames are not supported")
                pos += 4
            elif kind == ord("$"):
                x, y = data[pos + 1], data[pos + 2] | data[pos + 3] << 8
                w, h = data[pos + 4], data[pos + 5] | data[pos + 6] << 8
                pos += 7
                for plane in (bitmap, attrs):
                    run, pos = _rle(data, pos, w * h)
                    plane[y:y + h, x:x + w] = np.frombuffer(run, np.uint8).reshape(h, w)
            elif kind == ord("S"):
                samples = (data[pos + 5] | data[pos + 6] << 8) + 1
                pos += 7 + samples * (2 if data[pos + 4] == ord("S") else 1) * (2 if data[pos + 1] == ord("P") else 1)
            elif kind == ord("X"):
                if rate is not None:
                    close_frame()
                break
            else:
                raise ValueError(f"{path}: unknown chunk {chr(kind)!r} at {pos}")
    except IndexError:
        pass    # cut off mid-chunk: the frames closed so far are complete
    if not frames:
        raise ValueError(f"{path}: no frames in {start}..{'end' if end is None else end}")
    pixels = np.unpackbits(np.stack([f[0] for f in frames]), axis=-1).astype(bool)
    return Screen(pixels, np.stack([f[1] for f in frames]), Timing(tuple(TICK_MS * t for t in ticks)))
