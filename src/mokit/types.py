"""Port types of the conversion pipelines that do not depend on a platform: `docs/types.md` is the design.

Every visual type is batched (leading N axis) and carries a `Timing`; a still is N = 1. ZX raster types live in
`mokit.zx.screen` (`Patch`, `Screen`) and `mokit.zx.tiles` (`Tileset`, `TileMap`). Arrays make dataclass equality
ambiguous, so every type here is `eq=False` (identity); results are compared by graph key, never by value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

__all__ = ["Timing", "Image", "Stream", "Export", "per_frame", "TICK_HZ", "parse_timecode", "timecode"]

TICK_HZ = 50   # the Spectrum frame interrupt


def parse_timecode(text: str, fps: int = TICK_HZ) -> int:
    """Ticks from "ff", "ss:ff" or "mm:ss:ff" at `fps`; ValueError for anything else."""
    fields = [f.strip() for f in text.split(":")]
    if not 1 <= len(fields) <= 3 or not all(f.isdigit() for f in fields):
        raise ValueError(f"bad timecode {text!r}: use ff, ss:ff or mm:ss:ff")
    numbers = [int(f) for f in fields]
    if len(numbers) > 1 and numbers[-1] >= fps:
        raise ValueError(f"bad timecode {text!r}: frames run 0..{fps - 1}")
    frames = 0
    for n in numbers[:-1]:
        frames = frames * 60 + n
    return frames * fps + numbers[-1] if len(numbers) > 1 else numbers[-1]


def timecode(ticks: int, fps: int = TICK_HZ) -> str:
    """`ticks` as "m:ss:ff" at `fps`, the form `parse_timecode` reads back."""
    seconds, frames = divmod(int(ticks), fps)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}:{frames:02d}"


@dataclass(frozen=True)
class Timing:
    """Per-frame durations in ms are the truth (gif/apng give them); 0 means unknown. `fps` and 50 Hz ticks are
    derived. Rounding to ticks is an encoder decision (it changes bytes), so encoders take an override param."""
    durations_ms: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "durations_ms", tuple(float(d) for d in self.durations_ms))

    def __len__(self) -> int:
        return len(self.durations_ms)

    @classmethod
    def unknown(cls, n: int) -> "Timing":
        return cls((0,) * n)

    @classmethod
    def constant(cls, n: int, fps: float) -> "Timing":
        return cls((1000 / fps if fps > 0 else 0,) * n)

    @classmethod
    def from_ticks(cls, ticks: int, n: int, hz: int = TICK_HZ) -> "Timing":
        return cls((1000 * ticks / hz,) * n)

    @property
    def known(self) -> bool:
        return any(d > 0 for d in self.durations_ms)

    @property
    def fps(self) -> float:
        """1000 / median known duration; 0.0 when unknown."""
        known = [d for d in self.durations_ms if d > 0]
        return 1000.0 / float(np.median(known)) if known else 0.0

    @property
    def average_fps(self) -> float:
        """Known frames per second of known time (frames / total duration); 0.0 when unknown. Differs from `fps`
        (the median rate) on variable frame rate: one long hold pulls the average down, not the median."""
        known = [d for d in self.durations_ms if d > 0]
        return 1000.0 * len(known) / sum(known) if known else 0.0

    def ticks(self, hz: int = TICK_HZ, default: int = 2) -> Tuple[int, ...]:
        """Whole ticks per frame, rounding cumulative time so drift does not accumulate; `default` (25 fps at 50 Hz)
        for unknown frames; never below 1."""
        out, elapsed, previous = [], 0, 0
        for d in self.durations_ms:
            if d <= 0:
                out.append(default)
                continue
            elapsed += d
            current = round(elapsed * hz / 1000)
            out.append(max(1, current - previous))
            previous = current
        return tuple(out)

    def __getitem__(self, item) -> "Timing":
        picked = self.durations_ms[item]
        return Timing(picked if isinstance(picked, tuple) else (picked,))


def _check_timing(timing: Timing, n: int, what: str) -> None:
    if len(timing) != n:
        raise ValueError(f"{what}: {n} frames but timing has {len(timing)}")


@dataclass(frozen=True, eq=False)
class Image:
    """True-colour raster sequence. Always RGBA `(N,H,W,4) uint8`: readers normalise grayscale, RGB and palette
    files into it with alpha 255. Colour ops read and write `rgb` and leave alpha alone; resampling premultiplies
    alpha (`mokit.image`). Alpha becomes a `Patch` mask at the Image -> Patch step."""
    rgba: np.ndarray
    timing: Timing

    def __post_init__(self) -> None:
        if self.rgba.ndim != 4 or self.rgba.shape[-1] != 4 or self.rgba.dtype != np.uint8:
            raise ValueError(f"Image.rgba must be (N,H,W,4) uint8, got {self.rgba.shape} {self.rgba.dtype}")
        _check_timing(self.timing, len(self.rgba), "Image")

    def __len__(self) -> int:
        return len(self.rgba)

    @property
    def height(self) -> int:
        return self.rgba.shape[1]

    @property
    def width(self) -> int:
        return self.rgba.shape[2]

    @property
    def rgb(self) -> np.ndarray:
        """(N,H,W,3) view; `np.ascontiguousarray` it before handing it to cv2 / PIL."""
        return self.rgba[..., :3]

    @property
    def alpha(self) -> np.ndarray:
        return self.rgba[..., 3]

    @property
    def gray(self) -> np.ndarray:
        """(N,H,W) uint8 Rec. 601 luma."""
        rgb = self.rgb.astype(np.float32)
        return np.clip(rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32) + 0.5, 0, 255).astype(np.uint8)

    @classmethod
    def from_rgb(cls, rgb: np.ndarray, timing: Optional[Timing] = None, alpha: Optional[np.ndarray] = None) -> "Image":
        """Build from (N,H,W,3) or a single (H,W,3); alpha defaults to opaque."""
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim == 3:
            rgb = rgb[np.newaxis]
        if alpha is None:
            alpha = np.full(rgb.shape[:-1], 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, np.asarray(alpha, dtype=np.uint8)[..., np.newaxis]], axis=-1)
        return cls(np.ascontiguousarray(rgba), timing if timing is not None else Timing.unknown(len(rgb)))

    def with_rgb(self, rgb: np.ndarray) -> "Image":
        """The same frames with new colour and the old alpha: what every colour op returns."""
        rgba = self.rgba.copy()
        rgba[..., :3] = rgb
        return Image(rgba, self.timing)


def per_frame(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Lift a function over one frame (H,W,...) to the batch axis: `per_frame(f)(arr, *a) == stack(f(x, *a))`."""
    def lifted(frames: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.stack([fn(frame, *args, **kwargs) for frame in frames])
    lifted.__name__ = getattr(fn, "__name__", "per_frame")
    return lifted


@dataclass(frozen=True, eq=False)
class Stream:
    """Bytes for a player: what every encoder returns. Encoders subclass this with their blobs and stats; the base
    is what export, previews and the history read."""
    name: str
    format: str            # codec code: "v3.4", "anim"
    stats: Any             # self-describing (frames, total_compressed, rows(), SERIES...)

    @property
    def frames(self) -> int:
        return int(getattr(self.stats, "frames", 0))

    def decode(self):
        """The visual result the player produces from these bytes (a Patch or TileMap), computed on demand; None
        when the encoder cannot decode. The base returns None; encoders override."""
        return None


@dataclass(frozen=True)
class Export:
    """Files written by a writer op: (name, size) pairs, `index.inc` first when there is one."""
    out_dir: str
    files: Tuple[Tuple[str, int], ...]
