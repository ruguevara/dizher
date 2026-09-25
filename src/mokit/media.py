"""Read any still image, animated image, video or .scr folder into an `Image` (always RGBA, with `Timing`).

`av` (the `mokit[media]` extra) is only imported inside `read_image` for video containers, so the rest
of the package works without it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageSequence

from mokit.types import Image, Timing
from mokit.zx.screen import Screen, load_scr

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ANIMATED_SUFFIXES = {".gif", ".apng", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def natural_key(name: str) -> tuple:
    """Numbers inside a file name compare numerically (frame2 < frame10)."""
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", name))


def _is_animated_still(path: Path) -> bool:
    """A .png/.webp may or may not carry more than one frame; only trust n_frames when it says so."""
    with PILImage.open(path) as image:
        return getattr(image, "n_frames", 1) > 1


def _read_rgba(path: Path) -> np.ndarray:
    return np.asarray(PILImage.open(path).convert("RGBA"), dtype=np.uint8)


def read_image(path) -> Image:
    path = Path(path)
    suffix = path.suffix.lower()

    if path.is_dir():
        entries = sorted(path.iterdir(), key=lambda p: natural_key(p.name))
        scr_paths = [p for p in entries if p.suffix.lower() == ".scr"]
        if scr_paths:
            screen = Screen.load_folder(path)
            rgb = np.stack([screen.rgb(i) for i in range(len(screen))])
            return Image.from_rgb(rgb, Timing.unknown(len(screen)))
        rgba = np.stack([_read_rgba(p) for p in entries if p.suffix.lower() in IMAGE_SUFFIXES])
        return Image(rgba, Timing.unknown(len(rgba)))

    if suffix == ".scr":
        rgb = load_scr(path).rgb()[np.newaxis]
        return Image.from_rgb(rgb, Timing.unknown(1))

    if suffix in VIDEO_SUFFIXES:
        try:
            import av
        except ImportError:
            raise ImportError("install mokit[media] for video files")
        frames, fps = [], 0.0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate
            fps = float(rate) if rate else 0.0
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        rgb = np.stack(frames).astype(np.uint8)
        return Image.from_rgb(rgb, Timing.constant(len(frames), fps))

    if suffix in ANIMATED_SUFFIXES and _is_animated_still(path):
        image = PILImage.open(path)
        rgbas, durations = [], []
        for frame in ImageSequence.Iterator(image):
            rgbas.append(np.asarray(frame.convert("RGBA"), dtype=np.uint8))
            durations.append(frame.info.get("duration", 100) or 100)
        rgba = np.stack(rgbas)
        return Image(rgba, Timing(tuple(durations)))

    # single still image (.png/.jpg/.bmp/.webp not animated)
    rgba = _read_rgba(path)[np.newaxis]
    return Image(rgba, Timing.unknown(1))
