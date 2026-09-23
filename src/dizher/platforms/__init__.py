"""Devices and their graphics modes. A mode is the screen geometry the converter targets plus the
palette it may paint with; everything device-specific (native file format) hangs off it."""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from ..converter.palette import Palette


@dataclass
class Mode:
    name: str
    size: Tuple[int, int]       # screen (H, W) in pixels
    cell: Tuple[int, int]       # attribute cell (h, w) in pixels; every cell shows one (paper, ink) pair
    palette: Palette
    file_type: Optional[Tuple[str, str]] = None   # native screen file as (description, '*.ext'), if any
    encode: Optional[Callable[[np.ndarray, np.ndarray], bytes]] = None   # (bitmap (H, W) bool, idx_pairs (R, C, 2)) -> native bytes

    def __post_init__(self):
        assert self.size[0] % self.cell[0] == 0 and self.size[1] % self.cell[1] == 0, (self.size, self.cell)
