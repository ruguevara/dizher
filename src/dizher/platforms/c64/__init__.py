"""Commodore 64: 16 fixed colours, hires bitmap 320x200 with any two colours per 8x8 cell."""
import numpy as np

from .. import Mode
from ...converter.palette import Palette

# Pepto's measured VIC-II palette (pepto.de/projects/colorvic), index = VIC-II colour number.
PEPTO = [
    (0x00, 0x00, 0x00), (0xFF, 0xFF, 0xFF), (0x68, 0x37, 0x2B), (0x70, 0xA4, 0xB2),
    (0x6F, 0x3D, 0x86), (0x58, 0x8D, 0x43), (0x35, 0x28, 0x79), (0xB8, 0xC7, 0x6F),
    (0x6F, 0x4F, 0x25), (0x43, 0x39, 0x00), (0x9A, 0x67, 0x59), (0x44, 0x44, 0x44),
    (0x6C, 0x6C, 0x6C), (0x9A, 0xD2, 0x84), (0x6C, 0x5E, 0xB5), (0x95, 0x95, 0x95),
]


class C64Palette(Palette):
    # 0 black, 1 white, 11 dark grey, 12 grey, 15 light grey
    SUBSETS = {
        'Grayscale': frozenset({0, 11, 12, 15, 1}),
        'Mono': frozenset({0, 1}),
    }

    def __init__(self):
        super().__init__(PEPTO)


def to_art(bitmap: np.ndarray, idx_pairs: np.ndarray) -> bytes:
    """Art Studio hires file, the common one: load address $2000, the bitmap cell by cell (8 bytes each, top row
    first), screen RAM (high nibble the colour of set bits, low nibble of clear ones), border colour, 6 spare bytes.
    bitmap (200, 320) truthy = ink; idx_pairs (25, 40, 2) as (paper, ink) VIC-II colour numbers."""
    assert bitmap.shape == (200, 320) and idx_pairs.shape == (25, 40, 2)
    rows = np.packbits(bitmap.astype(bool), axis=1)   # (200, 40), bit 7 is the leftmost pixel
    cells = rows.reshape(25, 8, 40).transpose(0, 2, 1)   # (row, column, byte within the cell)
    screen = (idx_pairs[..., 1] << 4 | idx_pairs[..., 0]).astype(np.uint8)
    return b'\x00\x20' + cells.tobytes() + screen.tobytes() + bytes(7)   # border black


HIRES = Mode('C64 hires', size=(200, 320), cell=(8, 8), palette=C64Palette(),
             file_type=('Art Studio hires', '*.art'), encode=to_art)
