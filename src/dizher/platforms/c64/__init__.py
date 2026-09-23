"""Commodore 64: 16 fixed colours, hires bitmap 320x200 with any two colours per 8x8 cell."""
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
        'All colours': lambda i1, i2: True,
        'Grayscale': lambda i1, i2: {i1, i2} <= {0, 11, 12, 15, 1},
        'Mono': lambda i1, i2: {i1, i2} <= {0, 1},
    }

    def __init__(self, subset='All colours'):
        super().__init__(PEPTO, subset)


HIRES = Mode('C64 hires', size=(200, 320), cell=(8, 8), palette=C64Palette())
