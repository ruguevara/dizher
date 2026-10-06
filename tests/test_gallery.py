"""Run: .venv/bin/python tests/test_gallery.py (or pytest). The gallery's images and icons, without a window."""
import time
from dataclasses import replace

import numpy as np

from mokit.graph import Memo, evaluate

from dizher.converter.dither import ErrorDiffusion, Ordered
from dizher.ui.gallery import PICKERS, Thumbnails, halftoned, halftoner, ramp

from test_ops import pipeline


def test_ramp_goes_from_paper_to_ink():
    for label, (_, groups) in PICKERS.items():
        for name in (names[0] for names in groups.values() if names):
            r = ramp(label, name, 16)
            assert r.shape == (16, 64) and r.dtype == bool
            assert r[:, :4].mean() < 0.25 and r[:, -4:].mean() > 0.75, (label, name)


def test_gallery_image_of_the_current_pattern_is_the_halftone():
    """The current pattern's gallery image is Halftone's result; another pattern keeps the pairs."""
    memo = Memo()
    graph = pipeline()
    conv = evaluate(graph, 'halftone', memo)
    np.testing.assert_array_equal(halftoned(conv, halftoner(Ordered.label, 'Void dispersed dots')), conv.dithered_result)
    other = halftoned(conv, halftoner(Ordered.label, 'Bayer 4x4'))
    assert not np.array_equal(other, conv.dithered_result)
    assert ((other == conv.best_paper) | (other == conv.best_ink)).all()   # each pixel its cell's paper or ink
    assert conv.dithered_result is evaluate(graph, 'halftone', memo).dithered_result   # the stage's result untouched


def test_thumbnails_fill_in_and_drop_on_a_new_source():
    memo = Memo()
    conv = evaluate(pipeline(), 'halftone', memo)
    thumbs = Thumbnails()
    names = ('Bayer 2x2', 'Bayer 4x4')
    deadline = time.monotonic() + 30
    while set(thumbs.images) != set(names) and time.monotonic() < deadline:
        thumbs.want(conv, Ordered.label, (0, 0), names)
        time.sleep(0.01)
    assert set(thumbs.images) == set(names)
    assert thumbs.images['Bayer 2x2'].shape == (192, 256, 3) and thumbs.images['Bayer 2x2'].dtype == np.uint8
    thumbs.want(conv, Ordered.label, (1, 0), ())   # another origin
    assert thumbs.images == {}
    other = evaluate(pipeline().with_params('halftone', replace(pipeline()['halftone'].params, dithering=0.5)),
                     'halftone', memo)
    thumbs.want(other, ErrorDiffusion.label, (0, 0), ('Sierra lite',))
    deadline = time.monotonic() + 30
    while 'Sierra lite' not in thumbs.images and time.monotonic() < deadline:
        time.sleep(0.01)
    assert set(thumbs.images) == {'Sierra lite'}
    assert not thumbs.pending


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(name, 'ok')
