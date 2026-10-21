"""Conversion presets (src/dizher/ui/presets.py): storage, what applying one changes, and the Exact screen preset
giving a screen back pixel for pixel. Run: pytest tests/test_presets.py"""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from mokit.graph import Memo, evaluate
from mokit.zx.screen import load_scr

from dizher import ops
from dizher.ui import presets

IMAGES = Path(__file__).parent / 'images'


def edited():
    """A graph with an edit in a preset's node, the Target, the painted cells and Tune."""
    g = ops.make_graph()
    g = g.with_params('select', replace(g['select'].params, coherence=1.5))
    g = g.with_params('target', replace(g['target'].params, colours=(0, 7)))
    g = g.with_params('overpaint', replace(g['overpaint'].params, overrides=((0, 0, 1, 2),)))
    return g.with_params('contrast', replace(g['contrast'].params, contrast=20.0))


def test_roundtrip(tmp_path):
    presets.save(tmp_path, 'тест', edited())
    preset = presets.load(tmp_path, 'тест')
    assert preset.ids() == presets.NODES
    assert presets.apply(ops.make_graph(), preset)['select'].params.coherence == 1.5


def test_apply_keeps_the_rest():
    g = edited()
    applied = presets.apply(g, presets.BUILTIN['Default'][0])
    assert applied == g.with_params('select', ops.make_graph()['select'].params)
    assert presets.apply(applied, presets.of(applied)) == applied


def test_apply_skips_what_it_cannot_read():
    preset = presets.of(edited()).without('eye')   # a preset from before a stage
    g = ops.make_graph().with_params('eye', replace(ops.make_graph()['eye'].params, luma_scale=1.2))
    assert presets.apply(g, preset)['eye'] == g['eye']


def test_names_rename_delete(tmp_path):
    assert presets.names(tmp_path / 'none') == []
    for name in ('b', 'A', 'c'):
        presets.save(tmp_path, name, ops.make_graph())
    assert presets.names(tmp_path) == ['A', 'b', 'c']
    with pytest.raises(FileExistsError):
        presets.rename(tmp_path, 'A', 'b')
    presets.rename(tmp_path, 'A', 'd')
    presets.delete(tmp_path, 'b')
    assert presets.names(tmp_path) == ['c', 'd']
    assert presets.free_name(tmp_path, 'c') == 'c 2' and presets.free_name(tmp_path, 'Default') == 'Default 2'


def test_halftoned_is_the_method_preset():
    graph = presets.apply(ops.make_graph(), presets.BUILTIN['Halftoned (0.2.4)'][0])
    assert graph == ops.apply_preset(ops.make_graph(), ops.LEGACY)


@pytest.mark.parametrize('name', ['anubis', 'rocket-rackoon'])
def test_exact_screen_gives_the_screen_back(tmp_path, name):
    """A screen as a picture through the Exact screen preset: every pixel as it was."""
    screen = load_scr(IMAGES / name / 'reference.scr').rgb()
    picture = tmp_path / f'{name}.png'
    cv2.imwrite(str(picture), screen[..., ::-1])
    g = presets.apply(ops.make_graph(), presets.BUILTIN['Exact screen'][0])
    g = g.with_params('source', replace(g['source'].params, path=picture))
    out = tmp_path / f'{name}.scr'
    evaluate(g, 'optimise', Memo()).save(str(out))
    assert np.array_equal(load_scr(out).rgb(), screen)
