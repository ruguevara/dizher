"""Levels per channel, Curves and their palette-targeted eyedroppers (tone.py, ops.levels). Run: pytest tests/test_tone.py"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mokit import project
from mokit.graph import Op

from dizher import ops, tone
from dizher.platforms import c64, zxspectrum

IMAGES = Path(__file__).parent / 'images'
LEVELS = Op.resolve('dizher.ops:levels')
LSB = 1 / 255


def old_levels(rgb, in_black=0, in_white=255, gamma=1.0, out_black=0, out_white=255):
    """Levels as 0.2.5 had them: one set over all channels."""
    if (in_black, in_white, gamma, out_black, out_white) == (0, 255, 1.0, 0, 255):
        return rgb
    t = np.clip((rgb * 255 - in_black) / (in_white - in_black), 0, 1) ** (1 / gamma)
    return ((out_black + t * (out_white - out_black)) / 255).astype(np.float32)


def image(seed=0, shape=(24, 32)):
    return np.random.default_rng(seed).uniform(0, 1, (*shape, 3)).astype(np.float32)


def run(params, rgb):
    return LEVELS((rgb,), params)


def params(**kw):
    return replace(LEVELS.default_params(), **kw)


ZX, C64 = zxspectrum.STANDARD.palette.as_float(), c64.HIRES.palette.as_float()


def test_palette_roles():
    """Black, the grey nearest L* 50 and white among a palette's greys: the Spectrum's grey is its non-bright white,
    the C64's its middle grey; a measured palette's greys are not exactly neutral."""
    assert tone.palette_roles(ZX) == (0, 7, 15)
    assert tone.palette_roles(C64) == (0, 12, 1)
    tinted = np.array([(16, 13, 17), (102, 99, 97), (205, 20, 20), (216, 210, 206)]) / 255
    assert tone.palette_roles(tinted) == (0, 1, 3)


def test_old_params_give_the_same_image():
    rgb = image()
    for old in ((0, 255, 1.0, 0, 255), (22, 209, 1.2, 0, 255), (10, 200, 0.7, 30, 240), (0, 214, 1.0, 0, 255)):
        p = params(**dict(zip(('in_black', 'in_white', 'gamma', 'out_black', 'out_white'), old)))
        np.testing.assert_array_equal(run(p, rgb), old_levels(rgb, *old))


def test_project_from_before_channels_loads_the_same():
    """jojo's project was saved with the five composite fields only: it loads without diagnostics, Levels mode, the
    new fields at their neutral defaults, the same image."""
    data = json.loads((IMAGES / 'jojo' / 'project.json').read_text())
    stored = data['nodes']['levels']['params']
    assert set(stored) == {'in_black', 'in_white', 'gamma', 'out_black', 'out_white'}
    graph, _, diagnostics = project.loads(json.dumps(data), IMAGES / 'jojo')
    p = graph['levels'].params
    assert not diagnostics and p.mode == 'Levels' and p == params(**stored)
    rgb = image(1)
    np.testing.assert_array_equal(run(p, rgb), old_levels(rgb, *stored.values()))
    again, _, _ = project.loads(project.dumps(graph, {}, IMAGES / 'jojo'), IMAGES / 'jojo')   # and round-trips
    assert again['levels'].params == p


def test_session_from_before_channels_restores():
    """The window's session keeps a graph in the same JSON, with absolute paths: an old one restores the same way."""
    from dizher.ui.app import Pipeline
    data = project.graph_to_json(ops.make_graph(), None, None)
    data['nodes']['levels']['params'] = {'in_black': 5, 'in_white': 250, 'gamma': 0.9, 'out_black': 0, 'out_white': 255}
    host = Pipeline()
    host.restore(project.graph_from_json(json.loads(json.dumps(data)), None)[0])
    assert host.graph['levels'].params == params(in_black=5, in_white=250, gamma=0.9)
    host.close()


def test_channels_follow_the_composite():
    rgb = image(2)
    chans = ((10.0, 240.0, 1.3, 5.0, 250.0), tuple(map(float, tone.NEUTRAL)), (0.0, 200.0, 0.8, 0.0, 255.0))
    p = params(in_black=20, in_white=230, gamma=1.1, channels=chans)
    composite = old_levels(rgb, 20, 230, 1.1)
    want = np.stack([old_levels(composite[..., c], *chans[c]) for c in range(3)], -1)
    np.testing.assert_allclose(run(p, rgb), want, atol=1e-6)


@pytest.mark.parametrize('palette', [ZX, C64])
def test_levels_eyedroppers_hit_the_target(palette):
    """Black, then white, then grey, each on its own patch: every one lands within 1/255 of its palette colour."""
    black, grey, white = tone.palette_roles(palette)
    rgb = image(3)
    patches = {'black': (0.12, 0.10, 0.15), 'white': (0.85, 0.9, 0.8), 'grey': (0.45, 0.5, 0.38)}
    targets = {'black': palette[black], 'grey': palette[grey], 'white': palette[white]}
    p = params(in_black=4, in_white=250, gamma=1.1)
    for role in ('black', 'white', 'grey'):
        p = replace(p, channels=tone.levels_pick((p.in_black, p.in_white, p.gamma, p.out_black, p.out_white),
                                                 p.channels, role, patches[role], targets[role]))
    for role in ('black', 'white', 'grey'):   # grey set last must not move the ends
        got = run(p, np.array([[patches[role]]], np.float32))[0, 0]
        assert np.abs(got - targets[role]).max() <= LSB, (role, got * 255, targets[role] * 255)


def test_levels_grey_to_a_colour_clamps_what_it_cannot_reach():
    """A grey click aimed at non-bright cyan: green and blue land, red (0) cannot be reached by a gamma and goes as
    dark as the gamma range allows."""
    cyan = ZX[5]
    chans = tone.levels_pick(tone.NEUTRAL, (tone.NEUTRAL,) * 3, 'grey', (0.4, 0.5, 0.6), cyan)
    got = run(params(channels=chans), np.array([[(0.4, 0.5, 0.6)]], np.float32))[0, 0]
    assert np.abs(got[1:] - cyan[1:]).max() <= LSB and chans[0][2] == tone.GAMMA_RANGE[0], (got, chans)


def test_levels_to_curves():
    """Switching to Curves: the curves do what the levels did, within 1/255 at every 8-bit input (between those the
    256-entry LUT is straight, and a strong gamma near the black point is steeper than that)."""
    ramp = np.repeat(np.arange(256, dtype=np.float32)[:, None, None] / 255, 3, axis=2)
    rgb = np.concatenate([ramp, (np.round(image(4, (256, 1)) * 255) / 255)], axis=1)
    for composite, chans in (((0, 255, 1.0, 0, 255), ((0.0, 255.0, 1.0, 0.0, 255.0),) * 3),
                             ((22, 209, 1.2, 0, 255), ((10.0, 240.0, 0.6, 5.0, 250.0), (0.0, 255.0, 2.5, 0.0, 255.0),
                                                       (30.5, 200.2, 3.0, 20.0, 230.0))),
                             ((10, 245, 0.5, 30, 220), ((0.0, 255.0, 1.0, 255.0, 0.0),) * 3)):   # inverted output
        lv = params(**dict(zip(('in_black', 'in_white', 'gamma', 'out_black', 'out_white'), composite)), channels=chans)
        cv = params(mode='Curves', curves=tone.levels_to_curves(composite, chans))
        assert all(len(c) <= 2 * tone.MAX_POINTS for c in cv.curves)
        assert np.abs(run(lv, rgb) - run(cv, rgb)).max() <= LSB, composite


def test_curves_identity_and_pchip():
    rgb = image(5)
    assert run(params(mode='Curves'), rgb) is rgb
    lut = tone.curve_lut((30.0, 20.0, 100.0, 60.0, 110.0, 200.0, 220.0, 230.0))
    assert np.all(np.diff(lut) >= 0) and np.ptp(lut[:31]) == 0 and np.ptp(lut[220:]) == 0   # monotone, flat beyond the ends
    assert lut.max() <= 230 / 255 + 1e-6 and lut.min() >= 20 / 255 - 1e-6              # no overshoot
    with pytest.raises(ValueError):
        tone.curve_lut((0.0, 0.0, 0.0, 255.0))


def curves_run(curves, colour):
    return run(params(mode='Curves', curves=curves), np.array([[colour]], np.float32))[0, 0]


@pytest.mark.parametrize('palette', [ZX, C64])
def test_curves_eyedroppers(palette):
    """Black and white move the end points, grey adds a point per channel; each maps its sample onto the palette
    colour, up to the 256-entry LUT's straight steps (a kink between two entries: under 1/255). Several grey points pull several colours."""
    black, grey, white = tone.palette_roles(palette)
    curves = ((10.0, 0.0, 128.0, 140.0, 255.0, 255.0),) + (tone.IDENTITY,) * 3   # a composite to see through
    picks = ()
    curves, picks = tone.curves_end(curves, picks, 'black', (0.1, 0.12, 0.08), palette[black])
    curves, picks = tone.curves_end(curves, picks, 'white', (0.9, 0.85, 0.95), palette[white])
    samples = [((0.45, 0.5, 0.4), grey), ((0.3, 0.55, 0.7), 5), ((0.7, 0.4, 0.3), 2)]
    for s, i in samples:
        curves, picks = tone.curves_grey(curves, picks, s, palette[i], i)
    assert len(picks) == 3 and all(p[6] == i for p, (_, i) in zip(picks, samples))
    for s, t in [((0.1, 0.12, 0.08), palette[black]), ((0.9, 0.85, 0.95), palette[white])] + [(s, palette[i]) for s, i in samples]:
        got = curves_run(curves, s)
        assert np.abs(got - t).max() <= LSB, (s, got * 255, t * 255)
    assert curves[0] == (10.0, 0.0, 128.0, 140.0, 255.0, 255.0)   # the composite is left alone


def test_curves_grey_conflict_replace_and_drop():
    """A point whose samples and targets are ordered differently in a channel is added and marked; one within MERGE
    of another point's input replaces it there and says so; deleting a group takes its points away."""
    curves, picks = (tone.IDENTITY,) * 4, ()
    curves, picks = tone.curves_grey(curves, picks, (0.3, 0.3, 0.3), (0.5, 0.5, 0.5), 7)
    curves, picks = tone.curves_grey(curves, picks, (0.6, 0.301, 0.6), (0.4, 0.2, 0.7), 1)   # R: higher in, lower out
    assert [tone.pick_conflicts(curves, p) for p in picks] == [0b001, 0b001]
    assert picks[1][tone.PICK_REPLACED] == 0b010 and picks[0][tone.PICK_X][1] == -1   # G of the first was replaced
    assert len(tone.points(curves[2])) == 3
    got = curves_run(curves, (0.6, 0.301, 0.6))
    assert np.abs(got - (0.4, 0.2, 0.7)).max() <= 0.5 * LSB, got * 255   # a conflict still maps its own sample
    curves, picks = tone.curves_drop(curves, picks, 1)
    assert len(picks) == 1 and tone.pick_conflicts(curves, picks[0]) == 0
    assert curves[2] == tone.IDENTITY and len(tone.points(curves[1])) == 3   # G lost both groups' points
    curves, picks = tone.curves_drop(curves, picks, 0)
    assert curves == (tone.IDENTITY,) * 4 and picks == ()


def test_curve_edit_carries_the_group():
    curves, picks = tone.curves_grey((tone.IDENTITY,) * 4, (), (0.4, 0.5, 0.6), (0.5, 0.5, 0.5), 7)
    x = picks[0][tone.PICK_X][0]
    curves, picks = tone.curve_edit(curves, picks, 1, x, (x + 10, 140.0))
    assert picks[0][tone.PICK_X][0] == round(x + 10, 2) and (round(x + 10, 2), 140.0) in map(tuple, tone.points(curves[1]))
    for c in (1, 2, 3):
        curves, picks = tone.curve_edit(curves, picks, c, picks[0][tone.PICK_X][c - 1], None) if picks else (curves, picks)
    assert picks == ()   # a group with no points left goes


def test_sample_averages_and_clips():
    rgb = np.zeros((10, 10, 3), np.float32)
    rgb[:5, :5] = 1
    assert np.allclose(tone.sample(rgb, 2, 2), 1) and np.allclose(tone.sample(rgb, 0, 0), 1)   # 2x2 at the corner
    assert np.allclose(tone.sample(rgb, 4, 4), 4 / 9)   # 3x3, 2x2 of it lit
