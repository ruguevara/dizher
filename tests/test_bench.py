"""The pair-selection bench's helpers (tests/bench): screens, renders with rulers, variants, judgments, metrics."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench import render as R                                  # noqa: E402
from bench.scr import read_scr, shown, pairs_to_labels, label_pairs, parse_pair, pair_name, has_flash   # noqa: E402
from dizher.platforms.zxspectrum import ZXPalette              # noqa: E402
from dizher.platforms.zxspectrum.scr import to_scr             # noqa: E402
from dizher.platforms import Mode                              # noqa: E402
from dizher.converter.converter import Converter               # noqa: E402


def test_scr_round_trip_and_pairs(tmp_path):
    """to_scr -> read_scr gives the bitmap and (paper, ink) back; a labelling's pairs map back to the same labels,
    the two blacks one colour; a solid cell shows one colour twice."""
    rng = np.random.default_rng(0)
    bitmap = rng.random((192, 256)) < 0.5
    bitmap[:8, :8] = True                                        # a solid ink cell
    idx = rng.integers(0, 8, (24, 32, 2))
    idx[..., 1] += 8 * (idx[..., 0] % 2)                         # ink bright where paper's index is odd: no, keep legal
    idx[..., 1] = np.where(idx[..., 0] >= 8, idx[..., 1] % 8 + 8, idx[..., 1] % 8)
    path = tmp_path / 'x.scr'
    path.write_bytes(to_scr(bitmap, idx))
    back_bitmap, back_idx = read_scr(path)
    np.testing.assert_array_equal(back_bitmap, bitmap)
    np.testing.assert_array_equal(back_idx & 7, idx & 7)
    assert not has_flash(path)
    s = shown(back_bitmap, back_idx)
    assert s[0, 0, 0] == s[0, 0, 1]
    conv = Converter({'Luma': 1.0, 'Chroma': 1.0}, Mode('zx', (192, 256), (8, 8), ZXPalette()))
    labels = rng.integers(0, len(conv.color_pairs), (24, 32))
    back = pairs_to_labels(conv, np.array(list(conv.palette.iter_idxs_pairs()))[labels])
    np.testing.assert_array_equal(label_pairs(conv, back), label_pairs(conv, labels))
    assert pairs_to_labels(conv, np.array([[[8, 8]]]))[0, 0] == pairs_to_labels(conv, np.array([[[0, 0]]]))[0, 0]
    assert parse_pair('k/Y') == (0, 14) and pair_name(0, 14) == 'k/Y' and pair_name(9, 10) == 'B/R'


def test_ruled_keeps_the_pixels():
    """The picture's pixels sit at [MARGIN:, MARGIN:], enlarged without smoothing; only the grid lines on cell borders
    change, and only by blending; a crop with an origin gets the same numbers as the whole (same ruler pixels)."""
    rng = np.random.default_rng(1)
    img = rng.random((48, 64, 3)).astype(np.float32)
    zoom = 3
    plain = R.up(img, zoom)
    out = R.ruled(img, zoom)
    m = R.MARGIN
    assert out.shape == (48 * zoom + m, 64 * zoom + m, 3)
    pic = out[m:, m:]
    ys = np.arange(1, 6) * 8 * zoom
    xs = np.arange(1, 8) * 8 * zoom
    mask = np.ones(pic.shape[:2], bool)
    mask[ys] = False
    mask[:, xs] = False
    np.testing.assert_array_equal(pic[mask], plain[mask])
    assert (pic[ys].astype(int) >= plain[ys].astype(int)).all()      # blended towards white, never darker
    np.testing.assert_array_equal(R.ruled(img, zoom, grid=False)[m:, m:], plain)
    whole = R.ruled(img, zoom, origin=(0, 0))
    part = R.ruled(R.crop_cells(img, 2, 3, 4, 5), zoom, origin=(2, 3))
    # the column ruler over cells 3..5 of the whole equals the crop's, likewise the row ruler of cells 2..4
    np.testing.assert_array_equal(whole[:m, m + 3 * 8 * zoom:m + 6 * 8 * zoom], part[:m, m:])
    np.testing.assert_array_equal(whole[m + 2 * 8 * zoom:m + 5 * 8 * zoom, :m], part[m:, :m])


def test_sheet_layout():
    """Columns side by side, eye views under them, crops at twice the zoom; marked cells get a red outline."""
    img = np.full((16, 24, 3), 0.5, np.float32)
    m, z = R.MARGIN, 2
    tile_w = 24 * z + m
    assert R.sheet([('a', img), ('b', img)], z).shape[1] == 2 * tile_w + R.GAP
    sheet = R.sheet([('a', img), ('b', img)], z, eye=lambda p: p * 0.5, crops=[(0, 0, 1, 1)], marks={'b': [(1, 2)]})
    assert sheet.shape[1] == 2 * (16 * 2 * z + m) + R.GAP       # the crop row, 2 cells at twice the zoom, is the widest
    top = R.TITLE + m
    a = sheet[top:top + 16 * z, m:m + 24 * z]
    b = sheet[top:top + 16 * z, tile_w + R.GAP + m:tile_w + R.GAP + m + 24 * z]
    assert (a == 128).all(axis=-1).mean() > 0.9 and (b[8 * z, 16 * z] == (255, 0, 0)).all()
    eye_top = top + 16 * z + R.GAP + R.TITLE + m
    assert (sheet[eye_top + 4, m + 4] == 64).all()


def test_variants_sample_and_ids():
    """A sample covers every method and each param's range, starts with the presets, and ids are stable."""
    from bench import variants as V
    from dizher.converter.energy import METHODS
    s = V.sample(6, seed=3)
    assert len(s) == 6 + len(V.presets()) and {p['method'] for p in s[len(V.presets()):]} == set(METHODS)
    for k, (lo, hi) in V.RANGES.items():
        vals = [p[k] for p in s[len(V.presets()):]]
        assert all(lo <= v <= hi for v in vals) and max(vals) - min(vals) > (hi - lo) / 2
    assert V.sample(6, seed=3) == s and V.sample(6, seed=4) != s
    assert V.variant_id(s[0]) == V.variant_id(dict(s[0])) and V.variant_id(s[0]) != V.variant_id(s[1])
    assert V.variant_id(dict(chroma=1.00001)) == V.variant_id(dict(chroma=1.0))
    assert V.distance(dict(method='a', params={}), dict(method='a', params={})) == 0
    assert V.distance(dict(method='a', params={}), dict(method='b', params={'chroma': 3.0})) > 1


def test_judgments_parse_record_agree(tmp_path, monkeypatch):
    """Verdict text parses; pairs pick without repeats; verdicts land in the picture's judgments; agreement and
    kappa come out as known."""
    import json
    import pytest
    from bench import judge as J, variants as V
    p = J.parse('12 a hue, 13 same; 14 B clash noise -- the sky\n15 = ')
    assert p[12] == dict(verdict='a', tags=['hue'], note='') and p[13]['verdict'] == 'same'
    assert p[14] == dict(verdict='b', tags=['clash', 'noise'], note='the sky') and p[15]['verdict'] == 'same'
    with pytest.raises(ValueError):
        J.parse('12 a sky')
    with pytest.raises(ValueError):
        J.parse('a 12')
    # a fake picture with four cached variants and a reference
    monkeypatch.setattr(J, 'IMAGES', tmp_path)
    monkeypatch.setattr(V, 'IMAGES', tmp_path)
    import bench.project
    monkeypatch.setattr(bench.project, 'IMAGES', tmp_path)
    cache = tmp_path / 'pic' / 'cache' / 'variants'
    cache.mkdir(parents=True)
    for i, method in enumerate(('Exact mixture', 'Halftoned') * 2):
        meta = dict(id=f'v{i}', method=method, params=dict(chroma=0.5 + i))
        (cache / f'v{i}.json').write_text(json.dumps(meta))
        (cache / f'v{i}.scr').write_bytes(bytes(6912))
    (tmp_path / 'pic' / 'reference.scr').write_bytes(bytes(6912))
    rid = V.reference_id('pic')
    assert rid.startswith('ref-') and len(V.listing('pic')) == 5 and V.listing('pic')[rid]['reference']
    pairs = J.pick('pic', 6, seed=1)
    assert len(pairs) == 6 and len({frozenset(p) for p in pairs}) == 6
    assert sum(rid in p for p in pairs) == 2
    entries = J.add_pairs('pic', pairs)
    assert [e['k'] for e in entries] == [1, 2, 3, 4, 5, 6]
    assert (tmp_path / 'pic' / 'variants' / 'v0.scr').exists()             # judged variants are kept
    assert (tmp_path / 'pic' / 'variants' / f'{rid}.scr').exists()          # the reference snapshot too
    (tmp_path / 'pic' / 'reference.scr').write_bytes(bytes([1]) + bytes(6911))   # a repaint: a new snapshot id
    assert V.reference_id('pic') != rid and rid in V.listing('pic') and V.reference_id('pic') in V.listing('pic')
    more = J.pick('pic', 2, seed=1, judged=[(e['a'], e['b']) for e in entries])
    assert not {frozenset(p) for p in more} & {frozenset(p) for p in pairs}
    assert J.record('pic', '1 a hue, 2 b, 3 same, 4 a noise', 'user') == 4
    assert J.record('pic', '1 a, 2 a, 3 same, 4 b noise', 'claude') == 4
    with pytest.raises(AssertionError):
        J.record('pic', '99 a', 'user')
    a = J.agreement(['pic'])
    assert a['pairs'] == 4 and a['agree'] == 0.5 and a['opposed'] == 0.5
    assert a['tags']['hue'] == (1, 1.0) and a['tags']['noise'] == (1, 0.0) and a['tags']['clash'][0] == 0
    assert J.kappa(['a', 'b', 'same'], ['a', 'b', 'same']) == 1.0 and abs(J.kappa(['a', 'b'], ['b', 'a'])) == 1.0


def test_zxart_fetch_prepare_stats(tmp_path, monkeypatch):
    """The API's answer parses (a saved sample); fetch keeps only plain 6912-byte screens without flash and writes the
    index; prepare makes a source that the blur leaves close to the screen's mean colour per cell, and a project the
    bench can run; stats count what the artist did."""
    import json
    from bench import zxart as Z
    sample = (Path(__file__).parent / 'bench' / 'zxart-sample.json').read_bytes()
    entries = Z.api(0, 2, 4.0, fetch=lambda url: sample)
    assert len(entries) == 2 and entries[0]['id'] == 47111 and entries[0]['original'].endswith('.scr')
    assert "'" in Z.api(0, 2, 4.0, fetch=lambda url: sample)[1]['title']          # &#039; unescaped
    monkeypatch.setattr(Z, 'FOLDER', tmp_path / 'zxart')
    rng = np.random.default_rng(2)
    screens = {}
    for e in entries:
        bitmap = rng.random((192, 256)) < 0.5
        idx = np.broadcast_to((0, 5), (24, 32, 2)).copy()                           # k/c dithered all over
        idx[:12, :16] = (2, 6)                                                       # r/y in one quarter
        screens[e['original']] = bytearray(to_scr(bitmap, idx))
    screens[entries[1]['original']][6144] |= 0x80                                   # flash: skipped
    calls = []
    def get(url):
        calls.append(url)
        if url.startswith(Z.API[:30]):
            return sample if len(calls) == 1 else b'{"responseStatus": "success", "responseData": {"zxPicture": []}}'
        return bytes(screens[url])
    got = Z.fetch(5, 4.0, get=get, log=lambda s: None)
    assert [e['id'] for e in got] == [47111] and (tmp_path / 'zxart' / '47111' / 'reference.scr').exists()
    assert json.loads((tmp_path / 'zxart' / 'index.json').read_text())[0]['id'] == 47111
    assert (tmp_path / 'zxart' / '.gitignore').read_text().startswith('*')
    names = Z.prepare(got, log=lambda s: None)
    assert names == ['zxart/47111'] and (tmp_path / 'zxart' / '47111' / 'project.json').exists()
    import cv2
    src = cv2.cvtColor(cv2.imread(str(tmp_path / 'zxart' / '47111' / 'source.png')), cv2.COLOR_BGR2RGB) / 255
    bitmap, idx = read_scr(tmp_path / 'zxart' / '47111' / 'reference.scr')
    from bench.scr import render_scr
    screen = render_scr(bitmap, idx, ZXPalette())
    lin = lambda a: a ** 2.2
    cell_mean = lambda a: lin(a).reshape(24, 8, 32, 8, 3).mean(axis=(1, 3))
    inside = np.ones((24, 32), bool)                                                # away from the quarter's border
    inside[10:14] = False
    inside[:, 14:18] = False
    off = np.abs(cell_mean(src) - cell_mean(screen))[inside]    # a random 50% dither's cell mean itself wanders ~0.05
    assert off.mean() < 0.02 and off.max() < 0.1 and src.std() < 0.8 * screen.std()   # the 1 px lightness blur keeps texture
    s = Z.stats(got)
    assert s['pictures'] == 1 and s['cells'] == 768 and 0 < s['seam_changes'] < 0.1 and s['distinct'] == 2
    assert sum(f for _, f, _ in s['pairs']) > 0.99 and s['pairs'][0][0] == 'k/c' and s['pairs'][1][2] > 0


def synthetic_picture(tmp_path, monkeypatch):
    """A project in tmp_path: a smooth source; cached screens 'good' (the method's own result) and 'bad' (every
    cell magenta on black); the reference is the good one. Returns the metrics Picture."""
    import json
    import cv2
    from dataclasses import replace
    from mokit.project import save_project
    from dizher import ops
    from dizher.converter.dither import Stohastic
    from bench import metrics as M, variants as V, judge as J
    import bench.project
    for mod in (bench.project, V, J):
        monkeypatch.setattr(mod, 'IMAGES', tmp_path)
    y, x = np.mgrid[0:192, 0:256] / 255.0
    source = np.stack([x, y * 0.7, 0.3 + 0.3 * np.sin(6 * x)], -1).clip(0, 1).astype(np.float32)
    folder = tmp_path / 'pic'
    folder.mkdir()
    cv2.imwrite(str(folder / 'source.png'), cv2.cvtColor((source * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    graph = ops.make_graph()
    graph = graph.with_params('source', replace(graph['source'].params, path=folder / 'source.png'))
    graph = graph.with_params('framing', replace(graph['framing'].params, fit='Fit'))
    save_project(folder, graph, {})
    pic = M.Picture('pic')
    conv = pic.conv
    conv.dither(Stohastic())
    idx = np.array(list(conv.palette.iter_idxs_pairs()))[conv.best_attr_indexes]
    good = to_scr(conv.dithered_bitmap > 0.5, idx)
    bad = to_scr(conv.dithered_bitmap > 0.5, np.broadcast_to((0, 3), idx.shape).copy())
    cache = folder / 'cache' / 'variants'
    cache.mkdir(parents=True)
    for vid, data in (('good', good), ('bad', bad)):
        (cache / f'{vid}.scr').write_bytes(data)
        (cache / f'{vid}.json').write_text(json.dumps(dict(id=vid, method='Exact mixture', params={})))
    (folder / 'reference.scr').write_bytes(good)
    return pic


def test_metrics_zero_on_self_and_rank_by_closeness(tmp_path, monkeypatch):
    """Every metric scores a screen against itself at (or near) its floor; on pairs where the winner is the
    screen closer to the source, a distance-like metric agrees with the judge; the ranking report reads the
    judgments and reference of a picture."""
    from bench import metrics as M, judge as J
    pic = synthetic_picture(tmp_path, monkeypatch)
    self_case = pic.case('good')
    self_case = M.Case(self_case.result, self_case.result, self_case.pairs, self_case.conv, self_case.convs)
    distance_like = [m for m in M.METRICS if not m.startswith(('energy', 'label_noise', 'region_pairs', 'lpips', 'dists', 'mix_blur'))]   # mix_blur scores the pairs' mixture, never the screen
    for m in distance_like:
        assert abs(M.METRICS[m](self_case)) < 1e-4, m
    from bench import variants as V
    rid = V.reference_id('pic')
    values = M.scores(pic, ['good', 'bad', rid])
    colour_blind = ('gmsd', 'haarpsi')                        # gradient metrics on luminance: magenta is a fine grey
    for m in [m for m in distance_like if m not in colour_blind] + ['energy:Exact mixture', 'energy:Halftoned']:
        assert values['good'][m] < values['bad'][m], m
    assert values['bad']['label_noise'] == 0 and values['bad']['region_pairs'] == 0     # one pair everywhere is quiet
    J.add_pairs('pic', [('good', 'bad'), ('bad', 'good')])
    J.record('pic', '1 a hue, 2 b', 'user')
    result = M.rank(['pic'], J.load, 'user', ['opp_blur:2', 'de2000:2', 'label_noise'], log=lambda s: None)
    assert result['overall']['opp_blur:2'] == (1.0, 2) and result['by_tag']['hue']['de2000:2'] == (1.0, 1)
    assert result['overall']['label_noise'][0] == 0.0                                   # the spoiled copy is quieter
    assert result['reference']['pic']['opp_blur:2'] == (0.0, 2)                         # the reference is the good one
    M.print_rank(result, ['opp_blur:2', 'de2000:2', 'label_noise'])
    d = M.distances(pic.conv, pic.case('good').result, pic.case('bad').result, pic.case('bad').pairs)
    assert set(d) == set(M.DISTANCES) and all(v > 0 for v in d.values())
    same = M.distances(pic.conv, pic.case('good').result, pic.case('good').result, pic.case('good').pairs)
    assert all(abs(v) < 1e-4 for v in same.values())


def test_reference_frozen_from_the_project(tmp_path, monkeypatch):
    """A painted project's reference is the project converted as saved, frozen to its cache and refrozen only when
    project.json changes; a project with its own reference.scr keeps it; an unpainted project has none."""
    import json
    from dataclasses import replace
    from mokit.project import load_project, save_project
    import bench.project as P
    from bench import variants as V
    synthetic_picture(tmp_path, monkeypatch)
    (tmp_path / 'pic' / 'reference.scr').unlink()
    (tmp_path / 'pic' / 'cache' / 'variants' / 'good.json').unlink()
    assert not P.has_reference('pic') and V.reference_id('pic') is None
    project = load_project(tmp_path / 'pic')
    graph = project.graph.with_params('overpaint', replace(project.graph['overpaint'].params, overrides=((0, 0, 0, 14),)))
    save_project(project.folder, graph, project.view)
    assert P.has_reference('pic')
    frozen = P.reference_file('pic')
    assert frozen == tmp_path / 'pic' / 'cache' / 'reference.scr' and frozen.stat().st_size == 6912
    meta = json.loads((tmp_path / 'pic' / 'cache' / 'reference.json').read_text())
    assert meta['painted'] == 1 and meta['project'] == P.project_hash('pic')
    bitmap, idx, shown, painted = P.reference('pic')
    assert tuple(idx[0, 0] % 8) in ((0, 6), (6, 0)) and painted == {(0, 0)}      # black shares yellow's brightness
    stamp = frozen.stat().st_mtime_ns
    P.reference_file('pic')
    assert frozen.stat().st_mtime_ns == stamp                              # the same project: not refrozen
    graph = graph.with_params('overpaint', replace(graph['overpaint'].params, overrides=((0, 0, 0, 14), (0, 1, 0, 10))))
    save_project(project.folder, graph, project.view)
    assert P.reference('pic')[3] == {(0, 0), (0, 1)} and frozen.stat().st_mtime_ns != stamp
    g, conv, final, _ = P.convert_as_saved('pic')
    assert g['overpaint'].params.overrides == () and final.dithered_result.shape == (192, 256, 3)
    # the corrected cells: painted ones the painting changed away from the project's own conversion
    from bench.scr import read_scr as rd, shown as sh, matches as mt
    assert (tmp_path / 'pic' / 'cache' / 'unpainted.scr').exists()
    changed = P.changed_cells('pic')
    before, after = sh(*rd(tmp_path / 'pic' / 'cache' / 'unpainted.scr')), sh(*rd(frozen))
    assert changed.shape == (24, 32) and changed[2:].sum() == 0 and changed[:1, 2:].sum() == 0
    for r, c in ((0, 0), (0, 1)):
        assert changed[r, c] == (not mt(before, after)[r, c] or not mt(after, before)[r, c])
    assert changed.any()                                          # a smooth grey source never picks bright red at (0, 1)


def test_fit_energy_to_judgments(tmp_path, monkeypatch):
    """The energy under fitted params agrees with judgments that prefer the closer screen; the preset agrees too;
    the sample stays in range and the setup params rebuild the candidates only when they change; the dynamic
    energy metrics and the composite judge score."""
    import pytest
    from bench import fit as F, judge as J
    synthetic_picture(tmp_path, monkeypatch)
    J.add_pairs('pic', [('good', 'bad'), ('bad', 'good')])
    J.record('pic', '1 a, 2 b tone', 'user')
    s = F.sample(9, seed=1)
    assert len(s) == 9 and all(F.RANGES[k][0] <= p[k] <= F.RANGES[k][1] for p in s for k in F.FREE)
    assert {p['flare'] for p in s} == set(F.FLARES)
    from bench import metrics as M
    e = M.Energies('pic', 'Exact mixture', ['good', 'bad'])
    first = e(dict(s[0]))
    assert first['good'] < first['bad'] and e.conv.flare == s[0]['flare']
    calc_calls = []
    monkeypatch.setattr(e.conv.energy, 'calc', lambda: calc_calls.append(1))
    e(dict(s[0], chroma=1.5))
    assert not calc_calls                                                                # the same setup: no rebuild
    e(dict(s[0], flare=0.2))
    assert calc_calls == [1] and e.conv.flare == 0.2
    e(dict(s[0], flare=0.2, luma_scale=2.8))
    assert calc_calls == [1, 1] and e.conv.luma_scale == 2.8                            # the eye kernels rebuild too
    # the dynamic metrics: the preset's energy by either name, a wider blur another value, the composite judge
    method, params = M.parse_energy_metric('energy:Exact mixture:flare=1+luma_scale=2.8')
    assert method == 'Exact mixture' and params['flare'] == 1 and params['luma_scale'] == 2.8 and params['coherence'] == 6.0
    with pytest.raises(AssertionError):
        M.parse_energy_metric('energy:Exact mixture:zoom=2')
    pic = M.Picture('pic')
    v = M.scores(pic, ['good', 'bad'], ['energy:Exact mixture', 'energy:Exact mixture:', 'energy:Exact mixture:luma_scale=2.8',
                                        'judge:1', 'opp_blur:2:nogain'])
    assert v['good']['energy:Exact mixture'] == v['good']['energy:Exact mixture:'] != v['good']['energy:Exact mixture:luma_scale=2.8']
    assert v['good']['judge:1'] < v['bad']['judge:1'] and 'gmsd' not in v['good'] and 'opp_blur:2:nogain' in v['good']
    out = F.fit(['pic'], J.load, 'user', n=6, seed=1, holdout=False, log=lambda s: None)
    assert set(out) == {'Exact mixture', 'Halftoned'}
    for m, r in out.items():
        assert r['agreement'] == 1.0 and r['preset_agreement'] == 1.0 and r['pairs'] == 2, m
        assert all(F.RANGES[k][0] <= v <= F.RANGES[k][1] for k, v in r['params'].items())
