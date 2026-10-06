"""PLAN step 5b: seam visibility itself, on synthetic patches without picture content (the user, 2026-10-06). A patch
is 8 x 6 cells: the left four columns of cells one pair, the right four another, one threshold map over the whole
patch (the app's blue noise: DBS, on by default, breaks a regular pattern, and a regular one shows any break), so the
dots run on across the border, nested where the two sides' levels differ; the level has a gradient of a random
direction and strength (SLOPE; a strong one in the ramps) and varies slowly (FIELD), as a surface does, a uniform field
being the easiest to see a seam on (the user). Each change twice (INSTANCES), on another gradient and noise. Some
changes also with the dots breaking at the border (the right half's from another place in the tile), as they do on a
render, where each cell's pair has its own dots and DBS moves them. The user rates the seam down the
middle: 0 none, 1 faint, 2 clear, 3 jumps out.
The colour changes come at the same lightness (the user: else the lightness gives the seam away, and a real pair
switch on a smooth surface is near it, both pairs dithered to one target): the right side's level is set so that its
mean luminance in linear light is the left's (match). The lightness alone changes in a step of the level, the pair
the same. What changes: the ink's hue on black paper (near hues r-m, g-c, y-w; far r-g), its brightness, both; the
same in gradients of the level (black-white, red-yellow; the paper's or the ink's hue, brightness); the lightness
steps; the first version's hue changes at one level, the lightness changing too; the user's black border; none.

    python research/pairs/patches.py make     the patches and REPEATS of them again, in a random order:
                                              data/seams/patches/, the key in rounds/seams/patches-key.json
    python research/pairs/patches.py rate     the page: keys 0-3 rate and go on, Backspace back, +/- zoom; each
                                              rating saved at once to rounds/seams/patches.json
    python research/pairs/patches.py report [vN]   the ratings by group; the repeats; each measure's rank correlation
                                              with the ratings; vN, a kept round
Kept as the record: v1 (colours at one level, the lightness changing too; the user stopped it after 26), v2 (matched,
Bayer and blue noise on a uniform level; stopped after 3), v3 (blue noise, the slow variation; inks on black), v4 (and
gradients, two instances, dots breaking): rounds/seams/patches-vN.json, patches-key-vN.json, data/seams/patches-vN/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dizher.converter.energy import pair_dissimilarity   # noqa: E402
from dizher.halftoning.noise.noise import noise_dither   # noqa: E402
from dizher.halftoning.ordered.ordered import ordered_dither   # noqa: E402
from metrics import SRGB2XYZ, blur, encode, linear, xyz2lab   # noqa: E402
from seams import OUT as SEAMS, ROUND, ZX, auc, dots, measures, switch   # noqa: E402

OUT = SEAMS / 'patches'
KEY, RATINGS = ROUND / 'patches-key.json', ROUND / 'patches.json'
H, W = 48, 64   # 6 x 8 cells; the seam between cell columns 3 and 4
REPEATS = 10
STEP = 0.02   # the level of one up or down on a matched right side
HALFTONES = ('blue',)
LEVELS = (0.3, 0.5, 0.75)
FIELD = 0.08, 6.0   # the slow variation of the level: its sd and its scale (Gaussian sigma, px)
SLOPE = 0.4, 0.8    # the gradient's change of the level over the patch's width: at most, at random; in the ramps
INSTANCES = 2       # each change on this many gradients and noises; all once first, so a stop halfway covers them


def stimuli() -> list:
    """Each patch: its group, what changes at the border, the left and right pair (paper, ink by letter), the left's
    level (a number; 'ramp', a strong gradient; 'away', 'on': the user's dots), the right's ('match': the left's
    luminance; 'same'; a number: the left's plus it), the halftone and the dots (continue or break at the border).
    v5 (the user, 2026-10-06: v3 and v4 had only inks on black paper and near hues): a paper's change under one ink,
    blue, the same mean colour from other dots; v3 and v4's sets are in their keys."""
    out = []

    def add(group, change, left, right, level, right_level='match'):
        span = lambda pair: sorted(luminance(c) for c in pair)
        if right_level == 'match' and not span(right)[0] <= span(left)[0] <= span(left)[1] <= span(right)[1]:
            left, right = right, left   # the right follows the left: the left's luminance within the right's reach
        out.append(dict(group=group, change=change, left=left, right=right, level=level, right_level=right_level,
                        halftone='blue', dots='continue'))

    for left, right in (('CW', 'GW'), ('CW', 'BW'), ('YW', 'CW'), ('YW', 'GW'), ('MW', 'RW'), ('cw', 'gw')):
        add('paper under white', f'{left[0]}-{right[0]} paper', left, right, 0.5)
    for left, right, level in (('kb', 'kr', 0.75), ('kb', 'km', 0.75), ('kB', 'kb', 0.5), ('bc', 'bw', 0.5),
                               ('by', 'bw', 0.5), ('BC', 'BG', 0.5)):
        add('blue', f'{left}-{right}', left, right, level)
    for left, right in (('kw', 'by'), ('kw', 'rc'), ('kw', 'mg'), ('by', 'rc'), ('km', 'br'), ('kc', 'bg'),
                        ('ky', 'rg')):
        add('same mean, other dots', f'{left}-{right}', left, right, 0.5)
    add('none', 'CW', 'CW', 'CW', 0.5, 'same')
    add('none', 'by', 'by', 'by', 0.5, 'same')
    for step in (0.12, 0.25):
        add('lightness', f'CW +{step}', 'CW', 'CW', 0.5 - step / 2, step)
    add('anchor (v3, v4)', 'r-m', 'kr', 'km', 0.5)
    add('anchor (v3, v4)', 'r-g', 'kr', 'kg', 0.5)
    return out


def luminance(c: str) -> float:
    return float(linear(np.asarray(ZX[c])) @ SRGB2XYZ[1])


def levels(s: dict, shift=0.0) -> np.ndarray:
    """(H, W) each pixel's level: the left's spec, its level at the patch's centre, with the gradient (the patch's
    angle and slope) and the slow variation (FIELD), the same across the border; on the right its own. shift, the
    user's move of a matched right side from the luminance match (the minimally distinct border)."""
    y, x = np.mgrid[:H, :W]
    noise = blur(np.random.default_rng(s['seed']).normal(size=(H, W)), FIELD[1])
    across = ((x - (W - 1) / 2) * np.cos(s['angle']) + (y - (H - 1) / 2) * np.sin(s['angle'])) / (W - 1)
    base = 0.5 if s['level'] == 'ramp' else float(s['level'])
    row = np.clip(base + s['slope'] * across + FIELD[0] * noise / noise.std(), 0, 1)
    x = x[0]
    right = s['right_level']
    if right == 'match':   # the level of the right pair at the left's mean luminance
        (pl, il), (pr, ir) = ([luminance(c) for c in pair] for pair in (s['left'], s['right']))
        right = np.clip((pl + row * (il - pl) - pr) / (ir - pr) + shift, 0, 1)
    elif right == 'same':
        right = row
    else:
        right = row + right
    return np.ascontiguousarray(np.where(x < W // 2, row, np.clip(right, 0, 1)), dtype=np.float32)


def bitmap(s: dict, origin, shift=0.0) -> np.ndarray:
    """(H, W) True where the ink shows."""
    if s['level'] in ('away', 'on'):   # the user's case: dots in every other row, off the cells' border columns or on
        return np.tile(dots('k', 'r', (1, 2, 5, 6) if s['level'] == 'away' else (0, 3, 4, 7))(None)[..., 0] > 0,
                       (H // 8, W // 8))
    t = levels(s, shift)
    dither = lambda o: ordered_dither(t, 'Bayer 4x4', o) if s['halftone'] == 'bayer' else noise_dither(t, origin=o)
    if s['dots'] == 'break':   # the right half's dots from elsewhere in the tile
        return np.where(np.arange(W) < W // 2, dither(origin), dither((origin[0] + 17, origin[1] + 29)))
    return dither(origin)


def picture(s: dict, origin, shift=0.0) -> np.ndarray:
    """(H, W, 3) sRGB 0..1: the left pair's colours on the left half, the right pair's on the right, one bitmap."""
    side = np.arange(W) >= W // 2
    paper = np.where(side[:, None], ZX[s['right'][0]], ZX[s['left'][0]])
    ink = np.where(side[:, None], ZX[s['right'][1]], ZX[s['left'][1]])
    return np.where(bitmap(s, origin, shift)[..., None], ink, paper)


def cell_means(X) -> tuple:
    """(dL*, dab) between the mean colours, in linear light, of the two cells next to the border, mean down it;
    dL* signed, the right's less the left's."""
    m = lambda Y: xyz2lab(linear(Y).reshape(H // 8, 8, 8, 3).mean((1, 2)) @ SRGB2XYZ.T)
    d = m(X[:, W // 2:W // 2 + 8]) - m(X[:, W // 2 - 8:W // 2])
    return float(d[:, 0].mean()), float(np.linalg.norm(d[:, 1:], axis=-1).mean())


def make(seed=0):
    rng = np.random.default_rng(seed)
    ss = [dict(s, instance=n) for n in range(INSTANCES) for s in stimuli()]
    OUT.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(ss):
        s.update(file=f'p{i:03d}.png', origin=[int(v) for v in rng.integers(64, size=2)], seed=int(rng.integers(2 ** 31)),
                 angle=float(rng.uniform(0, 2 * np.pi)),
                 slope=SLOPE[1] if s['level'] == 'ramp' else float(rng.uniform(0, SLOPE[0])))
        X = picture(s, tuple(s['origin']))
        cv2.imwrite(str(OUT / s['file']), cv2.cvtColor(np.round(X * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    n = len(ss) // INSTANCES   # each instance's patches shuffled in turn, then the repeats of the first's
    order = [k for b in range(INSTANCES) for k in b * n + rng.permutation(n)]
    order += [int(k) for k in rng.choice(n, REPEATS, replace=False)]
    key = [dict(ss[k], id=f'q{j:03d}', stimulus=int(k)) for j, k in enumerate(order)]
    KEY.write_text(json.dumps(key, indent=1))
    print(f'{len(ss)} patches, {len(key)} to rate; the realised lightness step of the cells by the border, dL*:')
    for group in dict.fromkeys(s['group'] for s in ss):
        d = [abs(cell_means(picture(s, tuple(s['origin'])))[0]) for s in ss if s['group'] == group]
        print(f'  {group:20} mean {np.mean(d):5.1f}  max {np.max(d):5.1f}')


PAGE = """<!doctype html><meta charset="utf-8"><title>dizher seam patches</title>
<style>body{margin:0;background:#222;color:#ddd;font:15px system-ui;display:flex;flex-direction:column;align-items:center}
#info{padding:8px;align-self:stretch}#stage{margin-top:12vh;display:flex;flex-direction:column;align-items:center;gap:6px}
img{display:block;image-rendering:pixelated}.tick{width:2px;height:10px;background:#888}b{color:#fff}</style>
<div id="info"></div><div id="stage"><div class="tick"></div><img id="p"><div class="tick"></div></div>
<script>
const S = %s, $ = id => document.getElementById(id);
let zoom = +(localStorage.patchZoom || 3), k = S.k;
function draw() {
  $('p').style.width = 64 * zoom + 'px';
  $('p').src = `/img/${S.id}?k=${k}`;
  $('info').innerHTML = `<b>${S.i + 1} of ${S.n}</b>${S.rating === null ? '' : ' (rated ' + S.rating + ')'} &middot;
    zoom &times;${zoom} (+/&minus;)` + (S.adjust ? ` &middot; <b>the right side's lightness ${k > 0 ? '+' : ''}${k}</b>
    (&uarr;/&darr;): set it where the seam is least visible, then rate the seam there` : ' &middot; rate it as it is') +
    `<br>The seam down the middle, between the ticks: 0 none, 1 faint, 2 clear, 3 jumps out. Backspace: back`;
}
addEventListener('keydown', async e => {
  if (S.adjust && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
    e.preventDefault(); k += e.key === 'ArrowUp' ? 1 : -1; draw(); return;
  }
  if (e.repeat) return;
  if ('0123'.includes(e.key) && e.key.length === 1) {
    await fetch('/rate', {method: 'POST', body: JSON.stringify({id: S.id, rating: +e.key, k})});
    location = '/?i=' + (S.i + 1);
  }
  if (e.key === 'Backspace' && S.i > 0) { e.preventDefault(); location = '/?i=' + (S.i - 1); }
  if (e.key === '+' || e.key === '=' || e.key === '-') {
    zoom = Math.max(1, zoom + (e.key === '-' ? -1 : 1)); localStorage.patchZoom = zoom; draw();
  }
});
draw();
</script>"""


def rate(port=8768):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
    key = json.loads(KEY.read_text())
    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {}
    by = {k['id']: k for k in key}

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path.startswith('/img/'):
                s = by[Path(url.path).name]
                X = picture(s, tuple(s['origin']), int(parse_qs(url.query).get('k', [0])[0]) * STEP)
                png = cv2.imencode('.png', cv2.cvtColor(np.round(X * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))[1]
                return self.send(png.tobytes(), 'image/png')
            if url.path != '/':
                return self.send(b'', 'text/plain')
            todo = [i for i, k in enumerate(key) if k['id'] not in ratings]
            i = int(parse_qs(url.query).get('i', [todo[0] if todo else len(key)])[0])
            if i >= len(key):
                return self.send(f'<body style="background:#222;color:#ddd;font:15px system-ui">All {len(key)} rated, '
                                 f'thank you. <a style="color:#ddd" href="/?i={len(key) - 1}">back</a>'.encode(),
                                 'text/html')
            r = ratings.get(key[i]['id'], {})
            state = dict(i=i, n=len(key), id=key[i]['id'], rating=r.get('rating'), k=r.get('k', 0),
                         adjust=key[i]['right_level'] == 'match')
            self.send(PAGE.replace('%s', json.dumps(state), 1).encode(), 'text/html')

        def do_POST(self):
            v = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            ratings[v.pop('id')] = v
            RATINGS.write_text(json.dumps(ratings, indent=1))
            self.send(b'ok', 'text/plain')

        def log_message(self, *args):
            pass

    print(f'http://localhost:{port}  (Ctrl+C to stop; ratings in {RATINGS})', flush=True)
    webbrowser.open(f'http://localhost:{port}')
    HTTPServer(('localhost', port), Handler).serve_forever()


def seam_values(s: dict, V: dict, shift=0.0) -> dict:
    """{measure: value} at the patch's middle seam as the user rated it (the right side shifted): seams.measures (mean
    over its 6 cells, against a target blurred by 8 px, so it has no step of its own), the attributes (V of the two
    pairs, {(pair, pair): V}), the pair switch at the border's pixels, the cells' mean colours."""
    X = picture(s, tuple(s['origin']), shift)
    T = encode(blur(linear(X), 8.0))
    middle = lambda v: v[(H // 8 - 1) * (W // 8):].reshape(H // 8, W // 8 - 1)[:, W // 16 - 1].mean()
    out = {}
    for fam, ks in measures(X, T).items():
        out[fam + ' L'], out[fam + ' ab'] = middle(ks['L']), middle(ks['ab'])
    out['A attributes L'] = V[s['left'], s['right']]
    e = switch(X, np.array([[[ZX[c] for c in (s['left'] if j < W // 16 else s['right'])] for j in range(W // 8)]
                            for _ in range(H // 8)]))
    out['E switch L'], out['E switch ab'] = middle(e['L']), middle(e['ab'])
    for fam in {m.rsplit(' ', 1)[0] for m in out if m.endswith(' ab')}:
        out[fam + ' L+ab'] = out[fam + ' L'] + out[fam + ' ab']
    return out


def report(version=None):
    """version: 'vN', a kept round (patches-key-vN.json, patches-vN.json); the current by default."""
    key = json.loads((ROUND / f'patches-key-{version}.json' if version else KEY).read_text())
    ratings = json.loads((ROUND / f'patches-{version}.json' if version else RATINGS).read_text())
    got = {}   # stimulus: (its key entry, [(rating, k)])
    for k in key:
        if k['id'] in ratings:
            got.setdefault(k['stimulus'], (k, []))[1].append((ratings[k['id']]['rating'], ratings[k['id']]['k']))
    print(f'{sum(len(r) for _, r in got.values())} ratings of {len(got)} patches\n')
    spec = {i: got[i][0] for i in sorted(got)}
    mean = {i: float(np.mean([r for r, _ in got[i][1]])) for i in spec}
    shift = {i: round(float(np.mean([k for _, k in got[i][1]]))) * STEP for i in spec}
    column = lambda k: f"{'' if isinstance(k['right_level'], float) else k['level']}" + \
        (' break' if k['dots'] == 'break' else '')
    for group in dict.fromkeys(k['group'] for k in spec.values()):
        ids = [i for i in spec if spec[i]['group'] == group]
        cols = list(dict.fromkeys(column(spec[i]) for i in ids))
        matched = spec[ids[0]]['right_level'] == 'match'
        print(f"{group}: the mean rating" + (", and the right side's lightness the user set, dL* from the luminance "
                                             "match" if matched else ''))
        print(f"  {'':16}" + ''.join(f'{c:>14}' for c in cols))
        for change in dict.fromkeys(spec[i]['change'] for i in ids):
            cells = {}   # column: [(rating, moved)] over the instances
            for i in ids:
                if spec[i]['change'] == change:
                    s, o = spec[i], tuple(spec[i]['origin'])
                    moved = cell_means(picture(s, o, shift[i]))[0] - cell_means(picture(s, o))[0]
                    cells.setdefault(column(s), []).append((mean[i], moved))
            text = lambda v: f'{np.mean([r for r, _ in v]):.1f}' + (f' {np.mean([m for _, m in v]):+5.1f}' if matched else '')
            print(f'  {change:16}' + ''.join(f"{text(cells[c]) if c in cells else '':>14}" for c in cols))
        print()
    rep = [r for _, r in got.values() if len(r) > 1]
    if rep:
        d = np.array([abs(r[0][0] - r[1][0]) for r in rep])
        print(f'the repeats ({len(rep)}): the same rating {np.mean(d == 0):.0%}, off by one {np.mean(d == 1):.0%}, '
              f'mean difference {d.mean():.2f}; the lightness set, steps apart: '
              f"{' '.join(str(abs(r[0][1] - r[1][1])) for r in rep)}")
    pairs = sorted({k[side] for k in key for side in ('left', 'right')})
    V = pair_dissimilarity(np.array([[ZX[c] for c in pair] for pair in pairs]))
    V = {(a, b): float(V[i, j]) for i, a in enumerate(pairs) for j, b in enumerate(pairs)}
    values = {i: seam_values(spec[i], V, shift[i]) for i in spec}
    y = np.array([mean[i] for i in spec])
    print('\neach measure against the mean ratings: Spearman rho; AUC, any seam (rating >= 1) against none')
    for m in sorted(values[next(iter(values))]):
        x = np.array([values[i][m] for i in spec])
        print(f'  {m:32} rho {spearmanr(x, y)[0]:+.2f}   AUC {auc(x, y >= 1):.2f}')


if __name__ == '__main__':
    {'make': make, 'rate': rate, 'report': report}[sys.argv[1]](*sys.argv[2:])
