"""PLAN step 5: which measure of a seam's pixels tells where the user sees a cell seam the source does not have. The
coherence term prices a seam by the attributes alone (V, the CIELUV distance of the two cells' papers plus their
inks): red on black against magenta on black costs in full, though with black pixels along the border there is no
seam, and a border in ~50% noise does not show either (the user). The user paints the seams seen on whole renders;
each candidate measure is scored by how well it ranks the painted seams above the rest.

A seam is the border of two 4-neighbour cells: h (R-1, C) between (r, c) and (r+1, c), v (R, C-1) between (r, c) and
(r, c+1), as Lh, Lv in energy.py; a render's seams are one vector, h then v.

    python research/pairs/seams.py make     the renders to paint, DBS on: per training picture its gallery best and
                                            the same at coherence 0 (more seams), in a blind order, the two of a
                                            picture apart, then REPEATS again; data/seams/, the key in
                                            rounds/seams/key.json (commit it after the painting)
    python research/pairs/seams.py paint    the page in the browser: drag along a seam you see (shift: strong, alt:
                                            erase), Enter when done; each stroke saved at once to
                                            rounds/seams/marks.json; Ctrl+C stops, run it again to go on
    python research/pairs/seams.py check    the user's own cases against each measure, synthetic, no render
    python research/pairs/seams.py report   per measure how well it ranks the painted seams (AUC per render; the
                                            lightness-colour mix chosen with the picture left out), the repeats'
                                            agreement, and how often it rates the user's paintings below Select
                                            pairs' cells (build.py's segments, as score.py)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acuity import EYE   # noqa: E402
from common import DATA, Project   # noqa: E402
from dizher.converter.energy import pair_dissimilarity   # noqa: E402
from dizher.platforms.zxspectrum import ZXPalette   # noqa: E402
from metrics import SRGB2XYZ, encode, lab_blurred, linear, xyz2lab   # noqa: E402
from views import seen2   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'seams', HERE / 'rounds' / 'seams'
VARIANTS = {'best': {}, 'coherence0': {'select': {'coherence': 0.0}}}
REPEATS = 2
GROUPS = {'L': slice(0, 1), 'ab': slice(1, 3)}


# ---- the measures

def lab(X):
    """CIELAB of sRGB 0..1, pixel by pixel."""
    return xyz2lab(linear(X) @ SRGB2XYZ.T)


def eye_lab(X):
    """CIELAB of the picture through the user's eye (views.seen2 at 1x: lightness sigma 0.75, colour 1.0)."""
    return lab(seen2((np.clip(X, 0, 1) * 255).round().astype(np.uint8), *EYE, k=1) / 255)


def steps(Z, g):
    """Each pixel column border x of every cell row (x = 0, the picture's edge: none), in the channels g: (H, W) the
    pixel steps' size, and (R, C, 8) the size of the steps' mean along the cell row's 8 pixels, a line filter along
    the border where noise cancels; [..., 0] is the seam with the cell to the left, 1..7 the cell's own columns."""
    d = np.zeros(Z.shape[:2] + (g.stop - g.start,))
    d[:, 1:] = Z[:, 1:, g] - Z[:, :-1, g]
    H, W = Z.shape[:2]
    return np.linalg.norm(d, axis=-1), np.linalg.norm(d.reshape(H // 8, 8, W, -1).mean(1), axis=-1).reshape(
        H // 8, W // 8, 8)


def pixel_step(ZX, ZT, g):
    """(R, C-1) the pixel steps across each vertical seam beyond the target's, mean along the seam: seam_excess per
    seam."""
    (px, _), (pt, _) = steps(ZX, g), steps(ZT, g)
    ex = np.maximum(px - pt, 0)
    H, W = ex.shape
    return ex.reshape(H // 8, 8, W // 8, 8)[:, :, 1:, 0].mean(1)


def pixel_vs_texture(ZX, ZT, g):
    """(R, C-1) the pixel steps across each vertical seam beyond the target's, mean along it, less the cells' own pixel
    steps (the mean over the two cells' inner columns): the actual pixels at the border, as B, but not the dots."""
    (px, _), (pt, _) = steps(ZX, g), steps(ZT, g)
    H, W = px.shape
    cells = px.reshape(H // 8, 8, W // 8, 8)
    inside = cells[..., 1:].mean((1, 3))
    return np.maximum(cells[:, :, 1:, 0] - pt.reshape(H // 8, 8, W // 8, 8)[:, :, 1:, 0], 0).mean(1) - \
        (inside[:, :-1] + inside[:, 1:]) / 2


def line_step(ZX, ZT, g):
    """(R, C-1) the line filter at each vertical seam beyond the target's, less the cells' own texture (the filter's
    mean over the two cells' inner columns): a seam shows when it steps more than the dots around it."""
    (_, lx), (_, lt) = steps(ZX, g), steps(ZT, g)
    inside = lx[..., 1:].mean(-1)
    return np.maximum(lx[:, 1:, 0] - lt[:, 1:, 0], 0) - (inside[:, :-1] + inside[:, 1:]) / 2


def strips(ZX, ZT, g):
    """(R, C-1) how the 8 x 4 pixel strips on the two sides of each vertical seam differ, mean and spread per channel,
    beyond the target's: a change of dot texture."""
    def diff(Z):
        H, W = Z.shape[:2]
        z = Z[..., g].reshape(H // 8, 8, W // 8, 8, -1)
        a, b = z[:, :, :-1, 4:], z[:, :, 1:, :4]
        return sum(np.linalg.norm(f(a, axis=(1, 3)) - f(b, axis=(1, 3)), axis=-1) for f in (np.mean, np.std))
    return np.maximum(diff(ZX) - diff(ZT), 0)


def means_step(ZX, ZT, g):
    """(R, C-1) the step between the two cells' mean colours (in linear light) across each vertical seam, beyond the
    target's: what is left of a seam once the eye averages each cell's dots. ZX, ZT: linear RGB; CIELAB after the
    mean."""
    def step(Z):
        H, W = Z.shape[:2]
        m = xyz2lab(Z.reshape(H // 8, 8, W // 8, 8, 3).mean((1, 3)) @ SRGB2XYZ.T)[..., g]
        return np.linalg.norm(m[:, 1:] - m[:, :-1], axis=-1)
    return np.maximum(step(ZX) - step(ZT), 0)


def texture(ZX, ZT, g):
    """(R, C-1) the render's own pixel steps inside the two cells of each vertical seam, mean: how busy their dots
    are (not a seam: a modifier of one; ZT unused)."""
    px, _ = steps(ZX, g)
    H, W = px.shape
    inside = px.reshape(H // 8, 8, W // 8, 8)[..., 1:].mean((1, 3))
    return (inside[:, :-1] + inside[:, 1:]) / 2


def texture_step(ZX, ZT, g):
    """(R, C-1) how differently the two cells' dots stand out: the difference of their pixels' spread (sd, in the
    channels g) across each vertical seam. A cell whose two colours are near in lightness reads as solid next to one
    whose dots show (the user: dim yellow/white against any bright pair). ZT unused."""
    H, W = ZX.shape[:2]
    z = ZX[..., g].reshape(H // 8, 8, W // 8, 8, -1)
    sd = np.sqrt(z.var(axis=(1, 3)).sum(-1))
    return np.abs(sd[:, 1:] - sd[:, :-1])


SOLID = 3.0   # the lightness sd (L*) below which a cell's dots stop showing: exp(-sd / SOLID), its solidity


def solid_step(ZX, ZT, g):
    """(R, C-1) how differently solid the two cells look across each vertical seam: |exp(-sd_a / SOLID) - exp(-sd_b /
    SOLID)|, sd the cell's pixel spread in the channels g. A cell whose two colours are near in lightness (dim yellow
    and white, sd ~1) reads as one solid block, and stands out next to a cell with visible dots (the user); two dotted
    cells, whatever their contrast, do not. ZT unused."""
    H, W = ZX.shape[:2]
    z = ZX[..., g].reshape(H // 8, 8, W // 8, 8, -1)
    solid = np.exp(-np.sqrt(z.var(axis=(1, 3)).sum(-1)) / SOLID)
    return np.abs(solid[:, 1:] - solid[:, :-1])


def lightness(ZX, ZT, g):
    """(R, C-1) the two cells' mean L* (of their mean colour in linear light) / 100 across each vertical seam: where a
    seam is (not a seam: a modifier of one; ZX linear RGB, ZT and g unused)."""
    H, W = ZX.shape[:2]
    L = xyz2lab(ZX.reshape(H // 8, 8, W // 8, 8, 3).mean((1, 3)) @ SRGB2XYZ.T)[..., 0] / 100
    return (L[:, 1:] + L[:, :-1]) / 2


def both_ways(f, ZX, ZT, g):
    """The seams' vector of a vertical-seam measure: h from the pictures transposed, then v."""
    t = lambda Z: Z.transpose(1, 0, 2)
    return np.concatenate([f(t(ZX), t(ZT), g).T.ravel(), f(ZX, ZT, g).ravel()])


FAMILIES = {   # name: (the measure, the picture it looks at)
    'B pixel step raw': (pixel_step, lab),
    'B pixel step eye': (pixel_step, eye_lab),
    'B pixel step blur 2': (pixel_step, lambda X: lab_blurred(X, 2.0)),
    'B pixel step vs texture raw': (pixel_vs_texture, lab),
    'C line vs texture raw': (line_step, lab),
    'C line vs texture eye': (line_step, eye_lab),
    'D strips raw': (strips, lab),
    'M cell means': (means_step, linear),
    'T cell texture': (texture, lab),
    'K texture step': (texture_step, lab),
    'S solid step': (solid_step, lab),
    'Y cell lightness': (lightness, linear),
}


def measures(X, T) -> dict:
    """{family: {'L': seams, 'ab': seams}} of the render X against the target T, sRGB 0..1."""
    views = {}
    out = {}
    for name, (f, view) in FAMILIES.items():
        if view not in views:
            views[view] = view(X), view(T)
        out[name] = {k: both_ways(f, *views[view], g) for k, g in GROUPS.items()}
    return out


def switch(X, pairs) -> dict:
    """{'L': seams, 'ab': seams} the colour change a pair switch makes at the pixels along each border, the dots the
    same: each pixel of the column on each side, against the colour the neighbour's pair puts on its dot (its ink if
    the pixel is its own pair's ink, else its paper). Zero for the same pair, and where the border's pixels are a
    colour both pairs share in the same place (black paper); a selection term can have it, from each pair's dots.
    X (H, W, 3) and pairs (R, C, 2, 3) paper and ink, sRGB 0..1."""
    def vertical(X, pairs):
        paper, ink = (np.repeat(np.repeat(pairs[:, :, k], 8, 0), 8, 1) for k in (0, 1))
        bit = np.abs(X - ink).sum(-1) < 1e-3
        xl = np.arange(7, X.shape[1] - 8, 8)
        d = []
        for mine, theirs in ((xl, xl + 1), (xl + 1, xl)):
            other = np.where(bit[:, mine, None], ink[:, theirs], paper[:, theirs])
            d.append(lab(X[:, mine]) - lab(other))
        d = np.stack(d, -2).reshape(X.shape[0] // 8, 8, len(xl), 2, 3)
        return {k: np.linalg.norm(d[..., g], axis=-1).mean((1, 3)) for k, g in GROUPS.items()}
    v, h = vertical(X, pairs), vertical(X.transpose(1, 0, 2), pairs.transpose(1, 0, 2, 3))
    return {k: np.concatenate([h[k].T.ravel(), v[k].ravel()]) for k in GROUPS}


def attributes(z) -> dict:
    """The coherence term's own: {'A attributes': V, 'A x edge': V times the target's edge weight}, from make's npz."""
    v = np.concatenate([z['Vh'].ravel(), z['Vv'].ravel()])
    return {'A attributes': {'L': v}, 'A x edge': {'L': v * np.concatenate([z['Lh'].ravel(), z['Lv'].ravel()])}}


# ---- the user's cases

ZX = {ch: c for row, cs in zip(('kbrmgcyw', 'KBRMGCYW'), np.split(ZXPalette().as_float(), 2)) for ch, c in zip(row, cs)}
"""The app's Spectrum colours, sRGB 0..1, by letter: black, blue, red, magenta, green, cyan, yellow, white; capitals
bright (K, bright black, is black)."""


def dots(paper, ink, cols):
    """A cell of ink dots on paper in every other row, in these columns."""
    cell = np.empty((8, 8, 3))
    cell[:] = ZX[paper]
    for c in cols:
        cell[::2, c] = ZX[ink]
    return lambda rng: cell


def noise(paper, ink, share=0.5):
    return lambda rng: np.where(rng.random((8, 8, 1)) < share, ZX[ink], ZX[paper])


def flat(colour):
    return dots(colour, colour, ())


CASES = (   # (what, left cells, right cells, their pairs, the user's word)
    ('flat red | flat magenta', flat('r'), flat('m'), ('rr', 'mm'), 'shows'),
    ('red | magenta dots on black, black along the border', dots('k', 'r', (1, 2, 5, 6)), dots('k', 'm', (1, 2, 5, 6)),
     ('kr', 'km'), 'none'),
    ('the same, the dots on the border', dots('k', 'r', (0, 3, 4, 7)), dots('k', 'm', (0, 3, 4, 7)), ('kr', 'km'),
     ''),
    ('50% noise, white on black both', noise('k', 'w'), noise('k', 'w'), ('kw', 'kw'), 'none'),
    ('50% noise, bright | dim white on black', noise('k', 'W'), noise('k', 'w'), ('kW', 'kw'), 'none'),
)


def check(seed=0):
    """Each measure on each case: 4 x 4 cells, the left two columns of cells one kind, the right two the other, against
    a flat target of the patch's mean; the mean over the 4 seams between the kinds. A measure passes when every case
    the user called seamless scores below the one the user sees, lightness and colour summed (case 1 is a colour
    seam: on lightness alone nothing passes); its L and ab rows below it for the record."""
    rng = np.random.default_rng(seed)
    pairs = sorted({p for case in CASES for p in case[3]})
    V = pair_dissimilarity(np.array([[ZX[a], ZX[b]] for a, b in pairs]))
    rows, middle = {}, np.zeros((4, 4, 2), bool)   # the vertical seams' (R, C-1) = (4, 3), column 1
    for what, left, right, (pa, pb), _ in CASES:
        X = np.concatenate([np.concatenate([(left if c < 2 else right)(rng) for c in range(4)], 1)
                            for _ in range(4)], 0)
        m = measures(X, np.broadcast_to(encode(linear(X).mean((0, 1))), X.shape))
        n = 3 * 4   # the h seams first: (3, 4); then v (4, 3), the middle column 1
        pick = lambda s: s[n:].reshape(4, 3)[:, 1].mean()
        rows[what] = {('A attributes', ''): V[pairs.index(pa), pairs.index(pb)]}
        for f, ks in m.items():
            rows[what][f, ''] = pick(ks['L'] + ks['ab'])
            rows[what].update({(f, k): pick(s) for k, s in ks.items()})
    print(f"{'measure':28}" + ''.join(f'{i + 1:>8}' for i in range(len(CASES))) + '   pass')
    for f, k in rows[CASES[0][0]]:
        v = [rows[c[0]][f, k] for c in CASES]
        shows = min(x for x, c in zip(v, CASES) if c[4] == 'shows')
        ok = all(x < shows for x, c in zip(v, CASES) if c[4] == 'none')
        print(f"{'    ' + k if k else f:28}" + ''.join(f'{x:8.2f}' for x in v) + (k and ' ' or f"   {'yes' if ok else 'no'}"))
    print()
    for i, c in enumerate(CASES):
        print(f"{i + 1}: {c[0]}{f' (the user: {c[4]})' if c[4] else ''}")


# ---- the renders

def gallery_best(name: str) -> dict:
    """Project's node params at the picture's gallery best (the last pick), DBS on."""
    import gallery
    s = json.loads((gallery.ROUND / f'{name}.json').read_text())
    p = gallery.params(s, gallery.last(s))
    p['optimise']['enabled'] = True
    return p


def merged(a: dict, b: dict) -> dict:
    return {nid: {**a.get(nid, {}), **b.get(nid, {})} for nid in {*a, *b}}


def render(job):
    """Writes the render as png, and as npz the coherence term's seams (V of the two cells' pairs, the edge weights)
    and each cell's pair, (R, C, 2, 3) paper and ink, sRGB 0..1."""
    name, params, path = job
    p = Project(name, **params)
    X, sel = p.convert(), p.selection
    labels, V = sel.best_attr_indexes, sel.pair_dissimilarity
    Lh, Lv = sel.energy.seam_smoothness()
    np.savez_compressed(path.with_suffix('.npz'), Vh=V[labels[:-1], labels[1:]], Vv=V[labels[:, :-1], labels[:, 1:]],
                        Lh=Lh, Lv=Lv, pairs=sel.color_pairs[labels])
    png = lambda Y, f: cv2.imwrite(str(f), cv2.cvtColor(np.round(Y * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    png(X, path)
    png(p.target(), OUT / f'{name}-target.png')


def make(seed=0):
    """A blind order: each picture once in a random variant, then each again in the other, then REPEATS of the first
    block once more."""
    from tune import TRAIN
    rng, names = np.random.default_rng(seed), list(VARIANTS)
    first = rng.integers(len(names), size=len(TRAIN))
    block = lambda shift: [(TRAIN[i], names[(first[i] + shift) % len(names)]) for i in rng.permutation(len(TRAIN))]
    items = block(0) + block(1)
    key = [dict(id=f's{i:02d}', name=n, variant=v) for i, (n, v) in enumerate(items)]
    key += [dict(key[i], id=f's{len(key) + j:02d}', repeat_of=key[i]['id'])
            for j, i in enumerate(rng.choice(len(TRAIN), REPEATS, replace=False))]
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with Pool(len(items)) as pool:
        pool.map(render, [(n, merged(gallery_best(n), VARIANTS[v]), OUT / f'{n}-{v}.png') for n, v in items])
    (ROUND / 'key.json').write_text(json.dumps(key, indent=1))
    print(f'{len(items)} renders in {time.time() - t:.0f} s; {len(key)} to paint')


# ---- the page

PAGE = """<!doctype html><meta charset="utf-8"><title>dizher seams</title>
<style>body{margin:0;background:#222;color:#ddd;font:15px system-ui}b{color:#fff}#info{padding:6px 8px}
canvas{display:block;margin:0 8px;cursor:crosshair}</style>
<div id="info"></div><canvas id="c"></canvas>
<script>
const S = %s, R = S.R, C = S.C;
const $ = id => document.getElementById(id), cv = $('c'), ctx = cv.getContext('2d');
let marks = new Map(S.marks.map(([k, r, c, l]) => [`${k},${r},${c}`, l]));
const undo = [], img = new Image(), src = new Image(), t0 = Date.now();
let zoom = +(localStorage.seamZoom || 3), source = false, hide = false, stroke = null;
const q = () => Math.round(zoom * devicePixelRatio);   // screen pixels per Spectrum pixel
function draw() {
  const k = q();
  cv.width = 8 * C * k; cv.height = 8 * R * k; cv.style.width = 8 * C * k / devicePixelRatio + 'px';
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(source ? src : img, 0, 0, cv.width, cv.height);
  if (!source && !hide) {
    ctx.lineWidth = Math.max(1, Math.round(devicePixelRatio));
    for (const [key, l] of marks) {
      const [o, r, c] = key.split(',').map((x, i) => i ? +x : x);
      ctx.strokeStyle = l === 2 ? 'rgba(255,40,40,.85)' : 'rgba(255,220,0,.85)';
      ctx.beginPath();
      if (o === 'v') { ctx.moveTo(8 * (c + 1) * k, 8 * r * k); ctx.lineTo(8 * (c + 1) * k, 8 * (r + 1) * k); }
      else { ctx.moveTo(8 * c * k, 8 * (r + 1) * k); ctx.lineTo(8 * (c + 1) * k, 8 * (r + 1) * k); }
      ctx.stroke();
    }
  }
  $('info').innerHTML = `<b>${S.i + 1} of ${S.n}</b>${S.done ? ' (done)' : ''} &middot; ${marks.size} seams &middot;
    zoom &times;${zoom} (+/&minus;)<br>Mark every cell border that shows where the source has none (clash,
    blockiness): drag along it; shift-drag: strong; alt-drag: erase. Hold S: the source; hold H: hide the marks;
    Z: undo; Enter: done, next; Backspace: back`;
}
function seamAt(x, y, o) {   // the seam under (x, y) in Spectrum pixels; o, a stroke's orientation, or the nearer
  const cx = Math.round(x / 8), cy = Math.round(y / 8), dx = Math.abs(x - 8 * cx), dy = Math.abs(y - 8 * cy);
  const r = Math.floor(y / 8), c = Math.floor(x / 8);
  o = o || (dx <= dy ? 'v' : 'h');
  if (o === 'v' && dx <= 2.5 && cx >= 1 && cx < C && r >= 0 && r < R) return `v,${r},${cx - 1}`;
  if (o === 'h' && dy <= 2.5 && cy >= 1 && cy < R && c >= 0 && c < C) return `h,${cy - 1},${c}`;
  return null;
}
const at = e => [e.offsetX * devicePixelRatio / q(), e.offsetY * devicePixelRatio / q()];
function paint(x, y, o) {
  const key = seamAt(x, y, o);
  if (key) stroke.level ? marks.set(key, stroke.level) : marks.delete(key);
}
cv.onmousedown = e => {
  e.preventDefault();
  const [x, y] = at(e);
  stroke = {level: e.altKey ? 0 : e.shiftKey ? 2 : 1, x0: x, y0: y, x, y, before: new Map(marks)};
  paint(x, y, null); draw();
};
cv.onmousemove = e => {
  if (!stroke) return;
  const [x, y] = at(e), s = stroke, moved = Math.hypot(x - s.x0, y - s.y0) > 2;
  const o = moved ? (Math.abs(y - s.y0) > Math.abs(x - s.x0) ? 'v' : 'h') : null;   // along the stroke's direction
  const n = Math.ceil(2 * Math.hypot(x - s.x, y - s.y));
  for (let i = 1; i <= n; i++) paint(s.x + (x - s.x) * i / n, s.y + (y - s.y) * i / n, o);
  s.x = x; s.y = y; draw();
};
const send = (done = false) => fetch('/marks', {method: 'POST', body: JSON.stringify(
  {id: S.id, done, zoom, seconds: (Date.now() - t0) / 1000,
   marks: [...marks].map(([key, l]) => [...key.split(',').map((x, i) => i ? +x : x), l])})});
addEventListener('mouseup', () => { if (stroke) { undo.push(stroke.before); stroke = null; send(); } });
addEventListener('keydown', async e => {
  if (e.repeat) return;
  const key = e.key.toLowerCase();
  if (key === 's') { source = true; draw(); }
  if (key === 'h') { hide = true; draw(); }
  if (key === 'z' && undo.length) { marks = undo.pop(); draw(); send(); }
  if (e.key === '+' || e.key === '=' || e.key === '-') {
    zoom = Math.max(1, zoom + (e.key === '-' ? -1 : 1)); localStorage.seamZoom = zoom; draw();
  }
  if (e.key === 'Enter') { await send(true); location = '/?i=' + (S.i + 1); }
  if (e.key === 'Backspace' && S.i > 0) { e.preventDefault(); await send(); location = '/?i=' + (S.i - 1); }
});
addEventListener('keyup', e => {
  const key = e.key.toLowerCase();
  if (key === 's') { source = false; draw(); }
  if (key === 'h') { hide = false; draw(); }
});
addEventListener('blur', () => { source = hide = false; draw(); });
let loaded = 0;
img.onload = src.onload = () => ++loaded === 2 && draw();
img.src = '/img/' + S.id; src.src = '/target/' + S.id;
</script>"""

DONE_PAGE = """<!doctype html><meta charset="utf-8"><body style="background:#222;color:#ddd;font:15px system-ui">
All %d painted, thank you. Backspace: back.<script>addEventListener('keydown', e => e.key === 'Backspace' &&
(location = '/?i=%d'))</script>"""


def paint(port=8767):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
    key = {k['id']: k for k in json.loads((ROUND / 'key.json').read_text())}
    ids, path = list(key), ROUND / 'marks.json'
    marks = json.loads(path.read_text()) if path.exists() else {}
    file = lambda i: OUT / f"{key[i]['name']}-{key[i]['variant']}.png"
    R, C = (n // 8 for n in cv2.imread(str(file(ids[0]))).shape[:2])

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
                return self.send(file(Path(url.path).name).read_bytes(), 'image/png')
            if url.path.startswith('/target/'):
                return self.send((OUT / f"{key[Path(url.path).name]['name']}-target.png").read_bytes(), 'image/png')
            if url.path != '/':
                return self.send(b'', 'text/plain')
            todo = [i for i, k in enumerate(ids) if not marks.get(k, {}).get('done')]
            i = int(parse_qs(url.query).get('i', [todo[0] if todo else len(ids)])[0])
            if i >= len(ids):
                return self.send((DONE_PAGE % (len(ids), len(ids) - 1)).encode(), 'text/html')
            m = marks.get(ids[i], {})
            state = dict(i=i, n=len(ids), id=ids[i], R=R, C=C, marks=m.get('marks', []), done=m.get('done', False))
            self.send(PAGE.replace('%s', json.dumps(state), 1).encode(), 'text/html')

        def do_POST(self):
            v = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            m = marks.setdefault(v['id'], {})
            m.update(marks=v['marks'], zoom=v['zoom'], done=m.get('done', False) or v['done'])
            if v['done'] and 'seconds' not in m:
                m['seconds'] = round(v['seconds'], 1)
            path.write_text(json.dumps(marks, indent=1))
            self.send(b'ok', 'text/plain')

        def log_message(self, *args):
            pass

    print(f'http://localhost:{port}  (Ctrl+C to stop; marks in {path})', flush=True)
    webbrowser.open(f'http://localhost:{port}')
    HTTPServer(('localhost', port), Handler).serve_forever()


# ---- the report

def auc(score, y) -> float:
    """The chance a marked seam scores above an unmarked one, ties half."""
    pos = np.asarray(y) > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if not n1 or not n0:
        return np.nan
    return (rankdata(score)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def labels(m: dict, R: int, C: int) -> np.ndarray:
    """The seams' vector of the marks' levels, 0 unmarked."""
    y = np.zeros((R - 1) * C + R * (C - 1), int)
    for o, r, c, level in m['marks']:
        y[r * C + c if o == 'h' else (R - 1) * C + r * (C - 1) + c] = level
    return y


ANGLES = np.linspace(0, np.pi / 2, 9)   # the lightness-colour mix: cos L + sin ab


def mixed_auc(f: dict, y, keep) -> np.ndarray:
    """AUC at each mix of the family's L and ab (one score: the same at each)."""
    if 'ab' not in f:
        return np.full(len(ANGLES), auc(f['L'][keep], y[keep]))
    return np.array([auc((np.cos(a) * f['L'] + np.sin(a) * f['ab'])[keep], y[keep]) for a in ANGLES])


def report():
    key = {k['id']: k for k in json.loads((ROUND / 'key.json').read_text())}
    marks = json.loads((ROUND / 'marks.json').read_text())
    done = [i for i in key if marks.get(i, {}).get('done')]
    rows = []   # per first-pass render: its picture, labels, features, smooth seams
    for i in done:
        k = key[i]
        if 'repeat_of' in k:
            continue
        f = OUT / f"{k['name']}-{k['variant']}.png"
        rgb = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        X, T, z = rgb(f), rgb(OUT / f"{k['name']}-target.png"), np.load(f.with_suffix('.npz'))
        R, C = X.shape[0] // 8, X.shape[1] // 8
        y = labels(marks[i], R, C)
        smooth = np.concatenate([z['Lh'].ravel(), z['Lv'].ravel()]) >= 0.5
        changed = attributes(z)['A attributes']['L'] > 0
        rows.append(dict(name=k['name'], variant=k['variant'], y=y, smooth=smooth, changed=changed,
                         f={**attributes(z), **measures(X, T), 'E switch': switch(X, z['pairs'])}))
        print(f"{i} {k['name']:15} {k['variant']:11} {(y > 0).sum():4} marked ({(y > 1).sum()} strong) of {y.size}, "
              f"{marks[i].get('seconds', 0):.0f} s")
    pictures = sorted({r['name'] for r in rows})
    y, ch, flat = (np.concatenate([r[k] for r in rows]) for k in ('y', 'changed', 'smooth'))
    print(f"\n{(y[ch] > 0).sum()} of the {(y > 0).sum()} marks on the seams where the pair changes, {ch.sum()} of "
          f"{ch.size}: the score is on those (no measure sees a seam where the pair stays, and they flatter every "
          f"measure). Marked: {(y[ch & ~flat] > 0).mean():.2f} of them where the coherence term's edge weight "
          f"excuses them (< 0.5), {(y[ch & flat] > 0).mean():.2f} where it does not")
    print(f"AUC, mean over {len(rows)} renders: 'L', 'ab' each alone; 'both' the L-ab mix chosen on the other "
          f"pictures; 'top k' at that mix, the share of the k seams it ranks highest that the user marked (k, the "
          f"user's marks; the repeats' ceiling below); 'all' on all seams; 'flat' only where the target is flat (edge "
          f"weight >= 0.5); 'strong' strong against the rest")
    print(f"{'measure':24} {'L':>5} {'ab':>5} {'both':>5} {'top k':>5} {'all':>5} {'flat':>5} {'strong':>6}   both per render")
    for fam in rows[0]['f']:
        one = 'ab' not in rows[0]['f'][fam]
        mix = lambda f, a: f['L'] if one else np.cos(ANGLES[a]) * f['L'] + np.sin(ANGLES[a]) * f['ab']
        grid = {id(r): mixed_auc(r['f'][fam], r['y'], r['changed']) for r in rows}
        per = []
        for r in rows:   # the mix best on the other pictures' renders
            other = np.nanmean([grid[id(o)] for o in rows if o['name'] != r['name']], 0)
            per.append((r, int(np.nanargmax(other)) if len(pictures) > 1 else 0))
        both = [grid[id(r)][a] for r, a in per]
        top = [top_share(mix(r['f'][fam], a)[r['changed']], r['y'][r['changed']]) for r, a in per]
        every = [mixed_auc(r['f'][fam], r['y'], np.ones_like(r['changed']))[a] for r, a in per]
        smooth = [mixed_auc(r['f'][fam], r['y'], r['changed'] & r['smooth'])[a] for r, a in per]
        strong = [mixed_auc(r['f'][fam], (r['y'] > 1).astype(int), r['changed'])[a] for r, a in per]
        L = np.nanmean([g[0] for g in grid.values()])
        ab = np.nan if one else np.nanmean([g[-1] for g in grid.values()])
        print(f"{fam:24} {L:5.2f} {ab:5.2f} {np.nanmean(both):5.2f} {np.mean(top):5.2f} {np.nanmean(every):5.2f} "
              f"{np.nanmean(smooth):5.2f} {np.nanmean(strong):6.2f}   {' '.join(f'{b:.2f}' for b in both)}")
    print("\nthe repeats (the same render painted twice): the share of one painting's marks the other has, each way "
          "(the ceiling of 'top k'), and the AUC of one painting as a measure of the other on the changed seams")
    for i in done:
        k = key[i]
        if k.get('repeat_of') in done:
            f = OUT / f"{k['name']}-{k['variant']}.png"
            R, C = (n // 8 for n in cv2.imread(str(f)).shape[:2])
            a, b = labels(marks[k['repeat_of']], R, C) > 0, labels(marks[i], R, C) > 0
            c = np.concatenate([np.load(f.with_suffix('.npz'))[v].ravel() for v in ('Vh', 'Vv')]) > 0
            print(f"  {k['name']:15} {k['variant']:11} {(a & b).sum() / a.sum():.2f} {(a & b).sum() / b.sum():.2f}, "
                  f"AUC {(auc(a[c], b[c]) + auc(b[c], a[c])) / 2:.2f}")
    painted()


def top_share(score, y) -> float:
    """The share of the k seams the score ranks highest that the user marked; k, the user's marks."""
    marked = np.asarray(y) > 0
    return float(marked[np.argsort(-score, kind='stable')[:marked.sum()]].mean())


def painted_image(path: Path) -> list:
    """Per painted segment of build.py: per measure, the painting's (A) and Select pairs' (B) sums over the seams of
    the segment's window (its cells grown by 2, as score.py)."""
    z = np.load(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    f = lambda k: z[k].astype(np.float32) / 255
    T, fa = f('target'), measures(f('A'), f('target'))
    out = []
    for m in meta:
        s = m['segment']
        grown = cv2.dilate((z['diff'] & (z['segments'] == s)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        keep = np.concatenate([(grown[:-1] & grown[1:]).ravel(), (grown[:, :-1] & grown[:, 1:]).ravel()])
        fb = measures(f(f'B{s}'), T)
        out.append({(fam, k): (fa[fam][k][keep].sum(), fb[fam][k][keep].sum()) for fam in fa for k in fa[fam]})
    return out


def painted():
    """How often each measure rates the user's painting below Select pairs' cells: seam_excess, 0.65 (README 1)."""
    files = [p for p in sorted(DATA.glob('*.npz'))]
    with Pool(4) as pool:
        rows = [r for rs in pool.map(painted_image, files) for r in rs]
    print(f'\nthe painted segments ({len(rows)}): the share where the measure rates the painting below Select pairs '
          f'(seam_excess 0.65, README 1)')
    share = lambda a, b: ((a < b) + 0.5 * (a == b)).mean()
    for fam in dict.fromkeys(f for f, _ in rows[0]):
        (aL, bL), (aab, bab) = (np.array([r[(fam, k)] for r in rows]).T for k in GROUPS)
        print(f'  {fam:24} L {share(aL, bL):.2f}  ab {share(aab, bab):.2f}  L + ab {share(aL + aab, bL + bab):.2f}')


if __name__ == '__main__':
    {'make': make, 'paint': paint, 'check': check, 'report': report}[sys.argv[1]]()
