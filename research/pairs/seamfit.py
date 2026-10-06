"""PLAN step 5c: one seam score from all the user's seam data, to replace the coherence term's attributes. The score
is a weighted sum of the seam measures (seams.measures and seams.switch, lightness and colour apart), the weights
>= 0, fitted at once to:
  - the painted renders (seams.py): on the seams where the pair changes, marked or not; each render its own rate;
  - the synthetic patches (patches.py, rounds v3-v5): the order of their mean ratings, each round apart;
  - the ranked sheets the user sorted (sort): the order within each sheet.
Each data set's loss is its mean (a set of groups, the mean over them), so the 7197 seams do not drown the patches.

    python research/pairs/seamfit.py     the fits on all, on the renders alone, on the patches alone, and the joint
                                         fit with each picture or round left out: the weights, and per data set the
                                         renders' AUC and top k, the rounds' Spearman rho
    python research/pairs/seamfit.py ranked   per global lightness (LIGHTNESS, L*), every two Spectrum pairs that
                                         reach it side by side as a patch, ranked by SEAM; 16 from least to most
                                         visible on a sheet, data/seams/ranked/; the picks (each patch, its spec and
                                         score) in rounds/seams/ranked-key.json
    python research/pairs/seamfit.py sort     the page where the user drags the picks of each sheet into the order
                                         seen, least to most visible; each move to rounds/seams/ranked.json. A done
                                         round is kept as ranked-key-vN.json, ranked-vN.json (SORTED), its images in
                                         data/seams/ranked-vN/, before ranked runs again
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patches   # noqa: E402
import seams   # noqa: E402
from dizher.converter.energy import pair_dissimilarity   # noqa: E402

MEASURES = [f'{fam} {g}' for fam in ('E switch', 'M cell means', 'B pixel step raw', 'B pixel step blur 2',
                                     'C line vs texture raw', 'D strips raw', 'K texture step', 'S solid step',
                                     'T cell texture', 'Y cell lightness')
            for g in ('L', 'ab')]
PRODUCTS = [(f'{a} {g}', 'T cell texture L+ab') for a in ('E switch', 'M cell means') for g in ('L', 'ab')] + \
    [(f, 'Y cell lightness L') for f in ('E switch L', 'M cell means ab', 'S solid step L')]
FEATURES = [m for m in MEASURES if not m[0] in 'TY'] + [f'{a} x {b}' for a, b in PRODUCTS]


def features(m: dict) -> list:
    """FEATURES from {measure name: value or array}: the measures, and the colour changes times the cells' texture
    (L + ab, over its mean on the renders, TEXTURE)."""
    m = dict(m, **{'T cell texture L+ab': (m['T cell texture L'] + m['T cell texture ab']) / TEXTURE})
    return [m[f] for f in FEATURES if ' x ' not in f] + [m[a] * m[b] for a, b in PRODUCTS]


TEXTURE = 40.6   # the mean of the renders' T cell texture L + ab on the changed seams (dE): a unit


def score(m: dict):
    """SEAM of {measure name: value or array}, as features makes them."""
    f = dict(zip(FEATURES, features(m)))
    return sum(w * f[k] for k, w in SEAM.items())
ROUNDS = ('v3', 'v4', 'v5')
SEAM = {'M cell means ab': 0.0543, 'E switch L x Y cell lightness L': 0.1263,   # the score (README 17): per dE,
        'S solid step L x Y cell lightness L': 1.678}                            # lightness 0..1, solidity 0..1
LIGHTNESS = (15, 26, 37, 48, 59, 70, 81, 92)   # L* of the ranked sheets
REACH = 0.15, 0.85   # a pair reaches a lightness at a level within these, so the slow variation stays in its range
RANKED_KEY, RANKED = seams.ROUND / 'ranked-key.json', seams.ROUND / 'ranked.json'   # a new round
SORTED = ('v1',)   # the kept sorting rounds (ranked-key-vN.json, ranked-vN.json; v1, picked by the two-term score)


def renders() -> list:
    """Per first-pass painted render: its picture, the changed seams' features (n, k) and marks (n,)."""
    key = {k['id']: k for k in json.loads((seams.ROUND / 'key.json').read_text())}
    marks = json.loads((seams.ROUND / 'marks.json').read_text())
    rgb = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    out = []
    for i, k in key.items():
        if 'repeat_of' in k or not marks.get(i, {}).get('done'):
            continue
        f = seams.OUT / f"{k['name']}-{k['variant']}.png"
        X, T, z = rgb(f), rgb(seams.OUT / f"{k['name']}-target.png"), np.load(f.with_suffix('.npz'))
        m = {**seams.measures(X, T), 'E switch': seams.switch(X, z['pairs'])}
        changed = seams.attributes(z)['A attributes']['L'] > 0
        F = np.stack(features({f'{fam} {g}': v for fam, ks in m.items() for g, v in ks.items()}), 1)[changed]
        y = seams.labels(marks[i], X.shape[0] // 8, X.shape[1] // 8)[changed] > 0
        out.append(dict(name=k['name'], F=F, y=y.astype(float)))
    return out


def rounds() -> list:
    """Per round of patches a data set of one group: the patches' features (n, k) at the lightness the user set, their
    mean ratings; then the sorted sheets."""
    out = []
    for version in ROUNDS:
        key = json.loads((patches.ROUND / f'patches-key-{version}.json').read_text())
        ratings = json.loads((patches.ROUND / f'patches-{version}.json').read_text())
        got = {}
        for k in key:
            if k['id'] in ratings:
                got.setdefault(k['stimulus'], (k, []))[1].append((ratings[k['id']]['rating'], ratings[k['id']]['k']))
        pairs = sorted({k[side] for k in key for side in ('left', 'right')})
        V = pair_dissimilarity(np.array([[patches.ZX[c] for c in pair] for pair in pairs]))
        V = {(a, b): float(V[i, j]) for i, a in enumerate(pairs) for j, b in enumerate(pairs)}
        F, y = [], []
        for k, rs in got.values():
            shift = round(float(np.mean([s for _, s in rs]))) * patches.STEP
            v = patches.seam_values(k, V, shift)
            F.append(features(v))
            y.append(np.mean([r for r, _ in rs]))
        out.append(dict(name=version, groups=[(np.array(F), np.array(y))]))
    return out + [sheets()]


def sheets() -> dict:
    """The sheets the user sorted, a data set of a group each: the picks' features and the user's order (y, the
    position: the most visible highest)."""
    from collections import defaultdict
    groups = []
    for version in SORTED:
        key = json.loads((seams.ROUND / f'ranked-key-{version}.json').read_text())
        done = json.loads((seams.ROUND / f'ranked-{version}.json').read_text())
        for L, v in done.items():
            if v.get('done'):
                picks = {p['id']: p for p in key[L]}
                F = [features(patches.seam_values(picks[i], defaultdict(float))) for i in v['order']]
                groups.append((np.array(F), np.arange(len(F), dtype=float)))
    return dict(name='sorted', groups=groups)


def loss(w, rs, ps, scale):
    """The renders' logistic loss (each its own bias, fitted inside) and the rounds' pairwise ranking loss, each a
    mean; w >= 0 on the features divided by scale."""
    total = 0.0
    if rs:
        parts = []
        for r in rs:
            s = r['F'] / scale @ w
            b = minimize(lambda b: np.logaddexp(0, -(2 * r['y'] - 1) * (s - b[0])).mean(), [np.median(s)]).x[0]
            parts.append(np.logaddexp(0, -(2 * r['y'] - 1) * (s - b)).mean())
        total += np.mean(parts)
    if ps:
        parts = []
        for p in ps:
            within = []
            for F, y in p['groups']:
                s = F / scale @ w
                i, j = np.nonzero(y[:, None] > y[None, :])
                within.append(np.logaddexp(0, -(s[i] - s[j])).mean())
            parts.append(np.mean(within))
        total += np.mean(parts)
    return total + 1e-3 * w @ w


def fit(rs, ps, scale, use=None) -> np.ndarray:
    """w >= 0; use, the features allowed (all by default), the rest held at 0."""
    k = len(FEATURES)
    use = np.ones(k, bool) if use is None else use
    return minimize(loss, np.full(k, 0.1) * use, args=(rs, ps, scale), method='L-BFGS-B',
                    bounds=[(0, None) if u else (0, 0) for u in use]).x


def scores(w, rs, ps, scale) -> dict:
    """{data set: (AUC, top k) for the renders, mean over them; Spearman rho for each round, the mean over its
    groups}."""
    out = {}
    if rs:
        a, t = [], []
        for r in rs:
            s = r['F'] / scale @ w
            a.append(seams.auc(s, r['y']))
            t.append(seams.top_share(s, r['y']))
        out['renders'] = (np.mean(a), np.mean(t))
    for p in ps:
        out[p['name']] = np.mean([spearmanr(F / scale @ w, y)[0] for F, y in p['groups']])
    return out


def line(title, sc: dict) -> str:
    r = sc.get('renders')
    return f"{title:34} renders {r[0]:.2f} {r[1]:.2f}" * bool(r) + ''.join(
        f"  {k} {v:+.2f}" for k, v in sc.items() if k != 'renders')


def main():
    rs, ps = renders(), rounds()
    scale = np.concatenate([r['F'] for r in rs]).std(0) + 1e-9   # the renders' spread: weights comparable
    print(f"{len(rs)} renders, {sum(len(r['y']) for r in rs)} changed seams; rounds "
          + ', '.join(f"{p['name']} {sum(len(y) for _, y in p['groups'])}" for p in ps))
    print("\neach measure alone (L + ab, equal in dE): renders AUC, top k; rounds rho")
    for fam in dict.fromkeys(f.rsplit(' ', 1)[0] for f in FEATURES if ' x ' not in f):
        w = np.array([float(f.startswith(fam + ' ') and ' x ' not in f) for f in FEATURES]) * scale
        print(line('  ' + fam, scores(w, rs, ps, scale)))
    plain = np.array([' x ' not in f for f in FEATURES])
    fits = {'all': fit(rs, ps, scale), 'all, no texture': fit(rs, ps, scale, plain),
            'renders alone': fit(rs, [], scale), 'patches alone': fit([], ps, scale)}
    print('\nthe fits, scored on every data set (in sample for what they were fitted on)')
    for name, w in fits.items():
        print(line('  ' + name, scores(w, rs, ps, scale)))
    print('\nthe joint fit with a picture or a round left out, scored on it')
    held = {}
    for name in sorted({r['name'] for r in rs}):
        w = fit([r for r in rs if r['name'] != name], ps, scale)
        a = scores(w, [r for r in rs if r['name'] == name], [], scale)['renders']
        held[name] = a
    print('  renders, each picture out: AUC ' + ' '.join(f'{n} {a:.2f}' for n, (a, _) in held.items())
          + f'; mean {np.mean([a for a, _ in held.values()]):.2f}, top k {np.mean([t for _, t in held.values()]):.2f}')
    for p in ps:
        w = fit(rs, [q for q in ps if q is not p], scale)
        print(f"  {p['name']} out: rho {scores(w, [], [p], scale)[p['name']]:+.2f}")
    print('\nthe weights (per the renders\' sd of each measure), the joint fit:')
    for f, v in zip(FEATURES, fits['all']):
        print(f'  {f:28} {v:6.2f}')


def ranked(per_sheet=16, columns=4, zoom=3):
    """Per lightness in LIGHTNESS: every two Spectrum pairs (paper, ink, one brightness; not one colour twice) that
    reach its luminance at a level within REACH, as a patch (the left at that level, the right matched, the slow
    variation, no gradient), scored by SEAM on the patch as drawn; per_sheet of them evenly along the ranking."""
    from collections import defaultdict
    from dizher.platforms.zxspectrum import ZXPalette
    letter = lambda i: 'kbrmgcyw'[i & 7].upper() if i >= 8 else 'kbrmgcyw'[i]
    pairs = [letter(a) + letter(b) for a, b in ZXPalette().iter_idxs_pairs() if a != b]
    span = {p: [patches.luminance(c) for c in p] for p in pairs}
    out = seams.OUT / 'ranked'
    out.mkdir(parents=True, exist_ok=True)
    rng, key = np.random.default_rng(0), {}
    for L in LIGHTNESS:
        Y = ((L + 16) / 116) ** 3
        level = {p: (Y - lo) / (hi - lo) for p, (lo, hi) in span.items()}
        near = [p for p in pairs if REACH[0] <= level[p] <= REACH[1]]
        rows = []
        for i, a in enumerate(near):
            for b in near[i + 1:]:
                if not span[b][0] <= span[a][0] <= span[a][1] <= span[b][1]:
                    a, b = b, a   # the right follows the left: the left's range within the right's where it can be
                s = dict(left=a, right=b, level=level[a], right_level='match', halftone='blue', dots='continue',
                         angle=0.0, slope=0.0, seed=int(rng.integers(2 ** 31)),
                         origin=tuple(int(v) for v in rng.integers(64, size=2)))
                v = patches.seam_values(s, defaultdict(float))
                rows.append((score(v), v, s))
        rows.sort(key=lambda r: r[0])
        pick = [rows[round(k)] for k in np.linspace(0, len(rows) - 1, min(per_sheet, len(rows)))]
        tiles = []
        for n, (score, v, s) in enumerate(pick):
            X = patches.picture(s, s['origin'])
            key.setdefault(str(L), []).append(dict(s, id=f'L{L:02d}-{n:02d}', score=score, dL=v['E switch L'],
                                                   dab=v['M cell means ab']))
            cv2.imwrite(str(out / f'L{L:02d}-{n:02d}.png'), np.round(X * 255).astype(np.uint8)[..., ::-1])
            tile = np.repeat(np.repeat(np.round(X * 255).astype(np.uint8)[..., ::-1], zoom, 0), zoom, 1)
            label = np.full((34, tile.shape[1], 3), 34, np.uint8)
            cv2.putText(label, f"{s['left']} | {s['right']}   {score:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(label, f"dL {v['E switch L']:.1f}  dab {v['M cell means ab']:.1f}", (4, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
            tiles.append(np.pad(np.concatenate([tile, label], 0), ((6, 6), (6, 6), (0, 0)), constant_values=34))
        tiles += [np.full_like(tiles[0], 34)] * (-len(tiles) % columns)
        grid = np.concatenate([np.concatenate(tiles[r:r + columns], 1) for r in range(0, len(tiles), columns)], 0)
        title = np.full((30, grid.shape[1], 3), 34, np.uint8)
        cv2.putText(title, f"L* {L}: {len(rows)} pairs of pairs from {len(near)} pairs, least to most visible by the "
                    f"score, {len(pick)} evenly along the ranking", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f'L{L:02d}.png'), np.concatenate([title, grid], 0))
        print(f'L* {L}: {len(near)} pairs, {len(rows)} pairs of pairs; scores {rows[0][0]:.2f} .. {rows[-1][0]:.2f}')
    RANKED_KEY.write_text(json.dumps(key, indent=1))


SORT_PAGE = """<!doctype html><meta charset="utf-8"><title>dizher seams, sort</title>
<style>body{margin:0;background:#222;color:#ddd;font:15px system-ui}b{color:#fff}#top{padding:8px;display:flex;gap:8px;
flex-wrap:wrap;align-items:center}#top a{color:#ddd;text-decoration:none;padding:2px 8px;border:1px solid #555}
#top a.on{background:#fff;color:#000}#top a.done{border-color:#7c7}#grid{display:flex;flex-wrap:wrap;gap:10px;
padding:8px}.tile{position:relative;cursor:grab;padding:3px;border:1px solid #444}.tile.dragging{opacity:.35}
.tile img{display:block;image-rendering:pixelated}.n{position:absolute;left:4px;top:4px;background:#000c;color:#fff;
padding:0 5px;font-size:13px}button{font:inherit}</style>
<div id="top"></div><div id="info" style="padding:0 8px"></div><div id="grid"></div>
<script>
const S = %s, $ = id => document.getElementById(id), grid = $('grid');
let zoom = +(localStorage.sortZoom || 3), shuffled = S.shuffled;
$('top').innerHTML = S.levels.map(L => `<a href="/?L=${L}" class="${L === S.L ? 'on' : ''} ${S.done.includes(L) ? 'done' : ''}">L* ${L}</a>`).join('')
  + ' <button id="shuffle">Shuffle</button> <button id="reset">Back to the score order</button> <button id="fin">Done, next (Enter)</button>';
const tiles = new Map(S.ids.map(id => {
  const t = document.createElement('div'); t.className = 'tile'; t.draggable = true; t.dataset.id = id;
  t.innerHTML = `<span class="n"></span><img src="/img/${id}" draggable="false">`;
  t.ondragstart = e => { drag = t; t.classList.add('dragging'); e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', id); };
  t.ondragend = () => { t.classList.remove('dragging'); drag = null; save(false); };
  return [id, t];
}));
let drag = null;
function show(order) { grid.replaceChildren(...order.map(id => tiles.get(id))); draw(); }
function draw() {
  for (const img of grid.querySelectorAll('img')) img.style.width = 64 * zoom + 'px';
  [...grid.children].forEach((t, i) => t.querySelector('.n').textContent = i + 1);
  $('info').innerHTML = `<b>L* ${S.L}</b>: drag the patches into the order you see, <b>least visible seam first</b> (top left)
    to the most visible last; the seam is down the middle of each. ${S.ids.length} patches. +/&minus; zoom (&times;${zoom}).`;
}
grid.ondragover = e => {
  e.preventDefault();
  const t = e.target.closest('.tile');
  if (!drag || !t || t === drag) return;
  const r = t.getBoundingClientRect();
  grid.insertBefore(drag, e.clientX > r.left + r.width / 2 ? t.nextSibling : t);
  draw();
};
const order = () => [...grid.children].map(t => t.dataset.id);
const save = done => fetch('/order', {method: 'POST', body: JSON.stringify({L: S.L, order: order(), shuffled, done})});
$('shuffle').onclick = () => { const o = order(); for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; } shuffled = true; show(o); save(false); };
$('reset').onclick = () => { shuffled = false; show(S.ids); save(false); };
const next = async () => { await save(true); const i = S.levels.indexOf(S.L); location = i + 1 < S.levels.length ? `/?L=${S.levels[i + 1]}` : `/?L=${S.L}`; };
$('fin').onclick = next;
for (const b of document.querySelectorAll('button')) b.onmousedown = e => e.preventDefault();
addEventListener('keydown', e => {
  if (e.key === 'Enter') next();
  if (e.key === '+' || e.key === '=' || e.key === '-') { zoom = Math.max(1, zoom + (e.key === '-' ? -1 : 1)); localStorage.sortZoom = zoom; draw(); }
});
show(S.order);
</script>"""


def sort(port=8769):
    """The sorting page: per sheet the picks, in the score's order at first (or the user's, if saved), shown blind."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
    key = json.loads(RANKED_KEY.read_text())
    saved = json.loads(RANKED.read_text()) if RANKED.exists() else {}
    levels = [int(L) for L, picks in key.items() if len(picks) > 1]   # one patch: nothing to sort

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
                name = Path(url.path).name
                return self.send((seams.OUT / 'ranked' / f'{name}.png').read_bytes(), 'image/png')
            if url.path != '/':
                return self.send(b'', 'text/plain')
            todo = [L for L in levels if not saved.get(str(L), {}).get('done')]
            L = int(parse_qs(url.query).get('L', [(todo or levels)[0]])[0])
            ids = [p['id'] for p in key[str(L)]]
            mine = saved.get(str(L), {})
            state = dict(L=L, levels=levels, ids=ids, order=mine.get('order', ids), shuffled=mine.get('shuffled', False),
                         done=[int(k) for k, v in saved.items() if v.get('done')])
            self.send(SORT_PAGE.replace('%s', json.dumps(state), 1).encode(), 'text/html')

        def do_POST(self):
            v = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            m = saved.setdefault(str(v['L']), {})
            m.update(order=v['order'], shuffled=v['shuffled'], done=m.get('done', False) or v['done'])
            RANKED.write_text(json.dumps(saved, indent=1))
            self.send(b'ok', 'text/plain')

        def log_message(self, *args):
            pass

    print(f'http://localhost:{port}  (Ctrl+C to stop; the orders in {RANKED})', flush=True)
    webbrowser.open(f'http://localhost:{port}')
    HTTPServer(('localhost', port), Handler).serve_forever()


if __name__ == '__main__':
    {'ranked': ranked, 'sort': sort}.get(sys.argv[1] if sys.argv[1:] else '', main)()
