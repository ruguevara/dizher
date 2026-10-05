"""PLAN step 3: a gallery per picture. The converter's knobs (KNOBS, each over its slider in the app, scaled to 0..1)
searched by the user's eye: each round a 4x4 grid of renders (--side N, the same span) on a plane through the current
best point along two random directions; the user discards the bad ones, the rest laid out again as large as the window holds them, and
picks the best, or the last one left wins; the centre moves there, the step follows (GROW). When every direction is
bad, the round is drawn again from the same centre (REDRAW): other directions, or wider. Each render goes through
the project's whole pipeline, no cell painted (Project.convert); the start is the project's own settings, every knob
given, so the dots' weights do not follow the selection's. A new gallery's first round mixes the Metric methods: a
grid around each one's start (the project's own method at its settings, the other at its preset), shown together;
the pick's method is the gallery's from then on, and a redraw mixes them again.

    python research/pairs/gallery.py NAME [--no-dbs] [--side N]
                                              the gallery in the browser; every grid (its points, the
                                              layout) and each click saved at once to rounds/gallery/NAME.json,
                                              with the discards in order, renders in data/gallery/NAME/;
                                              Backspace takes the last click back (a discard, else a pick or redraw);
                                              Ctrl+C stops, run it again to go on. --no-dbs, for a new gallery:
                                              Optimise off and its three knobs out. --side N: an N x N grid from
                                              the next round (the pending one too, if untouched), over the 4x4's
                                              span: 3x3 sparser (steps of 0.3), 9x9 denser (0.075)
    python research/pairs/gallery.py best NAME   the last pick's settings against the start (NAME: the gallery's file)
    python research/pairs/gallery.py report      what the votes say: per picture a linear preference over the knobs and
                                              how much each knob changes the render, the knobs' effects alike
    python research/pairs/gallery.py defaults    per Metric method each knob's new default from the training galleries
"""
import json
import math
import sys
import time
import typing
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Project, project_graph   # noqa: E402
from dizher import ops   # noqa: E402
from dizher.converter.energy import METHODS   # noqa: E402
from mokit.graph import Memo   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'gallery', HERE / 'rounds' / 'gallery'
KNOBS = ('metric.chroma', 'metric.flare', 'select.coherence', 'select.edge', 'select.luma_noise',
         'select.chroma_noise', 'select.surface', 'halftone.chroma', 'optimise.luma_noise', 'optimise.chroma_noise',
         'optimise.structure', 'eye.luma_scale', 'eye.chroma_scale')
SIDE, STEP = 4, 0.2                # the grid's side by default; the first step on it, in the unit box. Any side spans
                                   # the same, STEP * (SIDE - 1) end to end along a direction: steps of 0.3 on a 3x3
SHRINK, GROW = 0.7, 1.25           # the span's factor when the pick is within a quarter of the grid's reach from the
                                   # centre (the grid reaches too far), or on its far edge (too short); else the same
REDRAW = {'directions': 1.0, 'wider': 1.5}   # the span's factor when a round is drawn again


def offsets(side: int) -> np.ndarray:
    """A direction's steps, so the centre is on the grid: -1, 0, 1, 2 on the 4x4 (the directions' signs are random),
    -4..4 on the 9x9."""
    return np.arange(side) - (side - 1) // 2


def side_of(r: dict) -> int:
    """The side of the round's grid (of each method's, in a mixed round)."""
    return math.isqrt(len(r['points']) // len(set(r.get('methods', [None]))))


def width_of(r: dict) -> float:
    """The round's grid end to end along a direction, in the unit box."""
    return r['step'] * (side_of(r) - 1)


def next_width(r: dict, k: int) -> float:
    """The next grid's width after point k of round r is picked."""
    n, o = side_of(r), offsets(side_of(r))
    k %= n * n   # its place on its method's grid
    far = max(abs(o[k % n]), abs(o[k // n])) / abs(o).max()
    return width_of(r) * (SHRINK if far <= 0.25 else GROW if far == 1 else 1.0)


def span(graph, knob: str) -> tuple:
    """(min, max) of the knob's slider in the app."""
    nid, k = knob.split('.')
    hints = typing.get_type_hints(getattr(ops, graph[nid].op.split(':')[1]), include_extras=True)
    m = hints[k].__metadata__[0]
    return float(m['min']), float(m['max'])


def params(s: dict, x, method=None) -> dict:
    """Project.convert's node params at the unit-box point x of gallery s, under method (a mixed round's point's), else
    the gallery's (None: the project's own)."""
    out = {'optimise': {'enabled': s['dbs']}}
    if method or s.get('method'):
        out['metric'] = {'method': method or s['method']}
    for knob, (lo, hi), v in zip(s['knobs'], s['ranges'], x):
        nid, k = knob.split('.')
        out.setdefault(nid, {})[k] = lo + v * (hi - lo)
    return out


def grid(centre, width: float, side: int, rng) -> dict:
    """A round: side x side points on the plane through centre along two random orthonormal directions, width end to
    end along each, clipped to the box, and the order they are shown in (shuffled, so the centre's place says
    nothing)."""
    d = np.linalg.qr(rng.normal(size=(len(centre), 2)))[0].T
    step = width / (side - 1)
    a, b = (o.reshape(-1, 1) for o in np.meshgrid(offsets(side), offsets(side)))
    points = np.clip(np.asarray(centre) + step * (a * d[0] + b * d[1]), 0, 1)
    return dict(centre=list(centre), step=round(step, 4), dirs=d.tolist(), points=points.tolist(),
                order=rng.permutation(len(points)).tolist(), discarded=[], pick=None, seconds=None)


def mixed(s: dict, width: float, side: int, rng) -> dict:
    """A new gallery's first round: a grid around each method's start, the points shown together, shuffled; methods
    says each point's."""
    rs = [grid(c, width, side, rng) for c in s['starts'].values()]
    points = [p for r in rs for p in r['points']]
    return dict(centre=[r['centre'] for r in rs], step=rs[0]['step'], dirs=[r['dirs'] for r in rs], points=points,
                methods=[m for m in s['starts'] for _ in range(side * side)],
                order=rng.permutation(len(points)).tolist(), discarded=[], pick=None, seconds=None)


def again(s: dict, r: dict, width: float, side: int, rng) -> dict:
    """Round r drawn again from the same centre (a mixed one from the starts), at this width and side."""
    return mixed(s, width, side, rng) if 'methods' in r else grid(r['centre'], width, side, rng)


def space(name: str, dbs: bool) -> dict:
    """A search over the picture's knobs: their ranges and each method's start in the unit box, the project's own
    method at its settings, the other at its preset (ops.apply_preset; the preset's values are all knobs); start, the
    project's own. The method is set by the first round's pick."""
    graph = project_graph(name)
    knobs = [k for k in KNOBS if dbs or not k.startswith('optimise.')]
    ranges = [span(graph, k) for k in knobs]
    unit = lambda g: [float(np.clip((getattr(g[k.split('.')[0]].params, k.split('.')[1]) - lo) / (hi - lo), 0, 1))
                      for k, (lo, hi) in zip(knobs, ranges)]
    own = graph['metric'].params.method
    starts = {m: unit(graph if m == own else ops.apply_preset(graph, m)) for m in METHODS}
    return dict(name=name, dbs=dbs, method=None, knobs=knobs, ranges=ranges, start=starts[own], starts=starts)


def start_of(s: dict) -> list:
    """Where the gallery's search started: its method's start once the first round is picked, else the project's."""
    return s['starts'][s['method']] if s.get('starts') and s.get('method') else s['start']


def new(name: str, dbs: bool, side: int, rng) -> dict:
    s = space(name, dbs)
    return dict(s, rounds=[mixed(s, STEP * (SIDE - 1), side, rng)])


_projects = {}


def render(job):
    """Writes one render (params None: the target) to path."""
    name, p, path = job
    if name not in _projects:
        _projects[name] = Project(name)
    project = _projects[name]
    project.memo = Memo()   # renders share only the Tune, cheap; a memo of them all would only grow
    X = project.target() if p is None else project.convert(**p)
    cv2.imwrite(str(path), cv2.cvtColor(np.round(X * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


PAGE = """<!doctype html><meta charset="utf-8"><title>dizher gallery</title>
<style>body{margin:0;background:#222;color:#ddd;font:15px system-ui}b{color:#fff}
#wrap{display:flex;align-items:flex-start;gap:12px;padding:8px;width:max-content}#left,#controls{display:flex;flex-direction:column;gap:8px}
#controls{align-items:flex-start}#grid{display:grid;gap:4px}img{display:block;image-rendering:pixelated}
#grid img{cursor:pointer}.discard #grid img:hover{outline:3px solid #d33}.pick #grid img:hover{outline:3px solid #fff}
button,input{font:inherit}input{width:3em}button.on{background:#fff;color:#000}
.busy{opacity:.3;pointer-events:none}</style>
<div id="wrap"><div id="left"><img id="target" src="/img/target.png"><div id="controls"><div id="info"></div>
<div><button id="discard">Discard bad</button> <button id="pick">Pick the best</button></div>
<button id="restore"></button>
<div><button id="directions">New directions</button> <button id="wider">Wider</button></div>
<div><label>min zoom <input id="minzoom" type="number" min="1" max="12" step="1"></label> <span id="zoom"></span></div>
<div>Space switches the mode, N new directions, W wider; Backspace takes the last click back</div></div></div>
<div id="grid"></div></div>
<script>
const R = %s, D = R.discarded;
const $ = id => document.getElementById(id), grid = $('grid'), points = R.images.map(([, p]) => p);
const imgs = new Map(R.images.map(([file, p]) => {
  const img = new Image(); img.src = '/img/' + file; img.onclick = () => click(p); return [p, img];
}));
let mode = 'discard';
const most = (xs, f) => { const m = Math.max(...xs.map(f)); return xs.filter(a => f(a) === m); };   // the best by f
function arrange(count) {   // the placement of the picture (beside the grid, or above it, the controls beside it) and
  // the columns that show count cells largest: zoom q in whole screen pixels per pixel, never below min zoom; raw, the
  // zoom that would just fit (below q: it overflows, scrolled)
  const d = devicePixelRatio, W = innerWidth - 16, H = innerHeight - 16, ctl = $('controls').offsetHeight, all = [];
  for (const top of [false, true]) {
    for (let cols = 1; cols <= count; cols++) {
      const rows = Math.ceil(count / cols), w = W - 4 * (cols - 1), h = H - 4 * (rows - 1);
      const raw = d * (top ? Math.min(w / (256 * cols), (h - 12) / (192 * (rows + 1)))
                           : Math.min((w - 12) / (256 * (cols + 1)), h / (192 * rows), (H - 8 - ctl) / 192));
      all.push({top, cols, rows, raw, q: Math.max(+$('minzoom').value, Math.floor(raw))});
    }
  }
  // at the largest zoom: one that fits, the picture beside, the most room each way (the grid shaped as the window);
  // within 5% of that the fewest rows, so the last few stand side by side
  let xs = most(most(most(all, a => a.q), a => +(a.raw >= a.q)), a => +!a.top);
  const room = Math.max(...xs.map(a => a.raw));
  xs = most(xs.filter(a => a.raw >= 0.95 * room), a => -a.rows);
  return most(xs, a => a.raw)[0];
}
function layout() {   // the ones left, side by side, so the last few are compared next to each other
  const shown = points.filter(p => !D.includes(p));
  const {top, cols, q} = arrange(shown.length), width = 256 * q / devicePixelRatio + 'px';
  $('wrap').style.flexDirection = top ? 'column' : 'row';
  $('left').style.flexDirection = top ? 'row' : 'column';
  grid.style.gridTemplateColumns = `repeat(${cols}, auto)`;
  grid.replaceChildren(...shown.map(p => imgs.get(p)));
  for (const img of [$('target'), ...imgs.values()]) img.style.width = width;
  $('left').style.width = top ? '' : width;
  $('zoom').textContent = `now ×${q}`;
  $('info').innerHTML = `<b>${R.name}</b> round ${R.round + 1}, step ${R.step}, ${R.seconds} s<br>${points.length - D.length} of ${points.length} left`;
  $('restore').textContent = `Return discarded (${D.length})`; $('restore').disabled = !D.length;
  $('discard').className = mode === 'discard' ? 'on' : ''; $('pick').className = mode === 'pick' ? 'on' : '';
  document.body.className = mode;
}
const send = (then = {}) => fetch('/round', {method: 'POST', body: JSON.stringify({round: R.round, discarded: D, ...then})});
async function next(then) {   // then: {pick: point} or {redraw: 'directions' | 'wider'}
  document.body.classList.add('busy'); $('info').textContent = 'rendering the next round...';
  await send(then); location.reload();
}
const choose = p => next({pick: p});
function click(p) {
  if (mode === 'pick') return choose(p);
  D.push(p);
  if (points.length - D.length === 1) return choose(points.find(q => !D.includes(q)));   // the last one left wins
  layout(); send();
}
const setMode = m => { mode = m; layout(); };
$('discard').onclick = () => setMode('discard'); $('pick').onclick = () => setMode('pick');
$('restore').onclick = () => { D.length = 0; layout(); send(); };
$('directions').onclick = () => next({redraw: 'directions'}); $('wider').onclick = () => next({redraw: 'wider'});
for (const b of document.querySelectorAll('button')) b.onmousedown = e => e.preventDefault();   // no focus: Space is ours
$('minzoom').value = localStorage.minzoom || Math.ceil(devicePixelRatio);   // kept over the rounds' reloads
$('minzoom').oninput = e => { localStorage.minzoom = e.target.value; layout(); };
addEventListener('resize', layout);
addEventListener('keydown', async e => {
  if (e.target.tagName === 'INPUT') return;   // typing a zoom
  if (e.key === ' ') { e.preventDefault(); setMode(mode === 'pick' ? 'discard' : 'pick'); }
  if (e.key === 'n' || e.key === 'w') return next({redraw: e.key === 'n' ? 'directions' : 'wider'});
  if (e.key !== 'Backspace') return;
  if (D.length) { D.pop(); layout(); send(); }
  else { await fetch('/undo', {method: 'POST'}); location.reload(); }
});
layout();
</script>"""


def serve(name: str, dbs: bool, side: int, port=8766):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    path, out, rng = ROUND / f'{name}.json', OUT / name, np.random.default_rng()
    s = json.loads(path.read_text()) if path.exists() else new(name, dbs, side, rng)
    r = s['rounds'][-1]
    if r['pick'] is None and not r.get('discarded') and side_of(r) != side:   # the pending round untouched: redrawn
        s['rounds'][-1] = again(s, r, width_of(r), side, rng)                 # at the side asked
    ROUND.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    save = lambda: path.write_text(json.dumps(s, indent=1))
    save()
    with Pool(SIDE * SIDE) as pool:   # a larger grid renders in batches
        def ready():
            """The last round rendered, if it is not yet."""
            i, r = len(s['rounds']) - 1, s['rounds'][-1]
            if r['seconds'] is None:
                t = time.time()
                pool.map(render, [(name, params(s, x, r['methods'][k] if 'methods' in r else None),
                                   out / f'r{i:02d}-{k:02d}.png') for k, x in enumerate(r['points'])])
                r['seconds'] = round(time.time() - t, 1)
                save()
                print(f'round {i + 1}: {r["seconds"]} s', flush=True)

        if not (out / 'target.png').exists():
            pool.apply(render, ((name, None, out / 'target.png'),))
        ready()

        class Handler(BaseHTTPRequestHandler):
            def send(self, body: bytes, kind: str):
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store')   # a round taken back is drawn again under its names
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith('/img/'):
                    return self.send((out / Path(self.path).name).read_bytes(), 'image/png')
                i, r = len(s['rounds']) - 1, s['rounds'][-1]
                state = dict(name=name, round=i, step=r['step'], seconds=r['seconds'], discarded=r.get('discarded', []),
                             images=[(f'r{i:02d}-{k:02d}.png', k) for k in r['order']])
                self.send(PAGE.replace('%s', json.dumps(state), 1).encode(), 'text/html')   # the script has its own %

            def do_POST(self):
                rounds = s['rounds']
                if self.path == '/round':   # the discards so far, and the pick or a redraw if there is one
                    v = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if v['round'] == len(rounds) - 1:   # not a late click on a round already picked
                        r = rounds[-1]
                        r['discarded'] = v['discarded']
                        k = v.get('pick')
                        if k is not None:
                            r['pick'] = k
                            if 'methods' in r:   # the first round: the pick's method is the gallery's
                                s['method'] = r['methods'][k]
                            rounds.append(grid(r['points'][k], next_width(r, k), side, rng))
                        elif v.get('redraw') in REDRAW:   # the shown grid stays in the log, marked
                            r['redrawn'] = v['redraw']
                            rounds.append(again(s, r, width_of(r) * REDRAW[v['redraw']], side, rng))
                        save()
                        ready()
                elif len(rounds) > 1:   # /undo: the pending round dropped, the last pick or redraw taken back
                    rounds.pop()
                    rounds[-1]['pick'] = None
                    rounds[-1].pop('redrawn', None)
                    if 'methods' in rounds[-1]:
                        s['method'] = None
                    save()
                self.send(b'ok', 'text/plain')

            def log_message(self, *args):
                pass

        print(f'http://localhost:{port}  (Ctrl+C to stop; picks in {path})', flush=True)
        webbrowser.open(f'http://localhost:{port}')
        HTTPServer(('localhost', port), Handler).serve_forever()


def last(s: dict) -> list:
    """The last pick's point, the gallery's best; the start if none."""
    picked = [r for r in s['rounds'] if r['pick'] is not None]
    return picked[-1]['points'][picked[-1]['pick']] if picked else start_of(s)


def method_of(s: dict) -> str:
    return s.get('method') or project_graph(s['name'])['metric'].params.method


def best(name: str):
    s = json.loads((ROUND / f'{name}.json').read_text())
    picked = [r for r in s['rounds'] if r['pick'] is not None]
    x = last(s)
    print(f'{name}: {len(picked)} rounds picked, DBS {"on" if s["dbs"] else "off"}, '
          f'method {s.get("method") or "the project one"}')
    settings(dict(s, start=start_of(s)), x)


def settings(s: dict, x):
    """Prints the knobs at the unit-box point x against the start."""
    for knob, (lo, hi), v, v0 in zip(s['knobs'], s['ranges'], x, s['start']):
        print(f'  {knob:22} {lo + v * (hi - lo):7.3f}   start {lo + v0 * (hi - lo):7.3f}')



def comparisons(r: dict) -> list:
    """[(better, worse)] point indexes a round says: each discard is worse than every render still left, the pick
    better than every one left with it."""
    left, out = set(range(len(r['points']))), []
    for d in r.get('discarded', []):
        left.discard(d)
        out += [(w, d) for w in left]
    if r['pick'] is not None:
        out += [(r['pick'], k) for k in left if k != r['pick']]
    return out


def preference(rounds: list, lam=1.0) -> np.ndarray:
    """w of a linear preference over the unit box, w . x, fitted to the rounds' comparisons (logistic, L2 lam)."""
    from scipy.optimize import minimize
    D = np.array([np.subtract(r['points'][a], r['points'][b]) for r in rounds for a, b in comparisons(r)])
    loss = lambda w: np.logaddexp(0, -D @ w).sum() + lam * w @ w / 2
    grad = lambda w: -(D * (1 / (1 + np.exp(D @ w)))[:, None]).sum(0) + lam * w
    return minimize(loss, np.zeros(D.shape[1]), jac=grad, method='L-BFGS-B').x


def effects(s: dict, renders: Path):
    """(knobs, pixels) each knob's change of the render per full slider range, and the share of the change it explains:
    the eye-blurred CIELAB render (4x4 means) less its round's centre's, regressed on the knobs' change over all the
    rounds' renders. A row's size is how much a knob changes the picture, two rows' cosine whether alike."""
    from metrics import SRGB2XYZ, linear, project_eye, xyz2lab
    lab = lambda f: xyz2lab(linear(project_eye(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB).astype(np.float32)
                                               / 255)) @ SRGB2XYZ.T)
    D, Y = [], []
    for i, r in enumerate(s['rounds']):
        centre = (side_of(r) - 1) // 2 * (side_of(r) + 1)   # the point at offsets (0, 0)
        if 'methods' in r or not (renders / f'r{i:02d}-{centre:02d}.png').exists():   # a mixed round: methods
            continue
        C = lab(renders / f'r{i:02d}-{centre:02d}.png')
        for k, x in enumerate(r['points']):
            if k != centre:
                d = lab(renders / f'r{i:02d}-{k:02d}.png') - C
                Y.append(cv2.resize(d, (d.shape[1] // 4, d.shape[0] // 4), interpolation=cv2.INTER_AREA).ravel())
                D.append(np.subtract(x, r['points'][centre]))
    D, Y = np.array(D), np.array(Y)
    J = np.linalg.lstsq(D, Y, rcond=None)[0]
    return J, 1 - ((Y - D @ J) ** 2).sum() / (Y ** 2).sum()


def trend(s: dict, rng, boots=200) -> tuple:
    """(w, lo, hi) the gallery's linear preference over the knobs and its 5-95% over its rounds resampled; a mixed
    round left out (it compares methods, not knobs)."""
    used = [r for r in s['rounds'] if comparisons(r) and 'methods' not in r]
    lo, hi = np.percentile([preference([used[i] for i in rng.integers(len(used), size=len(used))])
                            for _ in range(boots)], [5, 95], axis=0)
    return preference(used), lo, hi


def report(min_rounds=5, boots=200):
    """Per gallery of min_rounds or more: start and last pick, the preference w per full slider range with its 5-95%
    over rounds resampled (* when it excludes 0), each knob's rms dE per full range; then the knobs' effects alike
    (cosine, mean over the galleries) and the preferences alike across the pictures."""
    rng, prefs, sims = np.random.default_rng(0), {}, []
    for path in sorted(ROUND.glob('*.json')):
        s = json.loads(path.read_text())
        if len(s['rounds']) < min_rounds:
            print(f"{path.stem}: {len(s['rounds'])} rounds, left out\n")
            continue
        used = [r for r in s['rounds'] if comparisons(r) and 'methods' not in r]
        picked, x = [r for r in s['rounds'] if r['pick'] is not None], last(s)
        w, lo, hi = trend(s, rng, boots)
        J, explained = effects(s, OUT / path.stem)
        prefs[path.stem], knobs = w / np.linalg.norm(w), s['knobs']
        U = J / np.linalg.norm(J, axis=1, keepdims=True)
        sims.append(U @ U.T)
        print(f"{path.stem}: {len(picked)} picks, {sum(map(len, map(comparisons, used)))} comparisons; "
              f"the knobs' linear map explains {explained:.0%} of the renders' change")
        print(f"  {'knob':22} {'start':>7} {'last':>7}   {'preference [5..95%]':>22}   rms dE")
        for j, (knob, (a, b)) in enumerate(zip(knobs, s['ranges'])):
            mark = '*' if lo[j] > 0 or hi[j] < 0 else ' '
            print(f"  {knob:22} {a + start_of(s)[j] * (b - a):7.3f} {a + x[j] * (b - a):7.3f}   "
                  f"{w[j]:+5.1f} [{lo[j]:+5.1f} {hi[j]:+5.1f}]{mark}   {np.sqrt((J[j] ** 2).mean()):6.1f}")
        print()
    S = np.mean(sims, 0)
    print("the knobs' effects on the render alike, cosine, mean over the galleries:")
    short = [k.split('.')[1][:10] for k in knobs]
    print(f"  {'':22}" + ''.join(f'{k:>11}' for k in short))
    for a, k in enumerate(knobs):
        print(f'  {k:22}' + ''.join(f'{S[a, b]:+11.2f}' for b in range(len(knobs))))
    print('the preferences alike across the pictures, cosine:')
    for a in prefs:
        print(f'  {a:15}' + ''.join(f'{prefs[a] @ prefs[b]:+7.2f}' for b in prefs))


def defaults(min_picks=5):
    """Per Metric method each knob's new default: the median of its last picks over the training pictures (a picture
    once, its most picked gallery of the method), when the move to it agrees with the net direction its galleries of
    the method prefer (each gallery's sign where its 5-95% excludes 0); else the current default stays, as the picks
    there drift (from a floor, upward). Rounded to 2 digits, within 1% of the slider's min to the min."""
    from tune import TRAIN
    rng, by = np.random.default_rng(0), {}
    picks = lambda s: sum(r['pick'] is not None for r in s['rounds'])
    for path in sorted(ROUND.glob('*.json')):
        s = json.loads(path.read_text())
        if s['name'] in TRAIN and picks(s) >= min_picks:
            by.setdefault(method_of(s), []).append(s)
    for m, ss in sorted(by.items()):
        graph = ops.apply_preset(ops.make_graph(), m)   # a new project's, at the method
        once = {}
        for s in sorted(ss, key=picks):
            once[s['name']] = s   # the most picked last
        trends = [trend(s, rng) for s in ss]
        print(f"{m}: {', '.join(sorted(once))} ({len(ss)} galleries for the signs)")
        print(f"  {'knob':22} {'now':>7}  {'last picks':42} {'median':>7}  signs   {'new':>7}")
        for j, knob in enumerate(ss[0]['knobs']):
            nid, k = knob.split('.')
            now, (lo, hi) = getattr(graph[nid].params, k), ss[0]['ranges'][j]
            values = [lo + last(s)[j] * (hi - lo) for s in once.values()]
            signs = ''.join('+' if l[j] > 0 else '-' for _, l, h in trends if l[j] > 0 or h[j] < 0)
            med, net = float(np.median(values)), signs.count('+') - signs.count('-')
            new = med if net * (med - now) > 0 else now
            new = lo if new - lo < 0.01 * (hi - lo) else float(f'{new:.2g}')
            print(f"  {knob:22} {now:7.3f}  {' '.join(f'{v:6.3f}' for v in values):42} {med:7.3f}  "
                  f"{signs or '.':6}  {new:7.3f}{'' if new == now else '  *'}")
        print()


if __name__ == '__main__':
    if sys.argv[1] == 'report':
        report()
    elif sys.argv[1] == 'defaults':
        defaults()
    elif sys.argv[1] == 'best':
        best(sys.argv[2])
    else:
        side = int(sys.argv[sys.argv.index('--side') + 1]) if '--side' in sys.argv else SIDE
        serve(sys.argv[1], '--no-dbs' not in sys.argv, side)
