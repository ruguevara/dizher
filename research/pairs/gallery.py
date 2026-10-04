"""PLAN step 3: a gallery per picture. The converter's knobs (KNOBS, each over its slider in the app, scaled to 0..1)
searched by the user's eye: each round a 4x4 grid of renders on a plane through the current best point along two
random directions; the user discards the bad ones, the rest laid out again as large as the window holds them, and
picks the best, or the last one left wins; the centre moves there, the step follows (GROW). When every direction is
bad, the round is drawn again from the same centre (REDRAW): other directions, or wider. Each render goes through
the project's whole pipeline, no cell painted (Project.convert); the start is the project's own settings, every knob
given, so the dots' weights do not follow the selection's.

    python research/pairs/gallery.py NAME [--no-dbs]   the gallery in the browser; every grid (its points, the
                                              layout) and each click saved at once to rounds/gallery/NAME.json,
                                              with the discards in order, renders in data/gallery/NAME/;
                                              Backspace takes the last click back (a discard, else a pick or redraw);
                                              Ctrl+C stops, run it again to go on. --no-dbs, for a new gallery:
                                              Optimise off and its three knobs out
    python research/pairs/gallery.py best NAME   the last pick's settings against the start
    python research/pairs/gallery.py report      what the votes say: per picture a linear preference over the knobs and
                                              how much each knob changes the render, the knobs' effects alike
"""
import json
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
from mokit.graph import Memo   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'gallery', HERE / 'rounds' / 'gallery'
KNOBS = ('metric.chroma', 'metric.flare', 'select.coherence', 'select.edge', 'select.luma_noise',
         'select.chroma_noise', 'select.surface', 'halftone.chroma', 'optimise.luma_noise', 'optimise.chroma_noise',
         'optimise.structure', 'eye.luma_scale', 'eye.chroma_scale')
SIDE, STEP = 4, 0.2                # the grid's side; the first step, in the unit box
GROW = (0.7, 1.0, 1.25)            # the step's factor by the pick's distance from the centre, 0, 1 or 2 steps: the
                                   # centre picked, the grid reaches too far; its far edge, too short
REDRAW = {'directions': 1.0, 'wider': 1.5}   # the step's factor when a round is drawn again
OFFSETS = np.arange(SIDE) - 1      # -1, 0, 1, 2 steps, so the centre is on the grid; the directions' signs are random


def span(graph, knob: str) -> tuple:
    """(min, max) of the knob's slider in the app."""
    nid, k = knob.split('.')
    hints = typing.get_type_hints(getattr(ops, graph[nid].op.split(':')[1]), include_extras=True)
    m = hints[k].__metadata__[0]
    return float(m['min']), float(m['max'])


def params(s: dict, x) -> dict:
    """Project.convert's node params at the unit-box point x of gallery s."""
    out = {'optimise': {'enabled': s['dbs']}}
    for knob, (lo, hi), v in zip(s['knobs'], s['ranges'], x):
        nid, k = knob.split('.')
        out.setdefault(nid, {})[k] = lo + v * (hi - lo)
    return out


def grid(centre, step: float, rng) -> dict:
    """A round: SIDE x SIDE points on the plane through centre along two random orthonormal directions, clipped to the
    box, and the order they are shown in (shuffled, so the centre's place says nothing)."""
    d = np.linalg.qr(rng.normal(size=(len(centre), 2)))[0].T
    a, b = (o.reshape(-1, 1) for o in np.meshgrid(OFFSETS, OFFSETS))
    points = np.clip(np.asarray(centre) + step * (a * d[0] + b * d[1]), 0, 1)
    return dict(centre=list(centre), step=round(step, 4), dirs=d.tolist(), points=points.tolist(),
                order=rng.permutation(len(points)).tolist(), discarded=[], pick=None, seconds=None)


def new(name: str, dbs: bool, rng) -> dict:
    graph = project_graph(name)
    knobs = [k for k in KNOBS if dbs or not k.startswith('optimise.')]
    ranges = [span(graph, k) for k in knobs]
    start = [float(np.clip((getattr(graph[k.split('.')[0]].params, k.split('.')[1]) - lo) / (hi - lo), 0, 1))
             for k, (lo, hi) in zip(knobs, ranges)]
    return dict(name=name, dbs=dbs, knobs=knobs, ranges=ranges, start=start, rounds=[grid(start, STEP, rng)])


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


def serve(name: str, dbs: bool, port=8766):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    path, out, rng = ROUND / f'{name}.json', OUT / name, np.random.default_rng()
    s = json.loads(path.read_text()) if path.exists() else new(name, dbs, rng)
    ROUND.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    save = lambda: path.write_text(json.dumps(s, indent=1))
    save()
    with Pool(SIDE * SIDE) as pool:
        def ready():
            """The last round rendered, if it is not yet."""
            i, r = len(s['rounds']) - 1, s['rounds'][-1]
            if r['seconds'] is None:
                t = time.time()
                pool.map(render, [(name, params(s, x), out / f'r{i:02d}-{k:02d}.png') for k, x in enumerate(r['points'])])
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
                            far = max(abs(OFFSETS[k % SIDE]), abs(OFFSETS[k // SIDE]))
                            rounds.append(grid(r['points'][k], r['step'] * GROW[far], rng))
                        elif v.get('redraw') in REDRAW:   # the shown grid stays in the log, marked
                            r['redrawn'] = v['redraw']
                            rounds.append(grid(r['centre'], r['step'] * REDRAW[v['redraw']], rng))
                        save()
                        ready()
                elif len(rounds) > 1:   # /undo: the pending round dropped, the last pick or redraw taken back
                    rounds.pop()
                    rounds[-1]['pick'] = None
                    rounds[-1].pop('redrawn', None)
                    save()
                self.send(b'ok', 'text/plain')

            def log_message(self, *args):
                pass

        print(f'http://localhost:{port}  (Ctrl+C to stop; picks in {path})', flush=True)
        webbrowser.open(f'http://localhost:{port}')
        HTTPServer(('localhost', port), Handler).serve_forever()


def best(name: str):
    s = json.loads((ROUND / f'{name}.json').read_text())
    picked = [r for r in s['rounds'] if r['pick'] is not None]
    x = picked[-1]['points'][picked[-1]['pick']] if picked else s['start']
    print(f'{name}: {len(picked)} rounds picked, DBS {"on" if s["dbs"] else "off"}')
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


def effects(s: dict):
    """(knobs, pixels) each knob's change of the render per full slider range, and the share of the change it explains:
    the eye-blurred CIELAB render (4x4 means) less its round's centre's, regressed on the knobs' change over all the
    rounds' renders. A row's size is how much a knob changes the picture, two rows' cosine whether alike."""
    from metrics import SRGB2XYZ, linear, project_eye, xyz2lab
    lab = lambda f: xyz2lab(linear(project_eye(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB).astype(np.float32)
                                               / 255)) @ SRGB2XYZ.T)
    D, Y, centre = [], [], int(np.argmin(abs(OFFSETS))) * (SIDE + 1)   # the point at offsets (0, 0)
    for i, r in enumerate(s['rounds']):
        if not (OUT / s['name'] / f'r{i:02d}-{centre:02d}.png').exists():
            continue
        C = lab(OUT / s['name'] / f'r{i:02d}-{centre:02d}.png')
        for k, x in enumerate(r['points']):
            if k != centre:
                d = lab(OUT / s['name'] / f'r{i:02d}-{k:02d}.png') - C
                Y.append(cv2.resize(d, (d.shape[1] // 4, d.shape[0] // 4), interpolation=cv2.INTER_AREA).ravel())
                D.append(np.subtract(x, r['points'][centre]))
    D, Y = np.array(D), np.array(Y)
    J = np.linalg.lstsq(D, Y, rcond=None)[0]
    return J, 1 - ((Y - D @ J) ** 2).sum() / (Y ** 2).sum()


def report(min_rounds=5, boots=200):
    """Per gallery of min_rounds or more: start and last pick, the preference w per full slider range with its 5-95%
    over rounds resampled (* when it excludes 0), each knob's rms dE per full range; then the knobs' effects alike
    (cosine, mean over the galleries) and the preferences alike across the pictures."""
    rng, prefs, sims = np.random.default_rng(0), {}, []
    for path in sorted(ROUND.glob('*.json')):
        s = json.loads(path.read_text())
        used = [r for r in s['rounds'] if comparisons(r)]
        if len(s['rounds']) < min_rounds:
            print(f"{s['name']}: {len(s['rounds'])} rounds, left out\n")
            continue
        picked = [r for r in s['rounds'] if r['pick'] is not None]
        x = picked[-1]['points'][picked[-1]['pick']]
        w = preference(used)
        lo, hi = np.percentile([preference([used[i] for i in rng.integers(len(used), size=len(used))])
                                for _ in range(boots)], [5, 95], axis=0)
        J, explained = effects(s)
        prefs[s['name']], knobs = w / np.linalg.norm(w), s['knobs']
        U = J / np.linalg.norm(J, axis=1, keepdims=True)
        sims.append(U @ U.T)
        print(f"{s['name']}: {len(picked)} picks, {sum(map(len, map(comparisons, used)))} comparisons; "
              f"the knobs' linear map explains {explained:.0%} of the renders' change")
        print(f"  {'knob':22} {'start':>7} {'last':>7}   {'preference [5..95%]':>22}   rms dE")
        for j, (knob, (a, b)) in enumerate(zip(knobs, s['ranges'])):
            mark = '*' if lo[j] > 0 or hi[j] < 0 else ' '
            print(f"  {knob:22} {a + s['start'][j] * (b - a):7.3f} {a + x[j] * (b - a):7.3f}   "
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


if __name__ == '__main__':
    if sys.argv[1] == 'report':
        report()
    elif sys.argv[1] == 'best':
        best(sys.argv[2])
    else:
        serve(sys.argv[1], '--no-dbs' not in sys.argv)
