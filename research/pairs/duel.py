"""PLAN step 3b: a pairwise search over one picture's knobs (gallery.KNOBS, the unit box). The user picks the better of
two renders; a Gaussian process with a probit likelihood on the comparisons (Chu and Ghahramani 2005, by Laplace; its
scale and amplitude by the evidence) models the user's preference, and each next pair is the best point so far (the
highest posterior mean) against the point of the most expected improvement over it (Brochu et al. 2007), among random
points within a step of the best. The step follows the answers as a (1+1) evolution strategy's (Rechenberg's 1/5 rule):
larger when the challenger wins, smaller when it loses, so far challengers that the GP cannot relate to the rest come
nearer until it can. The picture's gallery, when it has the same knobs, seeds the comparisons.

    python research/pairs/duel.py NAME [--no-dbs]   the duel in the browser: ← and → pick (or a click), Space or the
                                            button can't tell (no comparison, another pair), Backspace takes the last
                                            answer back;
                                            each answer at once to rounds/duel/NAME.json, renders in data/duel/NAME/;
                                            Ctrl+C stops, run it again to go on. --no-dbs as in gallery.py
    python research/pairs/duel.py best NAME      the best point's settings against the start
    python research/pairs/duel.py check          the search against a simulated user, no renders
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA   # noqa: E402
from gallery import ROUND as GALLERY, STEP, comparisons, params, render, settings, space   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'duel', HERE / 'rounds' / 'duel'
SCALES = (0.1, 0.2, 0.3, 0.5, 0.8)   # the GP's length scale in the unit box, by the evidence
AMPS = (0.5, 1, 2, 4)                # its amplitude, in units of the probit noise on one render's value
# The galleries' evidence wants scales down to 0.03 and amplitudes up to 32; a grid that wide, or a prior over it, does
# no better in check, and alone (no prior) the evidence of a few answers swings from 0.03 (steps too small to tell
# apart) to 0.5 with amplitude 8 (every challenger far off, and worse)
GROW = 1.5   # the step after a challenger won, or the two could not be told apart; after it lost, / GROW ** (1/4):
             # steady when it wins 1 in 5


def data(s: dict):
    """(X, C): the distinct points compared and the comparisons [(better, worse)] as rows of X: the duels', and the
    picture's gallery's when it has the same knobs. X is the start alone when there is none."""
    pairs = [(d['points'][d['pick']], d['points'][1 - d['pick']]) for d in s['duels'] if d['pick'] is not None]
    path = GALLERY / f"{s['name']}.json"
    g = json.loads(path.read_text()) if path.exists() else {}
    if g.get('knobs') == s['knobs'] and np.array_equal(g['ranges'], s['ranges']):
        pairs += [(r['points'][a], r['points'][b]) for r in g['rounds'] for a, b in comparisons(r)]
    if not pairs:
        return np.array([s['start']]), np.zeros((0, 2), int)
    X, C = np.unique(np.round(pairs, 6).reshape(-1, len(s['start'])), axis=0, return_inverse=True)
    return X, C.reshape(-1, 2)


def kernel(X, Y, scale, amp):
    return amp ** 2 * np.exp(-((X[:, None] - Y[None]) ** 2).sum(-1) / (2 * scale ** 2))


def laplace(X, C, scale, amp):
    """The GP preference at X given the comparisons C, Laplace's approximation: the log evidence, the latent values f at
    X, and g, A with which the posterior mean at x is k(x, X) g and the covariance of x, y k(x, y) - k(x, X) A k(X, y)."""
    n = len(X)
    K = kernel(X, X, scale, amp) + 1e-6 * np.eye(n)
    D = np.zeros((len(C), n))   # z = (f_better - f_worse) / sqrt 2: unit noise on each render's value
    D[np.arange(len(C)), C[:, 0]] += 2 ** -0.5
    D[np.arange(len(C)), C[:, 1]] -= 2 ** -0.5
    f = np.zeros(n)
    for _ in range(100):   # Newton; the probit likelihood is log-concave
        z = D @ f
        r = np.exp(norm.logpdf(z) - log_ndtr(z))   # d log Phi(z) / dz
        g, W = D.T @ r, D.T @ (D * (r * (z + r))[:, None])
        B = np.eye(n) + W @ K
        new = K @ np.linalg.solve(B, W @ f + g)
        if np.abs(new - f).max() < 1e-8:
            break
        f = new
    evidence = log_ndtr(z).sum() - f @ g / 2 - np.linalg.slogdet(B)[1] / 2
    return evidence, f, g, W @ np.linalg.inv(B)


def propose(s: dict, rng, count=4000) -> dict:
    """The next duel: the best point so far against the candidate of the most expected improvement over it, within
    the step (from the gallery's first, then by the last answer)."""
    answered = [d for d in s['duels'] if d['pick'] is not None or d.get('tie')]
    step = answered[-1].get('step', STEP) * (GROW ** -0.25 if answered[-1]['pick'] == 0 else GROW) if answered else STEP
    X, C = data(s)
    # ponytail: one length scale for all knobs; a scale per knob (ARD) if the duels show the knobs differ much
    evidence, scale, amp = max((laplace(X, C, sc, a)[0], sc, a) for sc in SCALES for a in AMPS)
    _, f, g, A = laplace(X, C, scale, amp)
    b = X[np.argmax(f)]
    u = rng.normal(size=(count, len(b)))
    cand = np.clip(b + step * rng.random((count, 1)) * u / np.linalg.norm(u, axis=1, keepdims=True), 0, 1)
    kc, kb = kernel(cand, X, scale, amp), kernel(b[None], X, scale, amp)[0]
    m = (kc - kb) @ g   # the mean of f(x) - f(b), and its variance
    v = 2 * amp ** 2 - 2 * kernel(cand, b[None], scale, amp)[:, 0] - (((kc - kb) @ A) * (kc - kb)).sum(1)
    sd = np.sqrt(np.maximum(v, 1e-12))
    ei = m * ndtr(m / sd) + sd * norm.pdf(m / sd)
    return dict(points=[b.tolist(), cand[np.argmax(ei)].tolist()], left=int(rng.integers(2)), pick=None,
                seconds=None, compared=len(C), step=round(step, 4), scale=scale, amp=amp,
                evidence=round(float(evidence), 2))


def best(name: str):
    s = json.loads((ROUND / f'{name}.json').read_text())
    X, C = data(s)
    d = s['duels'][-1]
    _, f, _, _ = laplace(X, C, d['scale'], d['amp'])
    print(f"{name}: {sum(d['pick'] is not None for d in s['duels'])} duels picked, {len(C)} comparisons with the "
          f"gallery's, DBS {'on' if s['dbs'] else 'off'}; the GP scale {d['scale']}, amplitude {d['amp']}")
    settings(s, X[np.argmax(f)])


def check(duels=60, seeds=5):
    """A simulated user, a probit on a quadratic preference over 10 knobs that peaks off the start, the knobs mattering
    unequally: the duel more than halves the start's loss on average (a noiseless random search of 2 renders a duel, for
    scale)."""
    loss = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        peak, start, weight = rng.random(10), np.full(10, 0.5), rng.uniform(0.5, 3, 10)
        u = lambda x: -10 * (weight * (np.asarray(x) - peak) ** 2).sum()
        s = dict(name='', knobs=[], ranges=[], start=start.tolist(), duels=[])
        for _ in range(duels):
            s['duels'].append(propose(s, rng))
            a, b = s['duels'][-1]['points']
            s['duels'][-1]['pick'] = int(u(b) + rng.normal() > u(a) + rng.normal())
        X, C = data(s)
        found = X[np.argmax(laplace(X, C, s['duels'][-1]['scale'], s['duels'][-1]['amp'])[1])]
        random = max(np.random.default_rng(seed + 100).random((2 * duels, 10)), key=u)
        loss.append([-u(start), -u(found), -u(random)])
        print(f'seed {seed}: the loss at the start {-u(start):5.1f}, found {-u(found):5.1f}, random search {-u(random):5.1f}')
    loss = np.mean(loss, 0)
    assert loss[1] < loss[0] / 2, f'the duel leaves {loss[1]:.1f} of the start loss {loss[0]:.1f}'


PAGE = """<!doctype html><meta charset="utf-8"><title>dizher duel</title>
<style>body{margin:0;background:#222;color:#ddd;font:15px system-ui}b{color:#fff}
#row{display:grid;gap:12px;padding:8px;width:max-content}img{display:block;image-rendering:pixelated}
.r{cursor:pointer}.r:hover{outline:3px solid #fff}#info{padding:0 8px}.busy{opacity:.3;pointer-events:none}</style>
<div id="row"><img id="target" src="/img/target.png"><img class="r" id="L"><img class="r" id="R"></div>
<div id="info"></div><div style="padding:8px"><button id="tie">Can't tell</button></div>
<script>
const S = %s, $ = id => document.getElementById(id);
$('L').src = '/img/' + S.files[0]; $('R').src = '/img/' + S.files[1];
$('info').innerHTML = `<b>${S.name}</b> duel ${S.duel + 1}, ${S.compared} comparisons so far, ${S.seconds} s` +
  (S.same ? ', <b>the two renders are the same</b>' : '') +
  '<br>← and → pick the better (or a click), Space can\\'t tell, Backspace takes the last answer back';
let busy = false;
function layout() {   // the largest whole zoom: the three in a row, or the picture above the two
  const d = devicePixelRatio, W = innerWidth - 16, H = innerHeight - 70;
  const row = Math.min((W - 24) / 768, H / 192), two = Math.min((W - 12) / 512, (H - 12) / 384);
  const q = Math.max(1, Math.floor(d * Math.max(row, two)));
  $('row').style.gridTemplateColumns = `repeat(${row >= two ? 3 : 2}, auto)`;
  $('target').style.gridColumn = row >= two ? '' : '1 / 3';
  for (const img of document.images) img.style.width = 256 * q / d + 'px';
}
async function answer(side) {   // 0 left, 1 right, null can't tell
  busy = true; document.body.className = 'busy'; $('info').textContent = 'rendering the next pair...';
  await fetch('/pick', {method: 'POST', body: JSON.stringify({duel: S.duel, side})}); location.reload();
}
$('L').onclick = () => answer(0); $('R').onclick = () => answer(1); $('tie').onclick = () => answer(null);
$('tie').onmousedown = e => e.preventDefault();   // no focus, so Space is not a second click
addEventListener('resize', layout);
addEventListener('keydown', async e => {
  if (busy) return;   // its own flag: a browser or an extension may put a class on the body
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') answer(+(e.key === 'ArrowRight'));
  if (e.code === 'Space' || e.key === ' ') { e.preventDefault(); answer(null); }
  if (e.key === 'Backspace') { await fetch('/undo', {method: 'POST'}); location.reload(); }
});
layout();
</script>"""


def serve(name: str, dbs: bool, port=8767):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    path, out, rng = ROUND / f'{name}.json', OUT / name, np.random.default_rng()
    s = json.loads(path.read_text()) if path.exists() else dict(space(name, dbs), duels=[])
    ROUND.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    save = lambda: path.write_text(json.dumps(s, indent=1))
    files = lambda i: [out / f'd{i:03d}-{k}.png' for k in (0, 1)]   # by point: 0 the best, 1 the challenger
    with Pool(2) as pool:
        def ready():
            """The last duel proposed and rendered, if it is not yet."""
            if not s['duels'] or s['duels'][-1]['pick'] is not None or s['duels'][-1].get('tie'):
                s['duels'].append(propose(s, rng))
            i, d = len(s['duels']) - 1, s['duels'][-1]
            if d['seconds'] is None:
                t = time.time()
                pool.map(render, [(name, params(s, x), f) for x, f in zip(d['points'], files(i))])
                d['seconds'] = round(time.time() - t, 1)
                print(f"duel {i + 1}: {d['seconds']} s, {d['compared']} comparisons, scale {d['scale']}, "
                      f"amplitude {d['amp']}", flush=True)
            save()

        if not (out / 'target.png').exists():
            pool.apply(render, ((name, None, out / 'target.png'),))
        ready()

        class Handler(BaseHTTPRequestHandler):
            def send(self, body: bytes, kind: str):
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store')   # a duel taken back is drawn again under its names
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith('/img/'):
                    return self.send((out / Path(self.path).name).read_bytes(), 'image/png')
                i, d = len(s['duels']) - 1, s['duels'][-1]
                f = files(i)
                order = (d['left'], 1 - d['left'])
                state = dict(name=name, duel=i, compared=d['compared'], seconds=d['seconds'],
                             files=[f[k].name for k in order], same=f[0].read_bytes() == f[1].read_bytes())
                self.send(PAGE.replace('%s', json.dumps(state), 1).encode(), 'text/html')

            def do_POST(self):
                duels = s['duels']
                if self.path == '/pick':
                    v = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if v['duel'] == len(duels) - 1:   # not a late click on a duel already answered
                        d = duels[-1]
                        if v['side'] is None:
                            d['tie'] = True
                        else:
                            d['pick'] = d['left'] if v['side'] == 0 else 1 - d['left']
                        ready()
                elif len(duels) > 1:   # /undo: the pending duel dropped, the last answer taken back
                    duels.pop()
                    duels[-1]['pick'] = None
                    duels[-1].pop('tie', None)
                    save()
                self.send(b'ok', 'text/plain')

            def log_message(self, *args):
                pass

        print(f'http://localhost:{port}  (Ctrl+C to stop; answers in {path})', flush=True)
        webbrowser.open(f'http://localhost:{port}')
        HTTPServer(('localhost', port), Handler).serve_forever()


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        check()
    elif sys.argv[1] == 'best':
        best(sys.argv[2])
    else:
        serve(sys.argv[1], '--no-dbs' not in sys.argv)
