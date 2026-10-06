"""Session E step 3: labels from Claude judges where Select pairs is unsure of its own choice.

    python research/pairs/dp.py make       per picture, the alternatives the converter itself makes for each segment,
                                           kept where the energy's margin is small and the change is seen; data/dp/
    python research/pairs/dp.py wave N     N new pairs into round e1 (round robin over the pictures): sheets in
                                           data/dp/sheets/, key rounds/e1/key.json, a prompt file per judge
                                           data/dp/prompts/wW-I.txt, results rounds/e1/wW-I.json
    python research/pairs/dp.py score      per pair the judges' votes and the label by majority
    python research/pairs/dp.py user       step 4: blind sheets for the user in data/dp/user/, key rounds/e1/user-key.json
    python research/pairs/dp.py vote [ROUND SHEETS]   the user votes in the browser (keys left 1, right 2, Space =,
                                           x; Backspace back) on SHEETS (data/dp/user/) listed in ROUND/user-key.json
                                           (rounds/e1/), each answer saved at once to ROUND/user-verdicts.json

The base is the Exact mixture preset (tune.setting({})), no cell painted. Per segment (common.segments) the
alternatives: next, its cells on their next best pair by the energy's own term, showing other colours than the base
(as C in build.py); and NEAR, its labels from Select pairs at a neighbouring setting, put into the base labelling.
Every side is rendered by the base's pipeline, only the segment changed. Kept: changes of MIN_CELLS cells or more
(step 1: judges do not see 1 or 2), per picture the lowest half of the margins per changed cell, (energy(alternative)
- energy(base)) / cells (the total grows with the cells, so its lowest third were the smallest changes), then a change
through the eye of at least MIN_DE (step 2: one segment is decided from ~20 dE, below 10 it is even). Margin and
change go together (Spearman 0.45 over the 918 alternatives), so the plan's lowest third and 15 dE left 48 pairs;
lowest half and 10 dE leave 158, the pilot's votes by dE say where the labels are. The untuned loose pictures barely
change (median 3-6 dE): of them only vangog, gradient and tv_out give pairs. One null pair per picture (the base from another halftone origin) as a
control: judges should not call it.
Each pair 6 votes, 3 per order, fresh judges; answers 1, 2, = (both fine), x (both bad). A pair's label is the answer
of more than half its votes; = so labelled says the energy weighs what does not matter, x and splits are no labels."""
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acuity import EYE, regions   # noqa: E402
from build import NULL_ORIGIN, colours   # noqa: E402
from common import DATA, Project, segments   # noqa: E402
from curve import PROMPT as CURVE_PROMPT, deal   # noqa: E402
from metrics import seen_change   # noqa: E402
from tune import HELD, TRAIN, setting   # noqa: E402
from views import seen2   # noqa: E402
from mokit.graph import evaluate   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'dp', HERE / 'rounds' / 'e1'
LOOSE = ('goldhill', 'gradient', 'kotofey', 'landscape', 'tv_out', 'vangog')   # pictures without a project
PICTURES = TRAIN + LOOSE   # HELD stay out, as in session C
NEAR = {'coherence': {('select', 'coherence'): 12.0}, 'edge': {('select', 'edge'): 0.32},
        'eye': {('eye', 'chroma_scale'): 1.1}, 'luma_noise': {('select', 'luma_noise'): 0.1}}
KEEP, MIN_CELLS, MIN_DE, VOTES = 1 / 2, 3, 10.0, 3   # lowest share of margins; dE through the eye; votes per order
assert not set(PICTURES) & set(HELD)


def alternatives(name: str):
    """(project, base labels, segments, [alternative]) sorted by margin; each with its label map under 'labels'."""
    p = Project(name, **setting({}))
    T, base = p.target(), p.selection.best_attr_indexes.copy()
    seg, e0 = segments(T), p.energy(base)
    sets = colours(p.pairs, p.selection.palette.as_ubyte())
    sid = np.array([{s: i for i, s in enumerate(dict.fromkeys(sets))}[s] for s in sets])   # label -> colour set
    costs = p.selection.energy.unary().copy()
    costs[sid[:, None, None] == sid[base][None]] = np.inf   # the base's colours or the same in other order
    alts = {'next': costs.argmin(0),
            **{k: evaluate(p.changed(**setting(ch)), 'select', p.memo).best_attr_indexes for k, ch in NEAR.items()}}
    cands = []
    for s in range(seg.max() + 1):
        for k, L in alts.items():
            cells = (seg == s) & (sid[L] != sid[base])
            if cells.any():
                X = base.copy()
                X[cells] = L[cells]
                cands.append(dict(id=f'{name}/{s}/{k}', image=name, segment=s, alt=k, cells=int(cells.sum()),
                                  margin=round(p.energy(X) - e0, 4), labels=X))
    cands.sort(key=lambda c: c['margin'])
    return p, base, seg, cands


def make_job(name: str):
    p, base, seg, cands = alternatives(name)
    cands = sorted((c for c in cands if c['cells'] >= MIN_CELLS), key=lambda c: c['margin'] / c['cells'])
    T = p.target()
    A = p.render(base)
    out = dict(target=T, A=A, N=p.render(base, halftoner=NULL_ORIGIN), base=base, segments=seg)
    for i, c in enumerate(cands):   # all rendered, kept() picks the pairs
        X, cells = p.render(c['labels']), c['labels'] != base
        c['dE'] = round(float(seen_change(X, A, np.kron(cells, np.ones((8, 8), bool)))), 1)
        c['per_cell'] = round(c['margin'] / c['cells'], 4)
        out[f'X{i}'], out[f'L{i}'], c['n'] = X, c.pop('labels'), i
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'{name}.npz', **{k: np.round(v * 255).astype(np.uint8) if v.dtype.kind == 'f' else v
                                                for k, v in out.items()})
    (OUT / f'{name}.json').write_text(json.dumps(cands, indent=1))
    return name, len(cands), len(kept(cands))


def kept(cands: list) -> list:
    """A picture's pairs: the lowest KEEP of its margins per cell, of those the changes of MIN_DE or more."""
    return [c for c in cands[:max(1, round(KEEP * len(cands)))] if c['dE'] >= MIN_DE]


def make():
    with Pool(len(PICTURES)) as pool:
        for name, n, k in pool.imap_unordered(make_job, PICTURES):
            print(f'{name}: {n} alternatives, {k} kept', flush=True)


def survey_job(name: str) -> list:
    return [{k: v for k, v in c.items() if k != 'labels'} for c in alternatives(name)[3]]


def survey():
    """Every alternative's size and margin, unrendered, to data/dp/all.json; per kind and picture how many."""
    with Pool(len(PICTURES)) as pool:
        rows = [r for rs in pool.map(survey_job, PICTURES) for r in rs]
    (OUT / 'all.json').write_text(json.dumps(rows, indent=1))
    for alt in ['next', *NEAR]:
        c = np.array([r['cells'] for r in rows if r['alt'] == alt])
        m = np.array([r['margin'] for r in rows if r['alt'] == alt]) / c
        print(f'{alt:10} {len(c):3}, cells quartiles {np.percentile(c, [25, 50, 75])}, 3+ cells {(c >= 3).sum():3}, '
              f'margin per cell median {np.median(m):.3f}, of 3+ cells {np.median(m[c >= 3]):.3f}')
    for name in PICTURES:
        rs = [r for r in rows if r['image'] == name]
        print(f'  {name:15} {len(rs):3}, 3+ cells ' + ' '.join(f'{k} {sum(r["alt"] == k and r["cells"] >= 3 for r in rs)}'
                                                         for k in ['next', *NEAR]))


PROMPT = CURVE_PROMPT.replace(
    'If you honestly cannot tell them apart or they are equally good, say "=".{taste}',
    'If you cannot tell them apart or both are fine, say "="; if both are bad where they differ, say "x".').replace(
    '"verdict": "1" | "2" | "="', '"verdict": "1" | "2" | "=" | "x"')
assert PROMPT.count('"x"') == 2


def wave(n: int):
    """n new pairs, the pictures' kept alternatives shuffled and taken round robin, a picture's null after its
    pairs; sheets as in curve.make (order a: the alternative is colouring 1)."""
    key = json.loads((ROUND / 'key.json').read_text()) if (ROUND / 'key.json').exists() else {}
    w = 1 + max([k['wave'] for k in key.values()], default=0)
    rng = np.random.default_rng(w)
    used = {k['id'] for k in key.values()}
    by = {}
    for name in PICTURES:
        cs = [c for c in kept(json.loads((OUT / f'{name}.json').read_text())) if c['id'] not in used]
        rng.shuffle(cs)
        null = [] if f'{name}/null' in used or not cs else [dict(id=f'{name}/null', image=name, alt='null', cells=0)]
        by[name] = cs + null
    order = [v[i] for i in range(max(map(len, by.values()))) for v in by.values() if i < len(v)][:n]
    free = [i for i in range(1000) if f'e{i:03d}' not in key]
    sheets = OUT / 'sheets'
    (OUT / 'raw').mkdir(parents=True, exist_ok=True)
    sheets.mkdir(exist_ok=True)
    save = lambda f, img: cv2.imwrite(str(sheets / f), cv2.cvtColor(seen2(img, *EYE), cv2.COLOR_RGB2BGR))
    for c, i in zip(order, rng.choice(free, len(order), replace=False)):
        name, z = f'e{i:03d}', np.load(OUT / f'{c["image"]}.npz')
        T, X, Y = z['target'], z['N'] if c['alt'] == 'null' else z[f'X{c["n"]}'], z['A']
        np.savez_compressed(OUT / 'raw' / f'{name}.npz', T=T, X=X, Y=Y)   # X the alternative's side, Y the base
        save(f'{name}-0.png', T)
        for o, (a, b) in (('a', (X, Y)), ('b', (Y, X))):
            save(f'{name}{o}-1.png', a)
            save(f'{name}{o}-2.png', b)
        cells = np.zeros(z['base'].shape, bool) if c['alt'] == 'null' else z[f'L{c["n"]}'] != z['base']
        key[name] = dict(c, wave=w, truth=regions(cells))
    batches = deal([f'{name}{o}' for name, k in key.items() if k['wave'] == w for o in 'ab' for _ in range(VOTES)],
                   rng)
    ROUND.mkdir(parents=True, exist_ok=True)
    (ROUND / 'key.json').write_text(json.dumps(dict(sorted(key.items())), indent=1))
    (OUT / 'prompts').mkdir(exist_ok=True)
    for j, b in enumerate(batches):
        (OUT / 'prompts' / f'w{w}-{j}.txt').write_text(PROMPT.format(
            folder=sheets, items=' '.join(f'{x[:-1]} {x}' for x in b), results=ROUND / f'w{w}-{j}.json', first=b[0]))
    print(f'wave {w}: {len(order)} pairs, {len(batches)} judges, prompts data/dp/prompts/w{w}-*.txt')


def score():
    """Per pair the votes for the alternative (+), the base (-), = and x, and the label (more than half the votes)."""
    key = json.loads((ROUND / 'key.json').read_text())
    votes, side = {}, Counter()
    for f in sorted(ROUND.glob('w*-*.json')):
        for r in json.loads(f.read_text()):
            name, o, v = r['item'][:-1], r['item'][-1], r['verdict']
            side[v] += 1
            votes.setdefault(name, []).append(v if v in '=x' else '+' if v == ('1' if o == 'a' else '2') else '-')
    print(f'sides {dict(side)}')
    labels = Counter()
    for name, vs in sorted(votes.items(), key=lambda kv: key[kv[0]]['id']):
        k, c = key[name], Counter(vs)
        top, m = c.most_common(1)[0]
        label = top if m > len(vs) / 2 else 'split'
        labels[(k['alt'] == 'null', label)] += 1
        print(f'  {name} {k["id"]:28} cells {k["cells"]:3} margin {k.get("margin", 0):8.3f} dE {k.get("dE", 0):5.1f}'
              f'   {"".join(sorted(vs))}  {label}')
    for (null, label), n in sorted(labels.items()):
        print(f'  {"null" if null else "pairs":5} {label:5} {n}')
    groups = {}
    for name, vs in votes.items():
        k = key[name]
        if k['alt'] != 'null':
            d = k['dE']
            for g in (k['alt'], 'dE 10-15' if d < 15 else 'dE 15-20' if d < 20 else 'dE 20-30' if d < 30 else 'dE 30+'):
                groups.setdefault(g, []).append(vs)
    print('by kind and change: votes + alternative, - base, = even, x both bad; labels by majority')
    for g, vss in sorted(groups.items()):
        n, lab = Counter(v for vs in vss for v in vs), Counter()
        for vs in vss:
            top, m = Counter(vs).most_common(1)[0]
            lab[top if m > len(vs) / 2 else 'split'] += 1
        print(f'  {g:10} pairs {len(vss):2}  votes ' + ' '.join(f'{x}{n[x]:3}' for x in '+-=x')
              + '   labels ' + ' '.join(f'{x}{lab[x]:2}' for x in ('+', '-', '=', 'x', 'split')))


def labels() -> dict:
    """{pair name: label}: +, -, = or split, by more than half the votes."""
    votes = {}
    for f in sorted(ROUND.glob('w*-*.json')):
        for r in json.loads(f.read_text()):
            name, o, v = r['item'][:-1], r['item'][-1], r['verdict']
            votes.setdefault(name, []).append(v if v in '=x' else '+' if v == ('1' if o == 'a' else '2') else '-')
    out = {}
    for name, vs in votes.items():
        top, m = Counter(vs).most_common(1)[0]
        out[name] = top if m > len(vs) / 2 else 'split'
    return out


def user(n=30, repeats=5, seed=4):
    """The judges' own images of n labelled pairs (sides and =, no nulls or splits), round robin over the pictures, side
    by side in one file (the picture, 1, 2); repeats of them again in the other order. Names shuffled, so a and b do not
    give the side away; key rounds/e1/user-key.json (the alternative's side as 'x')."""
    from views import GAP
    rng = np.random.default_rng(seed)
    key, lab = json.loads((ROUND / 'key.json').read_text()), labels()
    by = {}
    for name in rng.permutation(sorted(lab)):
        if lab[name] != 'split' and key[name]['alt'] != 'null':
            by.setdefault(key[name]['image'], []).append(str(name))
    pairs = [v[i] for i in range(max(map(len, by.values()))) for v in by.values() if i < len(v)][:n]
    items = [(p, str(rng.choice(['a', 'b']))) for p in pairs]
    items += [(p, 'b' if o == 'a' else 'a') for p, o in items[:repeats]]
    out = OUT / 'user'
    out.mkdir(exist_ok=True)
    read = lambda f: cv2.imread(str(OUT / 'sheets' / f))
    user_key = {}
    for i, j in enumerate(rng.permutation(len(items))):
        p, o = items[j]
        imgs = [read(f'{p}-0.png'), read(f'{p}{o}-1.png'), read(f'{p}{o}-2.png')]
        gap = np.full((imgs[0].shape[0], GAP, 3), 90, np.uint8)
        cv2.imwrite(str(out / f'u{i + 1:02d}.png'), np.concatenate([imgs[0], gap, imgs[1], gap, imgs[2]], 1))
        user_key[f'u{i + 1:02d}'] = dict(pair=p, id=key[p]['id'], x=1 if o == 'a' else 2, judges=lab[p],
                                         repeat=bool(j >= n))
    (ROUND / 'user-key.json').write_text(json.dumps(user_key, indent=1))
    print(f'{len(user_key)} sheets in {out}: ' + str(Counter(lab[p] for p, _ in items[:n])))


PAGE = """<!doctype html><meta charset="utf-8"><title>dizher e1</title>
<style>body{margin:0;background:#222;color:#ddd;font:15px system-ui}#bar{padding:8px 12px}
img{display:block;margin:0 auto;image-rendering:pixelated}b{color:#fff}</style>
<div id="bar"></div><img id="img">
<script>
const S = %s, V = %s; let i = S.findIndex(s => !(s in V)); if (i < 0) i = S.length;
const img = document.getElementById('img'), bar = document.getElementById('bar');
function show() {
  const done = Object.keys(V).length;
  if (i >= S.length) { bar.innerHTML = `<b>All ${done} of ${S.length} done.</b> Backspace to go back.`; img.hidden = true; return; }
  img.hidden = false; img.src = '/img/' + S[i] + '.png';
  bar.innerHTML = `<b>${S[i]}</b> (${i + 1} of ${S.length}, ${done} answered${S[i] in V ? ', yours: ' + V[S[i]] : ''})` +
    ` &nbsp; left the picture, middle 1, right 2 &nbsp; keys: <b>&larr;</b> 1 &nbsp; <b>&rarr;</b> 2 &nbsp; <b>Space</b> both fine &nbsp; <b>x</b> both bad &nbsp; Backspace back`;
}
function fit() {   // the whole sheet as large as the window holds, up or down
  // whole screen pixels per sheet pixel, square (a fractional enlargement smoothed the raw sheets); smaller only to fit
  const d = devicePixelRatio, fit = d * Math.min(innerWidth / img.naturalWidth, (innerHeight - bar.offsetHeight) / img.naturalHeight);
  img.style.width = img.naturalWidth * (fit >= 1 ? Math.floor(fit) : fit) / d + 'px';
}
img.onload = fit; addEventListener('resize', fit);
addEventListener('keydown', async e => {
  if (e.key === 'Backspace') { i = Math.max(0, i - 1); return show(); }
  const v = {ArrowLeft: '1', ArrowRight: '2', ' ': '=', x: 'x'}[e.key];
  if (!v || i >= S.length) return;
  e.preventDefault();
  V[S[i]] = v;
  await fetch('/vote', {method: 'POST', body: JSON.stringify({sheet: S[i], verdict: v})});
  i++; show();
});
show();
</script>"""


def vote(round_dir, sheets_dir, port=8765):
    """The user's sheets one at a time in the browser; each key press saved to ROUND/user-verdicts.json."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    round_dir, sheets_dir = HERE / round_dir, HERE / sheets_dir   # relative to this folder
    sheets = sorted(json.loads((round_dir / 'user-key.json').read_text()))
    path = round_dir / 'user-verdicts.json'
    votes = json.loads(path.read_text()) if path.exists() else {}

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')   # sheets made again under the same names
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith('/img/'):
                return self.send((sheets_dir / Path(self.path).name).read_bytes(), 'image/png')
            self.send((PAGE % (json.dumps(sheets), json.dumps(votes))).encode(), 'text/html')

        def do_POST(self):
            v = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            votes[v['sheet']] = v['verdict']
            path.write_text(json.dumps(dict(sorted(votes.items())), indent=1))
            self.send(b'ok', 'text/plain')

        def log_message(self, *args):
            pass

    print(f'http://localhost:{port}  (Ctrl+C to stop; answers in {path})')
    webbrowser.open(f'http://localhost:{port}')
    HTTPServer(('localhost', port), Handler).serve_forever()


if __name__ == '__main__':
    {'make': make, 'survey': survey, 'wave': lambda: wave(int(sys.argv[2])), 'score': score, 'user': user, 'vote': lambda: vote(*sys.argv[2:4]) if len(sys.argv) >= 4 else
             sys.exit('usage: dp.py vote ROUND SHEETS (both: a missing SHEETS once showed another round\'s sheets)')}[sys.argv[1]]()
