"""Session E step 3: labels from Claude judges where Select pairs is unsure of its own choice.

    python research/pairs/dp.py make       per picture, the alternatives the converter itself makes for each segment,
                                           kept where the energy's margin is small and the change is seen; data/dp/
    python research/pairs/dp.py wave N     N new pairs into round e1 (round robin over the pictures): sheets in
                                           data/dp/sheets/, key rounds/e1/key.json, a prompt file per judge
                                           data/dp/prompts/wW-I.txt, results rounds/e1/wW-I.json
    python research/pairs/dp.py score      per pair the judges' votes and the label by majority

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


if __name__ == '__main__':
    {'make': make, 'survey': survey, 'wave': lambda: wave(int(sys.argv[2])), 'score': score}[sys.argv[1]]()
