"""Session E step 2: how often a Claude judge prefers what the user painted, by the size of the change.

    python research/pairs/curve.py make       renders and sheets in data/curve/, key and batches in rounds/curve/
    python research/pairs/curve.py prompts V  each batch's prompt for variant V (plain, taste), one per fresh subagent
    python research/pairs/curve.py score      rounds/curve/V-*.json against the key, by kind and size

Pairs with a known better side, the painting's:
- frac: per painted picture, Select pairs against Select pairs with a share f of the painted segments painted
  (f 0.1, 0.25, 0.5 random segments, at least one; f 1 is the whole painting);
- seg: the painting against itself with one segment given back to Select pairs (step 1's pairs of 3 cells or more);
- C: the painting against the segment's cells on their next best pair by the energy (build.py), assumed worse;
- null: the painting against itself from another halftone origin, even.
Each pair is judged in both orders by fresh judges, never twice by one; views are step 1's layout e (three files,
the user's eye, views.seen2). Variant plain is agents.PROMPT's task; taste adds what the user's choices so far say."""
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acuity import EYE, regions   # noqa: E402
from common import DATA, Project   # noqa: E402
from metrics import seen_change   # noqa: E402
from views import seen2   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'curve', HERE / 'rounds' / 'curve'
FRACS, NC = (0.1, 0.25, 0.5), 6
VOTES = {'plain': 3, 'taste': 2}   # per order
PER_JUDGE = 10


def frac_job(job):
    """Renders of Select pairs with the chosen segments painted: [(f, cells mask, uint8 render)]."""
    name, picks = job
    p, z = Project(name), np.load(DATA / f'{name}.npz')
    out = []
    for f, segs in picks:
        cells = z['diff'] & np.isin(z['segments'], segs)
        labels = z['bare'].copy()
        labels[cells] = z['painted'][cells]
        out.append((f, cells, np.round(p.render(labels) * 255).astype(np.uint8)))
    return name, out


def make(seed=0):
    rng = np.random.default_rng(seed)
    names = sorted(p.stem for p in DATA.glob('*.npz'))
    pairs = []   # (id, meta, T, better, worse, cells)
    picks = {}
    for name in names:
        touched = [m['segment'] for m in json.loads((DATA / f'{name}.json').read_text())]
        picks[name] = [(f, sorted(rng.choice(touched, max(1, round(f * len(touched))), replace=False).tolist()))
                       for f in FRACS]
    with Pool(len(names)) as pool:
        renders = dict(pool.map(frac_job, list(picks.items())))
    for name in names:
        z = np.load(DATA / f'{name}.npz')
        for f, cells, X in renders[name] + [(1.0, z['diff'], z['A'])]:
            pairs.append((f'{name}/f{f}', dict(kind='frac', image=name, f=f), z['target'], X, z['H'], cells))
    key = json.loads((HERE / 'rounds' / 'acuity' / 'key.json').read_text())
    segs = [(k['id'], k['cells']) for k in key.values() if k['cells'] >= 3]
    for pid, _ in segs:
        name, s = pid.split('/')
        z = np.load(DATA / f'{name}.npz')
        cells = z['diff'] & (z['segments'] == int(s))
        pairs.append((pid, dict(kind='seg', image=name), z['target'], z['A'], z[f'B{s}'], cells))
    for pid, _ in [segs[i] for i in rng.choice(len(segs), NC, replace=False)]:
        name, s = pid.split('/')
        z = np.load(DATA / f'{name}.npz')
        cells = z['diff'] & (z['segments'] == int(s))
        pairs.append((pid + '/C', dict(kind='C', image=name), z['target'], z['A'], z[f'C{s}'], cells))
    for k in key.values():
        if not k['truth']:
            name = k['id'].split('/')[0]
            z = np.load(DATA / f'{name}.npz')
            pairs.append((k['id'] + '/null', dict(kind='null', image=name), z['target'], z['A'], z['N'],
                          np.zeros_like(z['diff'])))
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    names = [f'p{i:02d}' for i in rng.permutation(len(pairs))]
    sheets, meta = {}, {}
    for (pid, m, T, X, Y, cells), name in zip(pairs, names):
        t, x, y = (seen2(img, *EYE) for img in (T, X, Y))
        cv2.imwrite(str(OUT / f'{name}-0.png'), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
        for order, (a, b) in ((1, (x, y)), (2, (y, x))):   # order 1: the better side is colouring 1
            for i, img in ((1, a), (2, b)):
                cv2.imwrite(str(OUT / f'{name}{"ab"[order - 1]}-{i}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        meta[name] = dict(id=pid, **m, cells=int(cells.sum()), truth=regions(cells),
                          dE=round(float(seen_change(X / 255, Y / 255, np.kron(cells, np.ones((8, 8), bool))))
                                   if cells.any() else 0, 1))
    (ROUND / 'key.json').write_text(json.dumps(dict(sorted(meta.items())), indent=1))
    batches = {}
    for v, n in VOTES.items():
        pool = [p for p in meta if v == 'plain' or meta[p]['kind'] in ('frac', 'null')]
        batches[v] = deal([f'{p}{o}' for p in pool for o in 'ab' for _ in range(n)], rng)
    (ROUND / 'batches.json').write_text(json.dumps(batches, indent=1))
    print(f'{len(pairs)} pairs; judges: ' + ', '.join(f'{v} {len(b)}' for v, b in batches.items()))


def deal(items: list, rng) -> list:
    """Items (pair + order) into batches of at most PER_JUDGE, no pair twice in one batch."""
    n = -(-len(items) // PER_JUDGE)
    while True:
        batches = [[] for _ in range(n)]
        for it in rng.permutation(items):
            free = [b for b in batches if len(b) < PER_JUDGE and all(x[:3] != it[:3] for x in b)]
            if not free:
                break
            min(free, key=len).append(str(it))
        else:
            return batches


PROMPT = """You are a blind judge of ZX Spectrum picture conversions. Judge by eye only: do NOT write or run any code, \
do not create any files except your one result file, do not crop or zoom with tools. Just open each image with the \
Read tool and look.

Each item is three images, shown as a viewer sees them (slightly blurred, enlarged): ORIG-0.png the original \
picture, NAME-1.png colouring 1, NAME-2.png colouring 2. Both colourings are Spectrum conversions of the same picture \
(two colours per 8x8 cell); they may differ in a few places only, or not at all. Decide which colouring a demoscene \
artist would rather keep as the conversion of this picture, looking at the whole picture. Typical faults: colours \
that do not read as the original's colours (wrong hue, drained to grey, garish), blocky cell seams or cells that \
clash with their neighbours where the picture is smooth, noisy cell-to-cell colour changes or specks, lost detail. \
If you honestly cannot tell them apart or they are equally good, say "=".{taste}

Items (folder {folder}/), each as ORIG NAME: {items}

Write a JSON list to {results}, one object per item:
{{"item": "{first}", "verdict": "1" | "2" | "=", "confidence": "high" | "medium" | "low", "problem": "colour" | \
"seams" | "noise" | "detail" | "none", "where": "the 4x3 grid square (columns A-D left to right, rows 1-3 top to \
bottom) of the main difference, or empty", "note": "one short sentence on what is wrong with the worse one"}}
Reply with just "done" when the file is written."""
TASTE = """

This artist's choices so far: a clean colour of the right hue family beats a closer mixture dithered from two far \
colours (flat red rather than red dithered with white into pink); a cell whose dot colours or texture clash with its \
neighbours where the picture is smooth is worse than a small error of colour; they give up colour accuracy to avoid \
seams, but not the picture's main hues or its detail."""


def prompts(variant: str):
    for i, b in enumerate(json.loads((ROUND / 'batches.json').read_text())[variant]):
        print(f'--- judge {variant}{i}\n' + PROMPT.format(
            taste=TASTE if variant == 'taste' else '', folder=OUT, items=' '.join(f'{x[:3]} {x}' for x in b),
            results=ROUND / f'{variant}-{i}.json', first=b[0]))


def score():
    key = json.loads((ROUND / 'key.json').read_text())
    for v in VOTES:
        votes = defaultdict(list)   # pair -> ['+' painting, '-' other, '=']
        where = defaultdict(list)
        side = defaultdict(int)
        for f in ROUND.glob(f'{v}-*.json'):
            for r in json.loads(f.read_text()):
                name, order = r['item'][:3], r['item'][3]
                better = '1' if order == 'a' else '2'
                vote = '=' if r['verdict'] == '=' else '+' if r['verdict'] == better else '-'
                votes[name].append(vote)
                side[r['verdict']] += 1
                if vote == '+' and key[name]['truth']:
                    where[name].append(r.get('where', '') in key[name]['truth'])
        if not votes:
            continue
        print(f'{v}: sides {dict(side)}')
        groups = defaultdict(list)
        for name, vs in votes.items():
            k = key[name]
            g = k['kind'] if k['kind'] != 'frac' else f'frac {k["f"]}'
            if k['kind'] == 'seg':
                g = 'seg ' + ('3-6' if k['cells'] <= 6 else '7-15' if k['cells'] <= 15 else '16+')
            groups[g].append((name, vs))
        for g, items in sorted(groups.items()):
            n = defaultdict(int)
            for _, vs in items:
                for x in vs:
                    n[x] += 1
            maj = sum(vs.count('+') > max(vs.count('-'), vs.count('=')) for _, vs in items)
            loc = [x for name, _ in items for x in where[name]]
            print(f'  {g:10} pairs {len(items):2}  votes painting {n["+"]:3} other {n["-"]:3} even {n["="]:3}'
                  f'   pairs won by majority {maj}/{len(items)}'
                  + (f'   located {sum(loc)}/{len(loc)}' if loc else ''))


if __name__ == '__main__':
    {'make': make, 'prompts': lambda: prompts(sys.argv[2]), 'score': score}[sys.argv[1]]()
