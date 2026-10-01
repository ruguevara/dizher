"""Session E step 1: what does a Claude judge see? A detection test, no preference: two colourings of a painted picture
that are the same or differ by one painted segment (A against B), the judge names where they differ on a 4x3 grid.
The same pairs in every layout, each layout to its own fresh judges.

    python research/pairs/acuity.py make     sheets in data/acuity/LAYOUT/, key and batches in rounds/acuity/
    python research/pairs/acuity.py prompts LAYOUT     each batch's prompt, one per fresh subagent
    python research/pairs/acuity.py score    rounds/acuity/LAYOUT-*.json against the key, by segment size

Layouts: a the judges' sheet so far (views.judge_sheet, 3088x768); c three files, each through the eye at 4x; r three
files, raw at 4x (nearest, as the user's app shows); e three files, 4x square pixels, then the eye the user picked
(views.seen2: lightness sigma 0.75, colour 1.0, in linear light). A text chart (data/acuity/chart/, 2026-10-01) showed the 3088 px
sheet shrinks little if at all on the way in: 8 px digits read at every width from 1000 to 3100 px, 6 px at 4-6 of 6."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA   # noqa: E402
from views import judge_sheet, seen, seen2, up   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'acuity', HERE / 'rounds' / 'acuity'
BINS, PER_BIN, NULLS, PER_JUDGE = ((1, 2), (3, 6), (7, 15), (16, 999)), 6, 4, 10
LAYOUTS = 'acre'
COLS, ROWS = 'ABCD', '123'
EYE = (0.75, 1.0)   # the user's pick on the blur grids (data/acuity/grid/): lightness and colour sigma, Spectrum px   # 4x3 grid of 8x8 cells each


def regions(cells: np.ndarray) -> list:
    """The grid squares (e.g. 'B2') holding any of the (24, 32) cells."""
    return sorted({COLS[c // 8] + ROWS[r // 8] for r, c in zip(*np.nonzero(cells))})


def write(layout: str, name: str, T, X1, X2) -> list:
    out = OUT / layout
    save = lambda f, img: cv2.imwrite(str(out / f), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if layout == 'a':
        save(f'{name}.png', judge_sheet(T, X1, X2))
        return [f'{name}.png']
    view = {'c': lambda img: up(seen(img, 1), 4), 'r': lambda img: up(img, 4),
            'e': lambda img: seen2(img, *EYE)}[layout]
    for i, img in enumerate((T, X1, X2)):
        save(f'{name}-{i}.png', view(img))
    return [f'{name}-{i}.png' for i in range(3)]


def make(seed=0):
    rng = np.random.default_rng(seed)
    segs = [(p.stem, m) for p in sorted(DATA.glob('*.npz')) for m in json.loads(p.with_suffix('.json').read_text())]
    picked = []
    for lo, hi in BINS:   # evenly over the images within a size bin
        pool = [s for s in segs if lo <= s[1]['cells'] <= hi]
        order = rng.permutation(len(pool))
        seen_img, first, rest = set(), [], []
        for i in order:
            (first if pool[i][0] not in seen_img else rest).append(pool[i])
            seen_img.add(pool[i][0])
        picked += (first + rest)[:PER_BIN]
    nulls = [s for s in segs if s[1]['null']]
    picked += [nulls[i] for i in rng.choice(len(nulls), NULLS, replace=False)]
    names = [f'q{i:02d}' for i in rng.permutation(len(picked))]
    key = {}
    for (im, m), name in zip(picked, names):
        z = np.load(DATA / f'{im}.npz')
        s, null = m['segment'], len(key) >= len(picked) - NULLS
        other = z['N'] if null else z[f'B{s}']
        X1, X2 = (z['A'], other) if rng.integers(2) else (other, z['A'])
        cells = z['diff'] & (z['segments'] == s)
        key[name] = dict(id=m['id'], cells=0 if null else m['cells'], truth=[] if null else regions(cells))
        for layout in LAYOUTS:
            (OUT / layout).mkdir(parents=True, exist_ok=True)
            key[name][layout] = write(layout, name, z['target'], X1, X2)
    ROUND.mkdir(parents=True, exist_ok=True)
    (ROUND / 'key.json').write_text(json.dumps(dict(sorted(key.items())), indent=1))
    order = list(rng.permutation(sorted(key)))
    n = -(-len(order) // PER_JUDGE)
    (ROUND / 'batches.json').write_text(json.dumps([order[i::n] for i in range(n)], indent=1))
    print(f'{len(key)} pairs x {len(LAYOUTS)} layouts, {n} judges a layout')


PROMPT = """You are testing what you can see in images. Judge by eye only: do NOT write or run any code, do not create \
any files except your one result file, do not crop or zoom with tools. Just open each image with the Read tool and look.

Each item is a picture and two ZX Spectrum conversions of it, 1 and 2 (two colours per 8x8 pixel cell), {how}. The two \
conversions are either identical apart from dither noise, or the colours of a few cells differ in one area (from one \
cell to a few dozen). Some items are identical: do not invent a difference. Think of each conversion as a grid of 4 \
columns A-D (left to right) by 3 rows 1-3 (top to bottom), equal squares. Name the square where 1 and 2 differ (two \
if the area straddles a border), or "same".

Items (folder {folder}/): {items}

Write a JSON list to {results}, one object per item:
{{"item": "{first}", "where": ["B2"] or [] for same, "confidence": "high" | "medium" | "low", "note": "what differs, \
one short sentence"}}
Reply with just "done" when the file is written."""
HOW = {'a': 'side by side in one image: LEFT the picture, MIDDLE 1, RIGHT 2, slightly blurred and enlarged',
       'c': 'as three images per item: NAME-0.png the picture, NAME-1.png conversion 1, NAME-2.png conversion 2, '
            'slightly blurred and enlarged',
       'e': 'as three images per item: NAME-0.png the picture, NAME-1.png conversion 1, NAME-2.png conversion 2, '
            'slightly blurred and enlarged',
       'r': 'as three images per item: NAME-0.png the picture, NAME-1.png conversion 1, NAME-2.png conversion 2, '
            'enlarged with square pixels'}


def prompts(layout: str):
    key = json.loads((ROUND / 'key.json').read_text())
    for i, b in enumerate(json.loads((ROUND / 'batches.json').read_text())):
        items = ' '.join(b) if layout != 'a' else ' '.join(key[n][layout][0] for n in b)
        print(f'--- judge {layout}{i}\n' + PROMPT.format(how=HOW[layout], folder=OUT / layout, items=items,
                                                         results=ROUND / f'{layout}-{i}.json', first=b[0]))


def score():
    key = json.loads((ROUND / 'key.json').read_text())
    label = lambda k: 'null' if not k['truth'] else next(f'{lo}-{hi}' for lo, hi in BINS if lo <= k['cells'] <= hi)
    for layout in LAYOUTS:
        res = {}
        for f in ROUND.glob(f'{layout}-*.json'):
            for r in json.loads(f.read_text()):
                res[r['item'].split('-')[0].removesuffix('.png')] = r
        if not res:
            continue
        hit, chance = defaultdict(list), defaultdict(list)
        for name, k in key.items():
            if name not in res:
                continue
            where = set(res[name]['where'])
            ok = not where if not k['truth'] else bool(where & set(k['truth']))
            hit[label(k)].append(ok)
            chance[label(k)].append(min(1, len(k['truth']) * max(1, len(where)) / 12) if k['truth'] else 0)
        line = '  '.join(f'{b} {sum(v)}/{len(v)}' + (f' (chance {np.mean(chance[b]):.2f})' if b != 'null' else '')
                         for b, v in sorted(hit.items()))
        print(f'{layout}: {line}')


if __name__ == '__main__':
    {'make': make, 'prompts': lambda: prompts(sys.argv[2]), 'score': score}[sys.argv[1]]()
