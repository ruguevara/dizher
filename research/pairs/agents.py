"""Do Claude judges see what the user saw in the full judges' optima (session B, rounds 3 and 4)?

    python research/pairs/agents.py make     judges' sheets in data/aopt/, key and batches in rounds/aopt/
    python research/pairs/agents.py prompts [TAG]  each batch's prompt, filled in, one per fresh subagent (aopt)
    python research/pairs/agents.py score    rounds/aopt/results-*.json against the user's verdicts (rounds/optN)

Per round and picture two pairs, the optimum against Select pairs and against the painting, each as two sheets with
the sides swapped (views.judge_sheet), never both with one judge; a few null pairs (the painting against another
halftone origin of itself). A pair's verdict: +, = or - for the optimum when both sheets agree, split otherwise."""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA   # noqa: E402
from views import judge_sheet   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'aopt', HERE / 'rounds' / 'aopt'
ROUNDS, NULLS, PER_JUDGE = (3, 4), 4, 10
PROMPT = """You are a blind judge of ZX Spectrum picture conversions. Judge by eye only: do NOT write or run any code, \
do not create any files except your one result file, do not crop or zoom with tools. Just open each image with the \
Read tool and look.

Each sheet is one PNG with three pictures side by side, shown as a viewer sees them (slightly blurred, enlarged): LEFT \
the original picture, MIDDLE colouring 1, RIGHT colouring 2. Both colourings are Spectrum conversions of the same \
picture (two colours per 8x8 cell). Decide which colouring a demoscene artist would rather keep as the conversion of \
this picture, looking at the whole picture. Typical faults: colours that do not read as the original's colours (wrong \
hue, drained to grey, garish), blocky cell seams or patches where the picture is smooth, noisy cell-to-cell colour \
changes or specks, lost detail. If you honestly cannot tell them apart or they are equally good, say "=".

Sheets (folder {folder}/): {files}

Write your verdicts as a JSON list to {results}, one object per sheet:
{{"file": "{first}", "verdict": "1" | "2" | "=", "confidence": "high" | "medium" | "low", "problem": "colour" | \
"seams" | "noise" | "detail" | "none", "note": "one short sentence on what is wrong with the worse one"}}
Reply with just "done" when the file is written."""


def write_round(tag: str, pairs: list, seed=0):
    """Judges' sheets of pairs [(id, meta, T, X, Y)] in data/TAG/, each pair two sheets with the sides swapped, never
    both in one batch; key (X's side as 'x') and batches in rounds/TAG/."""
    rng = np.random.default_rng(seed)
    out, folder = DATA / tag, HERE / 'rounds' / tag
    out.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    names = [f'{tag[0]}{i:03d}' for i in rng.permutation(2 * len(pairs))]
    key, sheets = {}, {}
    for pid, meta, T, X, Y in pairs:
        for side, (X1, X2) in ((1, (X, Y)), (2, (Y, X))):
            name = names.pop()
            cv2.imwrite(str(out / f'{name}.png'), cv2.cvtColor(judge_sheet(T, X1, X2), cv2.COLOR_RGB2BGR))
            key[name] = dict(id=pid, **meta, x=side)
            sheets.setdefault(pid, []).append(name)
    n = -(-len(key) // PER_JUDGE)
    batches = [[] for _ in range(n)]
    for j, i in enumerate(rng.permutation(len(pairs))):   # shuffled pairs dealt round robin
        a, b = sheets[pairs[i][0]]
        batches[(2 * j) % n].append(a)
        batches[(2 * j + 1) % n].append(b)
    for b in batches:
        rng.shuffle(b)
    (folder / 'key.json').write_text(json.dumps(dict(sorted(key.items())), indent=1))
    (folder / 'batches.json').write_text(json.dumps(batches, indent=1))
    print(f'{len(key)} sheets, {len(batches)} batches of {[len(b) for b in batches]}')


def make(seed=0):
    rng = np.random.default_rng(seed)
    images = sorted(f.stem for f in DATA.glob('*.npz'))
    pairs = []
    for r, im, vs in [(r, im, vs) for r in ROUNDS for im in images for vs in ('select', 'painting')] + \
            [(0, im, 'null') for im in rng.choice(images, NULLS, replace=False)]:
        z = np.load(DATA / f'{im}.npz')
        X, Y = (z['A'], z['N']) if vs == 'null' else \
            (np.load(DATA / 'optimum' / f'r{r}' / f'{im}.npz')['O'], z['H' if vs == 'select' else 'A'])
        pairs.append((f'{im}/r{r}/{vs}', dict(image=im, round=r, vs=vs), z['target'], X, Y))
    write_round('aopt', pairs, seed)


def prompts(tag='aopt'):
    folder = HERE / 'rounds' / tag
    for i, b in enumerate(json.loads((folder / 'batches.json').read_text())):
        print(f'--- judge {i}\n' + PROMPT.format(folder=DATA / tag, files=' '.join(f'{n}.png' for n in b),
                                                 results=folder / f'results-{i}.json', first=f'{b[0]}.png'))


def tally(tag: str) -> dict:
    """{pair id: (key entry, ['+' | '=' | '-' for X, one per judged sheet], [confidence...])}."""
    folder = HERE / 'rounds' / tag
    key, res = json.loads((folder / 'key.json').read_text()), {}
    for f in sorted(folder.glob('results-*.json')):
        for r in json.loads(f.read_text()):
            res[r['file'].removesuffix('.png')] = r
    by = {}
    for name, k in sorted(key.items()):
        if name in res:
            v = res[name]['verdict']
            e = by.setdefault(k['id'], (k, [], []))
            e[1].append('=' if v == '=' else '+' if int(v) == k['x'] else '-')
            e[2].append(res[name]['confidence'])
    return by


def score():
    user = {r: json.loads((HERE / 'rounds' / f'opt{r}' / 'verdicts.json').read_text()) for r in ROUNDS}
    agree, table = Counter(), []
    for pid, (k, vs, confs) in tally('aopt').items():
        if len(vs) < 2:
            continue
        agent = vs[0] if vs[0] == vs[1] else 'split'
        if k['vs'] == 'null':
            table.append(f'{pid:32} null      agents {vs}')
            agree['null ' + ('even' if agent == '=' else 'not even')] += 1
            continue
        u = user[k['round']][k['image']][0 if k['vs'] == 'select' else 1]
        agree[f'r{k["round"]} {k["vs"]}: ' + ('same' if agent == u else 'split' if agent == 'split' else 'other')] += 1
        table.append(f'{pid:32} user {u}   agents {vs} {confs}')
    print('\n'.join(table))
    for k, v in sorted(agree.items()):
        print(f'  {k:28} {v}')


if __name__ == '__main__':
    {'make': make, 'prompts': lambda: prompts(*sys.argv[2:]), 'score': score}[sys.argv[1]]()
