"""PLAN step 1, the competitor baseline: the same tuned pictures (each project's Tune, the target Select pairs aims at)
through img2spec, Image to ZX Spec and ZX-Paintbrush, against ours, the Exact mixture preset and surface dE 20 (their
renders from surface.py make). Per scripted tool a few settings; Claude judges pick each tool's best per picture, which
then meets ours before the user, blind.

    python research/pairs/rivals.py make      inputs data/rivals/in/NAME.png (and .bmp), every tool's settings, read
                                              back into our palette and checked (two colours a cell, one bright),
                                              data/rivals/NAME.npz and a sheet to look at data/rivals/NAME.png;
                                              ZX-Paintbrush by hand: import in/NAME.bmp at its defaults, save
                                              rounds/rivals/paintbrush/NAME.scr, then make again
    python research/pairs/rivals.py judges    each pair of a tool's settings per picture, 3 votes per order, sheets
                                              data/rivals/sheets/, key rounds/rivals/key.json, prompts
                                              data/rivals/prompts/jN.txt, results rounds/rivals/jN.json
    python research/pairs/rivals.py best      the judges' votes, each tool's best setting per picture
    python research/pairs/rivals.py user      the user's sheets, ours against each tool's best, data/rivals/user/, key
                                              rounds/rivals/user-key.json; vote:
                                              python research/pairs/dp.py vote rounds/rivals data/rivals/user
    python research/pairs/rivals.py tally     the user's votes per tool and per picture

Tools: img2spec, the macOS build of ../img2spec_video (v5.6) by its CLI with a workspace per setting, its ZX Spectrum
device at defaults (popular colour); Image to ZX Spec 2.3.0 through Izx.java (needs Java 17+; the jar from
gh release download v2.3.0 -R KodeMunkie/imagetozxspec -p 'imagetozxspec-macos-2.3.0.zip', into data/rivals/bin/).
"""
import itertools
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acuity import EYE   # noqa: E402
from common import DATA, ROOT   # noqa: E402
from curve import deal   # noqa: E402
from dp import PROMPT   # noqa: E402
from views import GAP, seen2, up   # noqa: E402
from mokit.zx.screen import load_scr   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, ROUND = DATA / 'rivals', HERE / 'rounds' / 'rivals'
PICTURES = ('jojo', 'rocket-rackoon', 'autumn', 'diver-sunset')   # the user's pick
OURS = {'Exact mixture': 'base', 'surface dE 20': 'v7'}   # keys in data/surface/NAME.npz (surface.VARIANTS)
IMG2SPEC = ROOT.parent / 'img2spec_video' / 'build-macos' / 'img2spec_video'
JAVA = '/opt/homebrew/opt/openjdk/bin/java'
JAR = OUT / 'bin' / 'imagetozxspec-2.3.0.jar'
TOOLS = {   # setting: an img2spec stack item (its modifier type and params), Image to ZX Spec's dither (Izx.java)
    'img2spec': {'fs': {'Type': 7, 'mModel': 0},          # error diffusion, Floyd-Steinberg
                 'bayer8': {'Type': 6, 'mMatrix': 3},      # ordered, Bayer 8x8
                 'dbs': {'Type': 15, 'mFamily': 6}},       # mono DBS, modulate (keeps hue)
    'izx': {'atkinson': ('ed', 0), 'fs': ('ed', 2), 'bayer4': ('ordered', 3)},
    'paintbrush': {'default': None}}
VOTES = 3   # per order


def spectrum(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 a tool's output in its own palette (normal 192..215, bright 255) -> our palette (205, 255);
    fails unless every cell shows at most two colours of one brightness."""
    on, bright = img > 96, (img > 230).any(-1)
    idx = on @ np.array([2, 4, 1]) + 8 * (bright & on.any(-1))
    cells = idx.reshape(24, 8, 32, 8).transpose(0, 2, 1, 3).reshape(24, 32, 64)
    for r, c in np.ndindex(24, 32):
        u = set(cells[r, c])   # black is 0 either way
        assert len(u) <= 2 and len({i >= 8 for i in u if i}) <= 1, (r, c, sorted(u))
    return (on * np.where(bright, 255, 205)[..., None]).astype(np.uint8)


def scr(path: Path) -> np.ndarray:
    """(192, 256, 3) uint8 an SCR in our palette."""
    s = load_scr(path)
    up8 = lambda a: np.repeat(np.repeat(a, 8, 0), 8, 1)
    bits = np.where(s.pixels[..., None], up8(s.attr_ink), up8(s.attr_paper))
    return (bits * np.where(up8(s.attr_bright), 255, 205)[..., None]).astype(np.uint8)


def run(tool: str, setting: str, src: Path) -> np.ndarray:
    """(192, 256, 3) uint8 RGB as the tool exports it."""
    with tempfile.TemporaryDirectory() as tmp:   # img2spec loads a conv.isw from its working folder
        out = Path(tmp) / 'out.png'
        if tool == 'img2spec':
            isw = Path(tmp) / 'w.isw'
            isw.write_text(json.dumps({'About': {'Magic': '0x50534D49', 'Version': 4}, 'Config': {'gDeviceId': 0},
                                       'Device': {'Name': 'ZX Spectrum'}, 'Stack': {'Item[0]': dict(
                                           mEnabled=1, mR_en=1, mG_en=1, mB_en=1, **TOOLS[tool][setting])}}))
            cmd = [IMG2SPEC, src, isw, '-p', out]
        else:
            kind, i = TOOLS[tool][setting]
            cmd = [JAVA, '-Djava.awt.headless=true', '-cp', JAR, HERE / 'Izx.java', src, out, kind, str(i)]
        subprocess.run(list(map(str, cmd)), cwd=tmp, check=True, capture_output=True)
        return cv2.cvtColor(cv2.imread(str(out)), cv2.COLOR_BGR2RGB)


def make():
    (OUT / 'in').mkdir(parents=True, exist_ok=True)
    for name in PICTURES:
        z = np.load(DATA / 'surface' / f'{name}.npz')
        T, src = z['target'], OUT / 'in' / f'{name}.png'
        cv2.imwrite(str(src), cv2.cvtColor(T, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(src.with_suffix('.bmp')), cv2.cvtColor(T, cv2.COLOR_RGB2BGR))
        out = dict(target=T, **{f'ours-{k}': z[v] for k, v in OURS.items()})
        for tool, settings in TOOLS.items():
            for s in settings:
                if tool == 'paintbrush':
                    path = ROUND / 'paintbrush' / f'{name}.scr'
                    if path.exists():
                        out[f'{tool}-{s}'] = scr(path)
                else:
                    out[f'{tool}-{s}'] = spectrum(run(tool, s, src))
        np.savez_compressed(OUT / f'{name}.npz', **out)
        gap = np.full((384, GAP, 3), 90, np.uint8)
        row = [x for v in out.values() for x in (up(v, 2), gap)]
        cv2.imwrite(str(OUT / f'{name}.png'), cv2.cvtColor(np.concatenate(row[:-1], 1), cv2.COLOR_RGB2BGR))
        print(f'{name}: {", ".join(out)}', flush=True)


def judges(seed=0):
    """Per picture and scripted tool every pair of its settings, both orders, VOTES each, fresh judges (curve.deal)."""
    rng = np.random.default_rng(seed)
    sheets = OUT / 'sheets'
    sheets.mkdir(parents=True, exist_ok=True)
    save = lambda f, img: cv2.imwrite(str(sheets / f), cv2.cvtColor(seen2(img, *EYE), cv2.COLOR_RGB2BGR))
    pairs = [(name, tool, a, b) for name in PICTURES for tool, ss in TOOLS.items() if len(ss) > 1
             for a, b in itertools.combinations(ss, 2)]
    key = {}
    for i, (name, tool, a, b) in zip(rng.permutation(len(pairs)), pairs):
        p, z = f'r{i:02d}', np.load(OUT / f'{name}.npz')
        save(f'{p}-0.png', z['target'])
        for o, (x, y) in (('a', (a, b)), ('b', (b, a))):
            save(f'{p}{o}-1.png', z[f'{tool}-{x}'])
            save(f'{p}{o}-2.png', z[f'{tool}-{y}'])
        key[p] = dict(image=name, tool=tool, a=a, b=b)   # order a: setting a is colouring 1
    batches = deal([f'{p}{o}' for p in key for o in 'ab' for _ in range(VOTES)], rng)
    ROUND.mkdir(parents=True, exist_ok=True)
    (ROUND / 'key.json').write_text(json.dumps(dict(sorted(key.items())), indent=1))
    (OUT / 'prompts').mkdir(exist_ok=True)
    for j, b in enumerate(batches):
        (OUT / 'prompts' / f'j{j}.txt').write_text(PROMPT.format(
            folder=sheets, items=' '.join(f'{x[:-1]} {x}' for x in b), results=ROUND / f'j{j}.json', first=b[0]))
    print(f'{len(key)} pairs, {len(batches)} judges, prompts data/rivals/prompts/j*.txt')


def best() -> dict:
    """{(picture, tool): setting}: per setting the votes for it less those against over its pairs; the most wins."""
    key, score = json.loads((ROUND / 'key.json').read_text()), {}
    for f in sorted(ROUND.glob('j*.json')):
        for r in json.loads(f.read_text()):
            k, o, v = key[r['item'][:-1]], r['item'][-1], r['verdict']
            if v in '12':
                win, lose = (k['a'], k['b']) if v == ('1' if o == 'a' else '2') else (k['b'], k['a'])
                for s, d in ((win, 1), (lose, -1)):
                    score.setdefault((k['image'], k['tool']), Counter())[s] += d
    for (name, tool), c in sorted(score.items()):
        print(f'{name:15} {tool:9} ' + '  '.join(f'{s} {c[s]:+d}' for s in TOOLS[tool]))
    return {nt: max(TOOLS[nt[1]], key=lambda s: c[s]) for nt, c in score.items()}


def user(repeats=5, seed=0):
    """Per picture, tool and ours one sheet: the picture, then ours and the tool's best in a random order, raw 3x as
    the app shows them; repeats of some again with the sides swapped; all shuffled."""
    rng, pick = np.random.default_rng(seed), best()
    items = [(name, tool, ours, int(rng.integers(1, 3))) for name in PICTURES for tool in TOOLS for ours in OURS]
    m = len(items)
    items += [(n, t, o, 3 - x) for n, t, o, x in (items[k] for k in rng.permutation(m)[:repeats])]
    out = OUT / 'user'
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    for i, j in enumerate(rng.permutation(len(items))):
        name, tool, ours, side = items[j]
        z = np.load(OUT / f'{name}.npz')
        s = pick.get((name, tool), next(iter(TOOLS[tool])))
        a, b = (z[f'ours-{ours}'], z[f'{tool}-{s}'])[::1 if side == 1 else -1]
        gap = np.full((576, GAP, 3), 90, np.uint8)
        sheet = np.concatenate([up(z['target'], 3), gap, up(a, 3), gap, up(b, 3)], 1)
        cv2.imwrite(str(out / f'u{i + 1:02d}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        key[f'u{i + 1:02d}'] = dict(id=name, tool=tool, setting=s, ours=ours, x=side, repeat=bool(j >= m))
    ROUND.mkdir(parents=True, exist_ok=True)
    (ROUND / 'user-key.json').write_text(json.dumps(key, indent=1))
    print(f'{len(key)} sheets in {out}')


def tally():
    """Per ours and tool, then per picture: + ours better, - the tool better, = both fine, x both bad; the repeats
    against the first answer."""
    key = json.loads((ROUND / 'user-key.json').read_text())
    votes = json.loads((ROUND / 'user-verdicts.json').read_text())
    mark = lambda u: votes[u] if votes[u] in '=x' else '+' if votes[u] == str(key[u]['x']) else '-'
    first = {(k['id'], k['tool'], k['ours']): mark(u) for u, k in key.items() if u in votes and not k['repeat']}
    again = [(first.get((k['id'], k['tool'], k['ours'])), mark(u)) for u, k in key.items()
             if u in votes and k['repeat']]
    print(f'repeats same {sum(a == b for a, b in again)} of {len(again)}: {again}')
    for group, at in (('tool', lambda n, t, o: (o, t)), ('picture', lambda n, t, o: (o, n))):
        rows = {}
        for (n, t, o), v in first.items():
            rows.setdefault(at(n, t, o), Counter())[v] += 1
        print(f'by {group}: ' + '; '.join(f'{o} vs {g}: ' + ' '.join(f'{m}{c[m]}' for m in '+-=x' if c[m])
                                          for (o, g), c in sorted(rows.items())))


if __name__ == '__main__':
    {'make': make, 'judges': judges, 'best': best, 'user': user, 'tally': tally}[sys.argv[1]]()
