"""Judgments: pairs of a picture's variants shown side by side as A and B, each judge's verdict (a, b or same) with
the faults named, and the agreement between judges. tests/images/NAME/judgments.json:

    {"pairs": [{"k": 1, "a": "<variant id>", "b": "<variant id>",
                "verdicts": {"user": {"verdict": "a", "tags": ["hue"], "note": ""}, "claude": {...}}}]}

k numbers the pairs of a picture across batches; the sheet of pair k is NAME-k.png. A verdict comes as text,
one pair per line, semicolon or comma: "12 a hue, 13 same; 14 b clash noise -- the sky, at the right", tags from TAGS,
a note after -- running to the end of the line or semicolon (commas inside it stay).
"""
import json
import re
from pathlib import Path

import numpy as np

from mokit.graph import Memo, evaluate


from . import render as R
from .project import IMAGES, DEFAULTS, project_graph
from .scr import render_scr
from .variants import listing, read_variant, keep, distance, reference_id

TAGS = ('hue', 'clash', 'noise', 'tone', 'other')   # wrong colour, block seams, pairs varying over one surface, lightness
VERDICTS = {'a': 'a', 'b': 'b', 'same': 'same', '=': 'same', 's': 'same'}


def judgments_file(name) -> Path:
    return IMAGES / name / 'judgments.json'


def load(name) -> dict:
    f = judgments_file(name)
    return json.loads(f.read_text()) if f.exists() else {'pairs': []}


def save(name, data) -> None:
    judgments_file(name).write_text(json.dumps(data, indent=1))


def pick(name, n, seed=0, judged=()):
    """n pairs of the picture's variants in turn: the farthest apart in settings, a random pair, the reference
    against a variant (when there is one); none repeated, A/B order random."""
    rng = np.random.default_rng(seed)
    metas = listing(name)
    rid = reference_id(name)
    ids = [i for i in metas if not metas[i].get('reference')]
    assert len(ids) >= 2, f'{name}: generate variants first'
    seen = {frozenset(p) for p in judged}
    pairs = []
    cands = [(distance(metas[a], metas[b]), a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    cands.sort(key=lambda t: -t[0])
    far = [(a, b) for _, a, b in cands]
    random = [far[i] for i in rng.permutation(len(far))]
    refs = [(rid, i) for i in rng.permutation(ids)] if rid else []
    pools = [far, random, refs]
    turn = 0
    while len(pairs) < n and any(pools):
        pool = pools[turn % 3]
        turn += 1
        if not pool:
            continue
        a, b = pool.pop(0)
        if frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        pairs.append((a, b) if rng.random() < 0.5 else (b, a))
    return pairs


def add_pairs(name, pairs) -> list:
    """The pairs appended to the picture's judgments with fresh numbers; their variants kept. Returns the entries."""
    data = load(name)
    k = max((p['k'] for p in data['pairs']), default=0)
    entries = []
    for a, b in pairs:
        k += 1
        entries.append(dict(k=k, a=a, b=b, verdicts={}))
        keep(name, a)
        keep(name, b)
    data['pairs'] += entries
    save(name, data)
    return entries


def sheet(name, a, b, zoom=2, eye=True):
    """target | A | B, the eye views under them; A and B unnamed."""
    graph = project_graph(name, DEFAULTS)
    conv = evaluate(graph, 'prepare', Memo())
    pics = [('target', conv.image_rgb)]
    for title, vid in (('A', a), ('B', b)):
        bitmap, idx = read_variant(name, vid)
        pics.append((title, render_scr(bitmap, idx, conv.palette)))
    return R.sheet(pics, zoom, conv.eye_view if eye else None)


def make_sheets(name, entries, out, zoom=2, eye=True, log=print) -> list:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for e in entries:
        path = out / f'{name.replace("/", "-")}-{e["k"]}.png'
        R.save(path, sheet(name, e['a'], e['b'], zoom, eye))
        paths.append(path)
        log(f'{name} #{e["k"]}: {path}')
    return paths


def parse(text: str) -> dict:
    """'12 a hue, 13 same, 14 b clash noise -- the sky' -> {12: {verdict, tags, note}, ...}."""
    out = {}
    items = []
    for chunk in re.split(r'[\n;]+', text):
        head, _, note = chunk.partition('--')
        parts = [p for p in head.split(',') if p.strip()]
        items += [(p, '') for p in parts[:-1]] + ([(parts[-1], note)] if parts else [])
    for body, note in items:
        words = body.split()
        if len(words) < 2 or not words[0].isdigit() or words[1].lower() not in VERDICTS:
            raise ValueError(f'cannot read {body.strip()!r}: expected "<pair> a|b|same [tags] [-- note]"')
        tags = [w.lower() for w in words[2:]]
        unknown = [t for t in tags if t not in TAGS]
        if unknown:
            raise ValueError(f'unknown tags {unknown} in {body.strip()!r}; tags are {TAGS}')
        out[int(words[0])] = dict(verdict=VERDICTS[words[1].lower()], tags=tags, note=note.strip())
    return out


def record(name, text, by) -> int:
    """A judge's verdicts from text into the picture's judgments; returns how many."""
    verdicts = parse(text)
    data = load(name)
    by_k = {p['k']: p for p in data['pairs']}
    missing = [k for k in verdicts if k not in by_k]
    assert not missing, f'{name} has no pairs {missing}'
    for k, v in verdicts.items():
        by_k[k]['verdicts'][by] = v
    save(name, data)
    return len(verdicts)


def kappa(x, y, classes=('a', 'b', 'same')) -> float:
    """Cohen's kappa of two judges' verdict lists."""
    n = len(x)
    if n == 0:
        return float('nan')
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in classes)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(names, judges=('user', 'claude')) -> dict:
    """Over the pairs both judges gave a verdict: agreement, kappa, the share of 'same', and agreement on the pairs
    the first judge tagged with each fault."""
    u, v, tagged = [], [], {t: [] for t in TAGS}
    for name in names:
        for p in load(name)['pairs']:
            vs = p['verdicts']
            if all(j in vs for j in judges):
                a, b = vs[judges[0]]['verdict'], vs[judges[1]]['verdict']
                u.append(a)
                v.append(b)
                for t in vs[judges[0]].get('tags', ()):
                    tagged[t].append(a == b)
    n = len(u)
    return dict(pairs=n, agree=sum(a == b for a, b in zip(u, v)) / n if n else float('nan'), kappa=kappa(u, v),
                same=(u.count('same') / n, v.count('same') / n) if n else (float('nan'),) * 2,
                # decided both ways: neither said same, and they disagree
                opposed=sum(a != b and 'same' not in (a, b) for a, b in zip(u, v)) / n if n else float('nan'),
                tags={t: (len(x), sum(x) / len(x) if x else float('nan')) for t, x in tagged.items()})
