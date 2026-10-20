"""Rounds for the voting page (vote/index.html): blind files, the db manifest, the key.

A round is a list of items; a pick item is an original and 2-4 options, each a 256x192 PNG. make_pick_round copies
them under random names (r/<round>/<token>.png), shuffles the options per item and repeats a share of the items with
the options in another order, then writes:

    data/vote/files/r/<round>/*.png   the files to publish beside the page (Artifact `files`)
    data/vote/<round>.manifest.json   the db document rounds/<round> (ArtifactData set)
    data/vote/<round>.key.json        which option is which, per item; out of git until the user has voted

    python research/metrics/rounds.py tally ROUND VOTES.json   the votes (ArtifactData list of `votes`) by the key
"""
import json
import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / 'data' / 'vote'


def _copy(src: Path, round_id: str) -> str:
    rel = f'r/{round_id}/{secrets.token_hex(6)}.png'
    dst = OUT / 'files' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return rel


def make_pick_round(round_id: str, title: str, items, zoom=None, repeats=0.1, seed=0, pilot=True):
    """items: [(item_id, original_path, {label: option_path, ...}), ...]. Returns the manifest."""
    rng = np.random.default_rng(seed)
    order = [(iid, orig, opts, False) for iid, orig, opts in items]
    for k in rng.choice(len(items), size=int(round(repeats * len(items))), replace=False):
        iid, orig, opts = items[k]
        order.append((f'{iid}-rep', orig, opts, True))
    first, rest = order[:2], order[2:]     # no repeat among the first two
    rng.shuffle(rest)
    order = first + rest
    manifest_items, key = [], {}
    shown = {}
    for iid, orig, opts, rep in order:
        labels = list(opts)
        rng.shuffle(labels)
        base = iid[:-4] if rep else iid
        if rep and labels == shown.get(base):          # a repeat shows the options in another order
            labels = labels[1:] + labels[:1]
        shown[base] = labels
        manifest_items.append({'id': iid, 'original': _copy(Path(orig), round_id),
                               'options': [_copy(Path(opts[l]), round_id) for l in labels]})
        key[iid] = {'options': labels, 'repeat_of': iid[:-4] if rep else None}
    manifest = {'title': title, 'task': 'pick', 'zoom': zoom or {'grid': 2, 'flip': 3}, 'w': 256, 'h': 192,
                'pilot': pilot, 'status': 'open', 'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'items': manifest_items}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{round_id}.manifest.json').write_text(json.dumps(manifest, indent=1))
    (OUT / f'{round_id}.key.json').write_text(json.dumps(key, indent=1))
    return manifest


def tally(round_id: str, votes_path: str):
    """The round's votes by the key: per item the label picked (or None for 0), its time; repeats paired."""
    key = json.loads((OUT / f'{round_id}.key.json').read_text())
    votes = json.loads(Path(votes_path).read_text())
    votes = {v['item']: v for v in (votes if isinstance(votes, list) else votes.values()) if v.get('round') == round_id}
    rows = {}
    for iid, k in key.items():
        v = votes.get(iid)
        rows[iid] = None if v is None else {'label': k['options'][v['pick'] - 1] if v['pick'] else None,
                                            'ms': v['ms'], 'flips': v.get('flips', 0), 'repeat_of': k['repeat_of']}
    return rows


if __name__ == '__main__':
    if sys.argv[1:2] == ['tally']:
        print(json.dumps(tally(sys.argv[2], sys.argv[3]), indent=1))
