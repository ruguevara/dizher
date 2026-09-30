"""Score judgements of blind sheets (views.py) against their key.

    python research/pairs/judge.py DIR       DIR holds key.json and results-*.json from the judges

A result is {"file", "verdict": "1" | "2" | "=", "confidence", "problem", "note"}. Per pair, the two sheets (sides
swapped) are one judgement: decisive when both pick the same colouring, a tie when both say "=", split otherwise.
Reported: how often a decisive judgement picks the painting, how many pairs are decisive, per image and per stated
problem; how null pairs (the same colouring) are called; and the side bias, the share of "1" among sheet verdicts."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(folder: Path):
    key = json.loads((folder / 'key.json').read_text())
    results = {}
    for f in sorted(folder.glob('results-*.json')):
        for r in json.loads(f.read_text()):
            results[r['file'].removesuffix('.png')] = r
    missing = sorted(set(key) - set(results))
    by_pair = defaultdict(list)
    for name, k in key.items():
        if name in results:
            by_pair[(k['id'], k['kind'])].append((k, results[name]))
    sheets = [r['verdict'] for r in results.values()]
    print(f'{len(results)} of {len(key)} sheets judged' + (f', missing {missing}' if missing else ''))
    print(f'side bias: "1" {sheets.count("1")}, "2" {sheets.count("2")}, "=" {sheets.count("=")}')

    def picks(k, r):   # 'user', 'other' or 'tie' for one sheet
        return 'tie' if r['verdict'] == '=' else 'user' if int(r['verdict']) == k['user'] else 'other'

    outcome, per_image, per_problem, conf = {}, defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for (pid, kind), sheets_ in by_pair.items():
        if len(sheets_) < 2:
            continue
        p = [picks(k, r) for k, r in sheets_]
        o = p[0] if p[0] == p[1] else 'split'
        outcome[(pid, kind)] = o
        if kind == 'user':
            per_image[pid.split('/')[0]][o] += 1
            for k, r in sheets_:
                per_problem[r.get('problem', '?')][picks(k, r)] += 1
                conf[r.get('confidence', '?')][picks(k, r)] += 1
    for kind in ('user', 'null'):
        c = Counter(o for (pid, k), o in outcome.items() if k == kind)
        n = sum(c.values())
        if not n:
            continue
        decisive = c['user'] + c['other']
        share = c['user'] / decisive if decisive else float('nan')
        print(f'\n{kind} pairs: {n}; painting picked {c["user"]}, the other {c["other"]}, tie {c["tie"]}, '
              f'split {c["split"]}' + (f'; of the decisive, the painting {share:.2f}' if kind == 'user' else ''))
    print('\nper image (painting / other / tie / split):')
    for img, c in sorted(per_image.items()):
        print(f'  {img:15} {c["user"]:3} {c["other"]:3} {c["tie"]:3} {c["split"]:3}')
    print('\nper sheet, by the stated problem of the worse one (painting / other / tie):')
    for prob, c in sorted(per_problem.items(), key=lambda x: -sum(x[1].values())):
        print(f'  {prob:8} {c["user"]:3} {c["other"]:3} {c["tie"]:3}')
    print('\nper sheet, by confidence (painting / other / tie):')
    for cf, c in sorted(conf.items()):
        print(f'  {cf:8} {c["user"]:3} {c["other"]:3} {c["tie"]:3}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
