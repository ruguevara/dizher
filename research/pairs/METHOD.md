# Method and tools: pair selection

## Rules agreed with the user

- A colouring is a label map. We judge it only after a render through the full pipeline of its project: Tune,
  Target, Halftoner, Metric, Select pairs, Halftone, Optimise, the whole field as an overpaint
  (`common.Project.render`). Outside its own settings a painting has no meaning. Until step 2 the dots' weights
  (chroma on Halftone, noise on Optimise) were the selection's weights. If a script sets a Metric or Select pairs
  weight and does not set the dots' weights, they move with it (`common.Project.changed`).
- We compare cells by the colours they show (`common.shown`). The ink/paper order and a colour that does not show do
  not count.
- There is no single correct colouring. The data are comparisons. The score of a metric is how frequently it agrees
  with the user's choice.
- The user's viewing conditions: the app at 2x–3x, a 21" 2560×1440 screen, ~65 cm, sometimes with a squint. This is
  ~26 Spectrum pixels per degree (`metrics.PPD`).
- We compare colourings through the eye, not as raw pixels (`FINDINGS.md`, "How to look and measure").
- The held-out pictures are andy, sunset, golden-axe, david, burning-hand. They are not used to tune, only for the
  user's final check.

## The user's round

- Whole pictures, blind: the source on the left, two colourings, random sides, shuffled sheet names. Raw at 3x as in
  the app (`surface.py user`), or exactly the judges' sheets (`dp.py user`). 5 repeats with the sides swapped check
  that the user agrees with the user's own answers.
- The vote: `python research/pairs/dp.py vote ROUND SHEETS` opens a page in the browser. Keys: ← the first, → the
  second, space both fine, x both bad, Backspace back. Each answer goes at once to `ROUND/user-verdicts.json`.
- Keep the key of a round out of git until the user votes. Commit it after.

## The Claude judge protocol

- Each judge is a fresh subagent with ≤10 items. One judge never gets the two orders of one pair (`curve.deal`).
- An item is three separate files (the source, colouring 1, colouring 2) through the user's eye
  (`views.seen2(img, *acuity.EYE)`: x4 with square pixels, then a blur of lightness σ 0.75 and of colour σ 1.0
  Spectrum pixel in linear light). No outlines and no crops.
- By eye only: "do not write or run code, do not make files other than the result". In the pilot the judges wrote
  scripts into one shared folder and overwrote the scripts of other judges.
- A simple prompt (`curve.PROMPT`). The answers are `1`, `2`, `=` (both fine), `x` (both bad) (`dp.PROMPT`). 6 votes
  per pair, 3 per order. A label is the answer of more than half the votes. "x" and splits are not labels.
- Write the prompts to files. The judge gets one line: "read the file and do it". Not more than 20 subagents at one
  time. For more than ~30 judges use a Workflow, if the user permits it ("use a workflow").

## Files

The code is in `research/pairs/` (pytest does not collect it). The commands are in `README.md`, "Run".

| file | what it does |
|---|---|
| `common.py` | the project (and a single picture without a project: the graph of a new project), a render of a label map through the project's pipeline (`Project.render`, `convert`, `energy`, with changes to node parameters), `shown`, segments, windows |
| `build.py` | 229 local pairs from 9 painted projects: A against B; the controls C and N |
| `variants.py`, `flat.py` | pairs against coherence ×3 and 0; the counterexample F: a segment on one pair |
| `metrics.py`, `score.py` | ~45 metrics, the converter's eye `project_eye`, the numeric judges `JUDGE`/`JUDGE_FAST`; a table of the metrics by pair |
| `fit.py`, `eyes.py` | the judge (a logistic regression on differences, bootstrap, anchor); the metrics through different eyes |
| `optimum.py` | the judge's optimum by a search cell by cell (session B) |
| `views.py`, `round.py`, `judge.py` | blind sheets for the user and the judges (old layouts), `seen2` (the judges' eye); the analysis of verdicts |
| `agents.py` | Claude judges on whole pictures (session B): sheets, prompts, `write_round`, `tally` |
| `tune.py` | the converter's knobs: sweep, fine, the sheets of round c1 (session C) |
| `acuity.py` | what a judge sees: sheet layouts, eye grids (session E, step 1) |
| `curve.py` | the threshold curve, the deal `deal`, the prompt `PROMPT`, sheets for the user (step 2) |
| `dp.py` | alternatives where the DP is unsure, waves of judges, the count, sheets for the user, the vote page `vote` (steps 3–4) |
| `smooth.py`, `surface.py` | the selection on a smoothed target; a pair per surface (after E) |
| `rivals.py`, `Izx.java` | the competitor baseline: img2spec, Image to ZX Spec (without its window), ZX-Paintbrush (the user's SCR); the best setting by the judges, sheets for the user, the count (plan step 1) |
| `gallery.py` | a gallery per picture (`defaults`: new defaults per method): a 4×4 grid of renders (`--side N`, the same span; the first round mixes the Metric methods, the pick sets it) on a plane through the best point in 13 knobs (10 without DBS); the user's discards and picks go to `rounds/gallery/NAME.json`; `report`, what they say (plan step 3) |
| `duel.py` | a pairwise search per picture: the best so far against the point of the most expected improvement, by a GP with a probit likelihood on the comparisons, the gallery's too; each answer to `rounds/duel/NAME.json`; `check` against a simulated user (plan step 3b, skipped) |
| `methods.py` | the Metric method, Exact mixture against Halftoned at their presets, without DBS, one blind sheet per training picture; `tally` (before more galleries, plan step 4) |
| `seams.py` | which seams the user sees: 12 renders with DBS (`make`), the page where the user paints the seams (`paint`, to `rounds/seams/marks.json`), the user's own cases against the measures (`check`), the measures' AUC on the painted seams, the repeats, the painted segments (`report`) (plan step 5) |
| `patches.py` | seam visibility on synthetic patches: two pairs on one threshold map, the colour changes at one luminance; the user sets the right side's lightness where the seam is least visible and rates it 0-3 (`make`, `rate` to `rounds/seams/patches.json`, `report`: by group with the lightness set, the repeats, each measure's rank correlation) (plan step 5b) |
| `seamfit.py` | one seam score: the seam measures' weights fitted at once to the painted renders and the patch rounds, each picture or round left out; `ranked`, per global lightness every two pairs ranked by the score, 16 on a sheet; `sort`, the page where the user drags them into the order seen, to `rounds/seams/ranked.json` (plan step 5c) |
| `selection.py` | the seam score in the selection in place of the coherence term: the per-border table from each pair's halftone, the six training pictures rendered both ways (`make`), blind sheets (`user`) (plan step 5d) |
| `zxart.py` | prior pairs from artists: `fetch`, the zxart.ee top 100 standard screens by votes (`rounds/zxart/top100.json`, the screens in `data/zxart/`); `stats`, the pairs, the borders by lightness in the seam measures against a null at the cell's level, the transition prior against V and the seam score; `recover [K]`, each screen through the user's eye (K times its sigmas) as a new project, Select pairs at the defaults without DBS, the converter's cells and borders against the artist's (plan step 6) |
| `vote.html` | the old vote page (an Artifact with a db, round cal1); `dp.py vote` replaces it |

The data in git are in `rounds/`: `pilot1`, `cal1`, `opt1`, `opt3`, `opt4`, `aopt`, `c1`, `acuity`, `curve`, `e1`,
`surface/r1`, `surface/r2`, `rivals`, `gallery`, `duel`, `methods`, `seams` (the keys, the verdicts of the judges and of the user). The renders in
`data/` are not in git; the scripts make them again:

- `build.py` ~15 min on 9 processes;
- `variants.py` ~3 min, `flat.py` ~7 min, `score.py` ~1 min;
- `optimum.py N` ~25 min (the full judge on the GPU ~1.5 h);
- `dp.py make` ~10 min with no other tasks on the machine (~900 renders, ~6 s each, on 12 processes);
- `surface.py make` some minutes;
- `seams.py make` ~30 s (12 renders with DBS, data/seams/);
- `zxart.py fetch` ~1 min (100 screens, 0.5 s between requests), `recover` ~50 s on 8 processes;
- `rivals.py make` ~30 s, after `surface.py make`. The competitors: the build `../img2spec_video/build-macos`; the
  Image to ZX Spec jar and ZX-Paintbrush in its own Wine prefix, both in `data/rivals/bin/`; the user's SCR files in
  `rounds/rivals/paintbrush/`.

Dependencies: `pip install -e . pytest`; for LPIPS/DISTS also `torch torchvision piq`.

## Cleanup

- `tests/images/pairs/*.png` (59 MB) are already in the history of `develop` (#13). It is too late to delete them. If
  they cause a problem, shrink them in a separate PR.
- `data/` is in `.gitignore`; the scripts make it again.
- The Artifact page of round cal1 (https://claude.ai/artifact/LMWE3KnHYA6ZcPGNMbhh9z) is not necessary now. Its votes
  are in `rounds/cal1/votes.json`.
- The user deleted `tests/images/jojo/reference.scr`. `tests/pair_bench.py` skips jojo (`pair_bench.py freeze NAME`
  makes the reference again). The painting is full (768 cells) on andy, autumn, diver-sunset, jojo, rocket-rackoon,
  sunset; partial on RC1 (543), anubis (366), golden-axe (323). burning-hand and david have no painting.
