# Log: pair selection

One entry for each session, the newest at the bottom: the question, the work, the result. The tables and protocols
are in `README.md` (the section is in parentheses). The conclusions are in `FINDINGS.md`. The next steps are in
`PLAN.md`.

## 2026-09-30 — metrics on the user's edits (README 1)

Which render metric selects the user's painting? `build.py` made 229 segments from 9 painted projects. Each compares
the painting (A) with the segment that Select pairs returns (B). The controls are C (the next pair by energy) and N (a
re-dither). `metrics.py` and `score.py` calculate ~45 metrics.
Result: colour fidelity to the source selects the painting in 31–36%. `seam_excess` and `neighbour_excess` select it
in 65% (on all 9 pictures), and in 77–81% against C. The project's energy selects it in 7%. No metric explains one
third of the edits.

## 2026-09-30 — pilot of the Claude judges (README 2; `rounds/pilot1`)

60 pairs and 7 null pairs, in both orders. The judges decided 78%, and 84% of these decisions are for the painting.
High confidence: 26 of 26. Low confidence: a coin toss. Null pairs: 7 of 7 "even". The best metric agrees with the
judges in 72%.

## 2026-09-30 — calibration round 1 (README 3; `rounds/cal1`)

49 sheets. The user voted blind, and 10 judges voted on the same sheets. The user confirmed the painting 18 to 1 (5
"even"), and gave the same answer on 4 of 5 repeats. Cells kept from Select pairs against coherence ×3 / 0: 14 to 1.
Smoothness against accuracy: 10 to 5. On the disputed pairs `seam_excess` agrees 16 of 19, S-CIELAB 9 of 19. On the
hard pairs the judges agree with the user in 72%.

## 2026-09-30 — session A: judge v1/v2 (README 4)

`fit.py`, `eyes.py`. Judge v2 = 0.810 `seam_excess` + 53.59 LPIPS through the eye. The fast judge = 0.973
`seam_excess` + 8.02 MS-SSIM of lightness through the eye (24 ms against 1.6 s). With one picture left out: B 0.64, C
0.77, the user's votes 0.73–0.79. On whole pictures the painting is better than Select pairs on 7 of 9, and a black
screen is last on all 9.
Result: one term holds, the steps at the seams. Raw LPIPS/DISTS are a coin toss; through a blur of 0.75–1.5 px LPIPS
gets 0.66.

## 2026-10-01 — session B: the judge's optimum (README 5–6; `rounds/opt1`, `opt3`, `opt4`, `aopt`)

A loop of 4 rounds: a judge, its optimum (`optimum.py`), the user, counterexamples, a new judge. Any optimisation
fools the fast judge (black and white cells, grey and yellow patches). Full v3: its optimum is better than Select
pairs on 5 of 9 and worse than the painting on 8. v4: its optimum is worse than Select pairs on 8 of 9, but the judge
put it higher on all 9. On the optima the Claude judges agree with the user in 27 of 32 decided pairs.
Result: a direct optimisation of the colouring against a judge does not work.

## 2026-10-01 — session C: the converter's knobs (README 7; `rounds/c1`)

`tune.py`. The search must keep DBS on: the rank correlation of the scores with and without DBS is 0.14–0.94. A sweep
of 8 knobs around the Exact mixture preset: no knob value is better than the default for all three judges on most
pictures. The candidate is an eye chroma blur of 1.1: better on 3 of 6 training pictures and on 3 of 5 held-out
pictures. The user, blind, on the held-out pictures: 3 "even", sunset better, golden-axe worse. The Claude judges
decided no pair.
Result: the preset is at the optimum for the user's eye.

## 2026-10-01 — session E, step 1: what a Claude judge sees (README 8; `rounds/acuity`)

`acuity.py`. On a sheet of three pictures the judge finds an edit of ≥3 cells in 7 of 18. On three files through the
eye it finds 14. Raw at 4x it finds 17, but it calls a re-dither a difference in 3 of 4. The user selected the judges'
eye from grids: lightness σ 0.75, colour σ 1.0 (`views.seen2`). Three files through this eye: 16 of 18, and the null
pairs are "even".
Result: in the earlier rounds the judges looked at a sheet where they did not see most edits.

## 2026-10-01 — session E, step 2: the threshold curve (README 9; `rounds/curve`)

`curve.py`, 6 votes per pair. Partial paintings against Select pairs: 193 votes to 19 (33 of 36 pairs by majority).
The painting against the next pair by energy: 30 to 0. A single segment is decided from ~20 ΔE through the eye. A
paragraph about the user's taste in the prompt makes the judges worse (0.85 against 0.91). The user on 12 pairs: where
the user selected a side, the judges agree 4 of 4. Where the judges were against the painting, the user said "both
fine" or "both bad" (4 of 5 partial paintings).

## 2026-10-01 — session E, step 3: where the DP is unsure (README 10; `rounds/e1`)

`dp.py`. The plan's filter (the lowest third of the total energy margin, 15 ΔE) took single-cell edits. We changed it
to ≥3 cells, the margin per cell, the lowest half, ≥10 ΔE: 158 pairs and 8 null pairs, 100 judges (a pilot of 24, then
a Workflow of 76). Labels: alternative better 34, base better 44, "=" 49, splits 31. Null pairs: 7 of 8 "=". The label
holds in both orders in 72 of 78.
Result: where the DP is unsure, the energy's choice is a coin toss for the judges. The energy margin does not predict
it (p 0.86).

## 2026-10-02 — session E, step 4: the user's check (README 10)

`dp.py user`, `dp.py vote` (a page in the browser). The user decided 4 of 30, all 4 as the judges did. "Both fine" 14,
"both bad" 11 (all on autumn, jojo, vangog). Repeats: 4 of 5 the same.
Result: where the DP is unsure, neighbouring colourings are equal for the user, or both are bad. Session E stopped;
step 5 (a fit of the energy weights) did not start.

## 2026-10-02 — after E: structural changes to the selection (README 11; `rounds/surface/r1`, `r2`)

`smooth.py`: the selection on a smoothed target. The patchwork stays, or the faces become grey. The user did not see
it. `surface.py`: one pair per surface. Hard (32 and 64 surfaces), soft (the cost to leave is a quantile of the gap),
and only the cells within 20 ΔE of the surface colour. The user, blind, two rounds of 14 sheets: jojo and vangog better
in all 4 variants, anubis and rocket-rackoon worse in all 4, diver-sunset and RC1 flip, autumn "both bad".
Result: the app has Select pairs: surface dE (default 0), PR #14 into `develop` (41927ed). The research branch is
rebased on it.

## 2026-10-02 — plan step 1: the competitor baseline (README 12; `rounds/rivals`)

`rivals.py`, `Izx.java`. jojo, rocket-rackoon, autumn, diver-sunset (the user's choice). Each tool gets the project's
tuned target. img2spec (CLI, 3 settings), Image to ZX Spec 2.3.0 (without its window, 3 settings), ZX-Paintbrush 2.6.1
(the user ran it under Wine, at its defaults). Claude judges selected each tool's best setting per picture (24 pairs,
15 judges). The user, blind, 24 sheets and 5 repeats (4 of 5 the same): ours better 15, both bad 9, the competitor
better 0.
Result: we do not lose anywhere. The coach's criterion holds on two pictures of four (jojo with surface dE 20,
rocket-rackoon with Exact mixture). autumn and diver-sunset are "both bad" against the dithering competitors.

## 2026-10-02 — plan step 2: the dithering gets its own weights (`feature/dbs-weights`, merged here)

DBS also read the chroma weight of Metric and the noise weights of Select pairs. Thus a change to the dots also moved
the pairs. Now Optimise has its own chroma, luma noise and chroma noise. The method's preset sets them together with
the selection weights; an older project takes the values of its Metric and Select pairs. The tests: at the preset
values DBS gives the same result as before; other weights move the dots, not the pairs. `common.Project.changed`
moves the Optimise weights together with the selection weights if a script does not set them. The renders before and
after the merge are the same. Result: the gallery (step 3) has its knobs. No time of the user was necessary.
Then chroma moved from Optimise to Halftone. The halftone without DBS (priorities 3–4) still used the Metric weight,
and this weight changes 0.4–4% of its pixels (jojo, rocket-rackoon, autumn, diver-sunset; the most on the last two).
Now one weight controls the dots: the halftone target and the DBS error. The renders again agree with `develop`.
The Projected view now comes from Halftone, with its chroma (before: from Overpaint, with the Metric weight). The
user: less chroma on Halftone sometimes makes the blockiness more even. This is a candidate for the seams
(`IDEAS.md`).

## 2026-10-04 — PLAN step 3: the gallery (README 13; `rounds/gallery`)

`gallery.py`: per picture, rounds of 16 renders on a plane through the current best in the knob space; the user
discards the bad ones and picks the best, or redraws the round (other directions, wider). Slow rounds (1-2 min) were
mostly the machine's load (arc, VS Code search): on a quiet machine a conversion without DBS took 1.0-1.2 s, and
`feature/faster-energy` (PR #15, merged into `develop`) made it 0.4-0.5 s: the selection energy's seam terms by matmul,
renders the same. The branch rebased on `develop`. The user: 5 galleries without DBS, ~30 picks. `gallery.py report`.
Result: chroma noise down on all four pictures that count; luma noise and coherence by the picture; the preferences
differ by picture; no two knobs change the render alike. Thin data. Next (the user): a pairwise search with a Gaussian
process per picture (PLAN step 3b).


## 2026-10-04 — PLAN step 3b: a pairwise search (`duel.py`)

A pairwise mode for one picture's knobs (the user's idea): the user picks the better of two renders; a GP with a probit
likelihood on the comparisons, the gallery's too, proposes the next pair, the best so far against the point of the most
expected improvement around it. One new render a duel, 0.6 s without DBS. A simulated user over 10 knobs, from no data:
60 duels cut the loss at the start from 16.5 to 5.9, level with a noiseless random search of 120 renders; other
acquisitions and hyperparameter grids, no better. The galleries' comparisons fit best with a length scale of 0.03-0.05
on diver-sunset and rocket-rackoon: their discard orders are not smooth in the knobs at the gallery's spacing.
The user's first 12 duels on autumn: the best won all 12 and the evidence did not move. Seeded by the gallery, the
GP's length scale is 0.1, and the challengers landed 0.27-0.40 off, unrelated to anything the GP knew. Now the
challenger is within a step that grows when it wins and shrinks when it loses (the 1/5 rule); a simulated user seeded
like a gallery: the challenger wins 17-24% instead of 2-3%, the result better. Space did not work for the user (not
reproduced): the page's busy check now has its own flag, Space is matched by its code too, and a Can't tell button.
jojo (no gallery): the first version's first challenger jumped 0.95 to surface dE 13 and Halftone chroma 0 and won (the
face not a patchwork); the duel stuck there, overall bad. Challengers along 1-3 knobs: no better on simulated users;
a can't-tell now grows the step. jojo to start again.
Result (the user, after duels on autumn 13, jojo 23 and 44, RC1 12): no use; step 3b skipped. `duel.py` and
`rounds/duel` stay as the record. The user: autumn's gallery best is good enough and counts for the criterion as it
is (the third picture, with jojo and rocket-rackoon). Step 4 on the data so far: 10 -> 6 knobs, no merges; the user:
too thin, more galleries first. Next: galleries without DBS, every training picture to ~10 picks (PLAN step 4).

## 2026-10-05 — the Metric method by the user (`methods.py`; `rounds/methods`)

The gallery's grid any size (`--side N`, the same span); the user's mode 3x3. Should the gallery vary the Metric method
too: no, a switch, and each method sets its own preset, so the same sliders mean other renders. anubis and
rocket-rackoon ran Halftoned, the other four Exact mixture; the two never compared by the user. `methods.py`: both at
their presets, without DBS, one blind sheet per training picture. The user: Halftoned on RC1, anubis, diver-sunset;
Exact mixture on autumn and rocket-rackoon (as in README 12); jojo both bad.
Result: the method is a setting by the picture. The user: the gallery picks it, its first round mixes the methods
(a grid around each one's start), the pick's method the gallery's. RC1, diver-sunset and rocket-rackoon start again
with it; their old galleries kept as NAME-method.json. Next: the galleries (PLAN step 4).

## 2026-10-05 — the galleries and new defaults per method (`gallery.py defaults`)

A gallery per training picture, 11-14 picks, 3x3, without DBS; the first round mixed the methods. The user picked
Exact mixture on RC1 (the sheet vote said Halftoned), autumn, jojo, rocket-rackoon; Halftoned on anubis, diver-sunset.
The user: more or less good. `gallery.py report`: chroma noise down on 4 of 6 and to ~0 on 5; luma noise at 0-0.06
everywhere; Metric chroma (0.2-3.75) and coherence by the picture; no two knobs change the render alike (|cos| <= 0.35).
New defaults per method (the user asked): the median of the last picks where it agrees with the preferred direction.
Result, robust over bootstrap seeds: Exact mixture chroma noise 0, eye luma blur 0.9; Halftoned chroma noise 0, eye
chroma blur 1.6; the rest stays (borderline ones in PLAN step 4).
In the app: PR #16 (the new defaults, a preset sets the Eye model too) and PR #17 (`feature/dbs-weights`, the dots' own
weights, reconciled with #16), both into `develop`. The research branch rebased on it, without those commits: no
changes of its own outside `research/pairs` but the unused `tests/images/pairs` removed, so rebases do not conflict.
