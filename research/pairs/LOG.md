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

## 2026-10-06 — PLAN step 5: which seams the user sees (`seams.py`; `rounds/seams`)

The user: the coherence term prices a seam by the attributes (V of the two cells' pairs); the user sees it by the
pixels (red | magenta dots on black with black along the border: no seam; ~50% noise along it: none). Which measure of
a seam's pixels tells where the user sees one? Candidates per seam: A the attributes; B the pixel steps across the
border; C a line filter along it less the cells' own texture; D the strips on the two sides (raw, through the eye).
`seams.py check`, the user's cases, synthetic: B and C through the eye fail the black border (the blur carries the
dots' colour onto it); raw B, C, D pass. The painted segments (229): C through the eye rates the painting below
Select pairs in 0.73, D 0.72, against `seam_excess` 0.65. The user paints the seams on 12 renders with DBS (each
training picture at its gallery best and at coherence 0) and 2 repeats (`seams.py paint`); the measures are scored by
AUC on the seams where the pair changes (`seams.py report`).
Result (README 14): 1338 of the 1342 marks on pair changes. The raw pixels' colour step across the border, beyond the
source's: AUC 0.79; the attributes 0.62, with the coherence term's edge weight 0.51 (it excuses the outlines'
staircases, which the user marks more). Through the eye, the texture terms, runs, mixes: no better. Top k 0.42 against
the repeats' 0.48-0.75. Next (the user): seam visibility itself on synthetic patches, without picture content.

## 2026-10-06 — PLAN step 5b: seam visibility on synthetic patches (`patches.py`; `rounds/seams/patches-*`)

The user: measure seam visibility itself, without picture content. Patches of two pairs on one threshold map. v1
(colours at one dot level) the user stopped: magenta is 34% brighter than red, the lightness gave the seams away. v2
(the same mean luminance, the user can move the right side's lightness to the least visible seam, Bayer and blue
noise) stopped: a regular texture and a uniform level show any break. v3 (blue noise, a slow variation of the level),
51 patches and 10 repeats (README 15): a hue change shows at the same lightness (~2 of 3), brightness alone faintly
(1), a lightness step up to 0.12 not at all; the user did not move the lightness. E, the pair switch at the border's
pixels (one implementation for the renders and the patches, `seams.switch`; the renders' npz now keep each cell's
pair, renders byte for byte the same), 0.75 on the patches, 0.71 on the renders; B raw 0.14 on the patches, 0.79 on
the renders; B less the cells' own pixel steps 0.50 and 0.73; the cells' mean colours add nothing on the renders. The
dots run on across the patches' border and break on a render's. v4: gradients of random direction and strength (the
user), each change twice, and the dots breaking at the border on some.
v4 (README 15): the dots breaking at the border makes no seam and changes none; the user set the brighter side 1-4
dL* darker than the CIE match; the attributes 0.83, E 0.81, B raw 0.20. The user: the patches had only inks on black,
near hues, no blue. v5: a paper's change under bright white, blue, the same mean colour from other dots (b/y, r/c, m/g
against k/w), each pair placed so that the luminance match never clips.
v5 (README 15): the same mean colour from other dots 1-2, a paper's change under white 1-2, blue on black 2.5-3;
repeats 100%. The pixel-level measures (E, the attributes) fall, the averaged colour ones lead.

## 2026-10-06 — PLAN step 5c: one seam score (`seamfit.py`)

The measures' weights (>= 0) fitted at once to the painted renders and the rounds v3-v5, each picture or round left
out (README 16). Joint, all measures: renders 0.73, rounds 0.82, 0.89, 0.70; two terms, E lightness (the pixels) and
the cells' mean colour: 0.68 and 0.79, 0.88, 0.77, against the coherence term's 0.51. The renders' own fit (0.80) does
not carry over to the patches (0.25-0.55); the renders' extra is partly the busy dots (a texture modifier, +0.02). The
cell mean step and the cells' texture are seams.measures now (one implementation for both).

## 2026-10-06 — the ranked sheets and the user's sorting (`seamfit.py ranked`, `sort`; `rounds/seams/ranked-v1`)

Per lightness L* 15-92 every two pairs that reach it, ranked by the two-term score, 16 on a sheet; the user did not
quite agree and sorted them (README 17). The user's rule: a dim pair near in lightness (yellow/white) reads as solid
and stands out next to any bright pair. A difference of dot contrast (sd) and its ratio: no; a solidity step, exp(-sd
/ 3), weight 0, because in the dark a solid cell next to dots is common and unmarked; the marked share rises with
lightness (1% to 36%). Times the cells' lightness, with the lightness term so too: three terms, held out renders 0.72,
v5 0.83, the sorting 0.85 (two terms 0.68, 0.77, 0.83); per sheet L* 81 0.66 -> 0.90, L* 26 0.87 -> 0.65. The sorting
round kept as v1 before `ranked` can run again.

## 2026-10-06 — PLAN step 5d: the seam score in the selection (`selection.py`; `rounds/seams/selection`)

The three-term score as the DP's pairwise table (each border, each two pairs, from the pairs' halftones in the two
cells), times coherence x 0.0048, the coherence term off; the six training pictures at their gallery best with DBS.
The copy of the selection gives the current labels with the old term. The score at the halftone predicts it after DBS
(Spearman 0.85-0.96). 35-184 cells changed; jojo's faces lose their yellow patchwork, greyer. The user's blind round.
The first vote was void: the page was started without SHEETS, and `dp.py vote` fell back to its default, session E's
eye-blurred judges' sheets, under the same names (the user: all three blurred). The 6 votes moved out of the round
(data/seams/selection/void-verdicts-session-e-sheets.json); `vote` needs both folders now. The sheets are 1x, the page
enlarges them by whole screen pixels with square pixels (before: a fractional enlargement, smoothed), no-store.
The vote again (README 18): the score better on 5 of 6 training pictures, rocket-rackoon both fine, the current never.
The held-out round: andy, sunset, golden-axe, david, burning-hand at their projects' settings with DBS, 5 sheets and 5
repeats with the sides swapped (`selection.py make held`, `user held`).
Held out (README 18): both bad on four (their own settings), the current better on david, every repeat the same; the
score wins none. The user: its contribution is low (the scale matched the old term's total cost). The strength sweep
x1-x16 on the training pictures (`selection.py strength`): the seams by the score fall 30-70% from x1 to x16.
The user's strengths: RC1 x2, anubis x8, autumn x4, diver-sunset x8, jojo the current, rocket-rackoon x1; times each
picture's coherence 16, 26, 23, 14, 0, 8: the coherence was the confound. The weight fixed at 16 (the geometric mean),
the coherence aside; the held-out check again at it (`held-fixed`).
Held out at the fixed weight (README 18): burning-hand the score better, david the current, three both bad; every
repeat the same. In all: training 5-0-1, held out 1-1-3.

## 2026-10-06 — PLAN step 6: prior pairs from artists, the zxart.ee top 100 (`zxart.py`, branch `research/zxart-top-100`)

The question (the user): priors from the artists' screens for a new, simpler pair selection that prevents clash,
measured as the seam research measures (step 5). `fetch`: the 100 best-voted standard screens by the API
(`filter:zxPictureType=standard`, `limit` before `start`, Cyrillic file names quoted; hobeta headers stripped, none
skipped). `stats` (README 19): the pair changes on half the borders, 87% with a shared colour (all in the dark), the
border's pixels in the shared colour 5x more than the cell's level gives (E = 0 on 25% against 5%). The transition
prior (count over the expected at random, per picture): what artists avoid, the seam score tells better than V (rho
-0.41 against -0.22); hue ramps (B/M | M/Y) are made 7x though the score charges them. `recover`: the screen through
the user's eye as a new project, Select pairs at the defaults without DBS, 0.5 s a screen; `common.project_graph`
takes an absolute path for a loose picture. At the eye's blur 82% of cells recovered, the borders like the artists';
at 2x and 3x the blur 74% and 63%, and the converter's changes without a shared colour grow to 10% of borders against
6.4%, unseen by the seam score (its changes score lower than the artists'). The branch is rebased on `develop` f4d422e
(#18-#21).
Step 6.1 (README 19): a shared-colour term on the current selection, W x coherence on a flat seam whose pairs share
no colour; at W 0.3 the changes without a shared colour fall to the artists' share (10.1 -> 5.5% at 3x), recovery
+1.1 at 3x, +0.3 at 2x, the hard constraint overshoots and loses 2.3 at 2x. Step 6.2: the artists' transition table
as V (in-sample): recovery +1.6 / +0.9, but the change rate 0.55 and bright/dim switches 0.17, twice the artists'; it
frees cheap changes rather than removing bad ones. Next: the shared-colour term at 0.3 (and 1) by the user, blind,
on the training pictures.
Step 6.3 (the user: the method is agreed before anything is built; the visibility measure from their votes, not an
attribute binary; README 19): the step 5d seam-score selection in the recovery test at weights 1, 4, 16, with and
without the coherence term, and the control, coherence 0. Every weight recovers fewer cells, monotonically; coherence
0 recovers the most (0.849 against 0.824 at x1). The count of recovered cells judges the unary fit on the artist's
own render; it cannot judge a seam cost. The borders' statistics can: the current is nearest the artists' on all,
coherence 0 doubles the bright/dim switches, the score at 16 makes the borders smoother than the artists' (its
changes 3.4-3.6 against 3.96), at 1-4 it stays in range. `extra`: the converter's own changes are the less visible
ones by the score (2.27 against the artist's own 1.81 at x1); at 3x the artist's own changes are the blur's lost
edges. The shared-colour term (6.1) is dropped: an attribute binary, its weight a sweep. The valid test of a seam
cost stays the user's blind vote on photographs.
Step 6.4 (the user: fit the score's weights on the artists with the pixels; `zxfit.py`, README 19): the energy's
weights by inverse optimisation, the artist's pair against the 16 colour-nearest alternatives at the converter's own
tables, a bounded ranking loss (the logistic one let the artist's content choices drag coherence to 54). The artists
imply coherence ~4-5 (the presets 2 and 6), eye seams 2-4 (chroma more), and a seam-score weight near 2 against the
user's 16, all on the solidity step; E and M nothing. The current energy at coherence 2 is at the artists' optimum
(x1 0.986 against the fit's 0.985); the artists' score terms on the user's data nearly match SEAM (renders 0.70
against 0.72). Both sources put the solidity step first. The user's part 2, a simple model of the artists' seams, next.
Step 6.6 (the user's part 2, a simple model; `zxart.py model`, README 19): P(change | the cells' colour step, Y),
0.08 under 5 dE to 0.89 over 40: artists change the pair where the picture changes colour; the masking as one logit
shift, +0.69 (odds x2; a colour +0.87, black +0.53), both border columns closed together (correlation +0.40, 25% both
wholly the shared colour). A prior for the dots at a pair change, and the range for a selection's changes.
