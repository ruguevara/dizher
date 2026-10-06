# Pair selection: a judge from the hand-painted projects

The aim is a measure of a colouring that agrees with the user's eye, found before any new selection method: the
last attempts tuned the energy on proxy counts (painted cells matched, magenta cells) and the user rejected results
that improved them. Plan and reasoning: the session's plan (methodology for finding the pair-selection metric).

This README is the detailed record: methods, tables, numbers, by section. The research subproject's other documents:

| document | what |
|---|---|
| `LOG.md` | the log: each session, its question, what was done, the result, the section here |
| `FINDINGS.md` | what is established, by theme, and the dead ends not to repeat |
| `IDEAS.md` | ideas: next, open, parked, closed |
| `PLAN.md` | the aim, the state, the next steps with their gates |
| `METHOD.md` | rules agreed with the user, the user's rounds, the Claude judges' protocol, files, data, cleanup |

## Ground rules

- A colouring is a label map, one (paper, ink) pair per cell. It is judged only as the project renders it: through
  the project's own pipeline, with its Tune, Target, Halftoner, Metric, Select pairs, Halftone and Optimise, painted
  as a whole field (`common.Project.render`). A painting means something only under the settings it was made with.
  The dots' weights (Halftone's chroma, Optimise's noise) were the selection's until PLAN step 2; a Metric or Select
  pairs weight changed in a script moves them too, unless they are given (`common.Project.changed`), so every round
  before renders as it did.
- Cells are compared by the colours they show in that render (`common.shown`): paper/ink order and a colour a cell
  does not show do not count.
- There is no single right colouring. The data are comparisons, and a metric is judged by how often it prefers what
  the user prefers.
- Colourings are compared through a blur of about a pixel (the converter's default eye, `metrics.project_eye`), not
  as raw pixels. At the user's viewing a Spectrum pixel is ~2.3' and the dots are visible, but the dither's grain is
  not what the user judges: raw, LPIPS and DISTS measure the grain, and a re-dither of the same colouring moves them
  about as much as a painted change. S-CIELAB at the user's viewing keeps the dots too (a re-dither 5.8 dE on the
  changed cells, against 2.0 through the eye, where a painted change is ~25). Metrics that blur or average on their
  own (the seams at 2 px, cell means, the coarse ones) need no eye in front: one costs them a little (section 4).

## Pairs

`build.py` renders each painted project twice: the painting (A) and Select pairs alone (Hide). Cells whose shown
colours differ are grouped by the picture's segments (k-means of cell colour and position, `common.segments`), one
pair per segment:

- **B**: A with that segment's changed cells given back to Select pairs. The user preferred A there.
- **C**: A with those cells on their next best pair by the project's energy, neither the painted nor the selected
  one. Assumed worse than A; a metric that prefers anything but Select pairs' choice fails here.
- **N**, a few per project: A rendered from another halftone origin, the same colouring. A metric should call it even.

DBS reaches past the changed cells (in autumn 281 cells change pixels, 71 change colours), so each pair is scored
over its cells grown by two cells (`common.window`).

## Run

    python research/pairs/build.py          # ~15 min on 9 processes: data/NAME.npz, data/NAME.json (not in git)
    python research/pairs/score.py          # data/metrics.csv and the table below
    python research/pairs/variants.py       # ~3 min: kept cells against other coherences, data/variants/
    python research/pairs/flat.py           # ~7 min: flat counterexamples, data/flat/
    python research/pairs/fit.py            # the judge: selection, anchor, whole pictures (section 4)
    python research/pairs/eyes.py           # ~10 min: every metric through 4 eye models; --deep: LPIPS/DISTS grid
    python research/pairs/optimum.py N [--full --mps]   # round N: a judge's optimum (~25 min fast, ~1.5 h full)
    python research/pairs/optimum.py N --pairs          # ~7 min: round N's optima as counterexamples for fit.py
    python research/pairs/views.py OUT [N]  # blind images of N pairs for judging by eye, with key.json
    python research/pairs/tune.py sweep|report|fine ...      # session C: the converter's settings (section 7)
    python research/pairs/acuity.py make|prompts L|score     # session E step 1: what a Claude judge sees (section 8)
    python research/pairs/curve.py make|prompts V|score|user # step 2: the threshold curve (section 9)
    python research/pairs/dp.py make|wave N|score|user       # steps 3-4: where Select pairs is unsure (section 10)
    python research/pairs/dp.py vote ROUND SHEETS            # the user's blind vote in the browser (both required)
    python research/pairs/surface.py make|user R V...        # pairs per surface (section 11)
    python research/pairs/rivals.py make|judges|best|user|tally   # the competitor baseline (section 12)
    python research/pairs/gallery.py NAME [--no-dbs] [--side N] | best NAME | report   # PLAN step 3: the gallery (section 13)
    python research/pairs/duel.py NAME [--no-dbs] | best NAME | check      # PLAN step 3b: a pairwise search, a GP
    python research/pairs/methods.py user | tally      # the Metric method by the user, before more galleries
    python research/pairs/seams.py make|paint|check|report     # PLAN step 5: which seams the user sees
    python research/pairs/patches.py make|rate|report [vN]     # step 5b: seam visibility on synthetic patches
    python research/pairs/seamfit.py                           # step 5c: one seam score from all the seam data
    python research/pairs/zxart.py fetch|stats|recover [K]     # step 6: priors from the zxart.ee top 100 (section 19)

## Results

### 1. Metrics on the painted segments (2026-09-30)

229 segments on 9 projects (RC1, andy, anubis, autumn, diver-sunset, golden-axe, jojo, rocket-rackoon, sunset), 36
null pairs. Share of segments where the metric rates the painting better, 95% bootstrap over images:

| metric | painting > Select pairs | painting > next best | null / real |
|---|---|---|---|
| scielab_dE | 0.36 (0.26-0.48) | 0.50 | 0.13 |
| scielab_dE_p95 | 0.46 | 0.51 | 0.33 |
| scielab_dL | 0.48 | 0.55 | 0.10 |
| scielab_dH | 0.31 (0.21-0.41) | 0.43 | 0.17 |
| chroma_excess | 0.44 | 0.41 | 0.13 |
| chroma_excess_neutral | 0.39 | 0.43 | 0.07 |
| chroma_deficit | 0.44 | 0.52 | 0.11 |
| hue_angle | 0.31 (0.22-0.40) | 0.41 | 0.08 |
| fine_chroma_excess | 0.33 | 0.43 | 0.10 |
| **seam_excess** | **0.65 (0.61-0.70)** | **0.77** | 0.15 |
| **neighbour_excess** | **0.65 (0.60-0.70)** | **0.81** | 0.12 |
| ms_dssim_L | 0.53 | 0.63 | 0.30 |
| ms_dssim_a | 0.46 | 0.52 | 0.07 |
| ms_dssim_b | 0.40 | 0.48 | 0.08 |
| blur_rmse | 0.34 (0.21-0.48) | 0.49 | 0.06 |
| energy (the project's) | 0.07 | 0.30 | 0 |

- Every fidelity measure against the tuned picture, the judge's S-CIELAB included, prefers Select pairs' cells in
  about two segments of three. The painting gives up colour accuracy.
- What it gains is structure: fewer false steps across cell seams and less cell-to-cell change where the picture is
  smooth. Both terms prefer the painting on every image (0.50-0.78), and over the next best pair four times in five.
- scielab_dE + lambda * (seam + neighbour), each standardised: 0.36 at lambda 0, 0.57 at 1, 0.64-0.65 from 2 on;
  the fidelity term adds nothing on these segments. A third of the painted segments stay unexplained.
- Caveat: these are the places the user overrode a colorimetric optimum, so fidelity measures lose here by
  construction. Where the user kept Select pairs' choice, fidelity may be what mattered. Pairs the user judges
  without having painted them (the calibration round) are needed to weigh the two.
- The project's own energy prefers the next best pair to the painting in 70% of segments.

### 2. Pilot: Claude as a judge (2026-09-30)

60 painted segments sampled evenly over the 9 images and 7 null pairs, each as two blind sheets with the sides
swapped (`views.py`), judged by 14 independent subagents, 10 sheets each, the two sheets of a pair never with one
judge. The prompt named the user's three faults (a colour that does not read as the original's, blocky seams where
the picture is smooth, noisy cell-to-cell change) and asked for 1, 2 or =, a confidence and the worse one's problem.
Key and verdicts: `rounds/pilot1/`; `python research/pairs/judge.py research/pairs/rounds/pilot1`.

- Both sheets of a pair agreed on 47 of 60 segments (78%): the painting 36, Select pairs 7, even 4; 13 split.
  Of the decisive, the painting in 0.84 (Wilson 95% about 0.70-0.92).
- By confidence, per sheet: high 26 painting, 0 other; medium 40 to 8 (0.83); low 18 to 17, a coin toss.
- Null pairs: 7 of 7 called even in both orders. Dither noise is not mistaken for a change.
- Side bias small: "1" 60 times, "2" 49.
- Against the judge's decisive verdicts: seam_excess agrees 0.72, neighbour_excess 0.65, scielab_dE 0.63.
- Where the judge went against the painting it mostly held to the picture's colour: diver-sunset/10, red dithered
  with white read as the sky's pink, where the user painted flat red; RC1's brown hair and fur, dark yellow over grey
  and white over magenta specks. These are the user's taste against fidelity, to put to the user.
- Protocol fault for the next round: some judges wrote helper scripts (zooms, crops) into one shared scratch folder
  and overwrote each other's. Next time each judge gets its own folder, or looks only.

Gate (plan: order agreement >= 85%, null pairs even >= 80%, decisive agreement >= 80% with the lower bound >= 70%):
nulls pass, decisive agreement passes at the edge, order agreement 78% falls short. Low-confidence verdicts carry no
information; counting medium and high only, 66 to 8 (0.89). The segments are easy ones, the user's own overrides, so
this is necessary, not sufficient: the calibration round with the user decides.

### 3. Calibration round 1 (2026-09-30)

49 blind sheets (`round.py`, key in `rounds/cal1/`): 24 pilot segments the judges disputed, 20 kept-cell pairs where
seam/neighbour and S-CIELAB disagree (half each way), 5 repeats. The user voted on the page (`vote.html`); 10 judges
saw the same sheets both ways, by eye only.

- The user confirmed the painting on the pilot's disputed segments 18 times, against 1, even 5. Repeats: 4 of 5 the
  same.
- Kept cells against coherence x3 / 0: kept 14, the variant 1, even 5; where structure and S-CIELAB disagreed, the
  user sided with structure 10 times, with S-CIELAB 5.
- Metrics against the user's decisive votes on the pilot segments: seam_excess 16/19, neighbour_excess 13/19,
  scielab_dE 9/19, hue_angle 8/19, ms_dssim_L 7/19.
- Judges on these hard pairs: split in 34% of pairs; where both they and the user decided, they agreed 13 of 18.

### 4. Judge v1 (2026-09-30)

The rebuild reproduces the 229 segments and 36 null pairs and section 1. New candidates in `metrics.py`: the seam
step split into lightness and colour (a, b) at blur 1, 2 and 4 px (`seam_L_*`, `seam_ab_*`); purity:
`two_colour_share` (cells showing two colours), `dot_contrast` (the spread of a cell's pixels in CIELAB),
`texture_excess` (that spread after the eye beyond the picture's); `hue_family_miss` (a cell's mean colour nearest
another Spectrum hue, or grey, than the picture's); coarse fidelity `blur_dE_4`, `blur_dE_8`; `lpips` and `dists`
(piq, VGG) over the window's bounding box at 2x. `flat.py` adds a counterexample per segment, **F**: the whole segment
on its most common painted pair, smooth by construction.

Single metrics, share where the metric prefers the painting (a tie counts half; the purity and hue family terms tie in
about two pairs of three):

| metric | B | C | cal | F |
|---|---|---|---|---|
| seam_excess | 0.66 | 0.78 | 0.76 | 0.41 |
| seam_L_1 / seam_ab_1 | 0.68 / 0.67 | 0.70 / 0.77 | 0.70 / 0.64 | 0.56 / 0.47 |
| scielab_dE | 0.37 | 0.50 | 0.42 | 0.74 |
| ms_dssim_L | 0.54 | 0.64 | 0.48 | 0.68 |
| lpips | 0.51 | 0.65 | 0.45 | 0.63 |
| dists | 0.50 | 0.50 | 0.52 | 0.61 |
| two_colour_share | 0.48 | 0.44 | 0.48 | 0.39 |
| hue_family_miss | 0.46 | 0.50 | 0.39 | 0.58 |
| blur_dE_8 | 0.22 | 0.34 | 0.19 | 0.65 |

- LPIPS and DISTS are no better than a coin on the painted segments and the user's votes, and dither noise moves them
  0.44 and 0.66 as much as the change does (`seam_excess` 0.15). Like MS-SSIM, they side with fidelity.
- Purity and hue family explain nothing: at chance on B, C and the votes. The unexplained third stays unexplained.

`fit.py` fits a logistic regression on the metrics' differences between the two sides, no intercept, L2, three
sources weighted alike: B (229), C (229) and the user's decisive votes of round 1 (cal, 33: the painting or kept cells
32, the other 1). F is only reported.

- These labels are one-sided: the painting always leaves the colorimetric optimum. Unconstrained, the fit's first
  term was coarse fidelity with a negative weight, "further from the picture is better" (B 0.77, F 0.35). Every error
  metric's weight is now >= 0; only the purity terms are free.
- Forward selection is noise after the first term: each step gains about one vote, and the path changed completely
  when one vote and two candidates were added. On 30 resamples of the images a seam term is picked first 27 times
  (`seam_excess` 17); every later term turns up in a minority. The judge keeps `seam_excess` only.
- Nothing in the labels speaks against a colouring that is smooth but wrong, so a fidelity anchor is fixed at the
  heaviest weight that costs at most 0.02 of the mean accuracy: S-CIELAB at 0.2 (standardised). The coarse anchors and
  `hue_family_miss` cost more at every weight.

**Judge v1** = 1.035 `seam_excess` + 0.054 `scielab_dE`, leaving one image out: B 0.62, C 0.74, cal 0.79, F 0.55. On
whole pictures it ranks a black screen last on all 9 images; the painting over Select pairs on 4, even on 2, under on 3.

#### Through the eye (`eyes.py`)

The metrics above compare the raw render, dither and all, except where they blur on their own (S-CIELAB, the seams,
the coarse ones). `eyes.py` shows both sides and the picture first through an eye model and recomputes every
candidate. Mean agreement over B, C and cal; F; noise (the null pairs' change over the real one):

| metric | none | project eye, Gaussian 1 px | random-portrait exp(-r^0.95) | S-CIELAB 26 ppd | S-CIELAB 13 ppd |
|---|---|---|---|---|---|
| seam_excess | 0.73 / 0.41 / 0.15 | 0.71 / 0.39 / 0.12 | 0.67 / 0.39 / 0.10 | 0.69 / 0.47 / 0.23 | 0.69 / 0.50 / 0.14 |
| scielab_dE | 0.43 / 0.74 / 0.12 | 0.41 / 0.72 / 0.13 | 0.40 / 0.70 / 0.10 | 0.39 / 0.73 / 0.12 | 0.39 / 0.69 / 0.12 |
| lpips | 0.54 / 0.63 / 0.44 | **0.66 / 0.72 / 0.31** | 0.66 / 0.70 / 0.34 | 0.55 / 0.70 / 0.44 | 0.56 / 0.67 / 0.52 |
| dists | 0.51 / 0.61 / 0.66 | 0.62 / 0.73 / 0.46 | 0.62 / 0.67 / 0.36 | 0.59 / 0.63 / 0.67 | 0.49 / 0.63 / 0.54 |

- Only the network metrics change: they see dither texture as distortion. Through a blur of 0.75-1.5 px LPIPS agrees
  0.65-0.66 (a grid of Gaussians 0.5-3 px and exp(-r^0.95) at scale 0.5-2; best 0.68, within a vote or two), and
  more blur loses again. S-CIELAB at the user's viewing leaves the dither nearly whole (a narrow luma filter with a
  sharpening lobe) and does not help. The rest, structure and fidelity alike, gain nothing from an eye in front.
- LPIPS through the project's eye (Gaussian 1 px, fixed before the grid) is the first metric that agrees with the
  labels and prefers the painting to the flat segment: B 0.57, C 0.71, cal 0.70, F 0.71.

**Judge v2** (`metrics.judge_score`) = 0.810 `seam_excess` + 53.59 `lpips_eye`, LPIPS at the heaviest anchor weight
within 0.02 of the best mean, B and C pairs within twice the dither noise left out (24 of 458, below): leaving one image
out B 0.64, C 0.77, cal 0.73, **F 0.68**. On whole pictures: a black screen last on all 9, the painting over Select
pairs on **7** (under on autumn and golden-axe). The seam term is still the only one picked first on resamples of the
images; LPIPS earns its place as the counterweight to smoothness, which the one-sided labels cannot select. The anchor
at half that weight scores the same within two votes (cal 0.79, F 0.68).

**Fast judge** (`metrics.judge_fast_score`, `fit.py --fast`; numpy and OpenCV only, for the app, which has no torch)
= 0.973 `seam_excess` + 8.02 `ms_dssim_eye` (multiscale SSIM of lightness through the converter's eye, 1-4 px):
leaving one image out B 0.64, C 0.74, cal 0.79, F 0.55; on whole pictures a black screen last on all 9, the painting
over Select pairs on 7 (under on RC1 and golden-axe). It decides as the full judge does on 0.78 of B, 0.80 of C, 0.70
of the votes and 0.66 of F. 24 ms a whole picture against 1.6 s. The full judge stays for tuning the converter and its
energy offline.

- Torch-free anchors tried: what the picture has that the render lacks (`detail_deficit_*`: F 0.73-0.74 alone, but
  the painting has less fine detail than Select pairs, so it costs B and C), GMSD and its mean GMSM through the eye,
  SSIM single and multiscale, S-CIELAB. Every anchor weight must also rank a black screen last on all 9 whole
  pictures: GMSD and GMSM fail that at most weights (GMSD is a spread, and a uniformly wrong screen is even).
- For the full judge DISTS through the eye at 0.2-0.3 does as well as LPIPS (B 0.68, C 0.77, cal 0.76-0.79, F
  0.60-0.65) and puts the painting over Select pairs on all 9 whole pictures; the rule kept LPIPS, and the difference
  is a few votes.

- The painted changes are plainly visible through the eye: on the changed cells a median 25 dE (the project's eye,
  1 px), where a re-dither of the same colouring moves 2.0. 15 of the 229 are within twice that; the 4 under it no
  metric gets right. A B or C pair within twice the dither's change is no label and `fit.py` leaves it out (15 B,
  9 C); the user's votes all count, a decisive vote being itself a difference seen.
- The judges' sheets are whole pictures through the eye (`views.judge_sheet`): the tuned picture and the two sides
  through the eye, then 4x, no crops, no outlined cells. The user's sheets keep the crops.
- F is not a clean label. By eye (RC1/2, andy/25): the flat segment is clearly worse where it turns the yellow fur
  grey, which S-CIELAB calls closer, and about as good where it makes andy's hair plain black and cyan. F near 0.5
  is no verdict; the judge's optimum (session B) and the user's votes on F pairs decide.
- Seven of round 1's kept-cell pairs no longer rebuilt as voted in the first rebuild, six in the next ones (other cell
  counts; one run of `variants.py` in three differed on diver-sunset and could not be repeated, pooled or not). Those
  are left out. `round.py` now writes each sheet's cells into the key.

### 5. Session B: the judges' optima (2026-10-01)

`optimum.py` looks for the colouring a judge likes best: from Select pairs, each cell in turn takes, of every distinct
colour pair (71), the one the judge rates best on the cells around it, for up to 4 passes, on renders without DBS (the
judge decides the painted segments the same way on them in 0.90-0.94, rank correlation 0.96); the result is rendered
through the project's pipeline. `--pairs` turns a round's optima into local labels (O): per picture segment where the
optimum shows other colours, the painting against itself with that segment from the optimum, the painting better
(assumed; visible changes only). The full judge's search runs LPIPS batched on the GPU (`--mps`, ~0.3 s a cell).

- Round 1, fast judge v2 (seams + SSIM of lightness): the optimum drains the colour, black and white cells and outlines
  where the picture is chromatic (andy's face, golden-axe's green and blue). The user: worse than the painting in 8 of
  9 pictures, slightly better in one. SSIM of lightness is colour-blind and the seam term rewards one hue.
- With round 1's counterexamples the full judge's data pick LPIPS through the eye by themselves: **full judge v3** =
  16.4 `lpips_eye` + 1.46 `seam_L_1`, leaving one image out B 0.68, C 0.77, cal 0.85, O 0.76, F 0.71; on whole
  pictures black last on 9, the painting over Select pairs on 7 and over the round 1 optimum on 8. New torch-free
  terms for a drained colouring: `grey_share` (cells shown near grey where the picture is chromatic),
  `chroma_deficit_eye` (the first colour term that agrees with the user and catches the optimum alone: B 0.58, C 0.63,
  cal 0.61, O 0.77), `colourfulness_deficit` (no use: a Spectrum render is always more colourful).
- **Fast judge v3** = 0.41 `seam_ab_1` + 1.23 `seam_L_1` + 13.95 `grey_share`: B 0.75, C 0.77, cal 0.73, O 0.65.
  Round 2, its optimum: flat light grey and yellow cells instead of grey ones, detail lost; the full judge rates it
  below the painting on 8 of 9. With both rounds' counterexamples no torch-free combination holds the user's labels
  and the counterexamples together: whatever resists the optima (detail lost, chroma lost) is what the paintings also
  give up against Select pairs. The fast judge stays v3, **for ranking finished conversions only, never as a target**
  (the user's decision). The full judge is the target for tuning the converter offline.
- Round 3, the full judge v3's optimum: colourful and detailed, closer to the painting (25% of painted cells get the
  painted colours, 10-15% for the fast judge's), but to Claude's eye with colour noise: saturated blocks and specks
  from cell to cell. Its seam term sees lightness only. The user: better than Select pairs in 5 of 9 pictures, even in
  3, worse in 1 (sunset); worse than the painting in 8, even in 1 (`rounds/opt3/verdicts.json`).
- **Full judge v4** = 18.06 `lpips_eye` + 0.62 `seam_ab_1` (B 0.71, C 0.80, cal 0.79, O 0.63), the colour seam's weight
  chosen to share most of the user's verdicts on round 3 (11 of 14). Round 4, its optimum: the user found it worse
  than Select pairs in 8 of 9 pictures (RC1 even) and worse than the painting in all 9 (`rounds/opt4/verdicts.json`);
  the judge had rated it above both on all 9.

**Conclusion: optimising the colouring directly against a judge does not work.** A free search over every cell and
every pair (30-65% of the cells moved) leaves the colourings the judge was fitted on and finds where it is blind, in
another place each round: black and white cells, flat grey and yellow, colour noise, a patchwork of cells. A judge that
agrees with the user on about three ordinary pairs in four is no target in a space that large: on its own optimum the
last one was wrong on all 9 pictures. Adding each round's optima as counterexamples does not converge: round 4, its
judge fitted to the user's round 3 verdicts, came out worse than round 3. The judges stay useful for comparing
colourings the converter itself makes; any optimisation against them has to stay in a small space (the converter's
few settings, or a few cells near the energy's choice) and be checked by the user on pictures it was not tuned on.

### 6. Claude judges on the optima (2026-10-01)

Could Claude judges, looking at whole pictures through the eye, have caught what the full judges missed? `agents.py`
put the full judges' round 3 and round 4 optima against Select pairs and against the painting on all 9 pictures, 36
pairs with the user's verdicts (`rounds/opt3`, `opt4`), and 4 null pairs, each pair as two sheets with the sides
swapped (`views.judge_sheet`), to 8 fresh subagents of 10 sheets, by eye only, the prompt in `agents.PROMPT`. Key and
verdicts: `rounds/aopt/`; `python research/pairs/agents.py score`.

| pairs | the user | agents agree | other | split |
|---|---|---|---|---|
| optimum against the painting, r3 and r4 | worse 17, even 1 | 17 | 1 | 0 |
| r4 optimum against Select pairs | worse 8, even 1 | 6 | 1 (rocket-rackoon) | 2 (andy, RC1) |
| r3 optimum against Select pairs | better 5, even 3, worse 1 | 4 | 3 | 2 |
| null pairs | | 4 of 4 even | | |

- Where judge v4 rated its optimum above Select pairs on all 9 pictures and the user found it worse on 8, the agents
  found it worse on 6, split on 2, and called it better once.
- Decisive verdicts (both orders agree) match the user in 27 of 32 (0.84, as in the pilot). Four of the five misses
  are harsher on the optimum than the user (even or better called worse); one approves what the user rejected.
  High-confidence sheets were wrong once.
- Close pictures stay hard: on r3 against Select pairs, 4 of 9.
- Caveat: the prompt was written after session B, and its faults (drained to grey, specks) partly name that session's
  failures.

So the agents work as a veto beside the numeric judges: a change counts as a candidate when the three judges agree it
is better and the agents, both orders, do not call it worse; the user's blind verdict on held-out pictures still
decides. They cost ~40 s per 10 sheets, enough for finalists, not for a sweep.

### 7. Session C: the converter's settings (2026-10-01)

`tune.py`. The base is the Exact mixture preset over each project's own Tune, Target, Halftoner, Eye and Optimise,
nothing painted; tuning on RC1, anubis, autumn, diver-sunset, jojo, rocket-rackoon; held out andy, sunset, golden-axe
(painted) and david, burning-hand (not).

- DBS stays in the search: the judges' ranks of 20 one-setting changes with DBS against without correlate 0.14-0.94;
  DBS moves a score about as much as the settings differ. 8.6 s a conversion.
- One setting at a time, 6 values over each slider, noise from the base at 4 halftone origins: no value is better by
  all three judges on most pictures. All three agree with the preset about what is worse (low chroma weight, flare 0,
  coherence 0, chroma_noise above 0.02, extreme eye blurs). `edge` 0.32 is better on RC1, jojo, autumn and worse on
  rocket-rackoon, anubis; `luma_noise` splits the judges (fast for, v4 against).
- Eye chroma blur, finely (0.7-1.2, every value at 4 origins): 1.1 better by all three judges beyond two standard
  errors on 3 of 6 training pictures, worse on none; held out, better on 3 of 5 (andy, golden-axe, burning-hand),
  david unchanged (almost no colour).
- Round c1 (`rounds/c1/`), held out, base against 1.1: the user, blind, even on andy, burning-hand, david, 1.1 better
  on sunset, worse on golden-axe, where all three judges had put it clearly better. The agents (20 pairs, both
  orders) decided no pair: ties and splits, nearly all low confidence.

**Conclusion: the settings are at their best for what the user sees.** What the judges still find around the preset is
below what the user or the agents can see, and on golden-axe against the user. By the plan, new energy terms (step 4)
were to follow only a gain here; there is none.

### 8. Session E step 1: what a Claude judge sees (2026-10-01)

`acuity.py`. A detection test, no preference: the painting against itself with one painted segment given back to
Select pairs (B), 6 segments in each size bin, and 4 null pairs (another halftone origin); the judge names the square
of a 4x3 grid where the two differ, or "same". The same 28 pairs in three layouts, each to 3 fresh judges.

First, does a sheet shrink on the way in? Rows of 6-digit codes at 4-30 px on white images 1000-3100 px wide, one
fresh agent each: 8 px reads whole at every width, 6 px 6/6 at 1000 and 4-5/6 from 1600 on, no digit misread. The
3088 px sheet shrinks little if at all.

| segment cells | a: the judges' sheet (one image, through the eye, 4x) | c: three files, through the eye, 4x | r: three files, raw 4x | e: three files, the user's eye |
|---|---|---|---|---|
| 1-2 | 0/6 | 0/6 | 0/6 | 0/6 |
| 3-6 | 1/6 | 5/6 | 6/6 | 6/6 |
| 7-15 | 2/6 | 5/6 | 5/6 | 4/6 |
| 16+ | 4/6 | 4/6 | 6/6 | 6/6 |
| null pairs called same | 4/4 | 3/4 | 1/4 | 4/4 |

Chance is 0.1-0.4 by bin.

- **The judges' sheet hides most changes from the judge.** On it a judge found 7 of 18 segments of 3 cells or more,
  nearly all at low confidence. The same pictures as separate files found 14 (blurred) and 17 (raw). The text test
  rules out shrinking, so the cause is the layout: three pictures side by side in one image. Rounds pilot1, cal1,
  aopt and c1 were judged on that sheet. c1's "no pair decided" says the judges did not see the change, not that it
  was invisible.
- Raw, the judge finds nearly every change of 3 cells or more but calls 3 of 4 dither-only pairs different. It
  compares pixels, dither included. Blurred separate files keep the nulls (3 of 4) and find 14 of 18.
- 1-2 cells: no layout. Two of the six give a render identical to the painting's (jojo/2, diver-sunset/15: B equals A
  pixel for pixel), so they are no change at all; `build.py` keeps such B pairs (fit.py drops them as within the
  dither noise).
- n is 6 a bin and one judge a pair: the layout effect (7 against 14-17 of 18) is clear, the bins are not.

The eye was the other fault. The converter's eye (sigma 1 px on everything) blurred at 1x and enlarged after does
two wrong things: a sigma of 1 on lightness averages a cell's dots into its mean, and with them the clash between
cells of different dot texture, which the user sees and dislikes; and enlarging after the blur turns every pixel
into a sharp block. `views.seen2` enlarges first (square pixels, as the screen shows them) and blurs after, in
linear light, the converter's opponent channels each with its own Gaussian, as S-CIELAB does. On 3x3 grids of every
picture (`data/acuity/grid/`, lightness and colour sigma 0.5, 0.75, 1) the user picked **lightness 0.75, colour
1.0**; colour below lightness gives false fringes. Layout e is that eye on three files: 16 of 18 segments of 3
cells or more and every null pair called same, the only layout that has both.

Step 2 takes layout e as the judges' view.

### 9. Session E step 2: how often a judge prefers the painting (2026-10-01)

`curve.py`. Pairs with the painting's side known better, through layout e: frac, Select pairs against Select pairs
with a share of the painted segments painted (0.1, 0.25, 0.5 random segments, and the whole painting; 36 pairs);
seg, the painting against itself with one segment given back (step 1's 18 segments of 3 cells or more); C, the
painting against the cells' next best pair by the energy (6); null, re-dithers (4). Each pair to 6 fresh judges, 3
in each order (39 judges, the prompt `curve.PROMPT`); frac and null also to 4 judges with a paragraph on the user's
taste (16 judges). Size is the mean dE through the project's eye on the changed cells.

| pairs | for the painting | against | even | pairs won by majority |
|---|---|---|---|---|
| frac, 10-500 cells | 193 | 19 | 4 | 33 of 36 |
| C | 30 | 0 | 6 | 5 of 6 |
| seg, under 10 dE | 4 | 1 | 19 | 0 of 4 (all even) |
| seg, 10-20 dE | 16 | 8 | 6 | 2 of 5, lost 1 |
| seg, 20 dE and more | 40 | 4 | 10 | 7 of 9, lost none |
| null | 7 | 2 | 15 | none decided |

- **With the user's eye and separate files the judges see the change and side with the painting**: nine votes in ten
  on the fractions at every size from 10 dE, 30 to 0 against the energy's next best pair. Sides are even (colouring
  1 156 votes, 2 168).
- A lone segment needs about 20 dE through the eye to be decided (7 of 9, none lost); below 10 the judges say even
  rather than guess. Several segments together are decided from 10 dE.
- Null pairs: no pair decided by the majority of 6, but 9 single votes of 24 pick a side (7 of them the painting's
  origin; on rocket-rackoon "a flat red sky with hard edges"). A label needs a majority of several judges, not one.
- Against the painting by majority: RC1/21 (6 of 6: "magenta specks in the dark trunk"), jojo at 0.5 (6 of 6: cyan
  cells on the jacket, yellow hair) while jojo at 0.1, 0.25 and 1 won 6 of 6, autumn at 0.5 (5 of 6: a pink patch on
  the path, green blocks). Half a painting may be worse than none or all; or the judges miss the user's taste. Only
  the user can say.
- The taste paragraph makes them worse: 119 to 21 on the same fractions (0.85 against 0.91); the whole autumn painting
  flips to 1 against 3. The plain prompt stays.

The user, blind, on 12 of these pairs (`curve.py user`: raw 3x as the app shows them, + the painting better, - worse,
= both fine, x both bad; `rounds/curve/user-verdicts.json`): nine the judges disputed (against the painting or split,
plain or taste prompt), three they agreed on.

| pair | the user | judges, plain (6) | taste (4) |
|---|---|---|---|
| autumn whole painting | + | 6 + | 1 +, 3 - |
| autumn 0.1 | + | 5 +, 1 - | 2 +, 2 - |
| jojo/10 | + | 6 + | |
| sunset whole painting | + | 6 + | 4 + |
| RC1/21 | = | 6 - | |
| diver-sunset/14 | = | 2 +, 2 -, 2 = | |
| golden-axe/13 | = | 1 +, 2 -, 3 = | |
| anubis/10 against C | x | 6 + | |
| jojo 0.5 | x | 6 - | 4 - |
| autumn 0.5 | x | 1 +, 5 - | 4 - |
| andy 0.5 | x | 5 +, 1 - | 2 +, 2 - |
| diver-sunset 0.1 | x | 2 +, 2 -, 2 = | 3 -, 1 = |

- Where the user prefers a side, the plain judges' majority agrees, 4 of 4; the taste prompt lost two of them.
- Where the judges went against the painting, the user did not prefer it either: both fine (RC1/21: "magenta specks"
  is a variant the user accepts) or both bad. No decisive disagreement in 12.
- **There is more than one good colouring**: three lone segments are "both fine". And half a painting is often worse
  than a whole one: 4 of 5 partial paintings (and anubis's painting against the next best pair) are "both bad", while
  the judges still gave them confident verdicts (jojo 0.5 6 of 6). A two-way choice cannot say "both bad"; the judges'
  verdict there is noise for the energy and must not become a label.

### 10. Session E step 3: Claude judges where Select pairs is unsure (2026-10-01)

`dp.py`. Base: Select pairs at the Exact mixture preset, nothing painted, on the 6 training pictures and the 6 loose
ones (goldhill, gradient, kotofey, landscape, tv_out, vangog: a new project's graph, untuned); the 5 held-out
pictures stay out. Per segment the alternatives the converter makes itself: next, its cells on their next best pair
by the energy's own term; coherence x2, edge 0.32, eye chroma blur 1.1, luma_noise 0.1, the segment's labels from
Select pairs at that setting put into the base. Each rendered by the base pipeline, only the segment changed.

- The plan's filter (the lowest third of energy margins, then 15 dE through the eye) picked the smallest changes: the
  total margin grows with the cells, and its lowest third had a median of 1 cell, which judges do not see (step 1).
  Now: 3 cells or more, the margin per changed cell, per picture its lowest half, then 10 dE or more. Margin and
  visible change go together (Spearman 0.45 over 918 alternatives): the lowest third and 15 dE leave 48 pairs, the
  lowest half and 10 dE 158.
- The untuned loose pictures barely change (median 3-6 dE): goldhill, kotofey and landscape give no pair, gradient
  and tv_out 4 each (gradient's all next); vangog 35.

158 pairs and 8 nulls (a picture's base from another halftone origin), each to 6 fresh judges, 3 per order (pilot 40
pairs, 24 judges; then 126, 76 judges in a workflow). Prompt `curve.PROMPT` with four answers: 1, 2, = both fine, x
both bad. A label is the answer of more than half the 6 votes.

| pairs | n | alternative better | base better | = | split |
|---|---|---|---|---|---|
| next | 20 | 3 | 13 | 2 | 2 |
| coherence x2 | 27 | 2 | 7 | 14 | 4 |
| edge 0.32 | 47 | 10 | 7 | 19 | 11 |
| eye chroma 1.1 | 20 | 7 | 1 | 6 | 6 |
| luma_noise 0.1 | 44 | 12 | 16 | 8 | 8 |
| 10-15 dE | 71 | 14 | 26 | 18 | 13 |
| 15-20 dE | 36 | 9 | 5 | 12 | 10 |
| 20-30 dE | 36 | 8 | 11 | 14 | 3 |
| 30 dE and more | 15 | 3 | 2 | 5 | 5 |
| **all** | **158** | **34** | **44** | **49** | **31** |
| null | 8 | | | 7 | 1 |

- **127 labels of 158**: 78 sides, 49 even; 31 splits. Nulls pass (7 of 8 even by majority, one split). Sides
  even (colouring 1 318 votes, 2 299); x almost unused (6 votes of 996).
- The labels hold across the orders: of the 78 side labels, 72 have both orders' own majority (3 votes each) on the
  label's side; 40 are 6 of 6.
- **Where Select pairs is unsure, the energy's choice is a coin toss for the judges**: base better 44, alternative
  better 34. The margin per cell does not tell them apart (median 0.036 both, Mann-Whitney p 0.86). The judges side
  with next least (3 of 16), with eye chroma 1.1 most (7 of 8). Of 9 alternatives with lower energy than the base
  (the DP not at the minimum), the judges prefer the base on 2 and the alternative on 1.
- Unlike step 2, 10-15 dE is decided as often as larger changes (40 sides of 71): these change several cells, a
  lone segment needed ~20 dE.

Labels and key: `rounds/e1/` (key out of git until the user's check, step 4); renders `data/dp/raw/`.

The user, blind, on 30 labelled pairs and 5 repeats, the judges' own images side by side (`dp.py user`, `dp.py vote`,
`rounds/e1/user-verdicts.json`):

| judges' label | user: alternative better | base better | = both fine | x both bad |
|---|---|---|---|---|
| alternative better (7) | 1 | 0 | 4 | 2 |
| base better (13) | 0 | 3 | 5 | 5 |
| = (10) | 1 | 0 | 5 | 4 |

- **The user decides 4 of 30, and agrees with the judges on all 4** (Wilson lower bound 51%, the gate's 65% not
  reached for want of decisions). The rest: both fine 14, both bad 11. Repeats 4 of 5 the same.
- Where Select pairs hesitates, its neighbours are equivalent to the user, or both bad: fitting the energy's weights
  to these labels (step 5) would choose among equals. A third of the places the DP is unsure of are bad whichever
  way it goes: the fault is outside the choice between neighbouring pairs. Session E stops after step 4.

### 11. After session E: bolder changes to pair selection (2026-10-02)

The user's "both bad" in step 4 are all on autumn, jojo and vangog, and show two faults no setting reaches: a
patchwork inside one surface (jojo's faces cell by cell flat pale yellow/white against red/yellow dither, vangog's
sky with stray cyan, magenta, green cells) and a wrong colour family where its mixture is nearest (a magenta
cypress). Every cell picks its own nearest mixture, so the cells of a smooth surface fall on both sides of the border
between two families.

- `smooth.py`: Select pairs on an edge-preserving smoothed picture (bilateral in CIELAB, 1-2 cells, 20-40 dE), DBS
  still toward the original. The patchwork stays (the border between families only moves) or the faces go grey.
  Not shown to the user.
- `surface.py`: pairs per surface (`common.segments`): the pair that covers the surface's cells at least own cost,
  every other pair priced out, then the usual DP. Hard (every cell bound) fixes the patchwork and loses the colours
  of surfaces that mix several (the raccoon black/white, RC1's greens). A soft cost of leaving only blends base and
  hard. Bound only the cells within 20 dE of their surface's mean colour, the rest free: the user's second round.

The user, blind, whole pictures raw 3x, the variant against the base (`rounds/surface/r1`, `r2`):

| picture | hard, 32 surfaces | hard, 64 | within 20 dE, 32 | within 20 dE, 64 |
|---|---|---|---|---|
| jojo | + | + | + | + |
| vangog | + | + | + | + |
| diver-sunset | + | - | - | + |
| RC1 | - | - | + | - |
| anubis | - | - | - | - |
| rocket-rackoon | - | - | - | - |
| autumn | x | x | x | x |

- **The first change of the research the user prefers consistently, on the pictures it was made for**: jojo and
  vangog better in all four. And consistently worse on anubis and rocket-rackoon: dense illustrations whose small
  accents (the torch's red glow, highlights on Anubis and the raccoon) a surface's pair erases. diver-sunset and RC1
  flip. 3 better, 3 worse, 1 both bad for each: a style that suits some pictures, not a new default.

### 12. The competitor baseline (2026-10-02)

The coach's "done" for the research: the conversion visibly better than the known converters on three reference
pictures, the user blind. Four pictures, the user's pick: jojo, rocket-rackoon, autumn, diver-sunset. Every tool gets
the picture Select pairs aims at (the project's Tune, 256x192, `data/rivals/in/`), so crop and tone are the same for
all (`rivals.py`):

- **img2spec** (`../img2spec_video` 5.6, the macOS build, its CLI with a workspace per setting): the ZX Spectrum
  device at its defaults (popular colour); Floyd-Steinberg, Bayer 8x8, mono DBS (modulate, keeps hue).
- **Image to ZX Spec** 2.3.0 (KodeMunkie; the plan's "image2zx"), without its window as its WorkProcessor converts
  one picture (`Izx.java`), its defaults (full palette, favour half bright, luminance distance); Atkinson (its
  default), Floyd-Steinberg, Bayer 4x4.
- **ZX-Paintbrush** 2.6.1 (Claus Jahn), by the user under Wine: import at the defaults, saved as SCR.

Every output is read back into our palette (205/255) and checked: two colours a cell, one brightness; all pass.

Each scripted tool's best setting per picture by Claude judges (the protocol of sections 9-10: pairs, three files
through the user's eye, 6 votes a pair; 24 pairs, 15 judges, 144 votes, 5 "x"); a setting's votes for less those
against over its two pairs, the most wins (no ties):

| picture | img2spec | Image to ZX Spec |
|---|---|---|
| jojo | Bayer 8x8 (+10) | Atkinson (+12) |
| rocket-rackoon | Floyd-Steinberg (+10) | Floyd-Steinberg (+10) |
| autumn | Floyd-Steinberg (+12) | Atkinson (+12) |
| diver-sunset | Bayer 8x8 (+12) | Bayer 4x4 (+8) |

img2spec's mono DBS is last on every picture: it drains the colour to grey and pink.

The user, blind, raw 3x as the app shows them (`rounds/rivals`): each of ours against each tool's best, 24 sheets, and
5 repeats with the sides swapped (4 of 5 the same; diver-sunset surface dE 20 against Image to ZX Spec x, then +).
+ ours better, x both bad, the first answer:

| picture | ours | img2spec | Image to ZX Spec | ZX-Paintbrush |
|---|---|---|---|---|
| jojo | Exact mixture | x | + | x |
| jojo | surface dE 20 | + | + | + |
| rocket-rackoon | Exact mixture | + | + | + |
| rocket-rackoon | surface dE 20 | + | x | + |
| autumn | Exact mixture | x | + | + |
| autumn | surface dE 20 | x | x | + |
| diver-sunset | Exact mixture | + | x | + |
| diver-sunset | surface dE 20 | x | x | + |

- **Ours is never worse**: 15 of 24 better, 9 both bad, none to the rival and none both fine. ZX-Paintbrush at its
  defaults loses 7 of 8.
- **The coach's criterion holds on two pictures, not three**: jojo with surface dE 20 and rocket-rackoon with Exact
  mixture beat all three tools, each with its own style of section 11 (a pair per surface on faces, free cells on a
  dense illustration); neither setting alone wins both.
- autumn and diver-sunset are both bad against the dithered tools at their best (Floyd-Steinberg, Atkinson, Bayer):
  ours is not worse there, but not good either. autumn has been both bad in every round; why is not in the votes.
- Nowhere do we lose, so the round names no fault of ours the rivals avoid (patchwork, accents, colour, seams); the
  seam terms parked on that condition stay parked.

### 13. The gallery (2026-10-04)

PLAN step 3 (`gallery.py`): per picture, rounds of 16 renders through the project's pipeline, here all without DBS (10
knobs, each over its slider scaled to 0..1), on a plane through the current best along two random directions. The user
discards the bad ones, the rest laid out again larger, and picks the best; a discard is worse than every render still
left, the pick better than all left with it (`gallery.comparisons`). autumn's 11 rounds are picks only (the page then
had no discards). Galleries: autumn 11 picks, rocket-rackoon 8, diver-sunset 7, anubis 3, david 1 (2 rounds, a held-out
picture, left out).

`gallery.py report`: per picture a linear preference over the knobs fitted to its comparisons (logistic, L2), 5-95% over
rounds resampled; each knob's change of the eye-blurred render per full slider range, a linear map over the picture's
75-165 renders that explains 45-49% of their change (a cell turning to another pair is a jump, not a slope). Start ->
last pick; - and + where the preference's 5-95% excludes 0:

| knob | autumn | rocket-rackoon | diver-sunset | anubis |
|---|---|---|---|---|
| Select chroma noise | 0.020 -> 0.017 - | 0.050 -> 0.040 - | 0.050 -> 0.000 | 0.050 -> 0.033 |
| Select luma noise | 0 -> 0.004 - | 0 -> 0 - | 0 -> 0.149 + | 0 -> 0.046 |
| Select coherence | 6.0 -> 5.7 - | 2.0 -> 1.9 | 2.0 -> 4.5 | 2.0 -> 5.6 + |
| Select edge | 0.10 -> 0.12 | 0.10 -> 0.11 | 0.10 -> 0.06 | 0.10 -> 0.22 + |
| Metric chroma | 2.0 -> 0.84 - | 1.0 -> 1.04 | 3.27 -> 4.0 | 1.0 -> 0.96 |
| Halftone chroma | 2.0 -> 1.77 | 1.0 -> 0.78 | 3.27 -> 3.61 - | 1.0 -> 1.02 |
| Eye luma blur | 1.4 -> 0.90 - | 1.4 -> 1.42 | 1.4 -> 1.78 | 1.4 -> 1.42 |
| Metric flare, surface dE, Eye chroma blur | small, mixed | | | |

- **Chroma noise down on all four**: the one knob that moved the same way everywhere; a candidate for a lower default.
- **Luma noise by the picture**: up on the smooth sky of diver-sunset, down to its floor on the dense rocket-rackoon
  and on autumn: a candidate per surface (`IDEAS.md`).
- **Coherence**: from 2 to ~5 on anubis and diver-sunset, autumn stayed near 6, rocket-rackoon indifferent.
- **The preferences differ by picture**: the cosines of their directions -0.27 to 0.39, but anubis and diver-sunset 0.65.
- **No two knobs alike**: every knob changes the render, ~10-50 dE rms per full range; the mean cosine of two knobs'
  effects is at most 0.48 (Select luma noise against Eye luma blur: one charges the dots' contrast, the other hides
  the dots; chroma noise against chroma blur -0.81 on anubis, weak elsewhere).
- Thin: each round spans 2 of 10 directions and ~10 rounds a picture, so the fitted preference sometimes points against
  the picks (diver-sunset's Eye luma blur, Halftone chroma); the picks count more than the fit. The user's earlier note
  that less Halftone chroma evens blockiness: down on rocket-rackoon and autumn, up on diver-sunset, not shown.

### 14. Which seams the user sees (2026-10-06)

PLAN step 5 (`seams.py`). The user painted the seams seen (where the source has none) on 12 whole renders with DBS,
at 3x: each training picture at its gallery best and at coherence 0, in a blind order, then 2 of them again; ~32 min,
27-245 marks a render, 1342 in all. 1338 of them are on seams where the pair changes (7197 of the 17760), so the score
is on those: where the pair stays no measure sees a seam, and those seams flatter every measure (the attributes 0.86
on all seams, 0.62 on the changed). `seams.py report`, AUC per render, mean; the lightness-colour mix chosen with the
picture left out; top k: the share of the k seams a measure ranks highest that the user marked (k, the user's marks):

| measure (changed seams) | L | ab | both | top k |
|---|---|---|---|---|
| A attributes, V | | | 0.62 | 0.34 |
| A x edge weight (the coherence term) | | | 0.51 | 0.24 |
| B pixel step, raw | 0.75 | 0.79 | 0.79 | 0.42 |
| B pixel step, through the eye | 0.66 | 0.68 | 0.67 | 0.26 |
| B pixel step, blur 2 | 0.56 | 0.61 | 0.60 | 0.23 |
| C line filter less the texture, raw | 0.64 | 0.74 | 0.74 | 0.42 |
| C, through the eye | 0.61 | 0.63 | 0.65 | 0.32 |
| D strips, raw | 0.65 | 0.74 | 0.73 | 0.38 |
| B less the cells' own pixel steps, raw | 0.68 | 0.74 | 0.73 | 0.41 |
| E the pair switch at the border's pixels, the dots the same | 0.70 | 0.71 | 0.71 | 0.31 |
| the user's repeats | | | | 0.48-0.75 |

- **The seams the user sees are pair changes** (1338 of 1342): a pairwise term on the labels is the right place.
- **The edge weight works against the user.** The user marked 0.22 of the changed seams where the coherence term
  excuses a seam (the target's block means step, weight < 0.5) and 0.15 where it does not. With the weight the
  attributes are a coin toss (0.51), without it 0.62. At an outline the render makes a staircase of cell borders,
  which shows though the source has an edge there; the source's own step lowers the share marked, but it does not
  remove it (an ad-hoc look: 0.58 to 0.27 among the largest pixel steps).
- **The raw pixels' colour step across the border, beyond the source's, is the measure**: AUC 0.79 on ab alone,
  above the attributes on 11 of 12 renders. Lightness adds nothing on these renders (a logistic over B, C, D and A,
  each picture left out: 0.76-0.78, the L weights ~0); the lightness steps are mostly the dots'.
- **The eye blur makes every measure worse** (0.67 and 0.60 against 0.79): the user sees a seam at the pixel level,
  sharper than the judges' eye, as `seams.py check` says (it carries the dots' colour onto a black border).
- The texture terms (C, D) and the runs along a border add nothing.
- **Far from the ceiling**: top k 0.42 against the repeats' 0.48-0.75.
- The painted segments (README 1) say the other: through the eye C rates the painting below Select pairs in 0.73,
  raw B in 0.61, `seam_excess` 0.65. Those comparisons mix the seams with the accuracy; the marks test the seams alone.

### 15. Seam visibility on synthetic patches (2026-10-06)

PLAN step 5b (`patches.py`): seam visibility itself, without picture content. A patch of 8 x 6 cells, the left half
one pair, the right another, one blue-noise threshold map over both, so the dots run on across the border; the level
varies slowly (sd 0.08, 6 px). The colour changes come at the same mean luminance, and the user could move the right
side's lightness to where the seam is least visible (the minimally distinct border) before rating it 0-3. Round v3,
51 patches and 10 repeats (v1, at one level, and v2, with Bayer on a uniform level, stopped by the user: the lightness
and the regular texture gave the seams away). `patches.py report v3`, the mean rating:

| at 30 / 50 / 75% dots | r-m | g-c | y-w | r-g |
|---|---|---|---|---|
| hue, the same luminance | 2 / 1 / 2 | 2 / 1 / 2 | 2 / 2 / 3 | 3 / 3 / 3 |
| brightness alone (the inks' capitals), the same luminance | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 2 | (w-W) 1 / 1 / 2 |

- **A colour change shows at the same lightness**: near hues ~2 (clear), red-green 3. Brightness alone is faint (1).
  Hue and brightness together 2.
- **Lightness alone hides in the surface's variation**: one pair, a step of the level of 0.06 or 0.12, rated 0 on all
  three pairs; 0.25 rated 1-2.
- The user never moved the lightness: the CIE match was left as it was.
- No change, 0; the user's black border, 1 (dots away from the border) against 2 (dots on it).
- Repeats: 80% the same rating, 20% one step off.
- The measures, Spearman with the ratings (AUC, any seam against none): E, the pair switch at the border's pixels,
  0.75 (0.95); D strips 0.75 (0.96); the attributes 0.72 (0.96); B raw 0.14 (0.54): with the dots running on, the step
  across the border is the dots', and B sees a dotted pair's own texture as a seam, even where the pair stays (0.15
  among the 37 patches where it changes, E 0.62).
- **The renders and the patches disagree**: on the renders B raw is the best (0.79, E 0.71), on the patches the worst.
  On a render the dots break at a pair change (each pair its own dots, then DBS); on the patches they run on.

Round v4 (the user): the same with a gradient of random direction and strength on each patch, each change twice, and
the dots breaking at the border on some; 124 patches and 10 repeats (`patches.py report v4`):

- **The dots breaking makes no seam**: no change with only the dots breaking, 0; the colour changes rated as with the
  dots running on (hue ~2, brightness ~1). It does not explain B raw on the renders.
- The user moved the lightness this time: the brighter side darker than the CIE match by 1-4 dL* (y-w, y-Y, g-G):
  bright and white dots look lighter than their luminance.
- Lightness alone: a step of 0.12, 0-0.5; 0.25, 0.5-1.5. Repeats 90% the same.
- The measures: the attributes 0.83, E 0.81, D 0.76, B raw 0.20.
- **The set was narrow** (the user): inks on black paper, near hues, no blue. The renders are full of what it lacks:
  a paper's change under one ink (bright white on cyan against on green: skies, foliage), blue, contrasting pairs, the
  same mean colour from other dots. Round v5 has them.

Round v5 (the user: the pairs the earlier rounds lacked), 25 changes twice and 10 repeats (`patches.py report v5`):
a paper's change under bright white 1-2 (cyan-green, magenta-red 1; cyan-blue, yellow-cyan, yellow-green 2); blue on
black against red 3, against magenta 2.5, bright against dim blue 0.5; blue paper with cyan against white ink 2, with
yellow against white 1; **the same mean colour from other dots 1-2** (black/white against blue/yellow 1, against
red/cyan and magenta/green 2). Repeats 100% the same. The measures turn over: E 0.36, the cells' mean colours 0.58,
the colour step at a blur of 2 0.58, B raw 0.01. The eye averages part of the dots: the colour change at the pixels
over-counts a change of dots that keeps the mean colour.

### 16. One seam score (2026-10-06)

PLAN step 5c (`seamfit.py`): the seam measures (lightness and colour apart), weights >= 0, fitted at once to the
painted renders (the seams where the pair changes, marked or not, each render its own rate) and to the rounds v3-v5
(the order of the mean ratings). Each picture or round left out in turn:

| | renders AUC (top k) | v3 | v4 | v5 |
|---|---|---|---|---|
| the coherence term (attributes x edge weight) | 0.51 (0.24) | | | |
| the attributes alone | 0.62 (0.34) | 0.72 | 0.83 | 0.48 |
| B raw, the best on the renders | 0.79 (0.42) | 0.14 | 0.20 | 0.01 |
| fitted on the renders alone (in sample) | 0.80 (0.46) | 0.47 | 0.55 | 0.25 |
| all the measures, joint | 0.73 (0.39) | 0.82 | 0.89 | 0.70 |
| **two: E lightness + the cells' mean colour** | 0.68 (0.35) | 0.79 | 0.88 | 0.77 |

- **The two-term score**: 0.075 x E_L + 0.055 x M_ab, per dE. E_L: the lightness change of the pixels along the
  border when the pair switches, the dots the same; M_ab: the colour difference of the two cells' mean colours (in
  linear light), beyond the source's. Lightness at the pixels, colour averaged: the eye's sharp lightness and coarse
  colour, from the user's data. Zero for the same pair; each part computable for any two pairs from their dots.
- What the patches teach carries over to the renders; the renders' own fit does not carry over to the patches. The
  renders' extra: the same change shows more among busy dots (the cells' own pixel steps alone, AUC 0.69; E times
  them 0.77; as a term in the joint fit, +0.02); perhaps attention in a whole picture rather than the seam.
- The joint fit of all the measures adds 0.05 on the renders over the two terms, with four more measures.

### 17. The user's sorting, and lightness (2026-10-06)

`seamfit.py ranked`: per global lightness (L* 15-92) every two Spectrum pairs that reach it, side by side as a patch
(blue noise, the slow variation, the lightness matched), ranked by the two-term score; 16 evenly along the ranking on a
sheet. The user did not quite agree and sorted the same patches (`seamfit.py sort`, blind, shuffled first on 6 of 7
sheets; `rounds/seams/ranked-v1.json`): Spearman with the two-term score 0.66-0.94. The largest misses are the user's
rule: **a dim pair whose colours are near in lightness (yellow and white; cyan, green) reads as one solid block and
stands out next to any bright pair**; dim and bright pairs with black, blue or magenta mix (dim blue/red against black/
bright magenta, the least visible at L* 37); dim green/blue against bright cyan/black too. At L* 81 the user put all
seven dim yellow/white patches at 10-16 of 16, the score some at 2 and 5.

- A cell's solidity, exp(-sd / 3) of its pixels' L* (dim yellow/white sd 1.2, bright magenta/green 12.5); the step of
  it across a seam. On the renders a solid cell next to a dotted one is common in the dark and not a seam there
  (below L* 30 the user marked 1% of the pair changes; AUC 0.35); the share marked rises with lightness: 1%, 16%, 25%,
  36% (L* < 30, 30-55, 55-75, > 75).
- **The score, three terms** (Y, the two cells' mean L* / 100): 0.054 x the cells' mean colour difference (ab) +
  0.126 x Y x the lightness change of the border's pixels + 1.68 x Y x the solidity step. Held out:

| | renders AUC (top k) | v3 | v4 | v5 | sorted |
|---|---|---|---|---|---|
| two terms (README 16) | 0.68 (0.34) | 0.79 | 0.88 | 0.77 | 0.83 |
| three terms | 0.72 (0.38) | 0.78 | 0.87 | 0.83 | 0.85 |

- Per sheet of the sorting: L* 81 0.66 -> 0.90, L* 92 0.89 -> 0.99; the darkest, L* 26, 0.87 -> 0.65: there the user
  ordered by lightness, and the lightness term scaled down lets colour lead. A compromise, by the renders.

### 18. The seam score in the selection (2026-10-06)

PLAN step 5d (`selection.py`): the three-term score as the DP's pairwise table, for each border and each two pairs from
the pairs' halftones in the two cells, times coherence x 0.0048 (the old term's sum over the current selections' pair
changes over the score's), the coherence term off; zero for the same pair. A copy of the selection with the old term
gives the current labels on every picture. The score at the halftone ranks the changed seams as after DBS: Spearman
0.85-0.96 on the training pictures (0.61-0.99 held out; david's few changes left are subtle). 35-184 cells changed a
training picture, 96-279 held out.

The user, blind, the six training pictures at their gallery best with DBS, raw at 1x enlarged by whole pixels (after
a void first vote, on session E's sheets: `dp.py vote` had fallen back to its default folder): **the score better on
5 of 6** (diver-sunset, autumn, jojo, RC1, anubis), rocket-rackoon both fine (35 cells changed), the current never.
The score was fitted partly on these pictures' renders, so the held-out pictures decide: their projects' settings with
DBS, 5 sheets and 5 repeats with the sides swapped.

The held-out round (`rounds/seams/selection/held`), 5 pictures and 5 repeats with the sides swapped, every repeat the
same: **both bad** on golden-axe, burning-hand, andy, sunset (their projects' own settings, no gallery: the conversions
as a whole); **the current better on david** (the score keeps one brightness where the current switches dim and
bright black/white, and the user prefers the switching there). The score wins none held out. The user: the score's
contribution is low. The scale matched the old term's total cost over the current pair changes, so it only moved the
cost from invisible changes to visible ones; at x1 the seams by the score even rise on anubis and diver-sunset.

`selection.py strength`: the score at x1-x16 of that scale on the training pictures, the seams' sum by the score over
the pair changes:

| picture | current | x1 | x2 | x4 | x8 | x16 |
|---|---|---|---|---|---|---|
| RC1 | 2194 | 1321 | 1122 | 888 | 671 | 590 |
| autumn | 1037 | 927 | 727 | 542 | 434 | 403 |
| jojo | 694 | 428 | 336 | 281 | 264 | 256 |
| anubis | 1233 | 1357 | 1208 | 1071 | 858 | 626 |
| diver-sunset | 698 | 730 | 666 | 608 | 558 | 521 |
| rocket-rackoon | 1214 | 1130 | 1019 | 866 | 738 | 594 |

The user's strengths on that page: RC1 x2, anubis x8, autumn x4, diver-sunset x8, jojo the current, rocket-rackoon x1.
**Times each picture's coherence** (the score was multiplied by it): 16, 26, 23, 14, (0), 8; the two Halftoned
pictures want x8 because their coherence is 2-3x lower. So the score's weight is fixed, the picture's coherence aside:
16 (the geometric mean of the picks, within a x2 step of four of them). The held-out check again at it
(`selection.py make held fixed`, `user held fixed`; `rounds/seams/selection/held-fixed`).

Held out at the fixed weight (the user asked to pick the less bad one when both are bad), every repeat the same:
**burning-hand the score better; david the current better; golden-axe, andy, sunset both bad.** In all: the training
pictures 5 better, 1 both fine, 0 worse (fitted partly on them); held out 1 better, 1 worse, 3 both bad. David is
greyscale: switching dim and bright white gives it one more tone, and the score charges the switch as a seam (by the
patches rightly, brightness alone ~1 against a hue's ~2); there the user wants the tone. The three both bad are their
projects' old settings (no gallery): the conversions as a whole.

### 19. Prior pairs from artists: the zxart.ee top 100 (2026-10-06)

`zxart.py`, branch `research/zxart-top-100`. The 100 best-voted standard (6912) screens by the site's API
(`export:zxPicture/filter:zxPictureType=standard/limit:100/start:N/order:votes,desc`; `limit` must come before
`start`); the index in `rounds/zxart/top100.json`, the screens in `data/zxart/` (not in git; `fetch` takes ~1 min).
The measures are the seam research's (14-17), on the artist's own pixels: a cell's shown set (as `common.shown`, from
the attributes: black's two indexes one, a colour with no pixel left out); per 4-neighbour border whether the set
changes, whether the two sets share a colour, Y, M_ab (no source to subtract), the solidity step, and E, the switch's
lightness change at the border's pixels (`selection.table`'s form, from the border columns' ink counts), with a null
E0: the same at the cell's mean ink level, so E against E0 says what the artist does at the border itself.

**The artists** (76,800 cells, 148,000 borders): 14% of cells show one colour, 48% are bright; the pairs most shown
k/b, k/W, k/B, k/r, k/w, k/c (5-7% each). **The pair changes on 50% of borders; 87% of the changes keep a shared
colour**, 9% switch bright/dim only. **The border hides**: on shared-colour changes E 12.8 against E0 15.7, and E = 0
(the border's pixels all the shared colour) on 25% against 5% at the cell's level. By lightness:

| Y (the cells' mean L*/100) | share of borders | change rate | shared | E / E0 | M | S | score / null |
|---|---|---|---|---|---|---|---|
| 0-0.25 | 0.24 | 0.38 | 1.00 | 3.7 / 5.0 | 45 | 0.49 | 2.61 / 2.64 |
| 0.25-0.5 | 0.25 | 0.59 | 0.91 | 15.2 / 18.5 | 59 | 0.11 | 4.03 / 4.19 |
| 0.5-0.75 | 0.30 | 0.57 | 0.77 | 20.4 / 23.1 | 53 | 0.08 | 4.53 / 4.75 |
| 0.75-1 | 0.21 | 0.44 | 0.89 | 16.6 / 19.8 | 34 | 0.34 | 4.12 / 4.46 |

Changes without a shared colour: 6.4% of borders, score 5.95 against the shared ones' 3.67.

**The transition prior**: 1281 kinds of change (two shown sets), 384 with 30+ counts; each one's count over the count
expected if a picture's cells changed at random (the picture's shown-set frequencies and change rate). The log ratio
against the coherence term's V (the sets as pairs, dark to bright): rho -0.22; against the seam score: **-0.41**; V
against the score 0.53. The most made (7-8x expected): a one-colour cell next to a pair that holds its colour (b/b |
b/w, k/r | r/r, k/M | M/M), and hue ramps (R/G | G/Y; B/M | M/Y 7x though the score gives it 7.5). The most avoided
(0.17-0.3x): no shared colour and far hues (k/R | B/C, k/y | B/C, k/r | b/c, k/k | r/y).

**The recovery test** (`recover [K]`): the artist's screen through the user's eye (`views.seen2` at 1x, K times its
sigmas) as a new project's source, Select pairs at the defaults, no DBS, 0.5 s a screen; the converter's shown sets
against the artist's, and its borders by the same measures:

| | cells recovered | change rate | shared | E = 0 | no shared colour (of borders) | score of changes | changed by the converter alone (its score; no shared) | by the artist alone |
|---|---|---|---|---|---|---|---|---|
| the artist | | 0.505 | 0.874 | 0.22 | 0.064 | 3.96 | | |
| the eye, x1 | **0.824** | 0.471 | 0.853 | 0.19 | 0.069 | 3.85 | 0.036 (2.27; 0.01) | 0.071 |
| x2 | 0.735 | 0.454 | 0.823 | 0.15 | 0.080 | 3.68 | 0.052 (2.07; 0.05) | 0.103 |
| x3 | 0.632 | 0.436 | 0.769 | 0.13 | **0.101** | 3.41 | 0.071 (2.11; 0.13) | 0.140 |

At x3 by lightness, shared on the changes: 0.64 against the artist's 0.77 at Y 0.5-0.75, 0.75 against 0.89 at
0.75-1, 0.99 against 1.00 in the dark. Through the user's eye the selection is nearly the artist: 82% of cells, the
borders alike, its own changes cheap and shared. As the blur grows (the picture less fitted to the grid, as a
photograph), it first loses the **shared colour**: changes without one grow from 6.9% to 10.1% of borders against the
artists' 6.4%, at mid and high lightness; and the seam score does not see it, the converter's changes score lower than
the artists' (the blurred target's cell means are nearer). A judge with an expert's ground truth that costs the user
no time, on art drawn to the grid; the pictures are not photographs.

**Step 6.1, a shared-colour term** (`recover K W`): on each seam whose two pairs share no colour (black's two indexes
one), W x coherence x the seam's smoothness weight (0 across the target's edges), added to the current selection; the
gate is the recovery at 2x and 3x the blur, the share of changes without a shared colour against the artists' 6.4%:

| W x coherence | x2: cells recovered / no shared colour | x3: cells recovered / no shared colour | change rate x3 | bright/dim only x3 |
|---|---|---|---|---|
| 0 (the current) | 0.735 / 0.080 | 0.632 / 0.101 | 0.436 | 0.095 |
| 0.3 | 0.738 / 0.055 | **0.643 / 0.055** | 0.435 | 0.088 |
| 1 | 0.731 / 0.047 | 0.639 / 0.044 | 0.436 | 0.087 |
| 100 (as good as forbidden) | 0.712 / 0.025 | 0.632 / 0.019 | 0.438 | 0.086 |
| the artists' V, W 0 (6.2) | 0.751 / 0.080 | 0.641 / 0.107 | 0.555 | 0.170 |
| the artists | / 0.064 | / 0.064 | 0.505 | 0.092 |

A soft weight (0.3) brings the share without a shared colour to the artists' and recovers one point more at 3x, the
same at 2x, the other border statistics untouched; the hard constraint overshoots (2%) and loses two points at 2x.

**Step 6.2, the artists' V** (`recover K 0 artists`): the transition table, -log(count / expected) clipped at 0 and
scaled to 0..1 by the pairs' shown sets, in place of the attributes' V (`artists_V`; fitted on the same 100 screens,
so in-sample). Recovery +1.6 at 2x and +0.9 at 3x, but the borders drift from the artists': the change rate 0.55
(the artists 0.505, the current 0.44), bright/dim-only switches 0.17 (0.09), the share without a shared colour as
before. Many adjacencies are free in the table (made more than expected), so it makes more cheap changes, not fewer
bad ones; the dim/bright switch the user kept on david (18) is what it multiplies.

Findings: `FINDINGS.md`; next steps: `PLAN.md`; the log: `LOG.md`.
