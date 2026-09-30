# Pair selection: a judge from the hand-painted projects

The aim is a measure of a colouring that agrees with the user's eye, found before any new selection method: the
last attempts tuned the energy on proxy counts (painted cells matched, magenta cells) and the user rejected results
that improved them. Plan and reasoning: the session's plan (methodology for finding the pair-selection metric).

## Ground rules

- A colouring is a label map, one (paper, ink) pair per cell. It is judged only as the project renders it: through
  the project's own pipeline, with its Tune, Target, Halftoner, the Metric and Select weights DBS reads, and
  Optimise, painted as a whole field (`common.Project.render`). A painting means something only under the settings it
  was made with.
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
    python research/pairs/views.py OUT [N]  # blind images of N pairs for judging by eye, with key.json

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
- The judges' sheets are whole pictures through the eye (`views.judge_sheet`): the tuned picture and the two sides at
  2x, no crops, no outlined cells. The user's sheets keep the crops.
- F is not a clean label. By eye (RC1/2, andy/25): the flat segment is clearly worse where it turns the yellow fur
  grey, which S-CIELAB calls closer, and about as good where it makes andy's hair plain black and cyan. F near 0.5
  is no verdict; the judge's optimum (session B) and the user's votes on F pairs decide.
- Seven of round 1's kept-cell pairs no longer rebuilt as voted in the first rebuild, six in the next ones (other cell
  counts; one run of `variants.py` in three differed on diver-sunset and could not be repeated, pooled or not). Those
  are left out. `round.py` now writes each sheet's cells into the key.

Summary and next steps: `HANDOFF.md`.
