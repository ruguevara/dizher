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

    python research/pairs/build.py          # ~30 min on 4 cores: data/NAME.npz, data/NAME.json (not in git)
    python research/pairs/score.py          # data/metrics.csv and the table below
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
