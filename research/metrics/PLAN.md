# Plan: an eye model and a metric for Spectrum conversions

## The aim

A model of what the viewer sees and a full-reference metric on it that rank Spectrum conversions as people do, with
the clash seen as people see it. First the metric, slow if it must be (seconds per picture); a fast approximation for
the interactive mode comes after it. The result goes into the converter, then into an article on Habr, with the
reasons for each choice. The eye model and the metric are written so that they carry over to other images with a
limited colour budget, e.g. a metric for lossy compression.

The criterion (agreed with the user, 2026-10-20): **near the original, with the artefacts weighed as people weigh
them**, the clash first. Style (checkers, dithering at 25-50%, no DBS, the source simplified or recoloured) is not in
the metric. It is in the generator's settings and presets, and the user picks it. The metric then has to rank
conversions within any one style correctly, so it must not charge a checker for being a different texture from the
source's gradient (see H4).

The user's priorities (2026-10-02, confirmed 2026-10-20): seams first; then colour and tone, the colour approximate,
no grey dither everywhere, local contrast and meaningful gradients over the overall tone; then form in the details;
then texture.

## The starting point: research/pairs, reviewed

`research/pairs/` (branches `feature/color-pair-selection-research-72ae52` and its earlier snapshot
`feature/gallant-wright-6dtgj6`, kept apart from `develop`) was an unsystematic search. `PAIRS-REVIEW.md` goes through
it: what does not hold and why, what holds, and the data this plan takes. In short:

- Not used: the paintings as ground truth, the fitted judges, Claude judges' labels, "the user's eye" (sigma 0.75 and
  1.0 picked by looking at blurred images), the whole-picture verdicts on 4-9 pictures.
- Kept as a rule: free optimisation against a fitted judge fails; it found a new blind spot in each of four rounds.
- Kept as hypotheses: seams are seen at the pixel level while the tone of a surface is judged at a coarser scale; a
  hue change across a seam shows, brightness alone is faint, a slow lightness step hides; a clean colour of the right
  family matters (H5); local contrast matters more than the overall tone (H7).
- Kept as data for stage A's secondary check: the seam marks and the patch ratings.
- Its lesson for the method: choosing between two bad conversions is hard, slow and gives "both bad". This plan does
  not ask people to do it (see "What people are asked to do").

## Viewing conditions

The main scenario is an emulator on a modern LCD: HP Z27k G3 (27", 3840x2160), at the native resolution, at 65 cm
(measured). The zooms the user finds comfortable, by the width of the 256-pixel screen:

| zoom | screen width | mm per Spectrum pixel | pixels per degree |
|---|---|---|---|
| x2 | 15 cm | 0.59 | 19.4 |
| x3 | 22.5 cm | 0.88 | 12.9 |
| x4 | 30 cm | 1.17 | 9.7 |

research/pairs worked at ~26 pixels per degree (21" 2560x1440, 2x-3x, 65 cm). Its data, where used, are at that value.

What the numbers mean:

- A checkerboard is at 7-14 cycles per degree, a 2-pixel line pattern at 5-10. The eye's luminance sensitivity is
  near its peak there, and black/white dots are far above threshold: the dither is seen as texture at every
  comfortable zoom.
- A blob of 8x8 cells alternating, the clash's scale, is at 0.6-1.2 cycles per degree. Chromatic sensitivity is
  highest at low frequencies. Colour differences at the cell scale are where the eye sees colour best, and colour
  differences between single dots are where it sees colour worst.
- Above threshold, perceived contrast hardly depends on frequency (contrast constancy, Georgeson & Sullivan 1975).
  A threshold CSF used as a filter (classic DBS, S-CIELAB) under-states how visible the luminance texture is here.

Two scales, then. The eye forms a tone from the dots: a checker or a dense blue noise reads as a grey while the
pattern is seen. The user (2026-10-20): a viewer used to Spectrum pictures reads tone over a wider area, so the tone
scale can be wider than the converter's sigma of 1 pixel. At the same time single pixels carry the details, and they
matter a great deal. The tone scale is a parameter fitted in B3; the details are the model's pixel-scale part.

The user's observation on the current energy: with a narrow blur DBS packs the dots in the deepest shadows into
clusters or moves them out. Our reading: the unblurred dot term, multiplied by the lightness gain (about 7x at
black against mid grey), charges each lone bright dot on black nearly its whole contrast, so removing or packing the
dots is cheaper than spreading them. In the model below the pixel scale charges a difference from the source's
structure (an edge lost, an edge added), not each dot's contrast; the dots in a flat shadow pay at the tone scale
only. A0 checks it.

The second scenario is a CRT: the user's CRT television (not a Trinitron) on a ZX Spectrum 128 over RGB. RGB has no
PAL chroma blur. The CRT adds a beam profile (a blur of about a pixel), scanlines, bloom (a bright pixel on black wider
and brighter than its area) and its own gamma. The user's expectation, which we share: the ranking does not turn over
between the LCD and the CRT. The CRT softens the dots, so conversions look more alike, and the clash, 8 pixels across,
stays fully visible. So the CRT is a display model in front of the same eye model and a robustness check (C6), not a
second metric.

The CRT is set up by the user's chart: brightness and contrast set so that a non-bright grey and bright dots on black
look the same tone. That makes one dither match one flat colour by construction, so H2 is not tested on the CRT, and
the setting is written down as part of the display.

The palette is part of the display model. The default is Pulsar's, which is the converter's own (`ZXPalette`, levels
205 and 255; the user, 2026-10-20). Other emulators' palettes and measurements of real machines are options. The rankings
under different palettes are compared by the metric alone (A5).

## Two references

The user's idea (2026-10-20): a conversion with the Spectrum's pixel and palette but without the attribute
constraint. Each source picture gets two references:

- **R0, the original**, 256x192 as prepared.
- **R1, the free conversion**: any of the palette's colours in every pixel, the same pixel grid. The best halftone
  the palette and the pixel allow, with no clash. Made by the methods in "The free conversions"; people pick the best
  (B3), which is an easy choice among good pictures.

The metric splits along them:

- **The halftoning part** (T and the details): how near R1 is to R0. Classic colour halftoning, where the literature
  applies, judged on good pictures, and on several platforms' palettes, not the Spectrum's alone (H9).
- **The attribute part** (A): what the attribute constraint adds, the Spectrum conversion against R1. The clash
  stands out against R1 far more clearly than against the original, and the difference is of one kind.

H8 below states the split as a hypothesis; stage C tests the whole metric against R0.

## The model to test

A display model, then an eye model with two scales, then pooling:

- **Display**: the palette in linear light, the pixel geometry at the zoom, the LCD as square pixels; for the CRT the
  beam profile, scanlines, bloom and gamma. Units: pixels per degree and cd/m², so the rest does not know about the
  Spectrum.
- **T, tone and colour of surfaces**: source and render each through a low-pass at the tone scale, mixed in linear
  light, compared in a uniform colour space (CIELAB with CIEDE2000, or CAM16-UCS), as local means and local contrasts
  (H7).
- **A, structure at the pixel scale**: edges and steps the render has and the source does not (seams, the clash,
  false contours) and edges and details the source has and the render lost, by their contrast, masked by the local
  texture activity and by darkness. A dither in a flat area is not an edge.
- **Pooling**: a Minkowski sum over the picture, its exponent fitted (an artefact in one place weighs more than its
  area).

Hypotheses:

- **H1, zoom**: within x2-x4 people see the same artefacts and pick the same R1. Then the model's scales are in
  Spectrum pixels, calibrated at x3. Otherwise they are in degrees.
- **H2, mixing**: a dithered patch looks like the linear-light mean of its dots, except for pairs of distant hues
  (blue on yellow), which look like two colours. The exceptions are what `dot_contrast` charges now.
- **H3, the trade-off**: one weight between T and A makes one threshold on the model's local map separate the cells
  people mark from the rest, for both kinds of marks (B4).
- **H4, style**: the same metric works within the checker style (Halftone: checkers, dithering 0.25-0.5, no DBS) as
  within the DBS style.
- **H5, colour family**: an error that changes the colour's name (red to pink, the magenta cypress) costs more than
  its CIEDE2000 says. Tested in B6; a term only if it holds.
- **H6, CRT**: through a CRT display model, the metric's ranking of conversions agrees with the LCD ranking (Kendall's
  tau 0.8 or more).
- **H7, local contrast**: people forgive a slow shift of tone or colour over a surface more than a loss of local
  contrast or of a meaningful gradient. Tested in B6.
- **H8, the split**: the halftoning part calibrated on R1 against R0 and the attribute part calibrated on Spectrum
  conversions against R1, added, rank Spectrum conversions against R0 as people do (C).
- **H9, palettes**: one set of the halftoning part's parameters ranks the free conversions as people do on every
  palette below, from CGA's 4 colours to the 16 of the C64 and EGA. A part tuned to one palette would carry its
  quirks; one that holds across them is the eye's, and carries over to other images.
- **H10, DBS**: DBS on top of the best positional or error-diffusion conversion brings out the details and does not
  spoil the tone: people pick it over its start, and mark fewer "lost" cells on it, none more for tone (B3b).

## The free conversions

The palettes, all on one grid: 256x192 square pixels, the prepared source itself (the user, 2026-10-20: pixels are
square). The platforms' own resolutions are left out on purpose: H9 is about the palette, and one grid keeps
everything else the same.

| platform | colours |
|---|---|
| ZX Spectrum | 15 (Pulsar) |
| Commodore 64 | 16 (Pepto's, `platforms/c64`) |
| MSX1 (TMS9918) | 15 |
| EGA | 16, the default of the 64 |
| CGA | 4: palette 0 (black, green, red, brown) and palette 1 (black, cyan, magenta, white), each in low and high intensity |

The methods, all for an arbitrary palette, first without DBS:

- **Positional dither**: Knoll's pattern dithering (the Adobe method, described with variants in Yliluoma's
  "Arbitrary-palette positional dithering algorithm"). For each pixel, a list of N palette colours whose mean
  approaches the target, built by accumulating the error, sorted by luminance; a threshold map picks one. Two maps:
  **Bayer** 8x8 (regular) and a **blue-noise** mask (stochastic; the repo has blue-noise textures in
  `halftoning/noise`).
- **Error diffusion**: Floyd-Steinberg and a wider kernel (Jarvis or Stucki), serpentine, the nearest colour and the
  error in linear light, the error clamped so that it cannot run away on a small palette (CGA).
- In each, the colour space of the mixing and the distance (linear RGB, sRGB, CIELAB) is a variant. Which one people
  prefer is H2 on whole pictures.
- **Palette DBS**, a separate step (B3b): from the best of the above, toggles among the palette's colours and swaps
  with neighbours, under the eye model with T's parameters fitted in B3.

The Spectrum's and the C64 hires' attribute constraint is the same (2 colours an 8x8 cell), so the attribute part can
be checked on the C64 later with no new method.

## How the rounds run

- **Where**: in the cloud session, which publishes the voting page and reads the answers; renders are small
  (256x192) and run on its 4 CPUs. A batch that needs more (palette DBS sweeps) can run on the user's machine.
- **The surround**: the page's background around the images is black, fixed for every round and written down. The
  surround changes how light and how contrasty a picture looks, so it is held constant, not studied.
- **A voting page** published as a claude.ai Artifact with a shared store. Each batch's images are published to it,
  the user opens one link and answers by keys, and the answers reach this session directly: no command line, no files
  to send. Each answer is saved at once, and the time it took is saved with it.
- The page draws the images at whole device pixels: it reads the device pixel ratio and refuses to show a round when
  the browser's zoom breaks the mapping. Picking: the original and a 2x2 grid at x2, or the original and one
  candidate at x3 with keys 1-4 to switch it in place. Marking: the conversion and R1 at x3, Space held shows the
  original, the mouse paints cells, keys 1 and 2 set the kind.
- **Small batches first**, 5-10 items, 2-3 minutes. After each, the spread of the variants is adjusted: picks that are
  all obvious and fast mean the variants are too far apart; repeats that disagree and slow answers mean too near.
  These batches are pilots, marked so and left out of the confirmatory data. When the spread is right, the stage's
  analysis and decision rule are committed, and its confirmatory batches follow.

## What people are asked to do

Three kinds of task, each easy and each giving many bits a minute. None of them is a choice between two bad
conversions.

- **Pick the best of a few good pictures** (B3, C): the free conversions R1 of one picture, made by different methods
  or eye models, side by side with the original; the person picks the one nearest it. The options are good, the
  differences are of one kind.
- **Mark what is wrong** (B1, B4): a Spectrum conversion and its R1 side by side at x3; holding a key shows the
  original in R1's place. The person paints the cells where the conversion looks worse, with two keys: 1, something
  added (seams, blocks, noise, false contours); 2, something lost or wrong (colour, tone, detail). 2-3 minutes a
  picture. The paradigm of local visible-difference maps (Wolski et al., "Dataset and metrics for predicting local
  visible differences", ACM TOG 2018), asked here for what bothers the viewer rather than what is visible. A metric is
  scored by how well its local map predicts the marks.
- **Match** (B2): set a flat patch to look like a dithered one.

Whole conversions against the original, forced choice, only in C, and only between conversions that are each the
best of their method: the question there is which method, not which of two failures.

## The candidates

Evaluation metrics, existing, each run with the display model in front where it takes one:

- **ColorVideoVDP** (Mantiuk et al., SIGGRAPH 2024): CSF, contrast masking and pooling, colour, the display's
  geometry and luminance as inputs. The main candidate: it is built for supra-threshold differences at a given
  pixels per degree, and it gives a local map.
- **SSIMULACRA2** and **Butteraugli**: made for compression artefacts, block artefacts included; multi-scale, in an
  opponent space. The nearest to the clash among the established metrics, and they carry over to compression.
- **S-CIELAB** at the scenario's pixels per degree (Zhang & Wandell 1997), the classic baseline.
- **MS-SSIM** on lightness, **DISTS** (texture-tolerant: two dither textures of one tone are near), **LPIPS** as a
  control.
- The converter's own energy, the baseline; from research/pairs, its seam score (step 5).
- **J, a Claude judge**: a multimodal subagent as one more observer, tested in stage J.

Our own model (T + A + pooling) is built in stage A only if no candidate passes, from the parts that work best.

## The pictures

All earlier pictures (`tests/images`, research/pairs' training and held-out sets) are set aside: the user's eye is
used to them, and not all of them are good test pictures. They stay for the tests and for stage A's secondary data.

A new collection, before any conversion:

1. 40 sources, by the real inputs: artists' illustrations (fantasy, sci-fi, portraits) 15, artistic and processed
   photo portraits 10, posters, film and game frames, game title screens, AI-made images 8, landscapes 5, plain
   photographs 2. Licensed for publication (own works, public domain, CC, made for the purpose).
2. Each tagged by a checklist of what makes conversions hard: smooth gradients (skies), faces, deep shadows, saturated
   colours, fine detail, text. Each tag on at least 6 pictures; the collection is filled to that, not picked by how
   the converter does.
3. Prepared as the user prepares sources: cropped and composed to 256x192, cleaned. Frozen: the files' hashes are
   committed before anything is converted.
4. Split at random with a committed seed: 24 for training (B, J), 16 for validation (C). The scripts refuse to convert
   a validation picture before C.

## Stage A: existing metrics on data with known answers (no user time)

A0, data that do not depend on research/pairs:

- Sanity checks a metric must pass, on synthetic images: monotone in a tone error on grey and colour ramps; a dither
  shifted by one pixel almost free; on a flat 50% grey at x3 the visibility order checkerboard < blue noise < 2x2
  clusters; an attribute clash on a smooth surface charged more than the same mean colour error spread evenly; a
  false contour (banding) on a ramp charged more than a dither of it; on a dark ramp, dispersed dots charged less
  than an empty or a clustered shadow; R1 charged less than any Spectrum conversion of the same picture.
- Public subjective data sets: TID2013's colour quantisation with dither (type 22) and its block-wise distortions
  (type 15), KADID-10k's quantisation and block types. Another regime (mild distortions, normal viewing), so a check
  that a candidate is sane, not a choice between candidates.

A candidate that fails A0's checks is out.

**Amendment, 2026-10-20, before any human data on it.** The first A0 run (`a0-results.md`) failed every candidate on
most checks. The checks' expectations are ours, not measured, so they are put to people before any metric is dropped:

- The shift check compared a dither with itself shifted. The halftoning part compares a conversion with the original,
  where a shift costs nothing to any metric, so the check does not apply there. It applies to the attribute part,
  where the reference R1 is itself a dither; B4 deals with it.
- The banding, dark-ramp, grey-50% and clash checks are shown to the user as a pick round (`a0h-1`: the original and
  the stimuli of each check, 6 items and a repeat). A check is kept as a rule only where the user's pick agrees with
  its expectation and the repeat agrees; otherwise the expectation is wrong and the check is dropped or reversed.
  Then each metric is scored on the kept checks.

Then, as a secondary check only, two of research/pairs' data sets, at 26 pixels per degree, where the stimuli can be
made again exactly. One observer, 6 pictures, so they rank candidates only together with B:

| data | what it measures | score |
|---|---|---|
| `rounds/seams/marks.json`, 1342 marked seams on 12 renders | where the clash is seen | AUC of the metric's local map at the marked seams; top-k against the user's own repeat agreement (0.48-0.75) |
| `rounds/seams/patches-v3..v5`, `ranked-v1` | seam visibility without picture content | rank correlation with the ratings and the sorting |

- A1. The data files are copied into `research/metrics/data-pairs/` with their source commit (`5810855`); nothing
  imports research/pairs' code. A round whose renders cannot be made again exactly is dropped, not approximated.
- A2. The table: metric x data set, with bootstrap intervals over pictures.
- A3. Where the best candidate fails, which part fails: the seams (A), the tone (T) or the colour family (H5).
- A4. If no candidate passes A0 and ranks the seam marks better than the converter's energy, build T + A + pooling
  from the parts that do, with no more than 4 free parameters, fitted on half the pictures and scored on the other.
- A5. Palettes: the best candidate's rankings of the same render pairs under Pulsar, the converter's palette and one
  more emulator's; tau 0.9 or more means the palette is a setting, not a research question.
- A6. H6 by the metric alone: the same with the CRT display model.

The outcome: one or two models for stage B, and what they cannot see.

## Stage B: the model's parts (~80 minutes of the user's time, in batches)

The monitor at 65 cm, sRGB mode, the brightness fixed and written down, the room light the same. Pulsar palette.
Sessions of at most 10 minutes. The training pictures only. Each task's analysis and decision rule are committed
before its data.

- **B1, zoom (H1)**, ~2 x 8 minutes on two days: 4 conversions marked (as B4) at x2 one day and at x4 the other, and
  the R1 picks of 6 pictures at both zooms. H1 holds if the marks agree between the zooms as well as between repeats
  at one zoom (AUC of one against the other, a difference under 0.05), and the picks agree.
- **B2, mixing (H2)**, ~10 minutes: 12 patches (6 pairs, 2 dot levels; blue/yellow and red/green among them) at x2
  and x4. The user sets a flat patch beside each to match (lightness, a*, b* by keys), or marks it "cannot match".
  The matches against the linear-light mean; the "cannot match" set against `dot_contrast`.
- **B3, the halftoning part (H2, H9)**, ~20 minutes in batches: 12 pictures x the 6 palettes (ZX, C64, MSX1,
  EGA, CGA 0, CGA 1), 4 free conversions each, without DBS, chosen where the candidate metrics disagree most; pick the
  best against the original, 10% repeated. Each candidate metric is scored by how often its best is the person's,
  per palette and over all; then the tone scale and the detail weight of T are fitted to the picks, once for all
  palettes and once per palette (H9 holds if the shared fit is within the per-palette fits' intervals).
- **B3b, DBS (H10)**, ~10 minutes: on the same pictures and palettes, the best non-DBS conversion against palette DBS
  started from it, under T as fitted; pick the better, then mark the "lost" and tone cells on both for 6 of them. The
  ZX pick of B3 or B3b is that picture's R1 for B4.
- **B4, the attribute part (H3, H4)**, ~3 x 8 minutes: 9 pictures (6 in the DBS style, 3 in the checker style), each
  converted at three coherence values (low, the preset's, high), marked against R1 as above; 2 repeated. The
  candidates' local maps against the marks (AUC per picture); then the T/A weight such that one threshold separates
  both kinds of marks (H3).
- **B5, seam thresholds**, ~15 minutes, only if A3 and B4 find the A part weak: two surfaces with one mean colour and
  different pairs, each pair with its own dots as in a render (research/pairs' patches ran one dot pattern across the
  border, and its measures did not carry over to renders), the step varied by a staircase (QUEST), forced choice
  "which has a seam"; pair types x 3 lightness levels.
- **B6, colour family and local contrast (H5, H7)**, ~8 minutes: on 6 pictures, forced choices between two edits of
  one R1 with equal mean CIEDE2000 from it: a colour moved within its name against one moved across it (red to pink);
  a slow shift of tone over a surface against a loss of its local contrast. 24 choices, 4 repeated.

## Stage J: the Claude judge

The question: does a multimodal subagent see conversions as people do, the clash included, and is it better than
the best metric from the literature? research/pairs used such judges without testing them against people on
stimuli made for the question. Here J gets exactly B3's, B4's and C's stimuli and is scored as a metric is.

- Delivery: each picture a separate file, enlarged by whole pixels (x4, 1024x768), the cell coordinates as a ruler in
  the margin, not over the picture. First a check of what the model actually sees: rows of codes at small sizes, as
  research/pairs did, and single-pixel and single-cell changes it must find.
- Tasks: B3, pick the best of 4 R1 against the original; B4, list the cells where the conversion looks worse than R1,
  with the same two kinds; C, the forced choices. A short prompt, no description of the user's taste. Every item 6
  times, 3 in each order of sides; a cell's score is the number of runs that list it.
- Scores, against the same people: B4, AUC of J's cell scores at the people's marks, next to the best metric's AUC
  and next to the user's repeat ceiling; B3 and C, agreement with the people's picks and choices, next to the
  metrics' and next to the agreement between people. J against the best metric, too: if they agree with each other
  and with people, J adds nothing a metric does not give for free. Consistency: the order of sides, the repeats.

What a good J would give: labels with no user time for choosing the informative stimuli (which pairs and pictures to
show people), a second observer on a larger set of pictures (reported apart, never in place of people), a check of
the converter's changes during development, and words for what it sees. What it would never give: a target for
optimisation, labels for a fit, the final decision. If J is below the best metric on B4 and C, it stays a source of
descriptions only.

## Stage C: validation on the validation pictures

- C1. The 16 validation pictures, converted for the first time now.
- C2. Per picture: the current converter at its default preset; the converter with the chosen metric (the slow mode:
  the metric in the selection and in DBS); the best competitor at its best setting (`rivals.py` scripts them); on 4
  pictures the checker style both ways (H4); and R1, as an upper anchor.
- C3. Forced choice between two conversions, the original on the left, raw at x3 (the app's view), random sides, no
  tie key; "both fine" and "both bad" go into a second, optional key after the choice. 6 pairs repeated.
- C4. Observers: the user, all pairs (~25 minutes in three sessions); 2-3 others, 15 minutes each on the 30 most
  informative pairs (where the candidates disagree). Spectrum-scene viewers are the product's audience, so experienced
  viewers are the right observers; the user's results are reported apart from the others'.
- C5. Analysis: the picture is the unit. Wins of the new metric's conversion over the current, a sign test over
  pictures; a Bradley-Terry scale of the methods (Pérez-Ortiz & Mantiuk's `pwcmp`), bootstrap over pictures; each
  metric's and J's agreement with the people on C's pairs (H8).
- C6. CRT: the C pairs whose order the CRT display model turns over (A6), and 6 pairs it does not, shown on the
  Spectrum 128 and the CRT, the original on the computer beside it. A script writes a 128K snapshot (.sna, to the
  divMMC's SD card) with a small Z80 viewer: the pairs' screens in the free banks, each pair copied into banks 5 and 7,
  a key switches the shown screen (bit 3 of port 0x7FFD), keys 1 and 2 answer, and at the end the answers are printed
  on the screen to be typed into the voting page. The screens packed with ZX0 (Saukas), unpacked into the
  banks by its small Z80 decompressor: about 10-12 pairs a snapshot; dithered screens pack worse, so the script fills
  the free banks and counts.

The decision, fixed now: the new metric goes into the converter if its conversions win on at least 10 of the 16
pictures, the sign test p < 0.05, no category loses the majority, and its agreement with people on C's pairs is not
below the current energy's. Otherwise the result is the table and what failed, and the converter stays as it is.

## Stage D: a fast approximation

After C. The converter's decomposable energy, its parameters fitted so that it ranks converter outputs as the chosen
metric does; the target, rank agreement 0.9 on the renders of held-out pictures. The slow mode keeps the metric itself.

## Rules

- A colouring is judged only after a render through its project's pipeline. The keys of a round stay out of git until
  the user votes. No free optimisation of a colouring against a metric outside the converter's structure without a
  stage C check.
- Each stage's hypotheses, measures and decision rule are committed before its data (the commit is the
  pre-registration).
- The picture is the statistical unit, not the trial.
- People are not asked to choose between two bad conversions.
- Claude judges are scored in stage J before any other use; never labels for a fit.
- The code goes into `research/metrics/` (pytest does not collect it); the renders into `data/`, out of git.

## Open decisions

- The palette DBS (any palette colour a pixel) is new code; the converter's DBS takes two colours a pixel. A research
  script first, for B3b.
- The monitor's brightness in sRGB mode, written down at B0.
