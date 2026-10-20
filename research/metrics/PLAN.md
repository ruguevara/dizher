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

## The starting point: research/pairs, reviewed

`research/pairs/` (branches `feature/color-pair-selection-research-72ae52` and its earlier snapshot
`feature/gallant-wright-6dtgj6`, kept apart from `develop`) was an unsystematic search. `PAIRS-REVIEW.md` goes through
it: what does not hold and why, what holds, and the data this plan takes. In short:

- Not used: the paintings as ground truth, the fitted judges, Claude judges' labels, "the user's eye" (sigma 0.75 and
  1.0 picked by looking at blurred images), the whole-picture verdicts on 4-9 pictures.
- Kept as a rule: free optimisation against a fitted judge fails; it found a new blind spot in each of four rounds.
- Kept as hypotheses: seams are seen at the pixel level (every blur made the seam measures worse) while the tone of a
  surface is judged at a coarser scale, so there are two scales in one judgement; a hue change across a seam shows,
  brightness alone is faint, a slow lightness step hides; a clean colour of the right family matters (H5); local
  contrast matters more than the overall tone (H7).
- Kept as data for stage A's secondary check: the seam marks and the patch ratings.

The change in method: measure the **parts** of one model on controlled stimuli, where a minute of the user's time
gives a threshold or a match point, take the parts' form from the vision literature, and use whole pictures only to
validate, on pictures not used before.

## Viewing conditions

The main scenario is an emulator on a modern LCD: HP Z27k G3 (27", 3840x2160), at the native resolution, at arm's
length (taken as 65 cm until measured). The zooms the user finds comfortable, by the width of the 256-pixel screen:

| zoom | screen width | mm per Spectrum pixel | pixels per degree (65 cm) |
|---|---|---|---|
| x2 | 15 cm | 0.59 | 19.4 |
| x3 | 22.5 cm | 0.88 | 12.9 |
| x4 | 30 cm | 1.17 | 9.7 |

research/pairs worked at ~26 pixels per degree (21" 2560x1440, 2x-3x, 65 cm; `metrics.PPD`). Its labels, where
used, are at that value.

What the numbers mean:

- A checkerboard is at 7-14 cycles per degree, a 2-pixel line pattern at 5-10. The eye's luminance sensitivity is
  near its peak there, and black/white dots are far above threshold: the dither is seen as texture at every
  comfortable zoom.
- A blob of 8x8 cells alternating, the clash's scale, is at 0.6-1.2 cycles per degree. Chromatic sensitivity is
  highest at low frequencies. Colour differences at the cell scale are where the eye sees colour best, and colour
  differences between single dots are where it sees colour worst.
- Above threshold, perceived contrast hardly depends on frequency (contrast constancy, Georgeson & Sullivan 1975).
  A threshold CSF used as a filter (classic DBS, S-CIELAB) under-states how visible the luminance texture is in this
  range.

The eye still forms a tone from the dots: people read a checker as a middle grey at x4 while they see the checker.
A model with no low-pass at all makes DBS degenerate to the nearest colour per pixel, and says the dither is worse
than banding. The blur in the model stands for the scale at which the viewer judges the tone and colour of a surface;
the visibility of the texture, the seams and false contours is a separate quantity at the pixel scale.

The second scenario is a CRT over RGB (Sony PVM, JVC, a Trinitron TV). RGB has no PAL chroma blur. The CRT adds a
beam profile (a blur of about a pixel), scanlines, bloom (a bright pixel on black wider and brighter than its area)
and its own gamma. The user's expectation, which we share: the ranking does not turn over between the LCD and the
CRT. The CRT softens the dots, so conversions look more alike, and the clash, 8 pixels across, stays fully visible.
So the CRT is a display model in front of the same eye model and a robustness check, not a second metric (C6).

The palette is part of the display model. The default is Pulsar's; the converter's own (`ZXPalette`, levels 205 and
255) is checked against it. Other emulators' palettes and measurements of real machines are options. The rankings
under different palettes are compared by the metric alone (A5).

## The model to test

A display model, then an eye model with two scales, then pooling:

- **Display**: the palette in linear light, the pixel geometry at the zoom, the LCD as square pixels; for the CRT the
  beam profile, scanlines, bloom and gamma. Units: pixels per degree and cd/m², so the rest does not know about the
  Spectrum.
- **T, tone and colour of surfaces**: source and render each through a low-pass at the surface scale, mixed in linear
  light, compared in a uniform colour space (CIELAB with CIEDE2000, or CAM16-UCS). Local differences.
- **A, visible artefacts**: edges and steps at the pixel scale that the render has and the source does not (seams,
  the clash, false contours) and edges the source has and the render lost, by their contrast, masked by the local
  texture activity and by darkness.
- **Pooling**: a Minkowski sum over the picture, its exponent fitted (an artefact in one place weighs more than its
  area).

Hypotheses:

- **H1, zoom**: within x2-x4 people's preferences between two conversions do not depend on the zoom. Then the model's
  scales are in Spectrum pixels, calibrated at x3. If they do depend, the scales are in degrees.
- **H2, mixing**: a dithered patch looks like the linear-light mean of its dots, except for pairs of distant hues
  (blue on yellow), which look like two colours. The exceptions are what `dot_contrast` charges now.
- **H3, the trade-off**: one weight between T and A explains where along the converter's coherence sweep people put
  their best conversion, on most pictures.
- **H4, style**: the same metric ranks conversions within the checker style (Halftone: checkers, dithering 0.25-0.5,
  no DBS) as well as within the DBS style.
- **H5, colour family**: an error that changes the colour's name (red to pink, the magenta cypress) costs more than
  its CIEDE2000 says. Tested in B5; added as a term only if it holds.
- **H7, local contrast**: people forgive a slow shift of tone or colour over a surface more than a loss of local
  contrast or of a meaningful gradient (the user's priorities, 2026-10-02). If it holds, T compares local means and
  contrasts (as SSIM's terms do) rather than absolute colour, with its own weight for the slow part.
- **H6, CRT**: through a CRT display model, the metric's ranking of conversions agrees with the LCD ranking (Kendall's
  tau 0.8 or more).

## The candidates

Evaluation metrics, existing, each run with the display model in front where it takes one:

- **ColorVideoVDP** (Mantiuk et al., SIGGRAPH 2024): CSF, contrast masking and pooling, colour, the display's
  geometry and luminance as inputs. The main candidate: it is built for supra-threshold differences at a given
  pixels per degree.
- **SSIMULACRA2** and **Butteraugli**: made for compression artefacts, block artefacts included; multi-scale, in an
  opponent space. The nearest to the clash among the established metrics, and they carry over to compression.
- **S-CIELAB** at the scenario's pixels per degree (Zhang & Wandell 1997), the classic baseline.
- **MS-SSIM** on lightness, **DISTS** (texture-tolerant: two dither textures of one tone are near), **LPIPS** as a
  control.
- The converter's own energy, the baseline; from research/pairs, its seam score (step 5).

Our own model (T + A + pooling) is built in stage A only if no candidate passes, from the parts that work best.

## Stage A: existing metrics on data with known answers (no user time)

A0, first, data that do not depend on research/pairs:

- Sanity checks a metric must pass, on synthetic images: monotone in a tone error on grey and colour ramps; a dither
  shifted by one pixel almost free; on a flat 50% grey at x3 the visibility order checkerboard < blue noise < 2x2
  clusters; an attribute clash on a smooth surface charged more than the same mean colour error spread evenly; a
  false contour (banding) on a ramp charged more than a dither of it.
- Public subjective data sets: TID2013's colour quantisation with dither (type 22) and its block-wise distortions
  (type 15), KADID-10k's quantisation and block types. Another regime (mild distortions, normal viewing), so a check
  that a candidate is sane, not a choice between candidates.

A candidate that fails A0's checks is out.

Then, as a secondary check only, two of research/pairs' data sets, at 26 pixels per degree, where the stimuli can be
made again exactly. One observer, 6 pictures, so they rank candidates only together with B:

| data | what it measures | score |
|---|---|---|
| `rounds/seams/marks.json`, 1342 marked seams on 12 renders | where the clash is seen | AUC of the metric's local map at the marked seams; top-k against the user's own repeat agreement (0.48-0.75) |
| `rounds/seams/patches-v3..v5`, `ranked-v1` | seam visibility without picture content | rank correlation with the ratings and the sorting |

- A1. The few data files needed are copied into `research/metrics/data-pairs/` with their source commit; nothing
  imports research/pairs' code. A round whose renders cannot be made again exactly (the pipeline changed since) is
  dropped, not approximated; the key files tell which.
- A2. The table: metric x data set, with bootstrap intervals over pictures.
- A3. Where the best candidate fails, which part fails: the seams (A), the tone (T) or the colour family (H5).
- A4. If no candidate passes A0 and ranks the seam marks better than the converter's energy,
  build T + A + pooling from the parts that do, with no more than 4 free parameters, fitted on half the pictures and scored on the
  other half.
- A5. Palettes: the best candidate's rankings of the same render pairs under Pulsar, the converter's palette and one
  more emulator's; tau 0.9 or more means the palette is a setting, not a research question.
- A6. H6 by the metric alone: the same with the CRT display model.

The outcome: one or two models for stage B, and what they cannot see.

## Stage B: the model's parts on controlled stimuli (~50 minutes of the user's time)

The new monitor, at the measured distance, sRGB mode, the brightness fixed and written down, the room light the same.
Pulsar palette. Sessions of at most 10 minutes. Each task's analysis and decision rule are committed before its data.

- **B1, zoom (H1)**, ~2 x 7 minutes on two days: 30 pairs of whole conversions of the training pictures, where the
  candidates disagree most, forced choice, original on the left, at x2 one day and x4
  the other, 6 pairs repeated within each session. H1 holds if the agreement between the zooms is not lower than the
  agreement between repeats at one zoom (a difference under 10 points).
- **B2, mixing (H2)**, ~10 minutes: 12 patches (6 pairs, 2 dot levels; blue/yellow and red/green among them) at x2
  and x4. The user sets a flat patch beside each to match (lightness, a*, b* by keys), or marks it "cannot match".
  The matches against the linear-light mean; the "cannot match" set against `dot_contrast`.
- **B3, the trade-off (H3, H4)**, ~2 x 8 minutes: 8 training pictures (5 in the DBS style, 3 in the checker style),
  each rendered along the coherence sweep (7 points, coherence 0 to the maximum, the rest of the preset fixed). The
  user's best point by a bisection with forced choices between points, 4-5 choices a picture; repeated on another
  day. Each candidate's own best point along each sweep is compared with the user's: the error in sweep steps. No
  fitting in this test. Then the model's one T/A weight is fitted to these points.
- **B4, seam thresholds**, ~15 minutes, only if A3 finds the A part weak: two surfaces with one mean colour and
  different pairs, each pair with its own dots as in a render (research/pairs' patches ran one dot pattern across the
  border, and its measures did not carry over to renders), the step varied by a staircase (QUEST), forced choice "which has a seam"; pair types x 3
  lightness levels. research/pairs' patch design is a draft for it.

- **B5, colour family and local contrast (H5, H7)**, ~8 minutes: on 6 training pictures, forced choices between two
  edits of one conversion with equal mean CIEDE2000 from it: a colour moved within its name against one moved across
  it (red to pink); a slow shift of tone over a surface against a loss of its local contrast. 24 choices, 4 repeated.

## Stage C: validation on new pictures

- C1. 16 new pictures, prepared as the user prepares sources (cropped and composed to 256x192, cleaned), in the
  proportions of the real inputs: artists' illustrations (fantasy, sci-fi, portraits) 6, artistic photo portraits and
  processed portraits 4, posters and film, game and AI-made images 3, landscapes 2, a plain photograph 1. Licensed for
  publication (own works, public domain, CC, made for the purpose). Not looked at through the converter before C.
- C2. Per picture: the current converter at its default preset; the converter with the chosen metric (the slow mode:
  the metric in the selection and in DBS); the best competitor at its best setting (from `rivals`); and on 4 pictures
  the checker style both ways (H4).
- C3. Forced choice, the original on the left, the two conversions raw at x3 (the app's view), random sides, `dp.py
  vote`'s page with the tie keys off. "Both fine" and "both bad" go into a second, optional key after the choice.
  6 pairs repeated.
- C4. Observers: the user, all pairs (~25 minutes in three sessions); 2-3 others, 15 minutes each on the 30 most
  informative pairs (where the candidates disagree). The user's results are reported apart from the others'.
- C5. Analysis: the picture is the unit. Wins of the new metric's conversion over the current, a sign test over
  pictures; a Bradley-Terry scale of the methods (Pérez-Ortiz & Mantiuk's `pwcmp`), bootstrap over pictures; each
  metric's agreement with the people on C's pairs.
- C6. CRT, if one is at hand: only the C pairs whose order the CRT display model turns over (A6), shown on the CRT.

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
- research/pairs' held-out pictures (andy, sunset, golden-axe, david, burning-hand) were used for its final checks:
  here they are training pictures. C's pictures are new.
- The picture is the statistical unit, not the trial.
- Claude judges only as a prefilter of easy cases, never as labels for a fit.
- The code goes into `research/metrics/` (pytest does not collect it); the renders into `data/`, out of git.

## Open decisions

- The viewing distance and the monitor's brightness, measured (B0).
- Whether a CRT is at hand for C6.
