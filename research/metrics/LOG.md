# Log: an eye model and a metric

Each session: what was decided or done, and the result. The plan is `PLAN.md`; what was taken from research/pairs and
why is `PAIRS-REVIEW.md`.

## 2026-10-20: the plan

The user and Claude (as the advisor) wrote the plan. Decisions by the user: the main scenario is an emulator on an
HP Z27k G3 at 65 cm, x2-x4; the CRT is a TV on a Spectrum 128 over RGB, set up by a chart, a robustness check; the
criterion is nearness to the original with the artefacts weighed as people weigh them, style left to the presets;
research/pairs taken critically; all earlier pictures set aside for a new collection of 40; the free conversion R1
as the second reference; the halftoning part on six palettes on one 256x192 grid of square pixels; positional and
error-diffusion dithers first, DBS as its own step; the C64 multicolour mode left out; answers through a voting page
in small batches; the CRT viewer as a 128K snapshot with ZX0-packed screens. The protocol is fixed at this commit.

Next, in order: stage A0 (the candidate metrics and the sanity checks, no user time) with the free-conversion
generators and the voting page; the user collects the 40 pictures meanwhile.

## 2026-10-20: stage A0, first run; tools

- `free.py`: the free conversions on 8 palettes (ZX at the app's 205/255 until Pulsar's values are known, C64
  Pepto, MSX1 TMS9918, EGA, CGA 0 and 1 low/high); Knoll with Bayer 8x8 and blue noise, Floyd-Steinberg and Jarvis
  (serpentine, the wanted value clamped to the palette's range), the mixing in linear RGB, sRGB or Lab; all under
  1 s a picture (numba). Knoll's n = 16 with an 8x8 map is the 4x4 pattern; the rounds use n = 64.
- `metrics.py`: ColorVideoVDP 0.5.7 at the ppd, SSIMULACRA2 and Butteraugli from libjxl 0.11.2 (built into /opt,
  rebuilt in a new container), S-CIELAB (our numpy, parameters not checked against the paper yet), MS-SSIM, DISTS,
  LPIPS (piq), the converter's eye baseline. Lower is nearer for all.
- `a0.py`, `a0-results.md`: no metric passes A0; SSIMULACRA2 3 of 6. All fail the shift and dark-ramp checks, all
  but the eye baseline prefer three flat bands to a dither of them. The checks' expectations are not measured, so
  PLAN's amendment puts them to the user first (round `a0h-1`).
- The voting page: https://claude.ai/artifact/HbWZvbKsQYAWcCVPpoy3Ey (db: `rounds/<id>`, `votes/<round>__<item>`).
  Rounds: `b3-pilot-1` (pilot: the 4 dither methods, linear, ZX and C64, on 5 set-aside pictures), `a0h-1`.

## 2026-10-20: round a0h-1, the user

- Banding: the dither over three flat bands, on both ramps (white 37 s, yellow 2 s). The expectation holds; every
  metric but the eye baseline fails a valid check.
- 50% grey: the checkerboard first. Holds as far as one pick tells (the order of the other two is not asked).
- Clash: on both surfaces a single pair over the whole surface, never the clash (grey: magenta/green; salmon:
  red/white, the repeat the same). Consistent with "the clash costs more than the pair picked".
- Dark ramp: not valid. The answer (0, none better) came while the page should have been blocked (browser zoom at
  2.19 device px a CSS px, a window too narrow for the grid): the keys still answered under the warning. Fixed (no
  answer while a warning hides the pictures); the item asked again as round `a0h-2`.
- The page waited for the write before showing the next item; now it shows it at once and writes behind it.
- The user: Pulsar's palette is the 205/255 levels the converter already uses.

## 2026-10-20: what clash means

The user: clash is a conspicuous seam where the pair changes and the original has no such structure; the first A0
"clash" check showed cell-scale colour noise. PLAN's Terms define both; the a0h-1 answers on it are kept under the
noise's name; amendment 2 adds clash checks (a seam on a smooth surface against a larger tone error, a seam off the
original's edge, the staircase), to be agreed with the user before they are made.

## 2026-10-20: round a0h-2; the clash stimuli

- Dark ramp (a0h-2): the dispersed dots over the empty shadow and the clusters, after 156 s. The expectation holds;
  every metric fails a check the user confirms.
- `a0clash.py` (amendment 2), round `a0h-3`: four gradients (sky, sunset, skin, field), each with one straight seam
  between the two pairs of least error against one pair over the whole surface (mean CIEDE2000 of the exact
  mixtures: seam 6.4-7.8, one pair 8.2-10.5; the sunset's seam shares red and is faint); an edge 3 px right of a
  cell border, the pair changing at the border to its left, to its right, or the edge drawn in the inks' pair; a
  diagonal edge as whole cells against drawn. Pair choice by CIEDE2000 of the linear-light exact mixture.

## 2026-10-20: round a0h-3 and the metrics against the user's picks

- Seams on gradients: one pair over the seam on all four (2.7-9.7 s), the sunset's faint shared-red seam included,
  though the seam's pairs are nearer in colour (mean CIEDE2000 of the mixtures 6.4-7.8 against 8.2-10.5). A straight
  seam the original lacks costs the user more than 2-3 CIEDE2000 over the whole surface.
- Edge 3 px right of a cell border: the pair changing at the border to its right (the sky 5 px into the sand) over
  the one to its left and over the drawn column, twice (the repeat had the options in the same order: rounds.py now
  shows a repeat in another order).
- Staircase: whole cells over the edge drawn in the inks' solid colours.
- The dark ramp's first answer was given again validly (Backspace, 2 device px a CSS px): dispersed, as in a0h-2.
- `a0score.py`, `a0-human.md`: each metric's nearest option against the user's pick on the 15 valid answers, at the
  user's viewing (x2, 19.4 ppd). Agreement: the eye baseline 7, cvvdp 6, SSIMULACRA2, Butteraugli max, LPIPS 5,
  S-CIELAB, DISTS 4, Butteraugli, MS-SSIM 3; chance about 6. Every metric picks the empty shadow and, but for the
  eye baseline, the bands; most pick the seam.
- A tone term alone, the linear-light blur then CIEDE2000 (sigma 1-4 px): 8/15 at sigma 1, 6/15 above. It gets the
  dark ramp, the banding, the 50% grey and the staircase right at every sigma, and every seam wrong, as it must:
  the seam's pairs are nearer in colour. Above sigma 1 it also prefers the cell-scale noise, which averages out.
- So stage A ends at A4: no candidate passes; our own model is built from a tone term of this kind and a seam term,
  the seam term's weight bounded below by the four gradients. 15 answers of one observer: a direction for the model,
  validated in B and C, not a fit.

## 2026-10-20: B3 pilot 1 (pilot data, set-aside pictures)

- The user's picks among Knoll-Bayer, Knoll-blue (n 64, multiplier 1), Floyd-Steinberg, Jarvis, all mixed in linear
  light: Floyd-Steinberg 6 of 9, Knoll-Bayer 2, Knoll-blue 1 (the repeat of autumn-c64, first answered
  Floyd-Steinberg), Jarvis never. 6-29 s an item: a real choice, the spread is right.
- The metrics' nearest option (`b3-pilot-1.md`, x2): our tone term (sigma 1 or 2) 6/9, the eye baseline 6/9;
  Butteraugli 3, SSIMULACRA2, LPIPS 2, cvvdp, Butteraugli max, DISTS 1, S-CIELAB, MS-SSIM 0 (chance 2.25). The
  literature metrics mostly pick Jarvis, the smoothest texture, which the user never picks.
- `metrics.py` gains `tone1`, `tone2` (the tone term at sigma 1 and 2 Spectrum pixels).
- B3 pilot 2 (`b3build.py`): per picture and palette 18 variants (Floyd-Steinberg, Jarvis in linear, sRGB, Lab;
  Knoll Bayer and blue noise in each space at multipliers 1 and 0.5), the four options the best by tone at sigma 1,
  2, 3, the eye baseline, cvvdp and SSIMULACRA2 in turn; 8 pictures over the six palettes (EGA, CGA 1 and 0 high,
  MSX1, ZX, C64) and a repeat.

## 2026-10-20: B3 pilot 2 (pilot data, set-aside pictures)

- Picks: Floyd-Steinberg linear 4 (andy ZX, landscape ZX, anubis EGA, the burning-hand CGA 0 repeat), Floyd-Steinberg
  Lab 2 (kotofey EGA, burning-hand CGA 0, which the repeat answered linear), Knoll-Bayer Lab at multiplier 0.5 2
  (golden-axe C64, rocket-rackoon CGA 1), Knoll-blue Lab at 1 (sunset MSX1). 9-46 s an item, the repeat split:
  options picked by the metrics' disagreement are close; for B3 proper, mix such items with ones of a clear spread.
- Agreement (`b3-pilot-2.md`, x2): tone at sigma 1 and the eye baseline 5/9; SSIMULACRA2 3; cvvdp, S-CIELAB, MS-SSIM,
  DISTS, tone at sigma 2 and 3 2; Butteraugli 1, LPIPS 0 (chance 2.25). Over both pilots: tone sigma 1 and the eye
  baseline 11/18, every literature metric 5/18 or less. Sigma 1 beats 2 and 3 here.

## 2026-10-20: zx.py, the marking page, B4 pilot 1

- `zx.py` (a subagent): the app's own pipeline (`ops.apply_preset(make_graph(), NEWEST)`, Exact mixture) on a
  prepared source, any node parameter settable; its default render equals the app's PNG and SCR export pixel for
  pixel. Defaults: coherence 6, DBS on (about 11 s a picture), the checker style = Halftone checker with dithering
  0.375 and DBS off.
- The voting page gains the marking task (B4): the conversion and R1 at x3, Space held shows the original, the mouse
  paints cells, key 1 something added, key 2 something lost or wrong, right button or Shift erases, Enter submits;
  an answer is the 768 cells as a string of 0/1/2.
- B4 pilot 1 (set-aside pictures, the default DBS style, R1 = Floyd-Steinberg linear on the ZX palette): jojo at
  coherence 0, autumn at 6 (the default), diver-sunset at 24.

## 2026-10-20: the collection's tags and manifest

`PICTURES.md`: the categories, ten difficulty tags, the manifest `pictures.csv`. A change to PLAN before any picture
is collected: film and game frames may be used for the research but not shown in the article (a `publish` field).
