# Plan: pair selection

## The aim and the criterion

A conversion that the user sees as better. The priorities (the user, 2026-10-02), from the highest:

1. **Seams**, the highest: clash and blockiness at the cell borders.
2. **Colour and tone.** The colour can be approximate. An exact hue is never necessary, but grey dither everywhere is
   not acceptable. Flat fills are not always necessary; frequently the picture needs gradients with dither. An exact
   overall tone is not necessary either. The important things are local contrast, and gradients where they exist
   globally and have a meaning.
3. **Form in details.** DBS dithering gives it already, but DBS needs a correct luma–chroma scale, possibly different
   for each surface. In some areas DBS places the pixels; in other areas keep the dither pattern and do not use DBS.
4. **Texture**: dithering and DBS by surface; some areas do not need DBS.

Items 3 and 4 are not pair selection, but dithering (Halftoner, DBS). Colorimetric closeness to the source is not an
aim (`FINDINGS.md`). Done, by the project's focus (the coach): the conversion is visibly better than the known
converters (ZX-Paintbrush, image2zx, img2spec) on three reference pictures, in a blind test by the user.

## The state (2026-10-06)

Step 6, the artists (below; README 19): the zxart.ee top 100 as priors, measured as step 5 measures seams. The
artists keep a shared colour on 87% of pair changes and hide it at the border's pixels; the current selection
recovers 82% of their cells through the user's eye and loses the shared colour first as the blur grows, which the
seam score does not see. Step 5, the seams (README 14-18): a seam score from the user's own data (painted seams,
rated patches, a sorting) replaces the coherence term's attributes in a research prototype of the selection. Better
blind on 5 of 6 training pictures, 1-1 held out (3 both bad at old settings). The user: more research before the app
(step 5's "Next").

The branch is `research/zxart-top-100`, from `feature/gallant-wright-6dtgj6` rebased on `develop` f4d422e; it has no
changes of its own outside `research/pairs` but the unused `tests/images/pairs` removed. Sessions A, B, C, E and steps 1 and 2 are closed
(`LOG.md`); step 3, the galleries, done. In the app: Select pairs: surface dE, one pair per surface, off by default
(PR #14); the selection energy's seam terms by matmul, a conversion without DBS ~2.4x faster (PR #15); new defaults per
method (PR #16); the dots' own weights, chroma on Halftone, noise on Optimise (step 2, PR #17).

Step 3b, a pairwise mode (`duel.py`), the user, 2026-10-04: no use, skipped. The user: autumn's gallery best is good
enough, and it counts for the criterion as it is (not blind against the competitors). Next: more galleries (step 3),
then step 4 (the user: the data is too thin to fix knobs yet).

Against the competitors (step 1, README 12; `rivals.py`, `rounds/rivals`): not worse anywhere. The criterion holds on
two of the three necessary pictures: jojo with surface dE 20, rocket-rackoon with Exact mixture, each with its own
style. autumn and diver-sunset: "both bad". The third: autumn by its gallery best (the user, 2026-10-04).

The steps (the user, 2026-10-02): for a time there are more knobs, so that later there are fewer, each with a meaning.

- Step 2: the dithering gets its own weights.
- Step 3: a gallery per picture (first autumn and diver-sunset, for the seams).
- Step 4: fewer knobs from the gallery data.
- Step 5 (the user, 2026-10-06), now: a seam score from the pixels, in place of the coherence term's attributes.

Seams in the algorithm and adaptive binding come after, from the gallery results and step 5.

## Step 2: the dithering gets its own weights (branch `feature/dbs-weights`), done 2026-10-02

- Before, the weights were shared. The chroma weight (Metric) scales the weighted opponent space
  (`Converter.opponent`). Both the selection (the candidate fit, the selection energy) and the dithering read it (the
  target of Halftone and DBS is the projection on the pair; the DBS error). luma_noise and chroma_noise of Select
  pairs also went to DBS. Flare and the eye model are also shared; structure is only on DBS.
- Now the dots have their own weights: chroma on Halftone (the halftone target and, through it, the weight of the DBS
  error), luma_noise and chroma_noise on Optimise. Metric and Select pairs stay with the selection. Flare and the eye
  stay shared: they model the viewer, not a taste. At first chroma was on Optimise, and Halftone without DBS stayed on
  the Metric weight. But this weight changes 0.4–4% of the halftone pixels (the most on autumn and diver-sunset), and
  the areas without DBS are in priorities 3–4. Thus chroma moved to Halftone: one weight for the dots, not two.
- Gate: the default render is the same. New projects get the preset values (`apply_preset` sets both Halftone and
  Optimise). Projects saved before this step get the values of their Metric and Select pairs. (`legacy` in mokit is a
  constant, but here the value of the adjacent node is necessary.) A test covers the two cases; pytest and
  `tests/test_ui.py` are green.
- The branch is from `develop`, merged here (the user, 2026-10-02): the gallery needs the knobs. In `develop` as PR #17
  (2026-10-05), reconciled with #16; the research branch dropped its commits in the rebase.
- Done: an older project gets the weights through `meta(legacy=Like(node, field))` in mokit. In the research,
  `common.Project.changed` moves the dots' weights together with the Metric and Select pairs weights, if a script does
  not set them. All earlier rounds give the same renders: rocket-rackoon at the `tune` base and with a shift of chroma
  and noise, and diver-sunset with its own. The render before and after the merge is the same byte for byte.

## Step 3: a gallery per picture (`gallery.py`, ~5 min of the user's time per picture)

- The knobs (13), each in the range of its slider, scaled to 0..1:
  - Metric: chroma, flare;
  - Select pairs: coherence, edge, luma noise, chroma noise, surface dE;
  - Halftone: chroma;
  - Optimise: luma noise, chroma noise, structure (DBS on);
  - the eye: luma and chroma blur.

  Set the dots' weights explicitly. If not, `common.Project.changed` moves them with the selection weights.
- The start is the project's settings. The user (2026-10-02): less chroma on Halftone sometimes makes the blockiness
  more even.
- A round: a plane through the current best point along two directions, and a 4×4 grid of renders through the
  project's full pipeline (`Project.render`). The user clicks the best render; the centre moves to it, and the step
  becomes smaller. At first the directions are random. Use a plane from Bayesian optimisation (Sequential Gallery,
  Koyama et al. 2020) only if the simple method does not converge in ~15 rounds. The page is like `vote.html`; each
  click goes at once to the json.
- First measure the time of a round: 16 renders with DBS. The estimate: a conversion with DBS takes ~8.6 s (README 7);
  on 12 cores (6 performance cores) a round takes ~30 s. If it takes longer, use a 3×3 grid. Measure on a quiet
  machine. On 2026-10-02 `arc mount` and the VS Code search (ripgrep) kept the load at ~58, and DBS was 3–8 times
  slower. Halftone chroma 0 adds one DBS pass (7 instead of 6).
- Done 2026-10-02, `gallery.py`:
  - the renders run in parallel, in 16 processes;
  - the grid is in random order, so the position of the centre tells nothing;
  - the step is ×0.7 if the user picks the centre, ×1.25 for the far edge, else the same;
  - 2026-10-03, at the user's request: first the user discards the bad renders. After each discard the remaining
    renders are laid out again, as large as a window of any proportions holds them (the zoom is an integer, not below
    min zoom), the last few side by side (2026-10-04: left dark in place, the last two could end up diagonal). Then
    the user picks the best, or the last remaining render wins. The json keeps the discard order;
  - 2026-10-04: when all directions are bad, the round is drawn again from the same centre, with new directions or
    wider (step ×1.5). The json keeps the shown grid with a mark;
  - Backspace: back;
  - `--no-dbs`: Optimise off and its knobs removed (the user permitted it);
  - 2026-10-05, at the user's request: `--side N`, an N×N grid over the 4×4's span; the user's mode is 3×3 (steps of
    0.3, not 0.2). The step shrinks when the pick is within a quarter of the reach from the centre, grows on the far
    edge: on the 3×3 the centre shrinks it, the other 8 grow it (steady when the centre wins ~38% of rounds); the 4×4
    as before.

  The centre of the grid is byte for byte the same as the project's render. Measured at a load of 60–160 (arc, VS
  Code ripgrep): one autumn render with DBS 118 s, without DBS 12 s; a round without DBS 55–108 s. On a quiet machine
  (2026-10-04) a conversion without DBS 1.0–1.2 s, 0.4–0.5 s after PR #15 (the seam terms by matmul).
- The order: autumn and diver-sunset (seams, priority 1), then jojo and rocket-rackoon, then the other training
  pictures (`tune.TRAIN`). Do not touch the held-out pictures.
- The log: each shown grid with its points and the pick (the pick is better than the other 15: these are
  comparisons), and a result per picture.
- The result (not a gate): the best from the gallery on autumn and diver-sunset goes into `rivals.py user` against the
  three competitors. If it wins, the settings found the third picture of the coach's criterion. If not, the problem is
  in the algorithm: see "Seams in the algorithm" below.
- So far (2026-10-04, README 13; `gallery.py report`): 5 galleries without DBS (autumn 11 picks, rocket-rackoon 8,
  diver-sunset 7, anubis 3, david 1). Select chroma noise went down on all four that count; luma noise up on
  diver-sunset, down (to its floor 0) on rocket-rackoon and autumn; coherence from 2 to ~5 on anubis and diver-sunset,
  autumn stayed near 6; the preferred directions differ by picture. No two knobs change the render alike (mean cosine
  at most 0.48, luma noise against luma blur). Thin: each round spans 2 of 10 directions, ~10 rounds a picture. The
  best on autumn and diver-sunset is not yet against the competitors. The user (2026-10-04): autumn's gallery best
  is good enough.

## Step 3b: a pairwise search for one picture's knobs, skipped 2026-10-04

`duel.py`: the user picks the better of two renders, a GP with a probit likelihood on the comparisons (the gallery's
too) proposes the next pair. The user, after duels on autumn (13), jojo (23, then 44) and RC1 (12): no use. The script
and `rounds/duel` stay as the record; the method and its checks in `LOG.md`.

## Step 4: fewer knobs from the gallery data

- First the Metric method (2026-10-05): a switch, and each method sets its own preset, so not a gallery knob; anubis
  and rocket-rackoon run Halftoned, the other four Exact mixture, and the two were never compared by the user.
  `methods.py`: both at their presets, without DBS, one blind sheet per training picture. If one is never worse, the
  galleries fix it (anubis and rocket-rackoon start again on it); if the pictures split, `--method` and a gallery per
  method, the bests against each other blind.
  The user (2026-10-05), one sheet each: Halftoned on RC1, anubis, diver-sunset; Exact mixture on autumn and
  rocket-rackoon; jojo both bad. Split, so the method is a setting by the picture. The user: the gallery picks it.
  A new gallery's first round mixes the methods: a grid around each one's start (the project's own method at its
  settings, the other at its preset; 18 renders on the 3×3); the pick's method is the gallery's from then on, a
  redraw mixes them again. The report leaves the mixed round out (it compares methods, not knobs). RC1, diver-sunset
  and rocket-rackoon start again with it, their old galleries kept as NAME-method.json (and their renders) for their
  method; anubis and autumn as they are.
- Then more galleries (the user, 2026-10-04): without DBS, as the five so far, so that they pool; every training
  picture (`tune.TRAIN`) to ~10 picks, 3×3 (the user): jojo new, RC1, diver-sunset and rocket-rackoon anew (the
  method in the first round), anubis 3 -> 10; autumn is done (11). Then `gallery.py report` over all six.
- Done 2026-10-05: a gallery per training picture, 11-14 picks, 3×3, the method by the first round (Exact mixture on
  RC1, autumn, jojo, rocket-rackoon; Halftoned on anubis, diver-sunset). The user: more or less good.
- New defaults per method (the user, 2026-10-05), `gallery.py defaults`: a knob moves to the median of its last picks
  over the method's pictures when the move agrees with the net direction its galleries prefer (5-95% excluding 0);
  else it stays (the picks drift). Robust over 4 bootstrap seeds: Exact mixture chroma noise 0.02 -> 0, eye luma blur
  1.4 -> 0.9; Halftoned chroma noise 0.05 -> 0, eye chroma blur 1.4 -> 1.6. On some seeds only: Exact mixture
  Halftone chroma 2 -> 1.8 (3 of 4), edge 0.1 -> 0.06 (2), Metric chroma 2 -> 1.8 (1); Halftoned flare 0.1 -> 0.12 (1).
  All without DBS; Optimise's weights have no data. In the app (the user, 2026-10-05): the four robust ones, PR #16
  into `develop` (1b0f386); a preset sets the Eye model too, and Optimise's noise as Select pairs' (PR #17), so DBS's
  chroma noise is 0 too.
- The data so far (README 13, 2026-10-04): no preference on any picture for Metric flare and Eye chroma blur; Select
  edge only on anubis (3 picks), Eye luma blur only on autumn (lower; its effect the closest to luma noise's, cosine
  -0.48); chroma noise down on all four; luma noise, coherence, Metric and Halftone chroma by the picture; surface dE
  flat in the galleries, but by the picture in README 11-12. No merges: the preferences point different ways by
  picture, no two knobs change the render alike. So far: 10 -> 6 knobs, flare, edge and the two eye blurs fixed (flare
  and the eye model the viewer, as in step 2). The Optimise knobs have no data. Before the gate, a cheap check: each
  gallery's best against it with the fixed knobs at their defaults, blind.
- From the log:
  - fix the knobs that the user does not move, or whose move does not change the pick;
  - merge the knobs that move together into one (the main directions of the picked moves across pictures);
  - keep the knobs that depend on the picture, or derive them from the picture (as the surface size from the colour
    spread, `IDEAS.md`).
- The result: a few knobs by the priorities of the aim (seams, colour and tone, form, texture). Each knob is a
  direction in the current knobs. The old knobs go to the advanced settings, or we delete them. A PR into `develop`.
- Gate: on two or three held-out pictures the best by the new knobs is not worse than the best by all knobs (blind,
  the user's vote).

## Step 5: which seams the user sees (`seams.py`, ~20 min of the user's time)

- Why (the user, 2026-10-06): the coherence term prices a seam by the attributes alone, V, the CIELUV distance of
  the two cells' papers plus their inks (`energy.pair_dissimilarity`). The user sees a seam by its pixels, and can
  tell steadily where one shows more or less: red on black against magenta on black, with mainly black pixels along
  the border, is no seam at all; ~50% noise along a border hides it; the colours, the brightness, the noise and the
  pixels along the border decide. First the measure, then it replaces V in the selection.
- The data: the user paints the seams seen (drag along a border; shift: strong) on 12 whole renders, DBS on, 3x as
  in the app: each training picture at its gallery best and the same at coherence 0 (more pair changes), in a blind
  order, the two of a picture apart; then 2 of the first 6 again, for the user's own agreement. Hold S for the source:
  only the seams the source does not have. Unmarked counts as no seam. The key (`rounds/seams/key.json`) goes into
  git after the painting.
- The candidates, each per seam, lightness and colour apart, their mix chosen with the picture left out:
  - A: the attributes, V, and V times the target's edge weight (the coherence term's own);
  - B: the pixel steps across the border beyond the target's, mean along it (`seam_excess` per seam): raw, through
    the user's eye (`views.seen2`), at a blur of 2;
  - C: a line filter along the border (the steps' mean along it, where noise cancels) beyond the target's, less the
    same filter over the two cells' own columns (a seam shows when it steps more than the dots around it): raw and
    through the eye;
  - D: the 8 x 4 strips on the two sides, their mean and spread per channel.
- The score: AUC per render, the marked seams above the rest; first on the seams where the pair changes (two thirds
  keep it, and no measure sees a seam there, so all seams flatter every measure, A too), also on flat target only,
  and strong against the rest. The ceiling: the repeats' agreement.
- Done 2026-10-06 without the user (LOG):
  - `seams.py check`, the user's cases, synthetic: the measures through the eye (B and C) fail red | magenta dots
    with black along the border: the blur carries the dots' colour onto it. Raw B, C and D pass, A passes but by a
    small margin.
  - The painted segments (229, README 1), the share where a measure rates the painting below Select pairs (L + ab):
    C through the eye 0.73, D 0.72, B at a blur of 2 0.69, against `seam_excess` 0.65; raw B and C 0.61-0.63.
  - The renders (`seams.py make`, 27 s) and the page (`seams.py paint`), checked headless; `seams.py report` on a
    simulated painter.
- The user's painting (2026-10-06, README 14): 1338 of 1342 marks on pair changes; B raw, the colour step, AUC 0.79
  against A's 0.62 (with the edge weight 0.51), above A on 11 of 12 renders; top k 0.42 against the repeats'
  0.48-0.75. Half the gate: clearly above A, not near the repeats. Lightness adds nothing on the renders, where its
  steps are mostly the dots'; whether the user does not see lightness seams or the measure misses them, the renders
  cannot tell.
- Step 5b (the user, 2026-10-06): seam visibility itself, on synthetic patches without picture content
  (`patches.py`, ~7 min of the user's time). A patch of 8 x 6 cells, the left half one pair, the right another, one
  threshold map over both (the app's blue noise), so the dots run on across the border, nested where the levels
  differ; the level varies slowly (sd 0.08, sigma 6 px), the same across the border; the user rates the middle seam
  0-3.
  - The first version changed the colours at one level, and the user stopped it after 26 ratings: magenta is 34%
    brighter than red, so the lightness gave every seam away (`patches-v1.json`). A pair switch on a smooth surface
    is near the same lightness, both pairs dithered to one target. Now the colour changes come at the same mean
    luminance (the right side's level set for it), and the user moves the right side's lightness (up/down) to where
    the seam is least visible and rates it there: the minimally distinct border (Boynton and Kaiser), so the
    perceived lightness, not the CIE one, is matched, and the move says how far they differ.
  - v2 (that, with Bayer and blue noise on a uniform level), the user after 3: a regular texture and a uniform field
    show any break, which no render has (DBS breaks the Bayer pattern). So blue noise only, and the slow variation
    of the level (`patches-v2.json`).
  - What changes: the ink's hue on black paper (near r-m, g-c, y-w; far r-g, its dots sparser), its brightness (at
    30, 50 and 75%), both (50%); gradients of the level (black-white, red-yellow; the paper's or the ink's hue, brightness);
    lightness alone, a step of the level of one pair (k/w, k/r, k/g; 0.06, 0.12, 0.25); the first version's hue
    changes at one level; the user's black border; none. 51 patches and 10 repeats, random order.
  - What it answers that the renders cannot: whether the user sees lightness seams (on the renders the lightness
    steps are mostly the dots'), how much the dots hide a colour change (the user: ~50% noise hides it), and how
    brightness counts.
  - v3 (blue noise, the slow variation), 51 patches and 10 repeats, done (README 15): hue at the same lightness ~2,
    brightness alone 1, lightness steps up to 0.12 0; E 0.75 on the patches, B raw 0.14; on the renders B raw 0.79,
    E 0.71. The dots run on across the patches' border and break on a render's.
  - v4 (the user, 2026-10-06): a gradient of random direction and strength on every patch (the ramps strong), each
    change twice, and the dots breaking at the border (the right half's from elsewhere in the tile) on the near
    hues, brightness and no change at 50%. 124 patches and 10 repeats, ~15 min; each instance in turn, so a stop
    halfway covers all.
  - v4 done (README 15): the dots breaking at the border makes no seam; the attributes 0.83, E 0.81, B raw 0.20 on the
    patches. The user: inks on black only, near hues, no blue; the renders are full of the rest.
  - v5 (2026-10-06): a paper's change under bright white (C-G, C-B, Y-C, Y-G, M-R, dim c-g), blue (b against r and m,
    B against b, blue paper under c, y, w; B/C against B/G), the same mean colour from other dots (b/y, r/c, m/g
    against k/w and each other; b/r against k/m, b/g against k/c, r/g against k/y), lightness on C/W, the v3-v4
    anchors r-m and r-g. 25 changes twice, 10 repeats, ~7 min.
  - A new candidate, E: the colour change the pair switch makes at the pixels along the border, the dots the same.
    Zero for the same pair and for paper on both sides; the form a selection term can take, from each pair's dots.
    B, the best on the renders, is not zero for the same pair: it charges a dotted pair's own steps at a border.
- Step 5c, done 2026-10-06 (`seamfit.py`, README 16): one score from all the data. Two terms: 0.075 x E_L (the
  lightness change of the border's pixels when the pair switches, the dots the same) + 0.055 x M_ab (the colour
  difference of the cells' mean colours, beyond the source's), per dE. Held out: renders 0.68, rounds 0.79-0.88;
  the coherence term 0.51. All the measures: 0.73 on the renders, four more terms.
- The user on the ranked sheets (`seamfit.py ranked`: per L* 15-92 every two pairs that reach it, 16 evenly along
  the score's ranking): not quite. The user sorts the same patches (`seamfit.py sort`, blind, starting in the score's
  order, a shuffle on offer); the orders become data for the fit (a ranking within each lightness).
- Done (README 17): the sorting, the user's rule (a solid-looking dim cell next to bright dots, where it is light),
  and the score with three terms: 0.054 x M_ab + 0.126 x Y x E_L + 1.68 x Y x the solidity step (Y, the cells' mean
  L* / 100). Held out: renders 0.72, rounds 0.78-0.87, the sorting 0.85. Each term from two pairs' dots and means.
- Step 5d, the prototype (`selection.py`, 2026-10-06): the score as the DP's pairwise table, per border and two
  pairs from each pair's halftone in each cell, times coherence x scale (0.0048: the old term's sum over the current
  selections' pair changes over the score's), the coherence term off. My copy of the selection with the old term
  gives the current labels on all six pictures. The score at the halftone against after DBS: Spearman 0.85-0.96 on
  the changed seams. 35-184 cells changed a picture. The user's blind round, one sheet a training picture (`selection.py
  user`, `dp.py vote rounds/seams/selection data/seams/selection/user`).
- The user's round (README 18): the score better on 5 of 6, both fine 1, the current 0. The gate: the held-out
  pictures (fitted partly on the training pictures' renders), 5 sheets and 5 repeats with the sides swapped. If it
  holds: the term in `src` on a `feature/*` branch from `develop` (in `SelectionEnergy`, from `realized` and
  `bitmaps`, `coherence` its weight), with tests (zero for the same pair, the user's cases), a PR into `develop`.
- Held out (README 18): both bad on four, the current better on david, every repeat the same; the gate fails at x1.
  The user: the score's contribution is low. The strength by the user on the training pictures (`selection.py
  strength`, x1-x16 side by side), then the held-out check again at that strength; the held-out pictures stay out of
  the choice.
- The user's strengths (README 18): times each picture's coherence 8-26, so the weight fixed at 16, the coherence
  aside. The held-out check at it (`held-fixed`). In `src` the seam term's weight would then not be the method
  presets' coherence (2 and 6), or both presets get one value.
- Held out at the fixed weight: 1 better (burning-hand), 1 worse (david, a tone from the brightness switch), 3 both
  bad (old settings). The gate in full is not met; the user decides the next step (an option off by default, as
  surface dE, or the default).
- **Status (2026-10-06): the user chose more research before the app.** What a next session needs:
  - The score: `seamfit.SEAM`, three terms, README 17; in the selection `selection.table` (per border, each two
    pairs, from `realized` and `bitmaps`), weight fixed at `selection.WEIGHT` = 16 x 0.0048 per score unit, the
    picture's coherence aside, the coherence term off (`select(c, scale, weight)`).
  - Results: README 14-18, FINDINGS "Seams". Training pictures blind 5 better, 1 both fine, 0 worse (at the first,
    weaker scale); held out at the fixed weight 1 better (burning-hand), 1 worse (david), 3 both bad.
  - The data, all in `rounds/seams/`: the painted seams (`key.json`, `marks.json`), the patch rounds (`patches-vN`,
    v3-v5 used), the sorting (`ranked-v1`), the selection rounds (`selection/`, `selection/held`, `held-fixed`, each
    `user-key.json` and `user-verdicts.json`). Renders in `data/` are rebuilt by `seams.py make`, `patches.py make`
    (keys hold the specs), `seamfit.py ranked`, `selection.py make [held] [fixed]`.
- Next, research (in this order):
  1. **Brightness as a tone** (david): a switch between the dim and bright variants of one pair's colours (k/w and
     K/W) gives a tone between them, and the user keeps it in greyscale; the score charges it as a seam (by the
     patches rightly, ~1 against a hue's ~2). Try: no lightness charge for such a switch where the target's tone lies
     between the two; check david against the training pictures, blind.
  2. **A fair held-out test**: three held-out pictures were both bad at their old project settings (no gallery),
     which hides the comparison. Give them the current method presets (not tuning on them), then the check again.
  3. The dark end: the lightness term times Y lets colour lead in the dark (the sorting at L* 26, 0.87 -> 0.65); a
     floor on Y, if the user's dark sheets matter more than the renders' few dark marks.
  4. The strength by picture: jojo wanted none, rocket-rackoon x1 (both had few changes); one weight may do, else a
     knob.
- Then the app: the term in `SelectionEnergy` on a `feature/*` branch from `develop`, its weight fixed (the method
  presets' coherence 2 and 6 would scale it apart), tests (zero for the same pair; the user's cases from `seams.py
  check`), as an option off by default or the default, by the user.

## Step 6: prior pairs from artists, the zxart.ee top 100 (`zxart.py`, no user time)

- Why (the user, 2026-10-06): priors for a new, simpler pair selection that prevents clash, from the only taste
  other than the user's; measured as step 5 measures a seam, so the two speak one language.
- Done 2026-10-06 (README 19): `fetch` (the 100 best-voted standard screens; `rounds/zxart/top100.json`, the screens
  in `data/zxart/`), `stats` (the pairs, the borders by lightness, E against the null E0 at the cell's level, the
  transition prior against V and the seam score), `recover [K]` (the screen through the user's eye at K times its
  sigmas as a new project, Select pairs at the defaults, no DBS; 0.5 s a screen). The artists change pairs on half
  the borders, 87% with a shared colour, the border's pixels in it 5x more than chance; what they avoid the score
  tells better than V (rho -0.41 against -0.22). Recovery 82% at the eye's blur, 74% at 2x, 63% at 3x; the converter
  loses the shared colour first (changes without one 6.9 -> 10.1% of borders against 6.4%), unseen by the score.
- Each candidate by recovery at 2x and 3x the blur (the gate: more cells recovered, the share of changes without a
  shared colour down to the artists' ~6%, on all 100), then the user blind on the training pictures:
  1. **A shared-colour term**, done 2026-10-06 (README 19): W x coherence x the seam's smoothness on a change of pair
     that keeps no colour of the neighbour's (`select_shared`). W 0.3 passes the gate (no shared colour 10 -> 5.5% at
     3x, recovery +1.1 at 3x, +0.3 at 2x, nothing else moved); W 100 (a hard constraint) overshoots to 2% and loses
     2.3 at 2x.
  2. **The transition table as V**, done (`artists_V`, in-sample): recovery +1.6 / +0.9 but the change rate and the
     bright/dim switches twice the artists' excess. Not taken; if ever, held out (the table from 50 screens, the test
     on the other 50) and with the bright/dim switch charged apart.
  3. The recovery's own blind spots: hue ramps the artists make and the score charges (B/M | M/Y), the one-colour cell
     next to its own colour (made 7-8x); whether the selection makes them at 3x.
- Next: the shared-colour term at W 0.3 and 1 against the current selection, blind, on the six training pictures at
  their gallery best with DBS (as `selection.py user`); then, if it holds, the term in `SelectionEnergy` on a
  `feature/*` branch from `develop` (a pairwise cost from the pairs' palette indexes, its weight a fraction of
  coherence), with tests (zero where a colour is shared, zero across an edge), a PR into `develop`.
- Caveats: the pictures are drawn to the grid, not photographs; a blur of 2-3x the eye is a stand-in for that. The
  recovery judges the selection against one expert's answer among the good ones (there is no correct colouring,
  `FINDINGS.md`), so a few points of recovery mean nothing, a shift of the shared-colour share does.

## After the gallery

### Seams in the algorithm: if the gallery does not fix autumn and diver-sunset

- The user (2026-10-02): autumn has clash on the ground, and grey blocks jump out; diver-sunset has clash in the sky,
  and blockiness. The problem is the cell seams on smooth surfaces, not the colour and not the patchwork of families.
- For this: an idea from `IDEAS.md` (the seam terms from parked, the surface size, or a new one) and a round against
  the base.
- Gate: that picture is better than all three competitors (`rivals.py user` with our new result); jojo and
  rocket-rackoon are not worse.

### Adaptive binding (~5 user minutes): if the gallery's surface dE splits by picture, as on jojo and rocket-rackoon

- A variant in `surface.py` (`allowed`): bind a surface only if the base keeps pairs of two or more families among its
  near cells (within 20 ΔE of the surface colour). Different families are different sets of visible colours
  (`build.colours`). A better rule: bind only if the share of the "other" family is above a threshold. Else the
  surface is free.
- `surface.py make`, `surface.py user r3 V` against the base on the same 7 pictures, the user's vote.
- Gate: jojo and vangog are better, anubis and rocket-rackoon are not worse. Then make it the default in the app in a
  separate PR into `develop` (as #14, with tests), with a check on the held-out pictures (andy, sunset, golden-axe,
  david, burning-hand). If not, write the result in README and `FINDINGS.md`; surface dE stays an option.
- If it passes: add the variant to `rivals.OURS` and run `rivals.py user`. Check that jojo and rocket-rackoon with this
  one variant are better than all three competitors. The judges for the best settings are ready; ~10 min of the user's
  time.

## Rules

- A change to the algorithm or the defaults goes on a separate `feature/*` branch from `develop`, with tests. Then a
  PR into `develop`, or, if the research needs it earlier, a merge here (as step 2). The research branch keeps no
  changes of its own in `src`.
- Each step has a gate. If a step fails its gate, write the result in README and `FINDINGS.md`, and do not go to the
  next step.
- The round protocols are in `METHOD.md`.
