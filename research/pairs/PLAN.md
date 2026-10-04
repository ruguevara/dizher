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

## The state (2026-10-04)

The branch is `feature/gallant-wright-6dtgj6`, rebased on `develop` 03d239b. Sessions A, B, C, E and steps 1 and 2
are closed (`LOG.md`); step 3, the gallery, has its first data (README 13). In the app: Select pairs: surface dE, one
pair per surface, off by default (PR #14); the selection energy's seam terms by matmul, a conversion without DBS ~2.4x
faster (PR #15). Merged here, not into `develop`: `feature/dbs-weights`, the dots have their own weights: chroma on
Halftone, noise on Optimise (step 2).

The next session (the user, 2026-10-04): a pairwise mode for one picture's knobs, below ("Step 3b").

Against the competitors (step 1, README 12; `rivals.py`, `rounds/rivals`): not worse anywhere. The criterion holds on
two of the three necessary pictures: jojo with surface dE 20, rocket-rackoon with Exact mixture, each with its own
style. autumn and diver-sunset: "both bad".

The steps (the user, 2026-10-02): for a time there are more knobs, so that later there are fewer, each with a meaning.

- Step 2: the dithering gets its own weights.
- Step 3: a gallery per picture (first autumn and diver-sunset, for the seams).
- Step 4: fewer knobs from the gallery data.

Seams in the algorithm and adaptive binding come after, from the gallery results.

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
- The branch is from `develop`, merged here (the user, 2026-10-02): the gallery needs the knobs. A PR into `develop`
  when the user decides.
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
  - `--no-dbs`: Optimise off and its knobs removed (the user permitted it).

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
  best on autumn and diver-sunset is not yet against the competitors.

## Step 3b: a pairwise search for one picture's knobs (the next session, the user's idea, 2026-10-04)

- The user picks one render of a pair, interactively as in the gallery; the backend fits a Gaussian process to the
  comparisons so far (preferential Bayesian optimisation: a GP with a probit likelihood on pairs, Chu and Ghahramani
  2005; Brochu et al. 2007; González et al. 2017) and proposes the next pair in the knob space, refining around the
  best.
- Known: the knobs and ranges as in the gallery (`gallery.KNOBS`, `span`, `params`, the unit box); preferences differ
  by picture, so a GP per picture; a render without DBS ~0.5 s on a quiet machine, so a pair in ~1 s; the galleries'
  comparisons (`gallery.comparisons`, `rounds/gallery/*.json`) can seed a picture's GP.
- Its result per picture goes where the gallery's would: autumn and diver-sunset against the competitors, the log to
  step 4.

## Step 4: fewer knobs from the gallery data

- From the log:
  - fix the knobs that the user does not move, or whose move does not change the pick;
  - merge the knobs that move together into one (the main directions of the picked moves across pictures);
  - keep the knobs that depend on the picture, or derive them from the picture (as the surface size from the colour
    spread, `IDEAS.md`).
- The result: a few knobs by the priorities of the aim (seams, colour and tone, form, texture). Each knob is a
  direction in the current knobs. The old knobs go to the advanced settings, or we delete them. A PR into `develop`.
- Gate: on two or three held-out pictures the best by the new knobs is not worse than the best by all knobs (blind,
  the user's vote).

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
