# Ideas: pair selection

The status: **next** — in `PLAN.md`; **open** — ready to take; **parked** — on hold, with a condition; **closed** —
tested, the result is in `LOG.md` and `FINDINGS.md`. The cost is the user's time and the order of work.

## next

- **Fewer settings, but each with a meaning** (the user, 2026-10-02). There are too many settings now, and it is hard
  to find a good combination for a picture. The path goes through more knobs for a time:
  - the dithering gets its own weights. Chroma and the noise weights were shared by the selection and the dots; now
    chroma is on Halftone and the noise is on Optimise (`PLAN.md`, step 2);
  - a mode to tune each picture: a gallery of renders, a person picks, the descent follows. Start with a simple plane
    through the best point. If the simple method does not converge, use Sequential Gallery (Koyama et al. 2020) or
    Sequential Line Search (SIGGRAPH 2017: one slider along a line that Bayesian optimisation selects). If the rounds
    are expensive, Claude judges can be a prefilter (`PLAN.md`, step 3);
  - from the gallery log: a few knobs by the priorities of the aim (seams, colour and tone, form, texture). Each knob
    moves several current weights together (`PLAN.md`, step 4).

  This is not the same as session C. Session C looked for a general preset (it is at the optimum); here we look for
  settings for each picture.
- **Seams on smooth areas: autumn and diver-sunset.** "Both bad" against the dithering competitors (README 12). The
  user: autumn has clash on the ground, and grey blocks jump out; diver-sunset has clash in the sky, and blockiness.
  The third picture of the coach's criterion is one of the two. First the gallery: can settings fix them? The user
  (2026-10-02): less chroma on Halftone (more lightness in the dots) sometimes makes the blockiness more even. If not,
  the candidates are the seam terms from parked. (`PLAN.md`, after the gallery)
- **Adaptive surface binding.** Bind a surface only where the base makes a patchwork: cells of almost one colour take
  pairs of two or more families. Where the cell colours are really different, keep the cells free. If jojo and vangog
  are better and anubis and rocket-rackoon are not worse, this is the default. Against the competitors (README 12)
  jojo wins against all only with binding, rocket-rackoon only without it; adaptive binding can possibly win both.
  ~5 min of the user's time. Take it if surface dE from the gallery splits by picture in the same way. (`PLAN.md`,
  after the gallery)

## open

- **A colour family metric.** A pair is permitted for a cell if its colours are in the family of the target colour.
  The family is more important than the saturation, the saturation is more important than the lightness, and greys
  are free. Against the magenta cypress and for "red, not pink" (`FINDINGS.md`, taste). A unary term in `energy.py`
  and a round with the user.
- **Prior pairs from artists (zxart.ee, the old session D).** A recovery test: blur an artist's screen, run the
  selection, compare the pairs (especially k/m against k/b in shadows). Also the statistics of neighbouring pairs as
  `V`. API: `https://zxart.ee/api/export:zxPicture/start:0/limit:N/order:votes,desc`; download to a gitignored folder.
  This is the only source of taste other than the user, and it needs no time of the user. It is the largest work.
- **The surface size.** 32 and 64 surfaces flip diver-sunset and RC1. We can select the number for each picture (from
  the colour spread) or give it to the user. Small.
- **Claude judges as a prefilter** of easy cases before the user (confident majority verdicts; they do not express
  "both bad"). A tool, not a method of selection.

## parked

- **New decomposable energy terms** (from session C, step 4): the colour step at a seam (the judges hold on to
  `seam_ab_1`; the DP already has the pair terms `S`); a `V` that knows the bright and dim variants (a change k/b↔K/B
  costs 0.06 against a median of 0.52). Take them if the competitor baseline shows a loss at the seams. README 12
  shows no losses, but the user explains "both bad" on autumn and diver-sunset by the seams (see next).
- **Cheaper numeric judges** (LPIPS on AlexNet, VGG conv2_2 through the eye). Necessary only if we need a numeric
  judge again; not as a target for optimisation.

## closed

- A free optimisation of the colouring against a judge (session B).
- Preset knobs one at a time, and a joint search (CMA-ES/TPE): session C.
- A fit of the energy weights to the Claude judges' labels where the DP is unsure (session E; step 5 did not start:
  there the labels are equal for the user).
- The selection on a smoothed target (`smooth.py`).
- The competitor baseline (step 1): not worse anywhere, the coach's criterion on 2 of 4 pictures (README 12).
- One pair per surface: in the app as the surface dE option (PR #14). For the default, see adaptive binding.
