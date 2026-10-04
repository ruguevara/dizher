# Findings: pair selection

What we know. The section of `README.md` with the evidence is in parentheses. The dead ends not to repeat are at the
bottom. How we found it: `LOG.md`.

## The user's taste

- **Smoothness is more important than colorimetry.** Metrics of colour accuracy against the source select the user's
  painting in 31–36%. The steps at the cell seams (`seam_excess`) select it in 65% on all edits, and in 84% on the
  hard ones. Where smoothness and accuracy disagree, the user selects smoothness 10 to 5. (1, 3)
- **A clean colour of the correct family is more important than an accurate mixture**: red, not a pink dither of red
  and white. No feature measures this. The metrics do not explain one third of the user's edits. (1, 4)
- **There is no correct colouring.** On close alternatives the user says "both fine" (14 of 30) or "both bad" (11 of
  30) more frequently than the user selects a side (4 of 30). (9, 10)
- Half a painting can be worse than all of it or none: 4 of 5 partial paintings are "both bad". (9)
- The user's "both bad" verdicts are on autumn, jojo and vangog. They show **a patchwork inside a surface** and **the
  wrong family** where its mixture is numerically nearer (the magenta cypress). Patchwork examples: faces with pale
  yellow/white cells among red-yellow dither, cell by cell; a sky with single cyan, magenta and green cells. (10, 11)
- **A pair bound to a surface is a style for each picture, not a default.** The user prefers it on faces and skies
  (jojo, vangog: in all 4 variants). The user rejects it on dense illustrations, where the surface's pair erases small
  accents (torch light, highlights: anubis, rocket-rackoon, in all 4). (11)

## How to look and measure

- Compare colourings through the eye (~1 Spectrum pixel), not as raw pixels. Raw LPIPS/DISTS and S-CIELAB at 26
  pixels per degree measure the dither grain. A re-dither is 5.8 ΔE raw against 2.0 through the eye, and an edit is
  ~25. (ground rules, 4)
- An eye with σ 1 on all channels erases the clash between cells, which the user sees. The eye that the user selected:
  x4 with square pixels, then a blur of lightness σ 0.75 and of colour σ 1.0 Spectrum pixel in linear light, on the
  channels of the converter's opponent space (`views.seen2`). (8)
- Metrics that blur on their own (2 px seams, cell means) do not need an eye in front. An eye makes them slightly
  worse. (4)
- Through the eye, the judges and the user do not see an edit below ~10 ΔE. A single segment is decided from ~20 ΔE,
  an edit of many cells from 10. (9, 10)

## Numeric judges

- The best single feature is `seam_excess` (selected first in 27 of 30 bootstraps). It needs a counterweight for
  accuracy at a coarse scale: LPIPS through the eye (judges v2–v4), or MS-SSIM of lightness through the eye for the
  fast judge. (4)
- The judges can compare the results of the converter itself, but they are not a target for optimisation. Each round
  finds a new blind spot, and the counterexamples do not converge. v4 put its optimum higher on all 9 pictures; the
  user put it lower on 8. (5)
- The data "the painting is better than the algorithm" are one-sided. Without a sign constraint the regression learns
  "further from the source is better". Thus the error weights are ≥ 0. (4)

## Claude judges

- The layout decides. On one sheet of three pictures the judges see an edit in 7 of 18. On three separate files
  through the user's eye they see it in 16 of 18. (8)
- A label is only a majority of 6 votes (3 for each order). On null pairs, single votes take a side (9 of 24); the
  majority never does. Majority labels are stable against the order (72 of 78). (9, 10)
- A simple prompt is better than a prompt that describes the user's taste (0.91 against 0.85). The judges almost never
  answer "both bad" (6 votes of 996); the user frequently does. (9, 10)
- Where the user selects a side, the judges agree with the user (4 of 4 in step 2, 4 of 4 in step 4). On close pairs
  the user usually does not select a side, but the judges do. There their labels are noise for the energy. (9, 10)

## The converter (Select pairs)

- The knobs of the Exact mixture preset are at the optimum for the user's eye. (7)
- Where the DP is unsure between neighbouring pairs, the choice is not important to the user, or both are bad. (10)
- **The patchwork is structural.** Each cell takes its nearest mixture, and the cells of a smooth surface fall on the
  two sides of the border between two families. Coherence (a local cost for a change of pair) does not remove it (×2
  makes no difference). A smoothed target only moves the border. A pair bound to the surface removes it. (10, 11)
- The binding fails where a surface is a mixture of several colours: the surface's pair becomes a grey compromise
  (the black and white raccoon, the greens of RC1). If only the cells near in colour (20 ΔE) are bound, the problem
  is smaller, but the accents are still lost. (11)
- Untuned pictures (a new project without Tune) give almost no visible alternatives (median 3–6 ΔE). (10)

## Against the competitors

- **Not worse anywhere**: img2spec and Image to ZX Spec at their best settings by the Claude judges, ZX-Paintbrush at
  its defaults. Of the user's 24 sheets: ours better 15, both bad 9, the competitor better 0. (12)
- **The coach's criterion holds on two pictures of four.** jojo with surface dE 20 and rocket-rackoon with Exact
  mixture are better than all three tools, each with its own style from section 11. One setting does not win both.
  autumn and diver-sunset are "both bad" against the dithering competitors (Floyd–Steinberg, Atkinson, Bayer). The
  user names the cause, seams on smooth areas: grey blocks on the ground in autumn, clash and blockiness in the sky of
  diver-sunset. (12)
- The mono DBS of img2spec is last on all pictures (the colour goes to grey and pink). Its strong settings are Bayer
  8×8 and Floyd–Steinberg. (12)

## The user as a source of labels

- Reliable: the user confirmed the painting 18 to 1, and the repeats agreed 4 of 5 two times. The user's time is the
  main resource. A round takes 5–15 minutes: whole pictures raw at 3x (as in the app) or the judges' sheets; the keys
  ←/→/space/x (`dp.py vote`). (3, 10, 11)

## Dead ends (do not repeat)

- A free optimisation of the colouring against a numeric judge. (5)
- Single preset knobs tuned against numeric judges; a joint search on them. (7)
- A fit of the energy weights to labels where the DP is unsure. (10)
- A selection of pairs by the total energy margin: it takes single-cell edits (the margin grows with the number of
  cells). (10)
- Judges on a sheet of three pictures in one file; an eye with σ 1 on all channels. (8)
- Partial paintings as labels. (9)
- The selection on a smoothed target. A soft cost to leave the surface's pair: it only blends the base and the hard
  binding. Two pairs per surface: they bring the patchwork back. (11)
