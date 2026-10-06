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

## Seams

- **The seams the user sees are where the pair changes** (1338 of 1342 marks): the selection's pairwise term is
  the place for a seam cost. (14)
- **The coherence term does not see them**: its attributes (V) rank the marked seams at 0.62, with its edge weight
  at 0.51, a coin toss. The edge weight excuses the seams on the source's outlines, and the user marks those more
  (0.22 against 0.15): the render makes a staircase of cell borders there. (14)
- **The colour of the pixels on the two sides of the border** decides: their step beyond the source's, raw, ranks
  the marks at 0.79, above the attributes on 11 of 12 renders. Black on both sides, no seam (the user). Lightness
  adds nothing on the renders. Through an eye blur every measure is worse: at a seam the user sees pixels. (14)
- Still far from the user's own agreement (top k 0.42 against 0.48–0.75). (14)
- **Without picture content** (synthetic patches, the same mean luminance on both sides): a change of hue shows
  (~2 of 3, red-green 3), brightness alone is faint (1), and a step of lightness alone of up to 0.12 of the dot level
  hides in a surface's slow variation (0). (15)
- **One seam score from all of it**: the lightness change of the border's pixels when the pair switches (the dots
  the same) plus the colour difference of the cells' mean colours, 0.075 and 0.055 per dE: lightness at the pixels,
  colour averaged. Held out: renders 0.68 against the coherence term's 0.51; the rounds 0.79-0.88. (16)
- **A solid-looking cell next to dotted ones is a seam where it is light** (the user): a dim pair near in lightness
  (yellow/white; cyan, green) against any bright pair. Dim and bright pairs with black, blue or magenta mix. In the
  dark the user hardly marks seams at all (1% below L* 30, 36% above 75). (17)
- **The score now**: 0.054 x the cells' mean colour difference + 0.126 x Y x the border pixels' lightness change +
  1.68 x Y x the solidity step, Y the cells' mean L* / 100. Held out: renders 0.72, rounds 0.78-0.87, the sorting
  0.85. (17)
- **The score in the selection** (its weight fixed, the coherence aside): blind, better on 5 of 6 training pictures
  and none worse; held out better on burning-hand, worse on david (greyscale: a dim/bright white switch is a tone
  there, which the user keeps), both bad on three at their old settings. It helps where seams are the trouble; it can
  cost a tone that a brightness switch gives. (18)
- The dots breaking at the border makes no seam (15). The same mean colour from other dots shows partly (1-2 of 3):
  the eye averages part of the dots. (15)
- The pixel step that leads on the renders (0.79) fails on the patches (0.01-0.20): it sees busy dots, and among
  busy dots a change does show more, but its fit does not carry over. (14, 16)

## Artists (the zxart.ee top 100)

- **Artists change the pair on half the borders.** Smoothness is not few changes: 87% of their changes keep a shared
  colour (100% in the dark), and the border's pixels carry it: the switch's lightness change at the border is 0 on
  25% of the shared-colour changes against 5% at the cell's own level. (19)
- **What artists avoid, the seam score tells better than the coherence term's V** (rho -0.41 against -0.22 on the
  transition prior); yet they make hue ramps the score charges high (B/M | M/Y 7x expected). The most made change is
  a one-colour cell next to a pair that holds its colour. (19)
- **Through the user's eye the current selection is nearly the artist**: 82% of cells, the borders alike. As the blur
  grows (less fitted to the grid) it loses the **shared colour** first: changes without one 6.9% -> 10.1% of borders
  against the artists' 6.4%, at mid and high lightness, and the seam score does not see it (the converter's changes
  score lower than the artists'). (19)
- Recovery of the artists' cells is a judge with an expert's ground truth and no user time, on art drawn to the grid;
  the pictures are not photographs. (19)
- **The count of recovered cells cannot judge a seam cost.** The target is the artist's own render through a mild
  blur, so the unary term fits it best: the current selection at coherence 0 recovers the most (0.849 against 0.824),
  and every weight of the seam score fewer, monotonically. The artists' pictures hold no clash for a seam cost to fix.
  (19)
- **The borders' statistics are the usable prior** (the artists' fingerprint: a change on 50% of borders, 9%
  bright/dim-only, E = 0 on 22%, the changes' score 3.96). The current selection is nearest on all; coherence 0
  doubles the bright/dim switches; the seam score at the user's weight 16 makes the borders smoother than the
  artists' (its changes 3.4-3.6), at weight 1-4 it stays in range. (19)
- By the user's measure, the changes the converter makes and the artist does not are the less visible ones (2.3
  against the artist's own 1.8 at the eye's blur). (19)
- **The artists' implied energy** (the artist's pair against the colour-nearest alternatives, the converter's own
  tables, the unary 1): coherence 4-5, eye seams 2-4 with chroma above luma, and the seam score near weight 2 (the
  user's pick 16), all of it on the solidity step; E and M nothing. The current energy at coherence 2 already ranks
  98.6% of the alternatives as the artists do; the fit 98.5%. (19)
- **Both sources put the solidity step first**: the artists' score terms, fitted without the user, rank the user's
  painted seams at 0.70 against the user's own 0.72, and need Y for the dark end. (19)
- **Artists change the pair where the picture changes colour**: 8% of flat borders (under 5 dE between the cells'
  means), 89% over 40 dE. **The masking is one number**: the border column holds the shared colour at twice the
  cell's odds (logit +0.69; a colour +0.87, black +0.53), both sides closed together (25% of shared changes have
  both border columns wholly in it). A prior for the dots at a pair change, not for the pair. (19)
- A shared-colour binary on the attributes (dropped: not a visibility measure, its weight a sweep) and the artists'
  transition table as V (in-sample; it frees cheap changes, doubling the bright/dim switches) are not the way. (19)

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
