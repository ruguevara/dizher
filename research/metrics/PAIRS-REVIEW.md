# research/pairs, reviewed

What this plan takes from the earlier work on pair selection and what it does not, and why.

## The two branches

- `feature/color-pair-selection-research-72ae52`: `research/pairs/`, sessions A-E and steps 1-7, 2026-09-30 to
  2026-10-08, ending in WIP commits that rebuild the painted projects as an oracle.
- `feature/gallant-wright-6dtgj6`: an earlier snapshot of the same work, up to step 5 (2026-10-06). Every commit of
  it is in the other branch; it has nothing of its own.

## How the work went

The question changed about every day: a numeric judge fitted to the user's paintings (sessions A-B), Claude as a
judge (pilot, E), the converter's knobs against the judges (C), per-picture galleries and duels (steps 3, 3b), seams
(step 5), artists' statistics (step 6), the paintings again as the oracle (step 7). Each step was planned after the
last one's data, with new measures and new weights. About 45 metrics, three judges in four versions, and a dozen
rounds with the user, most of them on 4-9 pictures.

## What does not hold, and why

1. **The paintings as ground truth.** The user's overpaints were made at older settings, only where the user noticed
   something, and they mix clash fixes with style. A painting against the algorithm is one-sided by construction
   (the painting is better), so a fit learned "further from the source is better" and needed a sign constraint.
   Step 7 found that under any energy the old paintings recover 11-46% of their cells on four pictures. They are
   not an oracle for a metric.
2. **The fitted judges (v1-v4, the fast judge).** Forward selection over ~45 features on a few hundred labels from
   one observer, refitted after each round's counterexamples. Four rounds of optimisation against them found a new
   blind spot each time. The features have no common model, so the fits do not carry over.
3. **Claude judges as labels.** They agree with the user where the user is decided (27 of 32, 4 of 4), but most
   labels used in fits were close pairs, where the user says "both fine" or "both bad" and the judges still take a
   side; the layout changed their detection from 7 to 17 of 18. Their labels measure the judge, not the viewer.
4. **"The user's eye", lightness sigma 0.75, colour 1.0 at x4.** It was picked by the user from 3x3 grids of blurred
   images on a monitor: the eye looks at a blurred picture through its own blur, so the pick is a preference for a
   look, not a measurement of what the eye does. It was then used as the judges' view and called the user's eye in
   FINDINGS. It also contradicts section 14: every seam measure is worse through it.
5. **Whole-picture tests of one component.** The seam score was tested held out on pictures whose conversions were
   bad as a whole at their old settings: 3 of 5 "both bad", no information on the seams. A component can be tested
   on whole pictures only where the rest of the conversion is good.
6. **Small numbers read as results.** "Better on 5 of 6 training pictures" (fitted partly on them), "1-1 held out",
   "the score's weight 16, the geometric mean of six picks" are not significant and were not tested as such. The
   held-out pictures were looked at in rounds c1, held and held-fixed, so they were no longer held out.
7. **The viewing conditions.** 26 pixels per degree on another monitor, "2x-3x, sometimes with a squint". A squint
   is a different blur. The labels hold for that setup only.
8. **The aim moved.** PLAN says colorimetric closeness is not an aim and lists local contrast and meaningful
   gradients over the overall tone; the measures were mostly fidelity terms all the same.

## What holds

- **Optimising freely against a fitted judge fails** (session B: the judge's optimum worse than Select pairs on 8 of
  9 pictures by the user, while the judge rated it above both on all 9). Four repetitions, one direction. This plan
  keeps it as a rule.
- **The seam marks** (section 14): 1342 seams marked by the user on 12 whole renders at 3x, 2 renders repeated. The
  densest local data from a person, on real renders. At a pair change the raw colour step across the border ranks
  the marks best (AUC 0.79), and every blur makes the measures worse. One observer and 6 pictures, but usable as a
  test of a metric's local map. That nearly all marks are on pair changes (1338 of 1342) is close to true by
  construction and is not a finding.
- **The patches** (section 15): a hue change at one luminance shows (rated ~2, red-green 3), brightness alone is faint
  (1), a slow lightness step hides (0); repeats 80% the same. Small, one observer, but a controlled design. Its flaw:
  the dots ran on across the border, unlike a render, where each pair has its own dots, and the measures that win on
  the renders lose on the patches. B4 here keeps the dots of each pair.
- **The artists' statistics** (section 19, `zxart.py`): the zxart.ee top 100, no observer. Pairs change on half the
  borders and keep a shared colour on 87% of changes, the border pixels in it twice as often as the cell's level
  gives. A description of how pixel artists work within the attribute grid: a prior for the generator, not a
  measure of what a viewer sees. These are drawn pictures, not conversions.
- **The user's priorities** (PLAN, 2026-10-02): seams first; then colour and tone with the colour approximate, no
  grey dither everywhere, local contrast over the overall tone. This plan tests the last as H7.
- **The competitor setup** (`rivals.py`): img2spec, Image to ZX Spec, ZX-Paintbrush scripted. Useful as anchors in
  stage C. The verdict there (ours better 15 of 24) was the author's, on 4 pictures.

## What this plan does differently

| research/pairs | here |
|---|---|
| labels from paintings and from Claude judges | labels from people only, on stimuli made for the question |
| whole-picture verdicts to fit weights | controlled stimuli for the parts (B), whole pictures only to validate (C) |
| features chosen by forward selection | one model, its parts' form from the vision literature, 4 free parameters at most |
| the question re-planned after each round | stages and decision rules committed before their data |
| held-out pictures reused | new pictures for C, looked at once |
| one observer | the user plus 2-3 others in C, reported apart |
| viewing as it happened | a measured setup, written down, the same in every session |

## The data carried over

Secondary checks in stage A, copied with their source commit (`5810855`), nothing imported from the code:
`rounds/seams/marks.json` with `key.json` (the 12 renders, made again exactly or dropped), `rounds/seams/patches-v3..v5`
with their keys, `rounds/zxart/top100.json` (for the generator later, not the metric). Nothing else.
