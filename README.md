# Dizher

Dizher (and AvtoDizher) is a 8-bit Graphics Converter for lovely retrocomputers with color-restricive graphics modes.

Turn on, tune in, drop out.

## Goal

A research prototype: can a converter driven by an eye model beat existing tools?

Done means:

* On 3 reference images, DBS with the eye model visibly beats known converters
  (ZX-Paintbrush, image2zx, image2spec) in both dithering and colour selection.
* The UI concept is clear and usable: adjustment layers for preprocessing, and a processing
  pipeline with a preview at each stage.

Out of scope: animation, non-8-bit platforms, compression.

## Supported 8-bit platforms and modes

* ZX Spectrum
  * Standard mode — 256x192 with 15 colors, two colors in one character 8x8 block. No flashing bit support.
* Commodore 64
  * HiRes mode — 320x200 with 16 colors, two colors in one character 8x8 block. PNG output only for now.

A mode is a `platforms.Mode`: screen size, attribute cell size, palette (with its allowed
paper/ink pairs) and the native screen file writer. The converter is generic over these; the ZX
Spectrum specifics live in `platforms/zxspectrum`.

## How it works

Dizher turns a full-colour picture into a ZX Spectrum screen: 256x192 pixels where every 8x8
block may use only two colours out of the 15-colour palette. The conversion is one optimisation
problem in two stages, colour selection and halftoning, both judged by the same model of the eye.

### Pipeline

Two optimisers in a row, both minimising the same eye-model error, each with its own extra term.
The GUI group that owns each knob is in brackets.

```
 source image ──► Tune (framing, light, levels, contrast, colour) ──► sRGB 256x192
                                                                        │
                       gamma 2.2 ──► linear RGB ──► opponent O1 O2 O3 (luma, red-green, blue-yellow)
                                                    scaled by sqrt(weights)         [Metric: chroma]
                                                                        │
   ┌────────────────────────────────────────────────────────────────────┼───────────────────────────┐
   │ Stage 1: SELECT PAIRS                one paper/ink pair per 8x8 block                          │
   │                                                                                                │
   │  candidates: for every allowed pair, project the block onto the paper-ink segment, blue-noise  │
   │  dither it                                    [Target: palette subset]  [Prepare: noise x, y]  │
   │                                                                                                │
   │  loss(labels) = Σ_ch || h_ch ∗ (composite − target) ||²    eye-blurred error, exact quadratic  │
   │               + Σ_ch noise_ch · || composite − target ||²  unblurred dot contrast              │
   │               + coherence · Σ_seams V(pair, pair') · exp(−step² / 2 edge²)   Potts prior       │
   │                                                                                                │
   │     h_ch      [Eye model: luma/chroma alpha, blur px]  support capped at 4 px (half a cell)    │
   │     noise_ch  [Select pairs: luma noise, chroma noise]                                         │
   │     V         CIELUV distance of papers + inks, fixed by the palette                           │
   │     coherence, edge  [Select pairs: coherence, edge]                                           │
   │                                                                                                │
   │  solver: rows then columns re-solved exactly by dynamic programming, until no label changes    │
   └──────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                              │ paper, ink per block
                                              ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ Stage 2: HALFTONE                  one bit per pixel: paper or ink                             │
   │                                                                                                │
   │  target: each pixel projected onto its block's paper-ink segment (the unreachable part is      │
   │  already paid for by stage 1, so it is not chased across seams)                                │
   │  start: blue-noise dither of that target                              [Prepare: noise x, y]    │
   │                                                                                                │
   │  loss(bitmap) = Σ_ch || h_ch ∗ (result − target) ||²    same kernels and noise weights         │
   │               + Σ_ch noise_ch · || result − target ||²                                         │
   │               + structure · Σ_p c_p (1 − SSIM_p(result, target))   luma only, 7x7 windows,     │
   │                                                                     c_p = local contrast       │
   │     structure  [Halftone: structure]                                                           │
   │                                                                                                │
   │  solver (DBS): every pixel tries a toggle and a swap with each of 8 neighbours, keeps the      │
   │  move that lowers the loss most; deltas are exact from running error and window statistics;    │
   │  a lattice of non-interacting pixels moves at once; stops when fewer than 0.1% of pixels move  │
   │  Alternatives for comparison  [Halftone: halftoner]: Stucki, ordered (matrix), stochastic      │
   └──────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                              │ bitmap + attributes
                                              ▼
                                  screen file (.scr) or PNG
```

Views in the Preview tab show what each stage sees: Projected is the stage-2 target, Eye is target
and result through h, Error is their difference, Energy is the stage-1 loss per block (own, seam,
coherence), Seams is where edge disables the coherence prior.

### Eye model

A screen viewed from a normal distance is blurred by the eye, so a fine mix of two colours reads
as their average. We model that with an isotropic kernel `exp(-(r / scale)^alpha)` applied in
linear light (alpha 2 is a Gaussian, alpha near 1 has a sharper peak and heavier tails, closer to
measured contrast sensitivity). Colour is compared in an opponent space, as in S-CIELAB:
one luminance channel and two chroma channels (red-green, blue-yellow), each with its own blur,
because the eye resolves luminance detail much finer than colour detail. Both stages use squared
error after this blur, plus an unblurred error penalty for visible dot noise, with the same
kernels and channel weights.

The converter caps kernel support at half the smallest cell dimension (radius 4 for
an 8x8 cell) before normalisation, so nearest-neighbour selection includes every interaction.
The `scale` and `alpha` controls still shape the finite support; wider or heavier tails are
deliberately truncated.

### Colour selection

For every block and every allowed pair of palette colours (72 pairs on the Spectrum), a candidate
block is realised by projecting the source onto the paper/ink segment in weighted linear opponent
colour space, then blue-noise dithering that mixture. The whole
screen is then the sum of one candidate per block, and the eye-model error of that composite is a
quadratic function of the block labels: a cost per block, plus a pairwise cost for every pair of
neighbouring blocks that measures the visible seam their two candidates paint across the border.
Because the blur is small, only the 8 surrounding blocks interact.

Blurred error alone is blind to texture: a lone block dithered in green dots among yellow-dotted
blocks is an obvious cell even when the mean colours match. A coherence prior therefore charges a
pair change between neighbours by how different the two pairs look (CIELUV distance of the papers
plus of the inks), scaled down where the original image itself has an edge across that seam.
Labels are optimised by line-wise dynamic programming: each row, then each column, is re-solved
exactly given the rest, repeated until nothing changes. The palette can be restricted to bright
only, not bright only, grayscale, or black and white.

### Halftoning

With paper and ink fixed per block, each pixel is set to one of them so that the blurred result
matches the blurred image. All halftoners consume the same colour-aware inputs. The default,
Direct Binary Search (DBS), starts from a blue-noise dither, then
repeatedly visits every pixel and either flips it or swaps it with one of its 8 neighbours,
whichever lowers the eye-model error most, until no move helps. A swap moves a dot without
changing the local tone; with flips alone the search stalls at about twice the error. The error
change of a move is computed exactly from a running error image, so a pass costs a few
convolutions. Pixels farther apart than the kernel do not interact, so a whole lattice of
pixels moves at once.

DBS uses the same per-channel blur and noise terms as colour selection, with SSIM applied to the
luminance channel.

Plain DBS reproduces tone but blurs faint edges and texture. Following structure-aware halftoning
(Pang et al. 2008), the energy also includes a structural similarity term (SSIM) between the
halftone and the image in small windows, weighted by the local contrast of the image (Jiang et al.
2023) so that flat areas do not grow holes. Its contribution to every move is also computed
exactly, and the weight is a GUI slider. Ordered dithering with a choice of 39 threshold matrices
(Bayer, dispersed and clustered dots, line screens and magic squares, the non-Bayer ones from
libdither), Stucki error diffusion and plain stochastic dithering remain available for comparison
and for their look; the Halftone block shows only the controls of the chosen method.

## Installation

Requires Python 3.10+. From the repository root:

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .

Run the GUI (imgui_bundle; the pipeline stages are mokit nodes):

    dizher [image.png]

## TODO

Done:

* [x] Error diffusion respecting the two colours of each character block
* [x] Direct Binary Search halftoner under the eye model, with a structure-aware (SSIM) term
* [x] Colour selection as one eye-model energy with a coherence prior
* [x] Adjustable metric weights (luma, chroma, coherence) and eye-model parameters
* [x] Selectable dithering methods
* [x] Tune stages: framing (fill or fit, scale, rotation, pixel shift and per-edge nudges onto the cell grid), exposure and white balance (temperature, tint), Photoshop-style Levels with Auto and histogram, contrast, vibrance and saturation
* [x] Palette subsets: bright only, not bright only, grayscale, black and white
* [x] Python installation package
* [x] Save as SCR and PNG

Planned:

* [ ] Curves (tone curve editor; subsumes Contrast)
* [ ] Highlights / Shadows (local tone mapping, Lightroom-style)
* [ ] Clarity (local contrast) and Sharpen (unsharp mask after the resize; matters at 256x192)
* [ ] Levels per channel (R, G, B) like Photoshop's channel menu
* [ ] Custom ZX Spectrum palettes (ZX Spectrum Next or another hardware enhanceds)
* [ ] ZX Spectrum MultiColor, GigaScreen and MultiGigaScreen software mode
* [ ] Commodore 64 HiRes: native file format (Art Studio)
* [ ] Commodore 64 LowRes 160x200 mode with 4 colors per character block
* [ ] Overpaint bitmap, attrs, bright
* [ ] Standalone binary package with PyInstaller for macOS, Windows, Linux
* [ ] Ability to save and load conversion projects with an image, settings and overpaint layers
