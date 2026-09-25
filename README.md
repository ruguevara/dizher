# Dizher

Dizher (and AvtoDizher) is an image converter for 8-bit computers with colour-restricted
graphics modes, such as the ZX Spectrum and the Commodore 64.

Turn on, tune in, drop out.

## Goal

A research prototype: can a converter driven by a model of the eye beat the existing tools?

It is done when:

* On three reference images, DBS with the eye model visibly beats the known converters
  (ZX-Paintbrush, image2zx, img2spec) in both dithering and colour selection.
* The UI concept is clear and usable: adjustment layers for preprocessing, and a processing
  pipeline with a preview at each stage.

Out of scope: animation, non-8-bit platforms, compression.

## Supported platforms and modes

* ZX Spectrum
  * Standard mode: 256x192, 15 colours, two colours per 8x8 character block. No flash attribute.
* Commodore 64
  * Hires mode: 320x200, 16 colours, two colours per 8x8 character block. PNG output only for now.

A mode is a `platforms.Mode`: screen size, attribute cell size, palette (with its allowed
paper/ink pairs) and the writer of the native screen file. The converter is generic over these;
the ZX Spectrum specifics live in `platforms/zxspectrum`.

## Installation

Requires Python 3.11 or newer. From the repository root:

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .

Run the GUI:

    dizher [image.png]

### Standalone app

`packaging/build.sh` builds a self-contained app with PyInstaller (`dist/Dizher.app` on macOS,
`dist/Dizher` elsewhere):

    pip install pyinstaller
    ./packaging/build.sh

GitHub Actions builds it for macOS (Apple Silicon and Intel), Windows and Linux on every push,
and a `v*` tag publishes the four archives as a release.

### Tests

    pip install pytest
    pytest                       # the converter and the pipeline
    python tests/test_ui.py      # the GUI, driven by the Dear ImGui test engine

### Layout

* `src/dizher`: the converter (colour selection, halftoning, eye model), the platforms and the GUI.
* `src/mokit`: the node graph, project files and UI widgets the GUI is built on; a copy of the
  Python part of mokit, the author's ZX Spectrum toolkit.

## How it works

Dizher turns a full-colour picture into a ZX Spectrum screen: 256x192 pixels where every 8x8
block may use only two colours out of the 15-colour palette. The conversion is one optimisation
problem in two stages, colour selection and halftoning, both judged by the same model of the eye.

### Pipeline

Two optimisers in a row, both minimising the same eye-model error, each with its own extra term.
The GUI block that owns each control is in brackets.

```
 source image ──► Tune (framing, light, levels, contrast, colour) ──► sRGB 256x192
                                                                        │
                       gamma 2.2 ──► linear RGB ──► opponent O1 O2 O3 (luma, red-green, blue-yellow)
                                                    scaled by sqrt(weights)         [Metric: chroma]
                                                                        │
                                                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ Stage 1: SELECT PAIRS                one paper/ink pair per 8x8 block                          │
   │                                                                                                │
   │  candidates: for every allowed pair, project the block onto the paper-ink segment, halftone    │
   │  it as stage 2 will          [Target: palette subset]  [Halftoner: halftoner, matrix, kernel]  │
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
   │  [Halftoner: halftoner] ordered (matrix; void-and-cluster dispersed dots by default, a fair    │
   │  stand-in for the DBS result), error diffusion (kernel), stochastic (blue noise); the tile     │
   │  is rolled by [Halftoner: noise x, y]                                                          │
   └──────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                              │ bitmap
                                              ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ Stage 3: OPTIMISE                  [Optimise: enabled]  off passes the halftone through        │
   │                                                                                                │
   │  loss(bitmap) = Σ_ch || h_ch ∗ (result − target) ||²    same kernels and noise weights         │
   │               + Σ_ch noise_ch · || result − target ||²                                         │
   │               + structure · Σ_p c_p (1 − SSIM_p(result, target))   luma only, 7x7 windows,     │
   │                                                                     c_p = local contrast       │
   │     structure  [Optimise: structure]                                                           │
   │                                                                                                │
   │  solver (DBS): from the halftone, every pixel tries a toggle and a swap with each of 8         │
   │  neighbours, keeps the move that lowers the loss most; deltas are exact from running error     │
   │  and window statistics; a lattice of non-interacting pixels moves at once; stops when fewer    │
   │  than 0.1% of pixels move                                                                      │
   └──────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                              │ bitmap + attributes
                                              ▼
                                  screen file (.scr) or PNG
```

The views in the Preview tab show what each stage sees: Projected is the stage 2 target,
Unoptimised is the halftone the optimiser started from, Eye shows the target and the result
through h, Error is their difference, Energy is the stage 1 loss per block (own, seam,
coherence), and Seams shows where edges switch the coherence prior off.

### Eye model

Seen from a normal distance, a screen is blurred by the eye, so a fine mix of two colours reads
as their average. Dizher models this with an isotropic kernel `exp(-(r / scale)^alpha)` applied
in linear light (alpha 2 is a Gaussian; alpha near 1 gives a sharper peak and heavier tails,
closer to the measured contrast sensitivity). Colours are compared in an opponent space, as in
S-CIELAB: one luminance channel and two chroma channels (red-green, blue-yellow), each with its
own blur, because the eye resolves luminance detail much more finely than colour detail. Both
stages use the squared error after this blur, plus an unblurred error term that penalises
visible dot noise, with the same kernels and channel weights.

The converter caps the kernel support at half the smallest cell dimension (radius 4 for an 8x8
cell) before normalisation, so that nearest-neighbour selection covers every interaction. The
`scale` and `alpha` controls still shape the finite support; wider or heavier tails are
deliberately truncated.

### Colour selection

For every block and every allowed pair of palette colours (72 pairs on the Spectrum), a candidate
block is made by projecting the source onto the paper/ink segment in weighted linear opponent
colour space and then halftoning that mixture with the chosen halftoner, so that pairs are chosen
for the dots they will actually get (error diffusion is batched over all pairs in one raster
pass). The whole screen is then the sum of one candidate per block, and the eye-model error of
that composite is a quadratic function of the block labels: a cost per block, plus a pairwise
cost for every two neighbouring blocks that measures the visible seam their candidates paint
across the border. Because the blur is small, only the 8 surrounding blocks interact.

Blurred error alone is blind to texture: a lone block dithered with green dots among blocks
dithered with yellow ones is an obvious cell even when the mean colours match. A coherence prior
therefore charges a change of pair between neighbours by how different the two pairs look (the
CIELUV distance of the papers plus that of the inks), scaled down where the original image itself
has an edge across that seam. Labels are optimised by line-wise dynamic programming: each row,
then each column, is re-solved exactly given the rest, until nothing changes. The palette can be
restricted to bright colours only, non-bright only, grayscale, black and white, or any custom set.

### Halftoning

With paper and ink fixed per block, each pixel is set to one of them so that the blurred result
matches the blurred image. All halftoners take the same colour-aware inputs:

* ordered, with a choice of 39 threshold matrices: Bayer, dispersed and clustered dots, line
  screens and magic squares (void-and-cluster dispersed dots by default);
* error diffusion, with 19 kernels from Floyd-Steinberg to Stevenson-Arce;
* stochastic, with blue noise.

The Halftoner block shows only the controls of the chosen method, and the same method paints the
pair candidates.

The Optimise stage, on by default, then runs Direct Binary Search (DBS) from that halftone: it
repeatedly visits every pixel and either flips it or swaps it with one of its 8 neighbours,
whichever lowers the eye-model error most, until no move helps. A swap moves a dot without
changing the local tone; with flips alone the search stalls at about twice the error. The error
change of a move is computed exactly from a running error image, so a pass costs a few
convolutions. Pixels farther apart than the kernel do not interact, so a whole lattice of pixels
moves at once. DBS uses the same per-channel blur and noise terms as colour selection.

Plain DBS reproduces tone but blurs faint edges and texture. Following structure-aware
halftoning (Pang et al. 2008), the energy also includes a structural similarity (SSIM) term
between the halftone and the image in small windows of the luminance channel, weighted by the
local contrast of the image (Jiang et al. 2023) so that flat areas do not grow holes. Its effect
on every move is also computed exactly, and its weight is a slider. With the optimiser off, the
halftone is the result, for its own look or for comparison; the Unoptimised view shows the
starting point under the optimised result.

## Roadmap

Done:

* [x] Error diffusion that respects the two colours of each character block
* [x] Direct Binary Search optimiser under the eye model, starting from any halftone, with a
  structure-aware (SSIM) term
* [x] Colour selection as one eye-model energy with a coherence prior
* [x] Adjustable metric weights (luma, chroma, coherence) and eye-model parameters
* [x] Selectable dithering methods
* [x] Tune stages: framing (fill or fit, scale, rotation, pixel shift and per-edge nudges onto
  the cell grid), exposure and white balance (temperature, tint), Photoshop-style Levels with
  Auto and a histogram, local tone (local contrast, shadows, highlights and clarity on an
  edge-preserving base layer), contrast, vibrance and saturation, texture and sharpening (unsharp
  mask at the screen's size)
* [x] Palette subsets: bright only, non-bright only, grayscale, black and white, or a custom set
  with any colour toggled on or off
* [x] Python installation package
* [x] Save as SCR and PNG
* [x] Projects: a folder with a `project.json` (every stage's params and the image path relative
  to the folder, in mokit's format, as in AmaZX); exports go to its `build/` by default, and the
  last session, unsaved edits included, comes back on start. Every image has its own project,
  the Lightroom way: File → Open image opens the project folder next to the image, named after
  it, or starts a new one there. Autosave (File menu, on by default) writes the project after
  every edit; with autosave off, opening another image over unsaved edits asks to save them.
  File → Open recent lists the last 20 images.
* [x] Undo/Redo (Edit menu, Cmd+Z / Shift+Cmd+Z): a slider drag is one step, and opening an image
  starts the history over. The History panel lists every step, newest on top, named after the
  params it changed; a click goes back or forward to it.
* [x] Overpaint: attributes and brightness painted by hand

### Alpha

Critical:

* [ ] Standalone binary packages with PyInstaller for macOS, Windows and Linux
* [ ] App icon
* [ ] .dmg for macOS
* [ ] Fix the middle column's width: the width of the preview plus padding
* [ ] Versioning
* [ ] About dialog with GitHub and web links, greetings, and automatic version bumping

### Beta

* [ ] A better loss function for pair selection, tuned on real artists' overpaint data
* [ ] Palette snapping prototype
* [ ] A panel of starred or named snapshots to compare and choose from
* [ ] Global presets of selected stages and params
* [ ] Curves (a tone curve editor; subsumes Contrast)
* [ ] Levels per channel (R, G, B), like Photoshop's channel menu

### Future versions

* [ ] Custom ZX Spectrum palettes (ZX Spectrum Next or other enhanced hardware)
* [ ] ZX Spectrum multicolor, GigaScreen and MultiGigaScreen software graphics modes
* [ ] Commodore 64 low-res mode: 160x200 with 4 colours per character block

## Credits

### Code and data

* [libdither](https://github.com/robertkist/libdither) by Robert Kist (MIT): the non-Bayer
  threshold matrices and the error-diffusion kernels.
* [Free blue noise textures](https://momentsingraphics.de/BlueNoise.html) by Christoph Peters
  (CC0): the blue noise.
* [img2spec](https://github.com/jarikomppa/img2spec) by Jari Komppa: the ZX Spectrum colours as
  measured from a real machine on a CRT.

### Libraries

[NumPy](https://numpy.org), [SciPy](https://scipy.org),
[scikit-image](https://scikit-image.org), [OpenCV](https://opencv.org),
[Pillow](https://python-pillow.org), [imageio](https://imageio.readthedocs.io),
[Dear ImGui Bundle](https://github.com/pthom/imgui_bundle) by Pascal Thomet, on top of
[Dear ImGui](https://github.com/ocornut/imgui) by Omar Cornut, and
[PyInstaller](https://pyinstaller.org) for the standalone app.

### References

* M. Analoui and J. P. Allebach, "Model-based halftoning using direct binary search", Proc. SPIE
  1666, 1992.
* R. Ulichney, "The void-and-cluster method for dither array generation", Proc. SPIE 1913, 1993.
* X. Zhang and B. A. Wandell, "A spatial extension of CIELAB for digital color image
  reproduction" (S-CIELAB), SID Symposium Digest, 1996.
* F. Durand and J. Dorsey, "Fast bilateral filtering for the display of high-dynamic-range
  images", SIGGRAPH 2002: the base/detail split behind Local tone.
* W.-M. Pang, Y. Qu, T.-T. Wong, D. Cohen-Or and P.-A. Heng, "Structure-aware halftoning",
  SIGGRAPH 2008.
* K. He, J. Sun and X. Tang, "Guided image filtering", ECCV 2010: the edge-preserving base layer.
* Jiang et al., 2023: contrast weighting of the structure term.

## Thanks

To Diver and Spke, for invaluable help and motivation while working on this project all these
years since 2020.

## Greetings

To sq, bfox, Grongy, Dalthon, Jammer, Vasyl, e!ghtbm, Gazela,
Wbcbz7, Kowalski, Volutar, Tmk, True-grue, and all retroscene pixel artists and demosceners.

## License

GNU General Public License v3.0 or later; see [LICENSE](LICENSE).
