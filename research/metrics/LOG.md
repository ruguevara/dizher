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
