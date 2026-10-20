# A0 results: synthetic sanity checks at 12.9 ppd, device factor 6

Made by `python research/metrics/a0.py` (2026-10-20, 765 s); the stimuli and the pass rules are in the docstrings of `a0.py`, the metrics' configuration in `metrics.py`. Every score is lower = more similar.

| check | cvvdp | ssimulacra2 | butteraugli | butteraugli_max | scielab | msssim | dists | lpips | eye |
|---|---|---|---|---|---|---|---|---|---|
| monotone in a tone error (grey, colour ramp) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| dither shifted 1 px < 0.5 x a 5% error | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| 50% grey: checker < blue noise < 2x2 | pass | pass | pass | pass | **FAIL** | pass | **FAIL** | pass | **FAIL** |
| clash > same error spread evenly | **FAIL** | pass | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | **FAIL** | **FAIL** |
| banding > dither of its levels | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass |
| dark ramp: dispersed < empty, < clustered | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| passed | 2/6 | 3/6 | 2/6 | 2/6 | 1/6 | 2/6 | 2/6 | 2/6 | 2/6 |

Clash stimuli, the largest dE00 between a cell's mean and the surface's: grey 0.00, salmon 5.78.

## Scores

| check | metric | result | scores |
|---|---|---|---|
| tone | cvvdp | pass | grey 0 < 0.0006905 < 0.01334 < 0.05802 < 0.1865; colour 0.05651 < 0.1086 < 0.2028 < 0.3649 < 0.6226 |
| tone | ssimulacra2 | pass | grey 8.369 < 13.57 < 27.29 < 54.5 < 97.31; colour 6.575 < 9.808 < 18.07 < 37.41 < 76.94 |
| tone | butteraugli | pass | grey 3.751 < 7.499 < 14.97 < 29.81 < 58.64; colour 5.149 < 10.27 < 20.41 < 40.24 < 77.71 |
| tone | butteraugli_max | pass | grey 4.328 < 8.651 < 17.28 < 35.34 < 78.03; colour 5.94 < 11.92 < 23.97 < 48.21 < 95.92 |
| tone | scielab | pass | grey 0.843 < 1.683 < 3.353 < 6.653 < 13.08; colour 0.9126 < 1.828 < 3.668 < 7.366 < 14.73 |
| tone | msssim | pass | grey 8.607e-05 < 0.0002633 < 0.0009606 < 0.003146 < 0.009075; colour 4.715e-05 < 0.0001613 < 0.00064 < 0.002261 < 0.007305 |
| tone | dists | pass | grey 0.001251 < 0.003899 < 0.01113 < 0.02849 < 0.06725; colour 0.0006846 < 0.002712 < 0.009939 < 0.02907 < 0.07062 |
| tone | lpips | pass | grey 0.0004809 < 0.001793 < 0.006285 < 0.02027 < 0.06509; colour 0.000146 < 0.0005599 < 0.002026 < 0.006691 < 0.02025 |
| tone | eye | pass | grey 2.372 < 9.746 < 41.13 < 182.7 < 893.9; colour 3.631 < 14.85 < 62.05 < 270.6 < 1280 |
| shift | cvvdp | FAIL | black/white shift / +5% = 1.12; red/yellow shift / +5% = 1.31 |
| shift | ssimulacra2 | FAIL | black/white shift / +5% = 7.9; red/yellow shift / +5% = 7.9 |
| shift | butteraugli | FAIL | black/white shift / +5% = 1.15; red/yellow shift / +5% = 1.27 |
| shift | butteraugli_max | FAIL | black/white shift / +5% = 0.998; red/yellow shift / +5% = 1.04 |
| shift | scielab | FAIL | black/white shift / +5% = 5.32; red/yellow shift / +5% = 4.57 |
| shift | msssim | FAIL | black/white shift / +5% = 4.44; red/yellow shift / +5% = 5.14 |
| shift | dists | FAIL | black/white shift / +5% = 0.917; red/yellow shift / +5% = 0.478 |
| shift | lpips | FAIL | black/white shift / +5% = 1.55; red/yellow shift / +5% = 2.11 |
| shift | eye | FAIL | black/white shift / +5% = 6.95; red/yellow shift / +5% = 3.47 |
| grey50 | cvvdp | pass | checker 6.396 < blue 6.995 < clusters 7.366 |
| grey50 | ssimulacra2 | pass | checker 824.1 < blue 1269 < clusters 1369 |
| grey50 | butteraugli | pass | checker 169.2 < blue 228.4 < clusters 233.9 |
| grey50 | butteraugli_max | pass | checker 216.6 < blue 322.8 < clusters 339 |
| grey50 | scielab | FAIL | checker 56.4 < blue 56.1 < clusters 56.05 |
| grey50 | msssim | pass | checker 0.9791 < blue 0.9891 < clusters 0.9914 |
| grey50 | dists | FAIL | checker 0.3897 < blue 0.4757 < clusters 0.4196 |
| grey50 | lpips | pass | checker 0.7916 < blue 0.8232 < clusters 0.8485 |
| grey50 | eye | FAIL | checker 0.9535 < blue 38.67 < clusters 30.29 |
| clash | cvvdp | FAIL | grey: clash 7.617 vs alone 7.597, 7.744; salmon: clash 7.082 vs alone 7.049, 7.366 |
| clash | ssimulacra2 | pass | grey: clash 1035 vs alone 908.1, 922; salmon: clash 1343 vs alone 1086, 1289 |
| clash | butteraugli | FAIL | grey: clash 116.5 vs alone 138.8, 73.85; salmon: clash 110.9 vs alone 133.6, 71.51 |
| clash | butteraugli_max | FAIL | grey: clash 212.1 vs alone 213, 127.3; salmon: clash 175.5 vs alone 183.1, 113 |
| clash | scielab | FAIL | grey: clash 29.67 vs alone 27.6, 31.7; salmon: clash 35.6 vs alone 32.06, 39.56 |
| clash | msssim | FAIL | grey: clash 0.9138 vs alone 0.9337, 0.8857; salmon: clash 0.9429 vs alone 0.9533, 0.9283 |
| clash | dists | pass | grey: clash 0.5308 vs alone 0.5227, 0.4814; salmon: clash 0.5243 vs alone 0.5136, 0.477 |
| clash | lpips | FAIL | grey: clash 0.9288 vs alone 0.8761, 0.9976; salmon: clash 0.8483 vs alone 0.8028, 0.8914 |
| clash | eye | FAIL | grey: clash 178.7 vs alone 22.21, 197.8; salmon: clash 199.9 vs alone 95.29, 249.4 |
| banding | cvvdp | FAIL | white: bands 4.076 vs dither 6.484; yellow: bands 4.416 vs dither 6.611 |
| banding | ssimulacra2 | FAIL | white: bands 161.6 vs dither 846.8; yellow: bands 163.2 vs dither 980.7 |
| banding | butteraugli | FAIL | white: bands 78.57 vs dither 120.2; yellow: bands 93.49 vs dither 120.7 |
| banding | butteraugli_max | FAIL | white: bands 115.7 vs dither 223.7; yellow: bands 141.6 vs dither 220.4 |
| banding | scielab | FAIL | white: bands 17.14 vs dither 27.95; yellow: bands 21.62 vs dither 35.28 |
| banding | msssim | FAIL | white: bands 0.1516 vs dither 0.7777; yellow: bands 0.1493 vs dither 0.7737 |
| banding | dists | FAIL | white: bands 0.1744 vs dither 0.4799; yellow: bands 0.1637 vs dither 0.5062 |
| banding | lpips | FAIL | white: bands 0.3221 vs dither 0.7443; yellow: bands 0.3073 vs dither 0.7667 |
| banding | eye | pass | white: bands 1466 vs dither 32.14; yellow: bands 3337 vs dither 71.46 |
| dark | cvvdp | FAIL | dispersed 6.917, empty 0.9135, clustered 6.947 |
| dark | ssimulacra2 | FAIL | dispersed 556.4, empty 145.9, clustered 261.9 |
| dark | butteraugli | FAIL | dispersed 104.1, empty 80.34, clustered 130 |
| dark | butteraugli_max | FAIL | dispersed 226, empty 130.3, clustered 305.6 |
| dark | scielab | FAIL | dispersed 19.26, empty 9.475, clustered 19.49 |
| dark | msssim | FAIL | dispersed 0.5885, empty 0.4082, clustered 0.5166 |
| dark | dists | FAIL | dispersed 0.4723, empty 0.2532, clustered 0.4785 |
| dark | lpips | FAIL | dispersed 0.5471, empty 0.2972, clustered 0.4735 |
| dark | eye | FAIL | dispersed 122.7, empty 64.36, clustered 535.9 |

## Runtime on one 256x192 pair

The grey ramp against its blue-noise dither, 4 CPU threads, no GPU; seconds, the first call (loading the models, writing PNGs) and the best of two more.

| metric | first | steady |
|---|---|---|
| cvvdp | 0.04 | 0.03 |
| ssimulacra2 | 0.64 | 0.60 |
| butteraugli | 1.05 | 0.91 |
| butteraugli_max | 1.02 | 0.90 |
| scielab | 0.07 | 0.07 |
| msssim | 0.80 | 0.43 |
| dists | 9.57 | 9.00 |
| lpips | 9.91 | 9.21 |
| eye | 0.01 | 0.01 |
