# Separation quality: production demucs vs stemsep

`cd bench && uv run modal run quality.py` (NVIDIA L4, htdemucs). SDR is the global
energy-ratio SDR ("nSDR") and cSDR the median over 1 s windows, from
`stemsep.quality.report`; a track's score is the mean over its four stems.
"prod" is production demucs (`apply_model(shifts=1, overlap=0.25)`), whose one random
shift makes it a different chunk grid per run; its per-track score here is the mean
of three seeds. Δ columns are paired per-track differences, mean ± standard error.
"delay k s" rows run on the input with k s of silence prepended (then trimmed), which
moves only the chunk grid. "demucs stride" is stemsep with its chunks cut every
`int(0.75 * segment)` samples, as demucs does, instead of spread evenly (last chunk
still flush with the end): a diagnostic, not an engine option.

## Full songs: MUSDB18-HQ test, first 12 by name (bench/fulltracks.py)

12 track(s), 46.9 min of audio.

| config | median SDR | mean SDR | Δ vs prod | Δ vs demucs s0 | cSDR | Δ cSDR vs prod | drums | bass | other | vocals | vs demucs s0 (dB) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| demucs shifts=0 | 8.54 | 7.62 | +0.00 ± 0.01 | – | 8.80 | +0.01 ± 0.01 | 7.84 | 9.21 | 6.49 | 9.43 | – |
| demucs shifts=1 seed 0 | 8.57 | 7.62 | -0.00 ± 0.01 | -0.00 ± 0.01 | 8.77 | +0.00 ± 0.01 | 7.88 | 9.23 | 6.51 | 9.42 | 22.1 |
| demucs shifts=1 seed 1 | 8.55 | 7.62 | +0.00 ± 0.01 | +0.00 ± 0.01 | 8.81 | -0.01 ± 0.01 | 7.77 | 9.22 | 6.50 | 9.45 | 21.8 |
| demucs shifts=1 seed 2 | 8.55 | 7.62 | +0.00 ± 0.00 | +0.00 ± 0.01 | 8.83 | +0.00 ± 0.01 | 7.85 | 9.23 | 6.51 | 9.44 | 22.1 |
| stemsep fp32 ov.25 | 8.58 | 7.61 | -0.01 ± 0.01 | -0.01 ± 0.01 | 8.82 | +0.02 ± 0.03 | 7.87 | 9.15 | 6.49 | 9.45 | 20.3 |
| stemsep fp32 ov.10 | 8.54 | 7.60 | -0.02 ± 0.01 | -0.02 ± 0.02 | 8.79 | -0.03 ± 0.02 | 7.74 | 9.26 | 6.54 | 9.45 | 19.0 |
| stemsep fp32 ov.05 | 8.46 | 7.57 | -0.06 ± 0.02 | -0.06 ± 0.03 | 8.78 | -0.02 ± 0.02 | 7.87 | 9.09 | 6.43 | 9.37 | 18.8 |
| stemsep fp16 ov.10 | 8.54 | 7.60 | -0.02 ± 0.01 | -0.02 ± 0.02 | 8.79 | -0.03 ± 0.02 | 7.74 | 9.26 | 6.54 | 9.45 | 19.0 |
| stemsep bf16 ov.10 | 8.54 | 7.60 | -0.02 ± 0.01 | -0.02 ± 0.02 | 8.79 | -0.03 ± 0.02 | 7.74 | 9.26 | 6.54 | 9.44 | 18.9 |
| stemsep fp32 ov.25 delay 0.1s | 8.58 | 7.62 | -0.01 ± 0.01 | -0.01 ± 0.01 | 8.78 | -0.00 ± 0.03 | 7.84 | 9.15 | 6.49 | 9.46 | 20.1 |
| stemsep fp32 ov.25 delay 0.2s | 8.59 | 7.62 | -0.01 ± 0.01 | -0.01 ± 0.01 | 8.81 | -0.00 ± 0.03 | 7.85 | 9.18 | 6.51 | 9.45 | 20.2 |
| stemsep fp32 ov.25 delay 0.3s | 8.61 | 7.62 | +0.00 ± 0.02 | +0.00 ± 0.02 | 8.78 | +0.00 ± 0.03 | 7.86 | 9.24 | 6.55 | 9.49 | 20.1 |
| stemsep fp32 ov.25 delay 0.4s | 8.59 | 7.63 | +0.01 ± 0.01 | +0.01 ± 0.01 | 8.79 | +0.00 ± 0.02 | 7.81 | 9.23 | 6.51 | 9.41 | 19.5 |
| stemsep fp32 ov.25 delay 0.5s | 8.59 | 7.63 | +0.01 ± 0.01 | +0.01 ± 0.01 | 8.82 | +0.02 ± 0.02 | 7.84 | 9.22 | 6.52 | 9.43 | 19.5 |
| stemsep fp32 ov.25 delay 1.5s | 8.60 | 7.61 | -0.01 ± 0.03 | -0.01 ± 0.03 | 8.83 | +0.01 ± 0.02 | 7.85 | 9.25 | 6.54 | 9.47 | 19.6 |
| stemsep fp32 ov.25 delay 3.0s | 8.59 | 7.61 | -0.01 ± 0.01 | -0.01 ± 0.02 | 8.84 | +0.02 ± 0.02 | 7.82 | 9.23 | 6.55 | 9.46 | 19.6 |
| demucs shifts=0 delay 1.5s | 8.54 | 7.62 | -0.00 ± 0.01 | -0.00 ± 0.01 | 8.84 | +0.02 ± 0.02 | 7.79 | 9.11 | 6.46 | 9.41 | 18.8 |
| demucs shifts=0 delay 3.0s | 8.59 | 7.63 | +0.01 ± 0.01 | +0.01 ± 0.01 | 8.82 | +0.02 ± 0.02 | 7.85 | 9.22 | 6.54 | 9.44 | 19.0 |
| stemsep fp32 ov.25 demucs stride | 8.54 | 7.62 | -0.00 ± 0.01 | -0.00 ± 0.00 | 8.81 | +0.01 ± 0.01 | 7.84 | 9.21 | 6.49 | 9.43 | 54.8 |

Chunk-grid noise: runs that differ only in where the grid falls, grouped by chunk spacing. Per track, the spread of mean SDR across the group, averaged over tracks.

| spacing | runs | mean SDR | std within track | range within track |
|---|--:|--:|--:|--:|
| stemsep spacing | 8 | 7.619 | 0.022 | 0.065 |
| demucs spacing | 7 | 7.622 | 0.023 | 0.063 |

Per track, mean SDR over stems:

| track | s | demucs s0 | prod (3 seeds) | stemsep fp32 ov.25 | delays min–max | demucs stride | fp32 ov.10 | fp16 ov.10 | bf16 ov.10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| AM Contra - Heart Peripheral | 210 | 4.71 | 4.71 | 4.72 | 4.69–4.73 | 4.71 | 4.65 | 4.65 | 4.65 |
| Al James - Schoolboy Facination | 201 | 6.96 | 7.01 | 6.94 | 6.96–7.07 | 6.96 | 7.01 | 7.01 | 7.01 |
| Angels In Amplifiers - I'm Alright | 180 | 9.33 | 9.33 | 9.34 | 9.32–9.35 | 9.33 | 9.33 | 9.33 | 9.32 |
| Arise - Run Run Run | 219 | 4.25 | 4.24 | 4.23 | 4.22–4.26 | 4.25 | 4.13 | 4.13 | 4.13 |
| BKS - Bulldozer | 337 | 9.28 | 9.28 | 9.25 | 9.24–9.28 | 9.28 | 9.27 | 9.27 | 9.26 |
| BKS - Too Much | 220 | 10.59 | 10.59 | 10.65 | 10.61–10.66 | 10.59 | 10.62 | 10.62 | 10.62 |
| Ben Carrigan - We'll Talk About It All Tonight | 255 | 8.98 | 8.99 | 9.04 | 9.02–9.03 | 8.98 | 8.92 | 8.92 | 8.92 |
| Bobby Nobody - Stitch Up | 221 | 8.10 | 8.12 | 8.12 | 8.12–8.18 | 8.10 | 8.17 | 8.17 | 8.17 |
| Buitraker - Revo X | 276 | 9.28 | 9.29 | 9.25 | 9.25–9.30 | 9.28 | 9.22 | 9.22 | 9.22 |
| Carlos Gonzalez - A Place For Us | 250 | 6.32 | 6.26 | 6.30 | 6.27–6.30 | 6.32 | 6.28 | 6.28 | 6.28 |
| Cristina Vane - So Easy | 254 | 9.35 | 9.35 | 9.32 | 9.32–9.35 | 9.35 | 9.31 | 9.31 | 9.31 |
| Detsky Sad - Walkie Talkie | 190 | 4.31 | 4.29 | 4.20 | 3.98–4.23 | 4.28 | 4.29 | 4.29 | 4.29 |

## Old set: 36 MUSDB18-7 test excerpts concatenated into one 245 s song

1 track(s), 4.1 min of audio.

| config | SDR | Δ vs prod | Δ vs demucs s0 | cSDR | Δ cSDR vs prod | drums | bass | other | vocals | vs demucs s0 (dB) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| demucs shifts=0 | 7.09 | -0.01 | – | 7.92 | -0.01 | 8.72 | 6.07 | 4.90 | 8.68 | – |
| demucs shifts=1 seed 0 | 7.11 | +0.01 | +0.02 | 7.90 | -0.03 | 8.79 | 6.10 | 4.91 | 8.64 | 19.4 |
| demucs shifts=1 seed 1 | 7.11 | +0.00 | +0.02 | 7.96 | +0.03 | 8.66 | 6.16 | 4.93 | 8.68 | 20.1 |
| demucs shifts=1 seed 2 | 7.09 | -0.01 | +0.00 | 7.92 | -0.01 | 8.70 | 6.08 | 4.91 | 8.67 | 19.5 |
| stemsep fp32 ov.25 | 6.94 | -0.17 | -0.15 | 7.94 | +0.01 | 8.35 | 5.88 | 4.92 | 8.60 | 16.7 |
| stemsep fp32 ov.10 | 6.92 | -0.18 | -0.17 | 7.76 | -0.17 | 8.43 | 5.88 | 4.81 | 8.55 | 16.7 |
| stemsep fp32 ov.05 | 6.86 | -0.25 | -0.23 | 7.72 | -0.21 | 8.27 | 5.86 | 4.83 | 8.45 | 15.4 |
| stemsep fp16 ov.10 | 6.92 | -0.18 | -0.17 | 7.76 | -0.17 | 8.43 | 5.88 | 4.81 | 8.55 | 16.7 |
| stemsep bf16 ov.10 | 6.93 | -0.17 | -0.16 | 7.78 | -0.15 | 8.45 | 5.91 | 4.82 | 8.55 | 16.7 |
| stemsep fp32 ov.25 delay 0.1s | 6.94 | -0.16 | -0.15 | 7.95 | +0.02 | 8.43 | 5.83 | 4.93 | 8.58 | 16.5 |
| stemsep fp32 ov.25 delay 0.2s | 6.94 | -0.16 | -0.15 | 7.94 | +0.01 | 8.39 | 5.86 | 4.93 | 8.60 | 16.5 |
| stemsep fp32 ov.25 delay 0.3s | 6.94 | -0.16 | -0.15 | 7.97 | +0.04 | 8.35 | 5.89 | 4.93 | 8.60 | 16.4 |
| stemsep fp32 ov.25 delay 0.4s | 6.93 | -0.17 | -0.16 | 7.99 | +0.06 | 8.33 | 5.90 | 4.92 | 8.59 | 16.3 |
| stemsep fp32 ov.25 delay 0.5s | 6.92 | -0.18 | -0.17 | 7.95 | +0.03 | 8.31 | 5.86 | 4.92 | 8.59 | 16.3 |
| stemsep fp32 ov.25 delay 1.5s | 6.91 | -0.19 | -0.18 | 7.89 | -0.03 | 8.24 | 5.91 | 4.92 | 8.58 | 15.8 |
| stemsep fp32 ov.25 delay 3.0s | 7.02 | -0.08 | -0.07 | 7.90 | -0.02 | 8.55 | 6.07 | 4.89 | 8.58 | 17.1 |
| demucs shifts=0 delay 1.5s | 7.06 | -0.04 | -0.03 | 7.97 | +0.05 | 8.54 | 6.21 | 4.90 | 8.59 | 16.7 |
| demucs shifts=0 delay 3.0s | 6.96 | -0.14 | -0.13 | 7.90 | -0.03 | 8.31 | 6.03 | 4.94 | 8.57 | 15.8 |
| stemsep fp32 ov.25 demucs stride | 7.11 | +0.01 | +0.02 | 7.95 | +0.02 | 8.72 | 6.11 | 4.92 | 8.68 | 30.8 |

Chunk-grid noise: runs that differ only in where the grid falls, grouped by chunk spacing. Per track, the spread of mean SDR across the group, averaged over tracks.

| spacing | runs | mean SDR | std within track | range within track |
|---|--:|--:|--:|--:|
| stemsep spacing | 8 | 6.945 | 0.034 | 0.112 |
| demucs spacing | 7 | 7.075 | 0.053 | 0.148 |
