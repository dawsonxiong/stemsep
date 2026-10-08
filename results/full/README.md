### T4 (Tesla T4, 14.6 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| demucs apply_model (production) | 9.19 | 1.00× | 26.6 | 1.61 | 0.053 | 7.14 | 7.93 | – | 10.08 |
| eager fp32 ov.25 b1 | 8.94 | 1.03× | 27.4 | 2.00 | 0.051 | 6.95 | 7.94 | – | 8.94 |
| eager fp32 ov.25 b8 | 6.95 | 1.32× | 35.3 | 5.45 | 0.040 | 6.95 | 7.94 | 122.3 | 7.00 |
| eager fp32 ov.10 b8 | 5.83 | 1.58× | 42.0 | 5.43 | 0.034 | 6.93 | 7.77 | – | 5.81 |
| eager tf32 ov.10 b8 | 5.84 | 1.58× | 42.0 | 5.43 | 0.034 | 6.93 | 7.77 | 129.6 | 5.82 |
| eager fp16 ov.10 b4 | 4.45 | 2.07× | 55.1 | 3.52 | 0.026 | 6.93 | 7.77 | 56.5 | 4.69 |
| eager fp16 ov.10 b8 | 3.91 | 2.35× | 62.6 | 4.96 | 0.022 | 6.93 | 7.77 | 57.4 | 4.01 |
| eager fp16 ov.10 b16 | 3.66 | 2.51× | 66.9 | 8.53 | 0.021 | 6.93 | 7.77 | 56.9 | 3.66 |
| eager fp16 ov.25 b8 | 4.67 | 1.97× | 52.4 | 4.98 | 0.027 | 6.95 | 7.94 | 58.1 | 4.71 |
| eager fp16 ov.05 b8 | 3.83 | 2.40× | 64.0 | 4.96 | 0.022 | 6.87 | 7.73 | – | 3.86 |
| compile fp16 ov.10 b8 | 2.47 | 3.72× | 99.1 | 4.41 | 0.014 | 6.93 | 7.77 | 60.1 | 283.66 |

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- lameenc q5: **8.51 s**, 0.199¢ per song (decode 0.68s, separate 3.87s, encode 3.95s)
- lameenc q2: **12.94 s**, 0.303¢ per song (decode 0.68s, separate 3.95s, encode 8.31s)
- ffmpeg q5: **9.63 s**, 0.225¢ per song (decode 0.73s, separate 3.91s, encode 4.99s)

### L4 (NVIDIA L4, 22.0 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| demucs apply_model (production) | 4.91 | 1.00× | 49.9 | 1.86 | 0.035 | 7.03 | 7.92 | – | 6.57 |
| eager fp32 ov.25 b1 | 4.54 | 1.08× | 53.9 | 2.04 | 0.033 | 6.94 | 7.94 | – | 4.54 |
| eager fp32 ov.25 b8 | 4.62 | 1.06× | 53.0 | 5.20 | 0.033 | 6.94 | 7.94 | 72.0 | 4.69 |
| eager fp32 ov.10 b8 | 4.03 | 1.22× | 60.8 | 5.18 | 0.029 | 6.92 | 7.76 | – | 4.11 |
| eager tf32 ov.10 b8 | 3.58 | 1.37× | 68.4 | 5.18 | 0.026 | 6.92 | 7.76 | 71.0 | 3.63 |
| eager fp16 ov.10 b4 | 3.02 | 1.63× | 81.2 | 3.53 | 0.022 | 6.92 | 7.76 | 57.2 | 3.35 |
| eager fp16 ov.10 b8 | 2.90 | 1.69× | 84.4 | 4.68 | 0.021 | 6.92 | 7.76 | 57.9 | 2.89 |
| eager fp16 ov.10 b16 | 2.78 | 1.77× | 88.1 | 8.62 | 0.020 | 6.92 | 7.76 | 57.8 | 2.79 |
| eager fp16 ov.25 b8 | 3.42 | 1.44× | 71.7 | 4.70 | 0.024 | 6.94 | 7.94 | 58.4 | 3.49 |
| eager fp16 ov.05 b8 | 2.80 | 1.75× | 87.6 | 4.67 | 0.020 | 6.86 | 7.72 | – | 2.80 |
| eager bf16 ov.10 b8 | 3.01 | 1.63× | 81.3 | 4.99 | 0.022 | 6.93 | 7.78 | 36.7 | 3.26 |
| compile bf16 ov.10 b8 | 1.74 | 2.81× | 140.4 | 4.39 | 0.013 | 6.92 | 7.76 | 40.3 | 452.47 |

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- lameenc q5: **8.13 s**, 0.237¢ per song (decode 0.86s, separate 2.87s, encode 4.41s)
- lameenc q2: **14.38 s**, 0.420¢ per song (decode 0.86s, separate 2.87s, encode 10.65s)
- ffmpeg q5: **8.99 s**, 0.263¢ per song (decode 0.88s, separate 2.86s, encode 5.24s)

### A10 (NVIDIA A10G, 22.1 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| demucs apply_model (production) | 5.47 | 1.00× | 44.8 | 1.92 | 0.050 | 7.11 | 7.93 | – | 6.66 |
| eager fp32 ov.25 b1 | 5.14 | 1.06× | 47.6 | 2.04 | 0.047 | 6.94 | 7.94 | – | 5.15 |
| eager fp32 ov.25 b8 | 2.90 | 1.88× | 84.4 | 5.53 | 0.027 | 6.94 | 7.94 | 73.7 | 2.92 |
| eager fp32 ov.10 b8 | 2.44 | 2.24× | 100.3 | 5.51 | 0.022 | 6.92 | 7.76 | – | 2.44 |
| eager tf32 ov.10 b8 | 2.38 | 2.30× | 103.1 | 5.51 | 0.022 | 6.92 | 7.76 | 71.2 | 2.38 |
| eager fp16 ov.10 b4 | 2.25 | 2.43× | 108.9 | 3.54 | 0.021 | 6.92 | 7.76 | 57.5 | 2.59 |
| eager fp16 ov.10 b8 | 1.88 | 2.90× | 130.0 | 4.64 | 0.017 | 6.92 | 7.76 | 57.5 | 1.89 |
| eager fp16 ov.10 b16 | 1.67 | 3.28× | 146.8 | 8.55 | 0.015 | 6.92 | 7.76 | 57.4 | 1.67 |
| eager fp16 ov.25 b8 | 2.24 | 2.45× | 109.5 | 5.00 | 0.021 | 6.94 | 7.94 | 58.4 | 2.29 |
| eager fp16 ov.05 b8 | 1.85 | 2.96× | 132.5 | 4.64 | 0.017 | 6.86 | 7.72 | – | 1.88 |
| eager bf16 ov.10 b8 | 1.93 | 2.84× | 127.0 | 4.68 | 0.018 | 6.93 | 7.77 | 36.5 | 2.01 |
| compile bf16 ov.10 b8 | 1.03 | 5.32× | 238.1 | 4.51 | 0.009 | 6.92 | 7.77 | 40.8 | 465.82 |

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- lameenc q5: **5.68 s**, 0.214¢ per song (decode 0.72s, separate 1.89s, encode 3.07s)
- lameenc q2: **10.04 s**, 0.378¢ per song (decode 0.72s, separate 1.89s, encode 7.43s)
- ffmpeg q5: **6.56 s**, 0.247¢ per song (decode 0.76s, separate 1.89s, encode 3.91s)
