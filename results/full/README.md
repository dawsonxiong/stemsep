### T4 (Tesla T4, 14.6 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32 (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| demucs apply_model (production) | 9.62 | 1.00× | 25.5 | 1.63 | 0.059 | 7.14 | 7.99 | – | 2.51 |
| eager fp32 ov.25 b1 | 9.57 | 1.00× | 25.6 | 2.14 | 0.059 | 6.95 | 7.94 | – | 0.94 |
| eager fp32 ov.25 b8 | 7.92 | 1.21× | 30.9 | 6.83 | 0.049 | 6.95 | 7.94 | 129.6 | 11.94 |
| eager fp32 ov.10 b8 | 6.95 | 1.38× | 35.2 | 5.71 | 0.043 | 6.93 | 7.77 | 16.6 | 9.02 |
| eager tf32 ov.10 b8 | 6.86 | 1.40× | 35.7 | 5.67 | 0.042 | 6.93 | 7.77 | 16.6 | 0.60 |
| eager fp16 ov.10 b4 | 4.86 | 1.98× | 50.4 | 3.63 | 0.030 | 6.93 | 7.77 | 16.6 | 2.21 |
| eager fp16 ov.10 b8 | 4.43 | 2.17× | 55.3 | 5.18 | 0.027 | 6.93 | 7.77 | 16.6 | 0.44 |
| eager fp16 ov.10 b16 | 4.26 | 2.26× | 57.5 | 9.44 | 0.026 | 6.93 | 7.77 | 16.6 | 0.44 |
| eager fp16 ov.25 b8 | 5.24 | 1.83× | 46.7 | 5.32 | 0.032 | 6.95 | 7.94 | 58.7 | 0.53 |
| eager fp16 ov.05 b8 | 4.29 | 2.24× | 57.1 | 5.71 | 0.026 | 6.87 | 7.73 | 16.8 | 0.44 |
| compile fp16 ov.10 b8 | 3.30 | 2.91× | 74.2 | 3.69 | 0.020 | 6.93 | 7.77 | 16.6 | 382.54 |

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- ffmpeg encoder: **18.91 s** (decode 1.44s, separate 4.46s, encode 13.01s)
- lameenc encoder: **15.09 s** (decode 0.98s, separate 4.24s, encode 9.86s)

### L4 (NVIDIA L4, 22.0 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32 (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| demucs apply_model (production) | 4.86 | 1.00× | 50.4 | 1.58 | 0.037 | 7.14 | 7.91 | – | 1.98 |
| eager fp32 ov.25 b1 | 4.46 | 1.09× | 55.0 | 2.13 | 0.034 | 6.94 | 7.94 | – | 0.44 |
| eager fp32 ov.25 b8 | 4.46 | 1.09× | 54.9 | 7.83 | 0.034 | 6.94 | 7.94 | 129.6 | 7.99 |
| eager fp32 ov.10 b8 | 3.76 | 1.29× | 65.1 | 5.72 | 0.029 | 6.92 | 7.76 | 16.5 | 3.66 |
| eager tf32 ov.10 b8 | 3.42 | 1.42× | 71.5 | 5.66 | 0.026 | 6.92 | 7.76 | 16.5 | 0.30 |
| eager fp16 ov.10 b4 | 3.00 | 1.62× | 81.8 | 3.44 | 0.023 | 6.92 | 7.76 | 16.5 | 8.54 |
| eager fp16 ov.10 b8 | 2.85 | 1.70× | 85.8 | 6.05 | 0.022 | 6.92 | 7.76 | 16.5 | 0.26 |
| eager fp16 ov.10 b16 | 2.75 | 1.76× | 88.9 | 12.71 | 0.021 | 6.92 | 7.76 | 16.5 | 0.26 |
| eager fp16 ov.25 b8 | 3.36 | 1.44× | 72.8 | 5.32 | 0.026 | 6.94 | 7.94 | 58.4 | 0.33 |
| eager fp16 ov.05 b8 | 2.76 | 1.76× | 88.6 | 5.36 | 0.021 | 6.86 | 7.72 | 16.6 | 0.26 |
| eager bf16 ov.10 b8 | 2.96 | 1.64× | 82.7 | 5.90 | 0.022 | 6.93 | 7.78 | 16.6 | 8.37 |
| compile bf16 ov.10 b8 | 1.71 | 2.84× | 143.4 | 3.71 | 0.013 | 6.92 | 7.76 | 16.5 | 431.72 |

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- ffmpeg encoder: **16.51 s** (decode 1.28s, separate 2.82s, encode 12.41s)
- lameenc encoder: **14.31 s** (decode 0.96s, separate 2.84s, encode 10.51s)

### A10 (NVIDIA A10, 22.1 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32 (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| demucs apply_model (production) | 5.47 | 1.00× | 44.8 | 1.93 | 0.053 | 7.11 | 7.95 | – | 1.45 |
| eager fp32 ov.25 b1 | 5.18 | 1.06× | 47.3 | 2.21 | 0.050 | 6.94 | 7.94 | – | 0.49 |
| eager fp32 ov.25 b8 | 3.06 | 1.79× | 80.0 | 5.50 | 0.030 | 6.94 | 7.94 | 129.6 | 5.71 |
| eager fp32 ov.10 b8 | 2.56 | 2.14× | 95.7 | 5.74 | 0.025 | 6.92 | 7.76 | 16.5 | 2.55 |
| eager tf32 ov.10 b8 | 2.30 | 2.38× | 106.5 | 5.69 | 0.022 | 6.92 | 7.76 | 16.5 | 0.22 |
| eager fp16 ov.10 b4 | 2.27 | 2.41× | 107.7 | 3.53 | 0.022 | 6.92 | 7.76 | 16.5 | 6.27 |
| eager fp16 ov.10 b8 | 1.90 | 2.88× | 129.2 | 5.07 | 0.018 | 6.92 | 7.76 | 16.5 | 0.21 |
| eager fp16 ov.10 b16 | 1.70 | 3.21× | 144.0 | 9.45 | 0.016 | 6.92 | 7.76 | 16.5 | 0.21 |
| eager fp16 ov.25 b8 | 2.25 | 2.43× | 108.7 | 5.33 | 0.022 | 6.94 | 7.94 | 58.5 | 0.25 |
| eager fp16 ov.05 b8 | 1.86 | 2.95× | 132.0 | 5.39 | 0.018 | 6.86 | 7.72 | 16.6 | 0.21 |
| eager bf16 ov.10 b8 | 1.94 | 2.83× | 126.6 | 5.05 | 0.019 | 6.93 | 7.78 | 16.6 | 6.04 |
| compile bf16 ov.10 b8 | 1.07 | 5.12× | 229.4 | 3.92 | 0.010 | 6.92 | 7.78 | 16.5 | 393.17 |

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- ffmpeg encoder: **12.36 s** (decode 1.69s, separate 1.88s, encode 8.79s)
- lameenc encoder: **9.76 s** (decode 0.51s, separate 1.88s, encode 7.37s)
