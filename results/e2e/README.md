### T4 (Tesla T4, 14.6 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32 (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- lameenc q5: **8.73 s**, 0.220¢ per song (decode 0.75s, separate 3.92s, encode 4.06s)
- lameenc q2: **13.26 s**, 0.334¢ per song (decode 0.75s, separate 3.93s, encode 8.58s)
- ffmpeg q5: **9.74 s**, 0.245¢ per song (decode 0.73s, separate 3.94s, encode 5.07s)

### L4 (NVIDIA L4, 22.0 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32 (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- lameenc q5: **8.90 s**, 0.276¢ per song (decode 1.12s, separate 3.01s, encode 4.77s)
- lameenc q2: **14.90 s**, 0.462¢ per song (decode 1.11s, separate 3.01s, encode 10.78s)
- ffmpeg q5: **9.76 s**, 0.303¢ per song (decode 1.21s, separate 2.99s, encode 5.56s)

### A10 (NVIDIA A10G, 22.1 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32 (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- lameenc q5: **5.80 s**, 0.229¢ per song (decode 0.85s, separate 1.90s, encode 3.06s)
- lameenc q2: **10.18 s**, 0.401¢ per song (decode 0.83s, separate 1.89s, encode 7.45s)
- ffmpeg q5: **6.83 s**, 0.269¢ per song (decode 0.93s, separate 1.89s, encode 4.01s)
