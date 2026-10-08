### T4 (Tesla T4, 14.6 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- streaming lameenc q5: **6.58 s**, 0.154¢ per song (decode 0.61s, separate 4.81s, encode 1.15s)
- streaming lameenc q5 b16: **7.39 s**, 0.173¢ per song (decode 0.59s, separate 4.61s, encode 2.19s)
- lameenc q5: **9.97 s**, 0.233¢ per song (decode 0.59s, separate 4.38s, encode 5.00s)
- lameenc q2: **14.65 s**, 0.343¢ per song (decode 0.58s, separate 4.38s, encode 9.69s)
- ffmpeg q5: **10.84 s**, 0.254¢ per song (decode 0.59s, separate 4.44s, encode 5.82s)

### L4 (NVIDIA L4, 22.0 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- streaming lameenc q5: **5.70 s**, 0.166¢ per song (decode 0.58s, separate 3.25s, encode 1.87s)
- streaming lameenc q5 b16: **6.23 s**, 0.182¢ per song (decode 0.56s, separate 2.90s, encode 2.76s)
- lameenc q5: **7.96 s**, 0.232¢ per song (decode 0.55s, separate 2.86s, encode 4.55s)
- lameenc q2: **13.90 s**, 0.406¢ per song (decode 0.56s, separate 2.85s, encode 10.49s)
- ffmpeg q5: **8.57 s**, 0.250¢ per song (decode 0.55s, separate 2.85s, encode 5.18s)

### A10 (NVIDIA A10, 22.1 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- streaming lameenc q5: **3.97 s**, 0.149¢ per song (decode 0.42s, separate 2.23s, encode 1.32s)
- streaming lameenc q5 b16: **4.28 s**, 0.161¢ per song (decode 0.39s, separate 1.93s, encode 1.96s)
- lameenc q5: **5.60 s**, 0.211¢ per song (decode 0.42s, separate 1.97s, encode 3.20s)
- lameenc q2: **9.83 s**, 0.370¢ per song (decode 0.43s, separate 1.97s, encode 7.43s)
- ffmpeg q5: **5.92 s**, 0.223¢ per song (decode 0.40s, separate 1.96s, encode 3.56s)
