### T4 (Tesla T4, 14.6 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- lameenc q5: **8.73 s**, 0.204¢ per song (decode 0.75s, separate 3.92s, encode 4.06s)
- lameenc q2: **13.26 s**, 0.311¢ per song (decode 0.75s, separate 3.93s, encode 8.58s)
- ffmpeg q5: **9.74 s**, 0.228¢ per song (decode 0.73s, separate 3.94s, encode 5.07s)

### L4 (NVIDIA L4, 22.0 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- streaming lameenc q5: **6.29 s**, 0.184¢ per song (decode 1.20s, separate 3.03s, encode 2.05s)
- streaming lameenc q5 b16: **6.63 s**, 0.194¢ per song (decode 0.91s, separate 2.95s, encode 2.76s)
- lameenc q5: **8.26 s**, 0.241¢ per song (decode 0.92s, separate 2.86s, encode 4.48s)
- lameenc q2: **14.33 s**, 0.419¢ per song (decode 0.92s, separate 2.86s, encode 10.55s)
- ffmpeg q5: **9.07 s**, 0.265¢ per song (decode 0.97s, separate 2.85s, encode 5.26s)

### A10 (NVIDIA A10G, 22.1 GiB) — torch 2.11.0+cu130, CUDA 13.0, 245 s song

| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min | SDR (dB) | cSDR (dB) | vs fp32, same overlap (dB) | first call (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|

End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:

- streaming lameenc q5: **4.37 s**, 0.164¢ per song (decode 0.86s, separate 2.00s, encode 1.50s)
- streaming lameenc q5 b16: **4.61 s**, 0.174¢ per song (decode 0.79s, separate 1.81s, encode 2.00s)
- lameenc q5: **5.82 s**, 0.219¢ per song (decode 0.82s, separate 1.90s, encode 3.10s)
- lameenc q2: **10.26 s**, 0.386¢ per song (decode 0.89s, separate 1.87s, encode 7.50s)
- ffmpeg q5: **6.78 s**, 0.255¢ per song (decode 0.87s, separate 1.90s, encode 4.01s)
