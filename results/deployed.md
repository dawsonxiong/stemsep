# Deployed jobs: stemplayer's worker, before and after

Two copies of stemplayer's Modal app were deployed side by side as throwaway apps on 2026-10-08:
- `stem-demucs-baseline`: `modal/app.py` from stemplayer `main` (demucs `apply_model`, L4, LAME q2)
- `stem-demucs-staging`: the `dawson/stemsep` branch (stemsep fp16, 10% overlap, batch 8, A10, LAME q5)

Each job was `Separator.separate(input_r2_key=...)` on the test set's 245 s `mixture.mp3`, called from a Modal function in the same region. It covered R2 download, decode, separate, encode four 192 kbps MP3s, and R2 upload. The webhook URL pointed at a closed local port, so reporting cost nothing. Times are as seen by the caller.

| | warm jobs (s) | cold jobs, after > scaledown_window idle (s) |
|---|---|---|
| baseline (main) | 18.39, 17.84, 18.75, 18.35, 19.05 | 29.17, 32.61 |
| staging (stemsep) | 7.92, 7.88, 7.95, 6.77, 7.91 | 16.76, 20.34 |

The first call after each deploy (47.5 s baseline, 29.2 s staging) took the memory snapshot. It isn't counted.

Every staging job wrote four stems of `audio/mpeg`, 5,878,804 bytes and 244.95 s each. A junk upload returned `status: error` with the user-facing "couldn't be read as audio" message, and the container kept serving.

Cost per warm job at Modal's prices for 4 cores and 8 GiB: baseline 18.4 s × $0.000292/s ≈ 0.54¢; staging 7.9 s × $0.000376/s ≈ 0.30¢.
