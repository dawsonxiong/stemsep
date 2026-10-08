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

## Second pass, per stage

Staging (`stem-demucs-staging`, A10) warm jobs, with webhooks to a Modal HTTPS endpoint and stage timings from the job's own log. Six warm jobs before this pass:

`r2_client 0.004, report_5 0.29, fetch 0.27, report_15 0.40, decode 0.82, separate 1.89, report_80 0.35, encode_upload 3.67, report_ready 0.30`: caller sees 8.0–8.2 s.

After (progress reports in the background, streaming encode, libsndfile decode, small first batch), five warm jobs:

`fetch 0.30–0.36, decode 0.55, separate 2.11, encode 1.37, upload 0.63–1.13, report_ready 0.21–0.23`: caller sees 5.3–5.8 s (other runs with slower R2 and webhook responses: 5.9–6.7 s).

Cold jobs (after > scaledown_window idle), separate including the wait for the background warm-up: 2.3–3.0 s (one 4.0 s), against 3.1–3.5 s with no warm-up.

## Third pass: in front of the GPU

The Modal API endpoint the Worker calls, timed with an unauthenticated POST (401):
- warm: 0.65 s
- cold: 5.65 s, then 3.2–4.2 s without a memory snapshot
- cold with a memory snapshot: 2.8–3.1 s, but 7.3 s and 10.4 s when the snapshot was being retaken

stemplayer keeps it warm 20 minutes after each request (`scaledown_window=1200`) and doesn't snapshot it.

Cold GPU jobs preceded by a `/warm` call 6 s earlier (standing in for the upload), after > scaledown_window idle: 7.96 s and 6.38 s seen by the caller, plus 20.71 s for the first cold start after a deploy, which took the snapshot. Without the head start: 9.4–12 s.

Final warm jobs on staging: 4.95, 5.15, 5.35, 5.97 s.

## Job status through Workers KV

A throwaway Worker with its own KV namespace (both deleted after) stood in for stemplayer's status path. This machine (Cloudflare location YYZ) wrote "processing" and kept reading the key, as `/api/separate` and the browser's polling do. A Modal function, whose requests reached Cloudflare at LAX like the job's webhook, then wrote "ready". Time until YYZ read "ready", six trials (`kv-propagation.jsonl`): 26.4, 6.8, 38.2, 18.3, 50.1, 30.0 s. KV caches a key at each location, so a person far from Modal's region sees progress and "ready" up to a minute after the stems exist.
