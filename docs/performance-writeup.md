# Making Stem Player fast

October 2026. A split used to take about a minute from pressing Split to hearing the stems. It now takes 6–9 s, and with live playback the song starts in about 1.5 s. The work covered [stemplayer](https://github.com/dawsonxiong/stemplayer), the app, and [stemsep](https://github.com/dawsonxiong/stemsep), the faster Demucs runner it uses.

## Results

Times are medians from Split until all four stems play in the browser, measured from Toronto.

| | Before | After |
|---|--:|--:|
| Warm split, 4 min MP3 | 60.6 s | 7.6 s |
| Cold split | 59.9 s | 9.3 s |
| First sound, with live playback | – | 1.5 s |
| YouTube link that YouTube refused | 47.6–60.8 s | 9.4–21.1 s |
| GPU job alone | 18.4 s | 5–6 s |
| Cost per song | 0.54¢ | ~0.15–0.30¢ |

Quality is unchanged: within 0.02 dB SDR of demucs on 12 MUSDB18-HQ songs.

## What changed

- **Stale job status.** Most of the old minute was Workers KV caching the job record for up to 60 s per location. Records moved to R2 and `/api/status` long-polls them, so "ready" arrives ~0.25 s after the job ends.
- **Faster separation.** stemsep runs the same `htdemucs` weights in fp16, 8 chunks per pass with 10% overlap, and encodes the MP3s as stems come off the GPU. Separation alone went from 5.47 s to 1.88 s on an A10.
- **Less waiting to start.** The GPU is pinned to the US (Modal had put it in Frankfurt). The page wakes Modal on load, picking a song warms a GPU, and the upload starts at pick rather than at Split. Each deploy warms the memory snapshots.
- **Live playback.** The GPU job streams each stem's MP3 frames over a WebSocket to a Durable Object, which relays them to the browser. An AudioWorklet plays all four stems in step while separation is still running, then hands over to the finished files without a gap.
- **Link downloads.** YouTube refuses many cloud IPs. Downloads moved to Google Cloud, with stand-ins in three other regions racing when one is refused. A residential proxy and the paid Apify actor come after that. The proxy is out of data for now and skips itself in ~2 s.

## Tried and dropped

`torch.compile`, ONNX/TensorRT, GPU memory snapshots, T4s (17.5 s cold starts), VBR and custom LAME builds, multipart stem uploads, a Media Source Extensions player (silent on iPhone), chunked POSTs to the relay (Cloudflare buffers the body), and retrying refused downloads in the same region.

## What's left

- **Slow uplinks.** The upload is now the biggest wait: 15 s on 25/5 Mbps and 30 s on poor mobile. Once it's in, the rest takes ~6.5 s. Possible fixes are separating while the file uploads, or shrinking WAV and FLAC in the browser.
- **Loud songs.** A stem that clips early stops streaming, so the song plays from the files instead. Phase 5 of the live playback plan fixes this.
- **Apify's $5 monthly cap**, about 60–80 refused songs, if the region stand-ins stop being enough.
- **Firefox** is untested with live playback.

Full details, measurements and gotchas are in stemplayer's [`docs/handoff.md`](https://github.com/dawsonxiong/stemplayer/blob/main/docs/handoff.md). Benchmarks and raw results are in stemsep's [README](https://github.com/dawsonxiong/stemsep#readme) and [`results/`](https://github.com/dawsonxiong/stemsep/tree/main/results).
