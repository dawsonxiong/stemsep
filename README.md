# stemsep

Faster [HTDemucs](https://github.com/facebookresearch/demucs) inference on Modal GPUs. It replaces `demucs.apply.apply_model` in [stemplayer](https://github.com/dawsonxiong/stemplayer)'s separation worker: same model, same weights, same quality, at about a third of the GPU time on an A10 (1.7–2.9× faster across T4, L4 and A10) and less than half the time per job.

```python
from stemsep import Config, Separator
from stemsep.audio import decode, encode_all

sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
stems = sep.separate(decode("song.mp3"))  # {"drums": [2, L], "bass": ..., "other": ..., "vocals": ...}
encode_all(stems, {s: f"{s}.mp3" for s in stems}, sep.samplerate)
```

## Results

The production job on Modal, for a 4-minute song: download from R2, decode, separate, encode four 192 kbps MP3s, upload. Each row was measured on a separately deployed copy of the app (`results/deployed.md`).

| | GPU | warm job | cost per song¹ |
|---|---|--:|--:|
| before: demucs `apply_model`, LAME q2 | L4 | 18.4 s | 0.54¢ |
| after: stemsep fp16, 10% overlap, batch 8, LAME q5 | A10 | **7.9 s** | **0.30¢** |

¹ GPU + 4 cores + 8 GiB at Modal's per-second prices, for the job alone. Idle time while a container waits for the next song costs more than the songs themselves, so `scaledown_window` also went from 300 s to 120 s.

Separation alone, 245 s song (`results/full/`):

| GPU | demucs `apply_model` | stemsep fp16 b8 ov.10 | speedup | peak VRAM |
|---|--:|--:|--:|--:|
| T4 | 9.19 s | 3.91 s | 2.4× | 5.0 GiB |
| L4 | 4.91 s | 2.90 s | 1.7× | 4.7 GiB |
| A10 | 5.47 s | 1.88 s | 2.9× | 4.6 GiB |

Quality on 12 full MUSDB18-HQ test songs, 47 minutes (`results/quality/`):

| | median SDR | Δ vs production, paired |
|---|--:|--:|
| demucs, production settings | 8.56 | – |
| stemsep fp32, 25% overlap | 8.58 | −0.01 ± 0.01 dB |
| stemsep fp16, 10% overlap (shipped) | 8.54 | −0.02 ± 0.01 dB |
| stemsep fp32, 5% overlap | 8.46 | −0.06 ± 0.02 dB |

Moving the chunk grid alone moves a song's SDR by ±0.02 dB, so the shipped config is within noise of production.

## What was tried, and what came of it

Measured against the plan's priorities.

**Profiling (P0).** `Profiler` times each stage with the GPU synchronized at the boundaries. Its first finding changed the plan: once separation got faster, **MP3 encoding was most of the job**: 10.7 s of 14.4 s on an L4 with demucs's settings (`results/full/L4.json`). LAME's `-q 5` encodes 1.6–1.7× faster than demucs's `-q 2` at the same SNR against the source (25.1 dB both, `results/encode-cpu4.json`). In-process lameenc beat ffmpeg subprocesses by ~1.3–1.5×; lameenc releases the GIL, so four threads encode four stems at once. 8 cores barely beat 4 (3.1 s against 3.4 s for four stems, `results/encode-cpu8.json`).

**Chunk overlap (P0).** 25% → 10% runs ~16% fewer chunks for −0.01 dB. 5% saves only another ~5% and costs a measurable −0.06 dB.

**Persistent model loading and cold starts (P0).** The model already loaded once per container from a memory snapshot. The surprise was cuDNN autotuning: in a fresh container, `cudnn.benchmark` made the first song ~13 s slower and later ones no faster (`results/smoke-L4.json`), so it's off. Cold start on an L4, from the request to a 10 s separation's result (median of 8, `results/coldstart.json`):

| | median | first call in the container |
|---|--:|--:|
| no snapshot | 11.5 s | 1.5 s |
| CPU memory snapshot (stemplayer's) | 8.6 s | 1.7 s |
| GPU memory snapshot (Modal alpha) | 6.8 s | 0.4 s |

The GPU snapshot restores the model already on the GPU and warmed up. It's the obvious next step, but it's alpha, so stemplayer keeps the CPU snapshot for now. Both kinds occasionally take 20–140 s while Modal takes a fresh snapshot on a new host. End to end, a cold job on staging took 17–20 s against 29–33 s for the old code (`results/deployed.md`).

**GPU type (P0).** Compared on cost per completed song, not raw speed. Separation is cheapest on the A10, and A10 hosts also have faster CPUs, which now matters more than the GPU: end to end it's 5.7 s on an A10 against 8.1 s on an L4 and 8.5 s on a T4, at 0.20–0.24¢ per song for all three, the A10 within 10% of the T4 (`results/full/`, billed at production's 4 cores and 8 GiB).

**Batching (P1).** demucs runs one 7.8 s chunk at a time. stemsep cuts the song up front and runs 8 chunks per forward pass. Every chunk is normalized on its own inside the model, so batching changes nothing numerically: batch 8 lands 72–122 dB from batch 1. Batch 16 is 4–13% faster than 8 for nearly twice the VRAM.

**Mixed precision (P1).** The STFT and iSTFT stay float32. Only the network runs under autocast. On music, fp16 lands 57–58 dB from fp32 and bf16 37 dB, with no NaNs, and both score identically on MUSDB. fp16 is the default because the T4 has no bf16 units.

**`torch.compile` (P1).** Compiling the network (max-autotune) is another 1.7–1.9× over eager bf16 on an L4 or A10 and 1.6× over eager fp16 on a T4, but takes 5–8 minutes per GPU type. Not shipped: it would need a compile cache or a GPU snapshot that's rebuilt per deploy. It's available as `Config(backend="compile")`.

**ONNX Runtime / TensorRT (P2).** The network exports cleanly (`onnx_backend.export`; the STFTs stay in torch). On ORT's CUDA provider it's 8–28% *slower* than eager fp16 and needs 1.6–3× the VRAM, since attention isn't fused. ORT's TensorRT provider is 1.8–1.9× faster than eager fp16 at batch 4, with a quarter less VRAM (1.6× at batch 8, with ~10% more), but builds an engine for 10–13 minutes per GPU type and batch size (timed once, when the cached engines were built) and adds TensorRT's libraries to the image. Not shipped: after these changes separation is about a third of a job, so TensorRT would save ~1 s per song. It's available as `Config(backend="onnx", onnx_options={"tensorrt": ...})`, with numbers in `results/onnx/`.

**A C++/Rust rewrite (P3).** Not worth it. The network runs in cuDNN, cuBLAS and fused attention kernels either way. What's left is MP3 encoding and decoding, which are already native code. A runtime of our own would cost months to save under a second.

**Vocal-only specialist.** `separate(wav, stems=[...])` runs only the sub-models of a bag (e.g. `htdemucs_ft`) that contribute to the requested stems. stemplayer needs all four, so it uses `htdemucs`.

## Second pass: the whole job

With separation down to ~2 s, the rest of the job mattered more. Each job now logs per-stage timings. These are stemplayer's warm jobs on an A10, uploading the same 245 s song, with webhooks going to a real HTTPS endpoint (`results/deployed.md`):

| stage | before this pass | after |
|---|--:|--:|
| progress webhooks (3, blocking) | 1.05 s | 0 s: sent from a thread, in order |
| fetch from R2 | ~0.3–0.6 s | same |
| decode | 0.82 s | 0.43 s: libsndfile in process |
| separate, then encode | 1.89 s + ~3.1 s | 2.1 s, then a 1.4 s tail: encoded while separating |
| upload to R2 | overlapped with encoding | ~0.6–1.4 s |
| final webhook | ~0.3 s | ~0.2–0.6 s |
| **warm job, as the caller sees it** | **~8.1 s** | **~5.0–6.0 s** |

**Encoding while separating.** `separate(wav, sink=...)` hands each stretch of the stems to a sink as soon as no later chunk can change it. `StreamingMP3` feeds those stretches to one LAME encoder per stem. The catch is demucs's clip guard, which scales a whole stem down when its peak would clip, and the peak isn't known until the end. So stems are encoded optimistically and re-encoded whole in the rare case they need it. No stem of 12 full MUSDB songs came within 1% of clipping (`results/peaks.json`), even though every mixture peaked at full scale. Either way the bytes are identical to encoding afterwards. End to end without network that's 5.82 → 4.37 s on an A10 and 8.26 → 6.29 s on an L4 (`results/e2e/`). A small first batch (2 chunks) gets the encoders going sooner.

**Decoding in process.** libsndfile reads MP3, WAV, FLAC and Ogg directly, a third faster than ffmpeg piping 86 MB of samples out, and lands within 3e-6 of ffmpeg's samples. Anything else, or anything that needs resampling, still goes through ffmpeg.

**Cold starts.** A cold container's first separation spent ~1.2 s loading CUDA kernels. A 20 s throwaway separation now runs on a background thread right after restore, alongside the first song's download and decode, which takes most of that cost off the song. Two more starts were hiding in front of the GPU. Every song goes through stemplayer's small API endpoint before a GPU starts, and a cold one took 3–6 s to answer against ~0.65 s warm. It now stays up 20 minutes after a request, for about 0.2¢. And the Worker now asks for a GPU container as soon as someone picks a song, so it boots while the song uploads. With a 6 s head start, cold jobs took 6.4–8.0 s, about warm speed, against 9–12 s without one. In practice GPU memory snapshots were worse: on staging, cold jobs took 19–44 s against 9–25 s with the CPU snapshot, while Modal took fresh snapshots on new hosts.

What didn't help:
- **More CPU cores.** 6 or 8 instead of 4 made no difference: each stem's LAME is single-threaded, ~2.4 s for the drums at `-q 5`, and that sets the floor.
- **A faster LAME.** LAME 3.100 built with `-O3 -march=x86-64-v3` was 1% faster than lameenc's, and adding fast-math only 9% (`results/lame-build.json`).
- **VBR.** `-V 2` shrank stems only 4% overall. The drums, the slowest stem to encode, grew to 206–229 kbps, so the critical path didn't move (`results/vbr.json`).
- **R2 transfers.** Ranged parallel downloads were slower than one GET. Uploads are ~5 MB/s per connection whatever the client (boto3, requests, httpx) or socket buffer size, and R2's 5 MiB minimum part size rules out parallel parts for a 6 MB stem.

## Third pass: what the person waits for

Timing the whole loop, from picking a song to hearing it, turned up costs outside the job itself:

- **Job status was up to a minute stale.** stemplayer kept job status in Workers KV. Its webhook writes land at the Cloudflare location nearest Modal, and the person polls from theirs, which caches the key. Writes landing at LAX took **7–50 s (median ~28 s)** to show in Toronto (`results/kv-propagation.jsonl`). Job records moved to R2, which is consistent everywhere, and the browser now long-polls (stemplayer #7).
- **Polling.** Before that, the browser polled every 2 s, backing off to 5 s. With a ~5 s job that found "ready" 2–5 s late. It now polls every 1 s, and with #7 a change arrives within a quarter second.
- **Loud songs encoded twice.** A loud commercial-style track's drums peaked at 1.64 and its bass at 1.02, where no MUSDB stem had come close. demucs's clip guard scales such a stem down on its own, which left the drums 4.4 dB too quiet against the rest, and it also meant the stem had to be encoded again after separation. Two fixes:
  - a streamed stem now stops encoding as soon as it's known to clip, and is re-encoded the moment separation ends: 5.6 → 4.1 s for separate plus encode on that track;
  - stemplayer #8 stores stems 6 dB down and has the player undo it, reporting each stem's exact gain. Stems below a ~2 peak never rescale, the balance matches the song, and the MP3's fidelity changes by 0.015 dB (`results/headroom.json`). The encode tail on that track went from 3.8 to 1.25 s.
- **Decoding while separating.** `decode_blocks` reads a file a block at a time on a thread, and `separate_stream` starts separating as soon as there's a batch of chunks. That means no song-level normalization, which needs the whole song: dropping it changed four full songs' SDR by under 0.001 dB. Chunks are cut at demucs's fixed stride plus one flush with the end, −0.004 dB on six songs. It's 0.25 s faster end to end (`results/stream-decode.json`). Anything libsndfile won't read exactly as `decode` would, from m4a and webm to a short or failing read, falls back to decoding whole.
- **Which GPU, counting the wait.** With streaming, the whole job (decode, separate, encode) is 4.0 s on an A10, 5.7 s on an L4 and 6.6 s on a T4, at 0.149¢, 0.166¢ and 0.154¢ (`results/e2e/`). But the minute a container idles after its last song costs 2.26¢ on an A10, 1.75¢ on an L4 and 1.40¢ on a T4. So for one song a session a T4 would be ~36% cheaper, at 2.6 s more per song, but its cold starts are much slower: a median of 17.5 s (9.4–94 s) against 5.5 s (3.5–16 s) on an A10, with CPU memory snapshots (`results/coldstart-T4.json`, `results/coldstart-A10.json`). With a one-minute idle window most sessions start cold, so stemplayer stays on the A10.
- **Where Modal runs the GPU.** Default placement put A10 jobs in Ashburn, Columbus, Frankfurt and Osaka. Against the ENAM bucket, uploads took 0.57–1.87 s. Pinning to `region="us"` (1.15× price) would save ~0.3–0.6 s a job; that's left as a cost call (`results/deployed.md`).
- **Tried and dropped:** fetching link songs on a separate CPU container. yt-dlp was 3× slower on one core, and the extra hop cost more than the GPU boot it hid (`results/deployed.md`).

## How it works

`model.py` splits `HTDemucs.forward` into `pre` (STFT and normalization, float32), `Core` (both encoders, the cross-transformer, both decoders) and `post` (mask and iSTFT, float32). Only `Core` changes per backend: eager, autocast, `torch.compile`, or an ONNX Runtime session bound zero-copy to torch's buffers. `engine.py` plans the chunks so the last one ends flush with the song, stacks them, runs them `batch_size` at a time, and cross-fades them back with demucs's triangular window, streaming finished stretches to a sink. `separate_stream` does the same for a song still being decoded. `audio.py` decodes with ffmpeg, mapping channels as demucs does, and encodes with lameenc.

Tests (`uv run pytest`) check the split forward against `HTDemucs.forward`, batched output against `apply_model`, edge cases (silence, one sample, mono, cancelling channels, padded batches) and the ONNX backend.

## Benchmarks

Everything runs on Modal (`uv run modal ...`, from `bench/`), against a Volume holding the test sets:

| | |
|---|---|
| `testset.py` | MUSDB18 7 s clips joined into one 245 s song with exact stems (timing; not for quality comparisons under ~0.2 dB, as its splices exaggerate chunk-grid effects) |
| `fulltracks.py` | 12 full MUSDB18-HQ test songs, fetched with HTTP range requests rather than the 23 GB zip |
| `gpu.py` | the config sweep per GPU, and end-to-end runs (`--sweep full`, `--sweep e2e`) |
| `quality.py` | SDR against production demucs on full songs |
| `encode.py` | MP3 encoders and LAME presets |
| `coldstart.py` | cold start with no snapshot, a CPU memory snapshot, and a GPU memory snapshot |
| `onnx_gpu.py` | ONNX Runtime CUDA and TensorRT providers |

## Using it from stemplayer

The repo is private, so stemplayer vendors the runtime modules: `scripts/vendor.sh ../stemplayer/modal/stemsep`.
