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

## How it works

`model.py` splits `HTDemucs.forward` into `pre` (STFT and normalization, float32), `Core` (both encoders, the cross-transformer, both decoders) and `post` (mask and iSTFT, float32). Only `Core` changes per backend: eager, autocast, `torch.compile`, or an ONNX Runtime session bound zero-copy to torch's buffers. `engine.py` plans the chunks so the last one ends flush with the song, stacks them, runs them `batch_size` at a time, and cross-fades them back with demucs's triangular window. `audio.py` decodes with ffmpeg, mapping channels as demucs does, and encodes with lameenc.

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
