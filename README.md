# stemsep

Faster [HTDemucs](https://github.com/facebookresearch/demucs) inference on Modal GPUs. It replaces `demucs.apply.apply_model` in [stemplayer](https://github.com/dawsonxiong/stemplayer)'s separation worker: same model, same weights, same quality, at about a third of the GPU time.

```python
from stemsep import Config, Separator
from stemsep.audio import decode, encode_all

sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
stems = sep.separate(decode("song.mp3"))  # {"drums": [2, L], "bass": ..., "other": ..., "vocals": ...}
encode_all(stems, {s: f"{s}.mp3" for s in stems}, sep.samplerate)
```

## Results

The production job on Modal, for a 4-minute song: download from R2, decode, separate, encode four 192 kbps MP3s, upload. Each row was measured on a separately deployed copy of the app.

| | GPU | warm job | cost per song¹ |
|---|---|--:|--:|
| before: demucs `apply_model`, LAME q2 | L4 | 18.4 s | 0.54¢ |
| after: stemsep fp16, 10% overlap, batch 8, LAME q5 | A10 | **7.9 s** | **0.30¢** |

¹ GPU + 4 cores + 8 GiB at Modal's per-second prices, for the job alone. Idle time while a container waits for the next song costs more than the songs themselves, so `scaledown_window` also went from 300 s to 120 s.

Separation alone, 245 s song (`results/full/`):

| GPU | demucs `apply_model` | stemsep fp16 b8 ov.10 | speedup | peak VRAM |
|---|--:|--:|--:|--:|
| T4 | 9.62 s | 4.43 s | 2.2× | 5.2 GiB |
| L4 | 4.86 s | 2.85 s | 1.7× | 6.1 GiB |
| A10 | 5.47 s | 1.90 s | 2.9× | 5.1 GiB |

Quality on 12 full MUSDB18-HQ test songs, 47 minutes (`results/quality/`):

| | median SDR | Δ vs production, paired |
|---|--:|--:|
| demucs, production settings | 8.55 | – |
| stemsep fp32, 25% overlap | 8.58 | −0.01 ± 0.01 dB |
| stemsep fp16, 10% overlap (shipped) | 8.54 | −0.02 ± 0.01 dB |
| stemsep fp32, 5% overlap | 8.46 | −0.06 ± 0.02 dB |

Moving the chunk grid alone moves a song's SDR by ±0.02 dB, so the shipped config is within noise of production.

## What was tried, and what came of it

Measured against the plan's priorities.

**Profiling (P0).** `Profiler` times each stage with the GPU synchronized at the boundaries. Its first finding changed the plan: once separation got faster, **MP3 encoding was most of the job** (10–13 s of 16 on an L4). LAME's `-q 5` encodes 1.7× faster than demucs's `-q 2` at the same SNR against the source (25.1 dB both, `results/encode-cpu4.json`). In-process lameenc beat ffmpeg subprocesses by ~1.5×; lameenc releases the GIL, so four threads encode four stems at once. 8 cores barely beat 4.

**Chunk overlap (P0).** 25% → 10% runs 20% fewer chunks for −0.01 dB. 5% saves only another ~5% and costs a measurable −0.06 dB.

**Persistent model loading and cold starts (P0).** The model already loaded once per container from a memory snapshot. The surprise was cuDNN autotuning: `cudnn.benchmark` added ~12 s to a fresh container's first song and saved ~2% after, so it's off. Modal's GPU memory snapshot (alpha), which restores the model on the GPU already warmed up, is measured in `results/coldstart.json`.

**GPU type (P0).** Compared on cost per completed song, not raw speed. Separation is cheapest on the A10, and A10 hosts also have faster CPUs, which now matters more than the GPU: end to end it's 5.8 s on an A10 against 8.9 s on an L4 and 8.7 s on a T4, at 0.23–0.28¢ per song for all three (`results/e2e/`).

**Batching (P1).** demucs runs one 7.8 s chunk at a time. stemsep cuts the song up front and runs 8 chunks per forward pass. Every chunk is normalized on its own inside the model, so batching changes nothing numerically (> 50 dB agreement). Batch 16 is ~4% faster than 8 for twice the VRAM.

**Mixed precision (P1).** The STFT and iSTFT stay float32. Only the network runs under autocast. fp16 is 45 dB from fp32 and bf16 31 dB, and both score identically on MUSDB. fp16 is the default because the T4 has no bf16 units.

**`torch.compile` (P1).** Compiling the network (max-autotune) is another 1.7–1.8× on top of eager fp16, but takes 6–8 minutes per GPU type. Not shipped: it would need a compile cache or a GPU snapshot that's rebuilt per deploy. It's available as `Config(backend="compile")`.

**ONNX Runtime / TensorRT (P2).** The network exports cleanly (`onnx_backend.export`; the STFTs stay in torch). On ORT's CUDA provider it's 7–25% *slower* than eager fp16 and needs 1.6–3× the VRAM, since attention isn't fused. ORT's TensorRT provider is 1.8–1.9× faster than eager fp16 with less VRAM, but builds an engine for 10–13 minutes per GPU type and batch size and adds ~2 GB of libraries. Not shipped: after these changes separation is about a third of a job, so TensorRT would save ~1 s per song. It's available as `Config(backend="onnx", onnx_options={"tensorrt": ...})`, with numbers in `results/onnx/`.

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
