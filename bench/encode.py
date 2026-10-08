"""MP3 encoding throughput on Modal CPUs: which encoder, how to run four at
once, and what LAME's quality preset costs.

    modal run bench/encode.py --cpu 4
"""

import json
import time
from pathlib import Path

import modal

from images import image, volume

app = modal.App("stemsep-encode")


def _lameenc_bytes(wav, samplerate, quality):
    import lameenc

    pcm = (wav.clamp(-1, 1) * 32767).short().T.contiguous().numpy()
    enc = lameenc.Encoder()
    enc.set_bit_rate(192)
    enc.set_in_sample_rate(samplerate)
    enc.set_channels(wav.shape[0])
    enc.set_quality(quality)
    return enc.encode(pcm.tobytes()) + enc.flush()


def _ffmpeg(wav, path, quality, fmt="f32le"):
    import subprocess

    import numpy as np

    if fmt == "s16le":
        raw = (wav.clamp(-1, 1) * 32767).short().T.contiguous().numpy()
    else:
        raw = wav.T.contiguous().numpy().astype(np.float32)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", fmt, "-ar", "44100",
           "-ac", "2", "-i", "-", "-c:a", "libmp3lame", "-b:a", "192k",
           "-compression_level", str(quality), "-f", "mp3", path]  # fmt: skip
    subprocess.run(cmd, input=memoryview(raw).cast("B"), check=True, capture_output=True)


def _snr(ref, path):
    """SNR of the decoded MP3 against the stem, aligned by cross-correlating
    the first second (LAME adds a delay)."""
    import numpy as np

    from stemsep import audio

    dec = audio.decode(path).numpy()
    ref = ref.numpy()
    n = 44100
    best, lag = -1, 0
    for k in range(0, 3000):
        c = float(np.dot(ref[0, n : 2 * n], dec[0, n + k : 2 * n + k]))
        if c > best:
            best, lag = c, k
    dec = dec[:, lag : lag + ref.shape[1]]
    m = min(dec.shape[1], ref.shape[1])
    err = ref[:, :m] - dec[:, :m]
    return round(float(10 * np.log10((ref[:, :m] ** 2).sum() / (err**2).sum())), 2)


@app.function(image=image, volumes={"/data": volume}, cpu=4.0, memory=8192, timeout=1800)
def bench(cpu: float) -> dict:
    import os
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from stemsep.quality import load_testset

    _, stems, sr = load_testset("/data/testset")
    names = list(stems)
    out = {"cpu": cpu, "os_cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))}

    def timed(label, fn):
        t0 = time.perf_counter()
        fn()
        out[label] = round(time.perf_counter() - t0, 2)
        print(label, out[label], flush=True)

    one = stems["other"]
    for q in (2, 5, 7):
        timed(f"lameenc 1 stem q{q}", lambda: _lameenc_bytes(one, sr, q))
    with tempfile.TemporaryDirectory() as tmp:
        for q in (2, 5, 7):
            Path(f"{tmp}/l.mp3").write_bytes(_lameenc_bytes(one, sr, q))
            out[f"lameenc snr q{q}"] = _snr(one, f"{tmp}/l.mp3")
    # lameenc releases the GIL while encoding, so threads run in parallel.
    for q in (2, 5):
        timed(f"lameenc 4 threads q{q}", lambda: list(ThreadPoolExecutor(4).map(
            lambda n: _lameenc_bytes(stems[n], sr, q), names)))  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        for q in (2, 5, 7):
            timed(f"ffmpeg 1 stem q{q}", lambda: _ffmpeg(one, f"{tmp}/one.mp3", q))
            out[f"snr q{q}"] = _snr(one, f"{tmp}/one.mp3")
        for q in (2, 5):
            for fmt in ("f32le", "s16le"):
                timed(f"ffmpeg 4 parallel q{q} {fmt}", lambda: list(ThreadPoolExecutor(4).map(
                    lambda n: _ffmpeg(stems[n], f"{tmp}/{n}.mp3", q, fmt), names)))  # fmt: skip
    return out


@app.local_entrypoint()
def main(cpu: float = 4.0):
    result = bench.with_options(cpu=cpu).remote(cpu)
    path = Path(__file__).resolve().parent.parent / "results" / f"encode-cpu{cpu:g}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
