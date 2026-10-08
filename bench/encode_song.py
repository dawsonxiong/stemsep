"""Encoding a real, dense song's stems: each alone, then four at once, at a
given CPU count. A SoundCloud track by default (yt-dlp's test track).

    modal run bench/encode_song.py --cpu 4
"""

import json
import time

import modal

from images import _base, volume

app = modal.App("stemsep-encode-song")
image = _base.uv_pip_install("yt-dlp==2026.8.19").add_local_python_source("stemsep", "images")


@app.function(image=image, gpu="A10", volumes={"/data": volume}, timeout=1800)
def run(cpu: float, page: str) -> dict:
    import os
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    import yt_dlp

    from stemsep import Config, Separator
    from stemsep.audio import decode, mp3_bytes

    with tempfile.TemporaryDirectory() as tmp:
        with yt_dlp.YoutubeDL({"quiet": True, "format": "bestaudio/best",
                               "outtmpl": os.path.join(tmp, "a.%(ext)s")}) as y:
            info = y.extract_info(page, download=True)
        wav = decode(info["requested_downloads"][0]["filepath"])
    sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
    stems = sep.separate(wav)
    out = {"cpu": cpu, "seconds": round(wav.shape[-1] / 44100, 1),
           "visible_cpus": len(os.sched_getaffinity(0))}
    for name, s in stems.items():
        t = time.perf_counter(); mp3_bytes(s, 44100); out[f"{name} alone"] = round(time.perf_counter() - t, 2)
        out[f"{name} peak"] = round(s.abs().max().item(), 3)
    out["mixture peak"] = round(wav.abs().max().item(), 3)
    for q in (5, 7):
        t = time.perf_counter()
        list(ThreadPoolExecutor(4).map(lambda s: mp3_bytes(s, 44100, quality=q), stems.values()))
        out[f"4 at once q{q}"] = round(time.perf_counter() - t, 2)

    import torch

    from stemsep.audio import StreamingMP3

    for pieces in (1, 8, 40):
        enc = StreamingMP3(sep.sources, 44100, 2)
        n = stems["drums"].shape[-1]
        bounds = [round(i * n / pieces) for i in range(pieces + 1)]
        t = time.perf_counter()
        for a, b in zip(bounds, bounds[1:]):
            enc.feed({k: v[:, a:b] for k, v in stems.items()})
        enc.finish(stems)
        out[f"StreamingMP3 alone, {pieces} pieces"] = round(time.perf_counter() - t, 2)

    default_threads = torch.get_num_threads()
    out["torch threads by default"] = default_threads
    for threads in (default_threads, 4, 2, default_threads):
        torch.set_num_threads(threads)
        enc = StreamingMP3(sep.sources, 44100, 2)
        t = time.perf_counter()
        result = sep.separate(wav, sink=enc.feed)
        torch.cuda.synchronize()
        sep_s = time.perf_counter() - t
        enc.finish(result)
        out[f"streaming, torch threads {threads}"] = {
            "separate": round(sep_s, 2), "total": round(time.perf_counter() - t, 2)}
    return out


@app.local_entrypoint()
def main(cpu: float = 4.0, page: str = "https://soundcloud.com/forss/flickermood"):
    print(json.dumps(run.with_options(cpu=cpu).remote(cpu, page)))
