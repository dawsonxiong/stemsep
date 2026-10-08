"""Stems encoded 6 dB down and turned back up on playback, against encoded
at full scale: SNR against the separated stem, after the MP3 round trip.

    modal run bench/headroom.py
"""

import json
import os
import tempfile

import modal

from images import _base, volume

app = modal.App("stemsep-headroom")
image = _base.uv_pip_install("yt-dlp==2026.8.19").add_local_python_source("stemsep", "images")


def _snr(ref, est):
    import torch
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[:, :n].double(), est[:, :n].double()
    return round(float(10 * torch.log10(ref.pow(2).sum() / (ref - est).pow(2).sum())), 2)


def _roundtrip(stem, gain):
    """Encode stem*gain, decode, divide by gain, align (LAME adds a delay)."""
    import numpy as np
    import torch

    from stemsep.audio import decode, mp3_bytes

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes(stem * gain, 44100))
    dec = decode(f.name) / gain
    os.unlink(f.name)
    ref = stem.numpy(); d = dec.numpy(); n = 44100
    # Align on the stem's loudest second: a quiet one can match any lag.
    energy = (ref[0, : len(ref[0]) // n * n].reshape(-1, n) ** 2).sum(1)
    a = int(energy[:-1].argmax()) * n
    lag = max(range(3000), key=lambda k: float(np.dot(ref[0, a:a + n], d[0, a + k:a + n + k])))
    return torch.from_numpy(d[:, lag:].copy())


@app.function(image=image, gpu="A10", volumes={"/data": volume}, timeout=3600)
def run(tracks: int, page: str) -> list[dict]:
    import yt_dlp

    from stemsep import Config, Separator
    from stemsep.audio import decode
    from stemsep.quality import load_fulltracks

    sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
    songs = []
    with tempfile.TemporaryDirectory() as tmp:
        with yt_dlp.YoutubeDL({"quiet": True, "format": "bestaudio/best", "outtmpl": os.path.join(tmp, "a.%(ext)s")}) as y:
            info = y.extract_info(page, download=True)
        songs.append(("soundcloud: " + info["title"], decode(info["requested_downloads"][0]["filepath"])))
    for i, (name, mix, _, _) in enumerate(load_fulltracks("/data/fulltracks")):
        if i >= tracks:
            break
        songs.append((name, mix))
    rows = []
    for name, mix in songs:
        stems = sep.separate(mix)
        for stem_name, stem in stems.items():
            peak = stem.abs().max().item()
            row = {"song": name, "stem": stem_name, "peak": round(peak, 3)}
            for gain in (1.0, 0.5):
                # Where full scale would clip, demucs rescales: compare like for like.
                scale = 1 / max(1.01 * peak * gain, 1.0)
                row[f"snr at gain {gain}"] = _snr(stem * scale * gain / gain, _roundtrip(stem, gain) * 1.0) if scale == 1 else None
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


@app.local_entrypoint()
def main(tracks: int = 4, page: str = "https://soundcloud.com/forss/flickermood"):
    rows = run.remote(tracks, page)
    from pathlib import Path
    (Path(__file__).resolve().parent.parent / "results" / "headroom.json").write_text(json.dumps(rows, indent=2))
    both = [r for r in rows if r["snr at gain 1.0"] is not None and r["snr at gain 0.5"] is not None]
    if both:
        d = [r["snr at gain 0.5"] - r["snr at gain 1.0"] for r in both]
        print(f"{len(both)} stems comparable; SNR change at -6 dB: mean {sum(d)/len(d):+.3f} dB, min {min(d):+.3f}, max {max(d):+.3f}")
