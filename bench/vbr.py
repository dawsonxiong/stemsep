"""CBR 192 kbps against LAME VBR on real separated stems: file size (what
R2 uploads and browsers download), encode time, and SNR against the stem.

    modal run bench/vbr.py
"""

import json
import time
from pathlib import Path

import modal

from images import image, volume

app = modal.App("stemsep-vbr")

MODES = {
    "cbr192 q5": dict(bitrate=192, quality=5),
    "vbr V2": dict(vbr=2),
    "vbr V3": dict(vbr=3),
}


def _encode(wav, sr, bitrate=None, quality=5, vbr=None):
    import lameenc

    from stemsep.audio import _rescale, _to_int16

    e = lameenc.Encoder()
    e.set_in_sample_rate(sr)
    e.set_channels(wav.shape[0])
    if vbr is None:
        e.set_bit_rate(bitrate)
        e.set_quality(quality)
    else:
        e.set_vbr(4)  # vbr_mtrh, LAME's default VBR
        e.set_vbr_quality(vbr)
        e.set_quality(quality)
    return bytes(e.encode(_to_int16(_rescale(wav.float())).tobytes()) + e.flush())


def _snr(ref, mp3: bytes, sr) -> float:
    import os
    import tempfile

    import numpy as np

    from stemsep.audio import decode

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3)
    dec = decode(f.name, sr, 2).numpy()
    os.unlink(f.name)
    ref = ref.numpy()
    n = sr
    lags = range(0, 3000)
    lag = max(lags, key=lambda k: float(np.dot(ref[0, n : 2 * n], dec[0, n + k : 2 * n + k])))
    dec = dec[:, lag:]
    m = min(dec.shape[1], ref.shape[1])
    err = ref[:, :m] - dec[:, :m]
    return round(float(10 * np.log10((ref[:, :m] ** 2).sum() / (err**2).sum() + 1e-20)), 2)


@app.function(image=image, gpu="A10", cpu=4.0, volumes={"/data": volume}, timeout=3600)
def run(tracks: int) -> list[dict]:
    from stemsep import Config, Separator
    from stemsep.quality import load_fulltracks

    sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
    rows = []
    for i, (name, mixture, _, sr) in enumerate(load_fulltracks("/data/fulltracks")):
        if i >= tracks:
            break
        stems = sep.separate(mixture)
        seconds = mixture.shape[-1] / sr
        for stem, wav in stems.items():
            row = {"track": name, "stem": stem, "seconds": round(seconds, 1)}
            for mode, kw in MODES.items():
                t0 = time.perf_counter()
                mp3 = _encode(wav, sr, **kw)
                row[mode] = {
                    "kbytes": len(mp3) // 1000,
                    "kbps": round(len(mp3) * 8 / seconds / 1000),
                    "encode_s": round(time.perf_counter() - t0, 2),
                    "snr_db": _snr(wav, mp3, sr),
                }
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


@app.local_entrypoint()
def main(tracks: int = 4):
    rows = run.remote(tracks)
    out = Path(__file__).resolve().parent.parent / "results" / "vbr.json"
    out.write_text(json.dumps(rows, indent=2))
    for mode in MODES:
        kb = sum(r[mode]["kbytes"] for r in rows)
        t = sum(r[mode]["encode_s"] for r in rows)
        snr = sum(r[mode]["snr_db"] for r in rows) / len(rows)
        print(f"{mode}: {kb / 1000:.1f} MB total, {t:.1f} s encoding, mean SNR {snr:.2f} dB")
