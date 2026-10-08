"""Decoding while separating (`separate_stream`) against decoding first,
end to end on the test song, and its quality on full songs.

    modal run bench/stream_decode.py
"""

import json
import statistics
import time
from pathlib import Path

import modal

from images import image, volume

app = modal.App("stemsep-stream-decode")


@app.function(image=image, gpu="A10", cpu=4.0, volumes={"/data": volume}, timeout=1800)
def run(tracks: int) -> dict:
    import torch

    from stemsep import Config, Separator
    from stemsep.audio import StreamingMP3, decode, decode_blocks
    from stemsep.quality import load_fulltracks, report

    sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
    path = "/data/testset/mixture.mp3"
    sep.separate(decode(path))  # warm
    out = {}

    def timed(fn):
        times = []
        for _ in range(4):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return round(statistics.median(times), 3)

    def decode_first():
        enc = StreamingMP3(sep.sources, 44100, 2, gain=0.5)
        enc.finish(sep.separate(decode(path), sink=enc.feed))

    def decode_streaming():
        enc = StreamingMP3(sep.sources, 44100, 2, gain=0.5)
        enc.finish(sep.separate_stream(decode_blocks(path), sink=enc.feed))

    out["decode, then separate + encode (s)"] = timed(decode_first)
    out["decode while separating + encode (s)"] = timed(decode_streaming)

    deltas = []
    for i, (name, mix, stems, sr) in enumerate(load_fulltracks("/data/fulltracks")):
        if i >= tracks:
            break
        a = report(stems, sep.separate(mix), sr)["mean"]["sdr"]
        b = report(stems, sep.separate_stream(mix[:, k : k + 88200] for k in range(0, mix.shape[-1], 88200)), sr)["mean"]["sdr"]
        deltas.append(round(b - a, 4))
    out["sdr stream - separate, per song (dB)"] = deltas
    return out


@app.local_entrypoint()
def main(tracks: int = 6):
    result = run.remote(tracks)
    (Path(__file__).resolve().parent.parent / "results" / "stream-decode.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
