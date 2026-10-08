"""Whether separating without the whole-song normalization (demucs's CLI
divides by the song's std first) changes the output, which decides whether
decoding could overlap separation.

    modal run bench/normalization.py
"""

import json

import modal

from images import image, volume

app = modal.App("stemsep-normalization")


@app.function(image=image, gpu="A10", volumes={"/data": volume}, timeout=1800)
def run(tracks: int) -> list:
    import torch

    from stemsep import Config, Separator
    from stemsep.quality import load_fulltracks, report, sdr

    sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
    raw_sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16", normalize=False))
    rows = []
    for i, (name, mix, stems, sr) in enumerate(load_fulltracks("/data/fulltracks")):
        if i >= tracks:
            break
        normal = sep.separate(mix)
        # Undo the engine's normalization from outside: hand it a song whose
        # mean is 0 and std is 1 already, then scale back. Same as skipping it.
        ref = mix.mean(0)
        mean, std = ref.mean(), ref.std()
        skipped = {k: v * std + mean for k, v in sep.separate((mix - mean) / std).items()}
        # And the plain un-normalized song, as a stream would have to hand it.
        raw = raw_sep.separate(mix)
        rows.append({
            "track": name,
            "sdr_normal": round(report(stems, normal, sr)["mean"]["sdr"], 3),
            "sdr_raw_input": round(report(stems, raw, sr)["mean"]["sdr"], 3),
            "raw_vs_normal_db": round(sum(sdr(normal[s], raw[s]) for s in normal) / 4, 1),
            "prenormalized_vs_normal_db": round(sum(sdr(normal[s], skipped[s]) for s in normal) / 4, 1),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


@app.local_entrypoint()
def main(tracks: int = 4):
    run.remote(tracks)
