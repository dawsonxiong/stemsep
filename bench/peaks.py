"""How often a separated stem would clip, which decides whether stems can be
encoded while they're still being separated (see audio.encode_mp3's rescale).

    modal run bench/peaks.py
"""

import json
from pathlib import Path

import modal

from images import image, volume

app = modal.App("stemsep-peaks")


@app.function(image=image, gpu="A10", volumes={"/data": volume}, timeout=1800)
def peaks() -> list[dict]:
    from stemsep import Config, Separator
    from stemsep.quality import load_fulltracks

    sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
    rows = []
    for name, mixture, _, _ in load_fulltracks("/data/fulltracks"):
        stems = sep.separate(mixture)
        rows.append({
            "track": name,
            "mixture_peak": round(mixture.abs().max().item(), 4),
            **{s: round(v.abs().max().item(), 4) for s, v in stems.items()},
        })
        print(rows[-1], flush=True)
    return rows


@app.local_entrypoint()
def main():
    rows = peaks.remote()
    out = Path(__file__).resolve().parent.parent / "results" / "peaks.json"
    out.write_text(json.dumps(rows, indent=2))
    stems = [(r["track"], s, r[s]) for r in rows for s in ("drums", "bass", "other", "vocals")]
    clipping = [x for x in stems if 1.01 * x[2] > 1]
    print(f"{len(clipping)} of {len(stems)} stems would be rescaled; "
          f"{len({t for t, _, _ in clipping})} of {len(rows)} songs")
