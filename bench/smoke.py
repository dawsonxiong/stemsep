"""Each config in a fresh container on the 245 s test song: its first call
(what a cold container pays), a warm call, and how far it lands from fp32 on
the same chunk grid.

    modal run bench/smoke.py --gpu L4 --out smoke-L4.json"""

import modal

from images import image, volume

app = modal.App("stemsep-smoke")


# One config per container, so cuDNN caches and global flags start clean.
@app.function(image=image, gpu="L4", volumes={"/data": volume}, timeout=1800,
              single_use_containers=True)
def smoke(configs: list[dict]) -> list[dict]:
    import time

    import torch

    from stemsep import Config, Separator
    from stemsep.quality import load_testset, sdr

    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    wav, _, _ = load_testset("/data/testset")
    rows = []
    for cfg in configs:
        try:
            t0 = time.perf_counter()
            sep = Separator("htdemucs", Config(**cfg))
            out = sep.separate(wav)
            first = time.perf_counter() - t0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = sep.separate(wav)
            torch.cuda.synchronize()
            second = time.perf_counter() - t0
            # fp32 on the same chunk grid, in this container (one config per
            # call), after the timed runs so they still start clean.
            del sep
            grid = {k: v for k, v in cfg.items() if k in ("overlap", "batch_size")}
            ref = Separator("htdemucs", Config(**grid)).separate(wav)
            fid = sum(sdr(ref[s], out[s]) for s in out) / len(out)
            nan = any(torch.isnan(v).any().item() for v in out.values())
            rows.append({**cfg, "first_s": round(first, 2), "warm_s": round(second, 3),
                         "sdr_vs_fp32": round(fid, 1), "nan": nan})
        except Exception as e:
            import traceback
            rows.append({**cfg, "error": traceback.format_exc()[-1500:]})
        print(rows[-1])
    return rows


@app.local_entrypoint()
def main(gpu: str = "L4", configs_json: str = "", out: str = ""):
    import json
    from pathlib import Path

    configs = json.loads(configs_json) if configs_json else [
        {"batch_size": 8},
        {"batch_size": 8, "precision": "tf32"},
        {"batch_size": 8, "precision": "fp16"},
        {"batch_size": 8, "precision": "bf16"},
        {"batch_size": 8, "precision": "fp16", "backend": "compile"},
    ]
    fn = smoke.with_options(gpu=gpu)
    results = []
    for rows in fn.map([[c] for c in configs]):
        for row in rows:
            print("RESULT", row)
            results.append(row)
    if out:
        path = Path(__file__).resolve().parent.parent / "results" / out
        path.write_text(json.dumps({"gpu": gpu, "rows": results}, indent=2))
        print(f"wrote {path}")
