"""Every precision and backend on one GPU, on noise: does it run, and how
far is it from fp32?   modal run bench/smoke.py --gpu L4"""

import modal

from images import image

app = modal.App("stemsep-smoke")


# One config per container, so cuDNN caches and global flags start clean.
@app.function(image=image, gpu="L4", timeout=1800, single_use_containers=True)
def smoke(configs: list[dict]) -> list[dict]:
    import time

    import torch

    from stemsep import Config, Separator
    from stemsep.quality import sdr

    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    torch.manual_seed(0)
    wav = torch.randn(2, 44100 * 240) * 0.1
    ref = None
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
            ref = ref or out  # with one config per call, always itself
            fid = sum(sdr(ref[s], out[s]) for s in out) / len(out)
            nan = any(torch.isnan(v).any().item() for v in out.values())
            rows.append({**cfg, "first_s": round(first, 2), "warm_s": round(second, 3),
                         "sdr_vs_first": round(fid, 1), "nan": nan})
        except Exception as e:
            import traceback
            rows.append({**cfg, "error": traceback.format_exc()[-1500:]})
        print(rows[-1])
    return rows


@app.local_entrypoint()
def main(gpu: str = "L4", configs_json: str = ""):
    import json

    configs = json.loads(configs_json) if configs_json else [
        {"batch_size": 8},
        {"batch_size": 8, "precision": "tf32"},
        {"batch_size": 8, "precision": "fp16"},
        {"batch_size": 8, "precision": "bf16"},
        {"batch_size": 8, "precision": "fp16", "backend": "compile"},
    ]
    fn = smoke.with_options(gpu=gpu)
    for rows in fn.map([[c] for c in configs]):
        for row in rows:
            print("RESULT", row)
