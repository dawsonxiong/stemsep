"""ONNX Runtime against PyTorch eager fp16, on one GPU type.

    cd bench && uv run modal run onnx_gpu.py --gpu L4
    cd bench && uv run modal run onnx_gpu.py --gpu L4 --trt   # TensorRT EP

Exports `Core` (fp32 and fp16, batch 4 and 8) into the stemsep-bench volume
once per torch version, then separates the ground-truth test song with each
config. Writes results/onnx/<gpu>.json (or <gpu>_trt.json). TensorRT engines
are cached in the volume too, per GPU type, since building them takes
minutes.

VRAM is read from the device (`mem_get_info`, sampled while separating),
since ORT allocates outside torch's allocator and max_memory_allocated
doesn't see it.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import threading
import time
from pathlib import Path

import modal

from images import onnx_image, trt_image, volume

app = modal.App("stemsep-onnx")

TESTSET = "/data/testset"
ONNX_ROOT = "/data/onnx"
REPEATS = 3
OVERLAP = 0.10
EXPORTS = [(b, fp16) for b in (4, 8) for fp16 in (False, True)]

REFERENCE = "eager fp16 b8"
CONFIGS = [
    {"name": REFERENCE, "batch_size": 8, "precision": "fp16"},
    {"name": "onnx fp32 b8", "batch_size": 8, "onnx": "fp32"},
    {"name": "onnx fp16 b4", "batch_size": 4, "onnx": "fp16"},
    {"name": "onnx fp16 b8", "batch_size": 8, "onnx": "fp16"},
    {"name": "onnx fp16 b8 graph", "batch_size": 8, "onnx": "fp16", "core": {"cuda_graph": True}},
    {"name": "onnx fp16 b8 heuristic", "batch_size": 8, "onnx": "fp16",
     "core": {"cudnn_conv_algo_search": "HEURISTIC"}},
    {"name": "onnx fp16 b8 nhwc", "batch_size": 8, "onnx": "fp16", "core": {"prefer_nhwc": "1"}},
    {"name": "onnx fp16 b8 pad_nc1d", "batch_size": 8, "onnx": "fp16",
     "core": {"cudnn_conv1d_pad_to_nc1d": "1"}},
]  # fmt: skip


def trt_configs(gpu: str, torch_version: str) -> list[dict]:
    """TensorRT on the fp32 export, with TensorRT choosing fp16 kernels where
    it likes; engines and timing caches kept in the volume."""

    def trt(batch: int) -> dict:
        cache = f"{ONNX_ROOT}/{torch_version}/trt/{gpu}/b{batch}_fp16"
        os.makedirs(cache, exist_ok=True)  # ORT won't
        return {"tensorrt": {"trt_fp16_enable": "True", "trt_engine_cache_enable": "True",
                             "trt_engine_cache_path": cache, "trt_timing_cache_enable": "True",
                             "trt_timing_cache_path": cache}}  # fmt: skip

    return [
        {"name": REFERENCE, "batch_size": 8, "precision": "fp16"},
        {"name": "trt fp16 b4", "batch_size": 4, "onnx": "fp16", "file": "fp32", "core": trt(4)},
        {"name": "trt fp16 b8", "batch_size": 8, "onnx": "fp16", "file": "fp32", "core": trt(8)},
        {"name": "trt fp16 b4 graph", "batch_size": 4, "onnx": "fp16", "file": "fp32",
         "core": {**trt(4), "cuda_graph": True}},
        {"name": "trt fp16 b8 graph", "batch_size": 8, "onnx": "fp16", "file": "fp32",
         "core": {**trt(8), "cuda_graph": True}},
    ]  # fmt: skip


def onnx_path(torch_version: str, batch: int, fp16: bool) -> str:
    return f"{ONNX_ROOT}/{torch_version}/htdemucs_b{batch}_{'fp16' if fp16 else 'fp32'}.onnx"


class VramSampler:
    """Peak device memory in use (every process and allocator), sampled on a
    thread while the block runs."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak = 0

    def _used(self) -> int:
        import torch

        free, total = torch.cuda.mem_get_info()
        return total - free

    def __enter__(self):
        self.peak = self._used()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self._used())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._used())


_function = dict(gpu="L4", volumes={"/data": volume}, cpu=4.0, memory=16 * 1024,
                 timeout=2 * 3600)  # fmt: skip


@app.function(image=onnx_image, **_function)
def bench(gpu: str) -> dict:
    return run(gpu, lambda version: CONFIGS)


@app.function(image=trt_image, **_function)
def bench_trt(gpu: str) -> dict:
    return run(gpu, lambda version: trt_configs(gpu, version))


def run(gpu: str, configs) -> dict:
    import onnxruntime as ort
    import torch

    from stemsep import model as htd
    from stemsep.quality import load_testset

    props = torch.cuda.get_device_properties(0)
    env = {
        "gpu": gpu,
        "device": props.name,
        "vram_gib": round(props.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "onnxruntime": ort.__version__,
        "ort_providers": ort.get_available_providers(),
    }
    print(env, flush=True)
    sub = htd.load("htdemucs").models[0]
    exports = export_all(sub, torch.__version__)
    print(exports, flush=True)

    mixture, stems, sr = load_testset(TESTSET)
    duration = mixture.shape[-1] / sr
    env["audio_seconds"] = round(duration, 2)
    core_diffs = compare_cores(sub, mixture, torch.__version__)
    print(core_diffs, flush=True)

    rows, reference = [], None
    for cfg in configs(torch.__version__):
        try:
            row, estimate = measure(cfg, mixture, stems, sr, duration, reference)
        except Exception as e:  # keep the rest of the sweep
            import traceback

            traceback.print_exc()
            rows.append({"name": cfg["name"], "error": repr(e),
                         "trace": traceback.format_exc()[-2000:]})  # fmt: skip
            continue
        if cfg["name"] == REFERENCE:
            reference = estimate
        rows.append(row)
        print(row, flush=True)
        volume.commit()  # keep any TensorRT engines built
    return {"env": env, "exports": exports, "core_diffs": core_diffs, "rows": rows}


def export_all(sub, torch_version: str) -> dict:
    """Export each file unless the volume already has it."""
    from stemsep.onnx_backend import export

    out = {}
    for batch, fp16 in EXPORTS:
        path = onnx_path(torch_version, batch, fp16)
        entry = {"cached": os.path.exists(path)}
        if not entry["cached"]:
            t0 = time.perf_counter()
            export(sub, path, batch, fp16=fp16)
            entry["export_s"] = round(time.perf_counter() - t0, 1)
            volume.commit()
        entry["mb"] = round(os.path.getsize(path) / 1e6, 1)
        out[os.path.basename(path)] = entry
    return out


def compare_cores(sub, mixture, torch_version: str) -> dict:
    """The network alone on the song's first four chunks: how far each
    backend lands from PyTorch fp32 (TF32 off). Batch 4, so fp32 ORT fits a
    T4."""
    import torch

    from stemsep import model as htd
    from stemsep.onnx_backend import OnnxCore

    sub = sub.to("cuda")
    seg = htd.segment_length(sub)
    mix = torch.stack([mixture[:, i * seg : (i + 1) * seg] for i in range(4)]).cuda()
    mix = (mix - mix.mean()) / mix.std()
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    out = {}
    with torch.inference_mode():
        spec = htd.pre(sub, mix)
        core = htd.Core(sub).eval()
        # On the CPU, so each backend has the GPU to itself.
        ref = [t.cpu() for t in core(spec.x, spec.xt)]
        with torch.autocast("cuda", dtype=torch.float16):
            candidates = {"eager fp16 (autocast)": [t.cpu() for t in core(spec.x, spec.xt)]}
        torch.cuda.empty_cache()
        for kind in ("fp32", "fp16"):
            try:
                onnx = OnnxCore(onnx_path(torch_version, 4, kind == "fp16"), "cuda", kind)
                candidates[f"onnx {kind}"] = [t.cpu() for t in onnx(spec.x, spec.xt)]
            except Exception as e:
                out[f"onnx {kind}"] = {"error": repr(e)[:300]}
            onnx = None
            gc.collect()
        for name, got in candidates.items():
            row = {}
            for key, a, b in zip(("x", "xt"), got, ref):
                a = a.float()
                err = (a - b).abs().max().item()
                row[key] = {"max_abs": err, "rel_to_peak": err / b.abs().max().item(),
                            "finite": bool(torch.isfinite(a).all())}  # fmt: skip
            out[name] = row
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    sub.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return out


def measure(cfg: dict, mixture, stems, sr: int, duration: float, reference):
    import torch

    from stemsep import Config, Profiler, Separator
    from stemsep.quality import report, sdr

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()
    kwargs = {"batch_size": cfg["batch_size"], "overlap": OVERLAP}
    t0 = time.perf_counter()
    if "onnx" in cfg:
        kind = cfg["onnx"]
        path = onnx_path(torch.__version__, cfg["batch_size"], cfg.get("file", kind) == "fp16")
        sep = Separator("htdemucs", Config(**kwargs, precision=kind, backend="onnx",
                                           onnx_paths=[path], onnx_options=cfg.get("core", {})),
                        device="cuda")  # fmt: skip
    else:
        sep = Separator("htdemucs", Config(**kwargs, precision=cfg["precision"]), device="cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    # The first call pays for cuDNN's algorithm search and graph capture.
    t0 = time.perf_counter()
    sep.separate(mixture[:, : sr * 20])
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - t0

    times, estimate = [], None
    with VramSampler() as vram:
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            estimate = sep.separate(mixture)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    prof = Profiler("cuda")
    sep.separate(mixture, profiler=prof)
    seconds = statistics.median(times)
    quality = report(stems, estimate, sr)
    nonfinite = any(not torch.isfinite(v).all() for v in estimate.values())

    row = {
        "name": cfg["name"],
        "load_s": round(load_s, 2),
        "first_call_s": round(first_call_s, 2),
        "separate_s": round(seconds, 3),
        "separate_runs_s": [round(t, 3) for t in times],
        "realtime_x": round(duration / seconds, 1),
        "vram_peak_device_gib": round(vram.peak / 2**30, 2),
        "vram_peak_delta_gib": round((vram.peak - (total - free_before)) / 2**30, 2),
        "stages_s": prof.as_dict(),
        "sdr": round(quality["mean"]["sdr"], 3),
        "nonfinite": nonfinite,
    }
    if reference is not None:
        row["sdr_vs_eager_fp16"] = round(
            statistics.mean(sdr(reference[s], estimate[s]) for s in estimate), 2
        )
    del sep
    return row, estimate


@app.local_entrypoint()
def main(gpu: str = "L4", trt: bool = False):
    result = (bench_trt if trt else bench).with_options(gpu=gpu).remote(gpu)
    out_dir = Path(__file__).resolve().parent.parent / "results" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{gpu}{'_trt' if trt else ''}.json"
    path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result["env"]))
    for name, entry in result["exports"].items():
        print(f"{name:28} {entry}")
    for name, row in result["core_diffs"].items():
        if "error" in row:
            print(f"{name:24} ERROR {row['error'][:100]}")
            continue
        print(f"{name:24} " + "  ".join(
            f"{k}: max {v['max_abs']:.2e} rel {v['rel_to_peak']:.2e} finite {v['finite']}"
            for k, v in row.items()))  # fmt: skip
    head = f"{'config':26} {'load':>6} {'first':>6} {'sep_s':>7} {'x_rt':>6} {'vram':>6} {'Δvram':>6} {'sdr':>6} {'vs_ref':>7}"  # noqa: E501
    print(head)
    for r in result["rows"]:
        if "error" in r:
            print(f"{r['name']:26} ERROR {r['error'][:100]}")
            continue
        print(f"{r['name']:26} {r['load_s']:6.2f} {r['first_call_s']:6.2f} {r['separate_s']:7.3f} "
              f"{r['realtime_x']:6.1f} {r['vram_peak_device_gib']:6.2f} {r['vram_peak_delta_gib']:6.2f} "
              f"{r['sdr']:6.2f} {r.get('sdr_vs_eager_fp16', float('nan')):7.2f}")  # fmt: skip
    print(f"wrote {path}")
