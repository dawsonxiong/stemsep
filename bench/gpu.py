"""Latency, VRAM, cost and quality of separation configs, per GPU type.

    modal run bench/gpu.py                       # every GPU, default sweep
    modal run bench/gpu.py --gpus T4,L4 --sweep quick

Each GPU runs in its own container, all at once. Results land in
results/<sweep>/<gpu>.json and are summarized by bench/report.py.
"""


import json
import os
import statistics
import time
from pathlib import Path

import modal

from images import CPU_CORE_PRICE, GPU_PRICE, MEMORY_GIB_PRICE, image, volume

app = modal.App("stemsep-bench")

CPU_CORES = 4.0
MEMORY_GIB = 16
TESTSET = "/data/testset"
REPEATS = 3

BASELINE = {"name": "demucs apply_model (production)", "baseline": True}


def sweep(name: str, gpu: str) -> list[dict]:
    """Configs to run, as stemsep.Config kwargs plus a display name."""
    if name == "e2e":
        return []
    half = "bf16" if gpu in ("L4", "A10", "L40S", "A100", "H100") else "fp16"
    if name == "quick":
        return [
            BASELINE,
            {"name": "eager fp32 ov.25 b1", "overlap": 0.25, "batch_size": 1},
            {"name": "eager fp16 ov.10 b8", "overlap": 0.10, "batch_size": 8, "precision": "fp16"},
        ]
    configs = [
        BASELINE,
        {"name": "eager fp32 ov.25 b1", "overlap": 0.25, "batch_size": 1},
        {"name": "eager fp32 ov.25 b8", "overlap": 0.25, "batch_size": 8},
        {"name": "eager fp32 ov.10 b8", "overlap": 0.10, "batch_size": 8},
        {"name": "eager tf32 ov.10 b8", "overlap": 0.10, "batch_size": 8, "precision": "tf32"},
        {"name": "eager fp16 ov.10 b4", "overlap": 0.10, "batch_size": 4, "precision": "fp16"},
        {"name": "eager fp16 ov.10 b8", "overlap": 0.10, "batch_size": 8, "precision": "fp16"},
        {"name": "eager fp16 ov.10 b16", "overlap": 0.10, "batch_size": 16, "precision": "fp16"},
        {"name": "eager fp16 ov.25 b8", "overlap": 0.25, "batch_size": 8, "precision": "fp16"},
        {"name": "eager fp16 ov.05 b8", "overlap": 0.05, "batch_size": 8, "precision": "fp16"},
        {"name": f"compile {half} ov.10 b8", "overlap": 0.10, "batch_size": 8,
         "precision": half, "backend": "compile"},
    ]  # fmt: skip
    if half == "bf16":
        configs.insert(-1, {"name": "eager bf16 ov.10 b8", "overlap": 0.10, "batch_size": 8,
                            "precision": "bf16"})  # fmt: skip
    return configs


# The container here has more memory to hold the test set; costs are for
# stemplayer's production container, which has 8 GiB.
BILLED_MEMORY_GIB = 8


def cost_per_second(gpu: str) -> float:
    return GPU_PRICE[gpu] + CPU_CORES * CPU_CORE_PRICE + BILLED_MEMORY_GIB * MEMORY_GIB_PRICE


@app.cls(
    image=image,
    # Overridden per run with with_options; see main.
    gpu="L4",
    volumes={"/data": volume},
    cpu=CPU_CORES,
    memory=MEMORY_GIB * 1024,
    timeout=3 * 3600,
)
class Bench:
    gpu: str = modal.parameter(default="L4")

    @modal.enter()
    def load_testset(self):
        from stemsep.quality import load_testset

        self.mixture, self.stems, sr = load_testset(TESTSET)
        assert sr == 44100, sr
        self.duration = self.mixture.shape[-1] / sr

    @modal.method()
    def run(self, sweep_name: str) -> dict:
        import torch

        props = torch.cuda.get_device_properties(0)
        env = {
            "gpu": self.gpu,
            "device": props.name,
            "vram_gib": round(props.total_memory / 2**30, 1),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "arch_list": torch.cuda.get_arch_list(),
            "audio_seconds": round(self.duration, 2),
        }
        rows = []
        # Per overlap, the first eager fp32 row's output: other rows compare
        # against the one with the same chunk grid, so "vs fp32" measures
        # precision, batching or compiling, not the grid (and b8 vs b1 at
        # 25% is the batching check).
        references = {}
        for cfg in sweep(sweep_name, self.gpu):
            try:
                row, estimate = self._measure(cfg)
            except Exception as e:  # keep the rest of the sweep
                import traceback

                rows.append({"name": cfg["name"], "error": repr(e),
                             "trace": traceback.format_exc()[-2000:]})  # fmt: skip
                continue
            overlap = cfg.get("overlap")
            fp32 = cfg.get("precision", "fp32") == "fp32" and cfg.get("backend", "eager") == "eager"
            if overlap in references:
                from stemsep.quality import sdr

                row["sdr_vs_fp32"] = round(
                    statistics.mean(sdr(references[overlap][s], estimate[s]) for s in estimate), 2
                )
            elif fp32 and not cfg.get("baseline"):
                references[overlap] = estimate
            rows.append(row)
            torch.cuda.empty_cache()
        e2e = self._end_to_end()
        return {"env": env, "rows": rows, "e2e": e2e}

    def _measure(self, cfg: dict):
        import torch

        from stemsep import Config, Profiler, Separator
        from stemsep.quality import report

        kwargs = {k: v for k, v in cfg.items() if k not in ("name", "baseline")}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if cfg.get("baseline"):
            sep = _DemucsBaseline()
        else:
            sep = Separator("htdemucs", Config(**kwargs), device="cuda")
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t0

        # The first call pays for cuDNN autotuning, and compiling. On the
        # whole song, so every batch shape the timed runs see is warm (a
        # short clip never fills a batch of 8).
        t0 = time.perf_counter()
        sep.separate(self.mixture)
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - t0

        # Drop what autotuning's trial workspaces left cached, so peak
        # reserved is what separating needs.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times, estimate = [], None
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            estimate = sep.separate(self.mixture)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        # Stages from a run of their own: the profiler synchronizes at every
        # stage boundary, which would slow stemsep's timed runs but not the
        # baseline's, which ignores it.
        prof = Profiler("cuda")
        sep.separate(self.mixture, profiler=prof)
        seconds = statistics.median(times)
        quality = report(self.stems, estimate, 44100)

        row = {
            "name": cfg["name"],
            "config": kwargs,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "load_s": round(load_s, 2),
            "first_call_s": round(first_call_s, 2),
            "separate_s": round(seconds, 3),
            "separate_runs_s": [round(t, 3) for t in times],
            "realtime_x": round(self.duration / seconds, 1),
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "stages_s": prof.as_dict(),
            "inference_cost_per_audio_min": round(
                seconds * cost_per_second(self.gpu) / (self.duration / 60), 6
            ),
            "quality": quality,
        }
        del sep
        return row, estimate

    def _end_to_end(self) -> dict:
        """Decode an MP3, separate, encode four MP3s: the whole job minus
        network transfers, with the production candidate config."""
        import tempfile

        import torch

        from stemsep import Config, Profiler, Separator, audio

        sep = Separator("htdemucs", Config(overlap=0.10, batch_size=8, precision="fp16"))
        sep.separate(self.mixture)  # warm every batch shape, as in _measure
        out = {}
        variants = {
            "lameenc q5": (audio.encode_mp3, 5),
            "lameenc q2": (audio.encode_mp3, 2),
            "ffmpeg q5": (audio.encode_mp3_ffmpeg, 5),
        }
        for encoder, (fn, quality) in variants.items():
            prof = Profiler("cuda")
            with tempfile.TemporaryDirectory() as tmp:
                t0 = time.perf_counter()
                with prof.stage("decode"):
                    wav = audio.decode(f"{TESTSET}/mixture.mp3")
                with prof.stage("separate"):
                    stems = sep.separate(wav)
                with prof.stage("encode"):
                    audio.encode_all(
                        stems,
                        {s: os.path.join(tmp, f"{s}.mp3") for s in stems},
                        sep.samplerate,
                        quality=quality,
                        encoder=fn,
                    )
                total = time.perf_counter() - t0
            out[encoder] = {"total_s": round(total, 3), "stages_s": prof.as_dict()}
        torch.cuda.empty_cache()
        return out


class _DemucsBaseline:
    """What stemplayer's Modal app runs today: apply_model with one random
    shift and 25% overlap, batch size 1, fp32."""

    def __init__(self):
        from demucs.pretrained import get_model

        self.model = get_model("htdemucs").eval().to("cuda")
        self.sources = list(self.model.sources)

    def separate(self, wav, profiler=None):
        import torch
        from demucs.apply import apply_model

        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std()
        with torch.inference_mode():
            out = apply_model(
                self.model, ((wav - mean) / std)[None].to("cuda"),
                shifts=1, split=True, overlap=0.25,
            )[0].cpu() * std + mean  # fmt: skip
        return dict(zip(self.sources, out))


@app.local_entrypoint()
def main(gpus: str = "T4,L4,A10", sweep: str = "full"):
    out_dir = Path(__file__).resolve().parent.parent / "results" / sweep
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = {
        g: Bench.with_options(gpu=g)(gpu=g).run.spawn(sweep) for g in gpus.split(",")
    }
    for gpu, call in calls.items():
        result = call.get()
        path = out_dir / f"{gpu}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"wrote {path}")
