"""Cold-start latency: how long a request waits when no container is up.

    modal deploy bench/coldstart.py && python bench/coldstart.py

Three ways to bring a container up, as three classes:

- `NoSnapshot`: import torch, load weights, move to the GPU, every time.
- `CpuSnapshot`: what stemplayer does today. A memory snapshot restores
  torch and the weights on the CPU; they move to the GPU after restore.
- `GpuSnapshot`: Modal's GPU memory snapshot (alpha). Weights load straight
  to the GPU and a warm-up pass runs before the snapshot, so cuDNN's chosen
  algorithms and the CUDA context come back already set up.

Snapshots are only taken for deployed apps, so this deploys, then calls each
class repeatedly with gaps longer than its scaledown window, so every call
lands on a fresh container. The first call of each class takes the snapshot
and isn't counted.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import modal

from images import image

APP = "stemsep-coldstart"
app = modal.App(APP)

GPU = "L4"
SCALEDOWN = 2
COMMON = dict(image=image, gpu=GPU, cpu=4.0, memory=8192, scaledown_window=SCALEDOWN, timeout=600)
WARM_SECONDS = 20


def _config():
    from stemsep import Config

    return Config(overlap=0.10, batch_size=8, precision="fp16")


def _probe(sep, timings: dict) -> dict:
    """A short separation, so the reported time includes the first real
    call's extra costs (cuDNN autotuning) as well as container start."""
    import torch

    t0 = time.perf_counter()
    sep.separate(torch.randn(2, 44100 * 10) * 0.1)
    torch.cuda.synchronize()
    return {**timings, "first_call_s": round(time.perf_counter() - t0, 3), "now": time.time()}


@app.cls(**COMMON)
class NoSnapshot:
    @modal.enter()
    def load(self):
        t0 = time.time()
        from stemsep import Separator

        self.sep = Separator("htdemucs", _config(), device="cuda")
        self.timings = {"enter_s": round(time.time() - t0, 3), "enter_done": time.time()}

    @modal.method()
    def probe(self) -> dict:
        return _probe(self.sep, self.timings)


@app.cls(**COMMON, enable_memory_snapshot=True)
class CpuSnapshot:
    @modal.enter(snap=True)
    def load(self):
        from stemsep import Separator

        self.sep = Separator("htdemucs", _config(), device="cpu")

    @modal.enter(snap=False)
    def to_gpu(self):
        t0 = time.time()
        self.sep.to("cuda")
        self.timings = {"enter_s": round(time.time() - t0, 3), "enter_done": time.time()}

    @modal.method()
    def probe(self) -> dict:
        return _probe(self.sep, self.timings)


@app.cls(
    **COMMON,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class GpuSnapshot:
    @modal.enter(snap=True)
    def load(self):
        import torch

        from stemsep import Separator

        self.sep = Separator("htdemucs", _config(), device="cuda")
        # Autotune before the snapshot, so restores skip it.
        self.sep.separate(torch.randn(2, 44100 * WARM_SECONDS) * 0.1)
        torch.cuda.synchronize()

    @modal.enter(snap=False)
    def restored(self):
        self.timings = {"enter_s": 0.0, "enter_done": time.time()}

    @modal.method()
    def probe(self) -> dict:
        return _probe(self.sep, self.timings)


def measure(name: str, calls: int, gap: float) -> dict:
    cls = modal.Cls.from_name(APP, name)
    samples = []
    for i in range(calls + 1):
        t0 = time.time()
        result = cls().probe.remote()
        total = time.time() - t0
        sample = {
            "total_s": round(total, 2),
            # Clocks differ a little between here and the container.
            "until_ready_s": round(result["enter_done"] - t0, 2),
            "enter_s": result["enter_s"],
            "first_call_s": result["first_call_s"],
        }
        print(name, i, sample, flush=True)
        if i:  # the first call takes the snapshot
            samples.append(sample)
        time.sleep(gap)
    return {
        "samples": samples,
        "median_total_s": statistics.median(s["total_s"] for s in samples),
        "median_until_ready_s": statistics.median(s["until_ready_s"] for s in samples),
    }


if __name__ == "__main__":
    calls = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    results = {"gpu": GPU}
    for name in ("NoSnapshot", "CpuSnapshot", "GpuSnapshot"):
        try:
            results[name] = measure(name, calls, gap=SCALEDOWN + 25)
        except Exception as e:
            results[name] = {"error": repr(e)}
            print(name, "failed:", e)
    out = Path(__file__).resolve().parent.parent / "results" / "coldstart.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v.get("median_total_s") if isinstance(v, dict) else v
                      for k, v in results.items()}, indent=2))  # fmt: skip
