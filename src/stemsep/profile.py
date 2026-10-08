"""Wall-clock time per stage, with the GPU synchronized at each boundary so
asynchronous kernels are charged to the stage that launched them."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class Profiler:
    def __init__(self, device=None, sync: bool = True):
        self.device = torch.device(device) if device is not None else None
        self.sync = sync and self.device is not None and self.device.type == "cuda"
        self.seconds: dict[str, float] = defaultdict(float)

    @contextmanager
    def stage(self, name: str):
        if self.sync:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize(self.device)
            self.seconds[name] += time.perf_counter() - start

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] += seconds

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.seconds.items()}


class NullProfiler(Profiler):
    """Times nothing and never synchronizes: for production, where a sync per
    stage would cost throughput."""

    def __init__(self, device=None):
        super().__init__(device, sync=False)

    @contextmanager
    def stage(self, name: str):
        yield
