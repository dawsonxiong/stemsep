"""Batched, overlap-added HTDemucs separation.

`demucs.apply.apply_model` cuts a song into training-length chunks and runs
them through the model one at a time, which leaves most of a GPU idle. Here
the chunks are cut up front, stacked, and run `batch_size` at a time; the
results are cross-faded back together with the same triangular window demucs
uses. Each chunk is normalized on its own inside the model, so batching
doesn't change what any chunk sees.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import torch

from . import model as htd
from .profile import NullProfiler, Profiler

Precision = Literal["fp32", "tf32", "fp16", "bf16"]
Backend = Literal["eager", "compile", "onnx"]


@dataclass
class Config:
    # Fraction of each chunk shared with the next. demucs defaults to 0.25.
    overlap: float = 0.25
    # Chunks per forward pass.
    batch_size: int = 8
    # fp16 and bf16 run the network under autocast; the STFTs stay float32.
    # tf32 keeps float32 tensors but lets matmuls use TF32 units too. Convs
    # use them in every mode on GPUs that have them, as torch (and so demucs)
    # does by default, through torch.backends.cudnn.allow_tf32.
    precision: Precision = "fp32"
    backend: Backend = "eager"
    # torch.compile mode, for backend="compile".
    compile_mode: str = "max-autotune-no-cudagraphs"
    # Path to an exported ONNX file per sub-model, for backend="onnx".
    onnx_paths: Sequence[str] = field(default_factory=tuple)
    # Passed to OnnxCore: cuda_graph, tensorrt, and provider options.
    onnx_options: dict = field(default_factory=dict)
    # Pad the last batch up to batch_size, so a compiled or exported network
    # only ever sees one shape.
    static_batch: bool | None = None
    # Let cuDNN time its convolution algorithms on the first batch. On an L4
    # that added ~12 s to a fresh container's first song and saved ~2% after,
    # so it's off unless containers live long or snapshot the GPU.
    cudnn_benchmark: bool = False

    def __post_init__(self):
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.static_batch is None:
            self.static_batch = self.backend != "eager"


def plan_offsets(length: int, segment: int, overlap: float) -> list[int]:
    """Chunk starts covering [0, length), each chunk `segment` long.

    The chunks are spread evenly so the last one ends exactly at `length`,
    rather than overhanging into padding the way demucs's do. The overlap
    between neighbours is at least `overlap`."""
    if length <= segment:
        return [0]
    stride = max(1, int((1 - overlap) * segment))
    n = math.ceil((length - segment) / stride) + 1
    return [round(i * (length - segment) / (n - 1)) for i in range(n)]


def window(segment: int, device) -> torch.Tensor:
    """demucs's triangular cross-fade weight, peaking mid-chunk."""
    half = segment // 2
    w = torch.cat(
        [
            torch.arange(1, half + 1, device=device),
            torch.arange(segment - half, 0, -1, device=device),
        ]
    ).float()
    return w / w.max()


class Separator:
    """Splits songs into stems with a pretrained HTDemucs model.

    Load once and keep: construction loads weights, and the first call on a
    compiled backend compiles."""

    def __init__(self, name: str = "htdemucs", config: Config | None = None, device=None):
        self.config = config or Config()
        self.bag = htd.load(name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sources: list[str] = list(self.bag.sources)
        self.samplerate: int = self.bag.samplerate
        self.audio_channels: int = self.bag.audio_channels
        self._cores: list = []
        # Process-wide, so set either way rather than inherit a neighbour's.
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        # Also process-wide. Set here, once, rather than set and restored per
        # call, which races when separations run concurrently in one process.
        torch.set_float32_matmul_precision(self._matmul_precision())
        self.to(self.device)

    # ---- setup -----------------------------------------------------------

    def to(self, device) -> "Separator":
        self.device = torch.device(device)
        self.bag.to(self.device)
        self._cores = [self._make_core(i, m) for i, m in enumerate(self.bag.models)]
        return self

    def _make_core(self, index: int, model):
        cfg = self.config
        core = htd.Core(model).eval()
        if cfg.backend == "eager":
            return core
        if cfg.backend == "compile":
            return torch.compile(core, mode=cfg.compile_mode, dynamic=False, fullgraph=False)
        if cfg.backend == "onnx":
            from .onnx_backend import OnnxCore

            if index >= len(cfg.onnx_paths):
                raise ValueError("backend='onnx' needs an onnx_paths entry per sub-model")
            return OnnxCore(
                cfg.onnx_paths[index], self.device, cfg.precision, **cfg.onnx_options
            )
        raise ValueError(f"unknown backend {cfg.backend!r}")

    def _autocast(self):
        cfg = self.config
        if cfg.backend == "onnx" or cfg.precision not in ("fp16", "bf16"):
            # Not autocast(enabled=False): older torch rejects device type "mps".
            return nullcontext()
        dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
        return torch.autocast(self.device.type, dtype=dtype)

    def _matmul_precision(self):
        return "high" if self.config.precision == "tf32" else "highest"

    # ---- separation ------------------------------------------------------

    @torch.inference_mode()
    def separate(
        self,
        wav: torch.Tensor,
        stems: Sequence[str] | None = None,
        progress: Callable[[float], None] | None = None,
        profiler: Profiler | None = None,
    ) -> dict[str, torch.Tensor]:
        """Separate `wav`, a [channels, length] float tensor at
        `self.samplerate`, into {stem: [channels, length]} float32 tensors on
        the CPU.

        `stems` limits the work to the sub-models that contribute to those
        stems, which only saves anything for a bag of specialists such as
        htdemucs_ft. `progress` is called with the fraction of chunks done.
        """
        prof = profiler or NullProfiler()
        # `stems or ...` would turn an empty list into every stem.
        wanted = list(self.sources if stems is None else stems)
        if not wanted:
            raise ValueError("no stems requested")
        unknown = set(wanted) - set(self.sources)
        if unknown:
            raise ValueError(f"unknown stems {sorted(unknown)}")
        if wav.dim() != 2 or wav.shape[0] != self.audio_channels:
            raise ValueError(f"expected [{self.audio_channels}, length], got {list(wav.shape)}")
        if wav.shape[-1] == 0:
            raise ValueError("empty audio")

        with prof.stage("to_device"):
            wav = wav.to(self.device, torch.float32, non_blocking=True)
            # Normalize the song as a whole, as demucs's CLI does. It divides
            # by the std alone, which is 0 for silence or channels that
            # cancel (and NaN for one sample), and every stem comes out NaN.
            # Undoing with the same `scale` keeps the round trip exact.
            ref = wav.mean(0)
            mean = ref.mean()
            scale = (ref.std() if ref.numel() > 1 else ref.new_zeros(())) + 1e-8
            wav = (wav - mean) / scale

        plan = self._plan(wanted)
        total_chunks = sum(len(self._offsets(wav.shape[-1], self.bag.models[i])) for i, _ in plan)
        done = 0

        def tick(n: int) -> None:
            nonlocal done
            done += n
            if progress:
                progress(done / total_chunks)

        estimate = None
        totals = torch.zeros(len(self.sources), device=self.device)
        for index, weights in plan:
            out = self._separate_one(self.bag.models[index], self._cores[index], wav, prof, tick)
            w = torch.tensor(weights, device=self.device)
            out *= w[:, None, None]
            estimate = out if estimate is None else estimate + out
            totals += w
        assert estimate is not None
        estimate /= totals.clamp_min(1e-8)[:, None, None]

        with prof.stage("to_host"):
            estimate = (estimate * scale + mean).cpu()
        return {name: estimate[self.sources.index(name)] for name in wanted}

    def _plan(self, wanted: list[str]) -> list[tuple[int, list[float]]]:
        """(sub-model index, weights) for each sub-model that adds to
        `wanted`, with the weights of other stems zeroed."""
        plan = []
        for index, weights in enumerate(self.bag.weights):
            weights = [
                float(w) if name in wanted else 0.0 for name, w in zip(self.sources, weights)
            ]
            if any(weights):
                plan.append((index, weights))
        return plan

    def _offsets(self, length: int, sub) -> list[int]:
        return plan_offsets(length, htd.segment_length(sub), self.config.overlap)

    def _separate_one(self, sub, core, wav, prof: Profiler, tick) -> torch.Tensor:
        cfg = self.config
        seg = htd.segment_length(sub)
        length = wav.shape[-1]
        offsets = self._offsets(length, sub)

        with prof.stage("chunk"):
            padded = wav if length >= seg else torch.nn.functional.pad(wav, (0, seg - length))
            chunks = torch.stack([padded[:, o : o + seg] for o in offsets])

        out = torch.zeros(len(self.sources), wav.shape[0], max(length, seg), device=self.device)
        weight_sum = torch.zeros(max(length, seg), device=self.device)
        weight = window(seg, self.device)

        for start in range(0, len(offsets), cfg.batch_size):
            batch = chunks[start : start + cfg.batch_size]
            n = batch.shape[0]
            if cfg.static_batch and n < cfg.batch_size:
                fill = batch.new_zeros(cfg.batch_size - n, *batch.shape[1:])
                batch = torch.cat([batch, fill])

            with prof.stage("stft"):
                spec = htd.pre(sub, batch)
            with prof.stage("network"), self._autocast():
                x, xt = core(spec.x, spec.xt)
            with prof.stage("istft"):
                result = htd.post(sub, spec, x, xt)[:n]
            with prof.stage("overlap_add"):
                for i, o in enumerate(offsets[start : start + n]):
                    out[..., o : o + seg] += result[i] * weight
                    weight_sum[o : o + seg] += weight
            tick(n)

        out /= weight_sum
        return out[..., :length]

