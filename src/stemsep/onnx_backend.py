"""HTDemucs's network (`model.Core`) on ONNX Runtime.

Only `Core` is exported: the STFTs in `pre` and `post` are complex-valued and
stay in PyTorch. `pre` also does the normalization whose std overflows in
half precision, so `Core` converts to fp16 as a whole, with no float32
islands, and keeps float32 inputs and outputs so it drops into the same place
in `Separator` either way.

Shapes are fixed at export, batch included: `Separator` pads the last batch
of a song for non-eager backends, so the graph only ever sees one shape, and
ORT can plan memory and capture a CUDA graph once.

onnx and onnxruntime are optional dependencies (`stemsep[onnx]`), imported
only when used. On a GPU, install onnxruntime-gpu built for the same CUDA
major version as torch.
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager

import numpy as np
import torch

from . import model as htd

INPUTS = ("x", "xt")
OUTPUTS = ("x_out", "xt_out")


def export(model_sub, path: str, batch_size: int, fp16: bool = False, dynamo: bool = True,
           opset: int | None = None) -> str:  # fmt: skip
    """Write `Core(model_sub)` to `path` as ONNX for batches of exactly
    `batch_size` chunks. With `fp16`, weights and compute are float16 but the
    inputs and outputs stay float32.

    `dynamo` picks torch's export-based exporter, which folds the shape
    arithmetic the TorchScript one (opset 17 only) leaves in the graph; both
    compute the same thing. `opset` defaults to 18 for it. At 23 the
    attention, GroupNorm and GELU export as single ONNX ops rather than their
    decompositions, which only helps where the runtime has kernels for them.
    The file appears atomically, so an existing one is always complete."""
    sub = copy.deepcopy(model_sub).cpu().float().eval()
    core = htd.Core(sub).eval()
    inputs = htd.example_inputs(sub, batch_size)
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with _no_mha_fastpath(), torch.no_grad():
        kwargs = dict(input_names=list(INPUTS), output_names=list(OUTPUTS), dynamo=dynamo)
        if dynamo:
            kwargs.update(opset_version=opset or 18, external_data=False, verbose=False)
        else:
            kwargs.update(opset_version=opset or 17)
        torch.onnx.export(core, inputs, tmp, **kwargs)
    if fp16:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        graph = convert_float_to_float16(onnx.load(tmp), keep_io_types=True)
        onnx.save(graph, tmp)
    os.replace(tmp, path)
    return path


@contextmanager
def _no_mha_fastpath():
    """The fused nn.MultiheadAttention fast path has no ONNX export; the
    plain one does, and computes the same thing."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


class OnnxCore:
    """Runs an exported `Core` like the module: `core(x, xt)` takes float32
    torch tensors on `device` and returns float32 torch tensors there.

    On CUDA, inputs and outputs are bound to torch's memory with io_binding,
    so nothing is copied. ORT runs on a stream of its own, ordered after
    whatever torch queued before the call and before whatever it queues
    after, so callers needn't synchronize.

    The returned tensors are views of output buffers that every call reuses:
    they are valid until the next call, which overwrites them. `Separator`
    reads them in `post` before it calls again, and anything queued on torch's
    stream before a call finishes before ORT starts writing, so that's safe.
    Clone them to keep them longer.

    `precision` follows `Config.precision`: "fp32" turns TF32 off, "tf32"
    allows it. Whether the network runs in fp16 is fixed by the file (see
    `export`), so "fp16" just means "this is an fp16 export". There's no bf16.

    `cuda_graph` captures the whole network once, at construction, and
    replays it on every call, which saves the kernel launches; inputs are then
    copied into fixed buffers, since a replay reads from the addresses it
    captured. It applies to TensorRT too: don't set trt_cuda_graph_enable
    directly, or calls rebind the inputs and the replay reads stale memory.

    `provider_options` go to the CUDA execution provider as-is (for example
    cudnn_conv_algo_search="HEURISTIC", prefer_nhwc="1"). `tensorrt`, a dict
    of TensorRT execution provider options (possibly empty), puts TensorRT in
    front of CUDA; it needs TensorRT's libraries on the loader path. Give it
    the fp32 export with {"trt_fp16_enable": "True"} (its flags take
    "True"/"False", not "1"/"0"), and cache the engine
    (trt_engine_cache_enable, trt_engine_cache_path): building one takes
    minutes.

    Not for concurrent use: calls share the output buffers and the binding.
    Give each thread its own OnnxCore.
    """

    def __init__(self, path: str, device, precision: str = "fp32", cuda_graph: bool = False,
                 tensorrt: dict | None = None, **provider_options):  # fmt: skip
        import onnxruntime as ort

        if precision == "bf16":
            raise ValueError("the ONNX backend has no bf16; export with fp16=True instead")
        self.device = torch.device(device)
        self.cuda_graph = cuda_graph
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The memory pattern plans one block for every intermediate up front;
        # for HTDemucs at batch 4 that held 14 GiB of an L4 against 6 GiB
        # without it, at the same speed.
        options.enable_mem_pattern = False

        if self.device.type == "cuda":
            if hasattr(ort, "preload_dlls"):
                # Load the CUDA and cuDNN libraries torch already loaded, not
                # whatever else is on the library path.
                ort.preload_dlls()
            index = self.device.index
            if index is None:
                index = torch.cuda.current_device()
            self.device = torch.device("cuda", index)
            self.stream = torch.cuda.Stream(self.device)
            cuda = {
                "device_id": str(index),
                "has_user_compute_stream": "1",
                "user_compute_stream": str(self.stream.cuda_stream),
                "use_tf32": "0" if precision == "fp32" else "1",
                "enable_cuda_graph": "1" if cuda_graph and tensorrt is None else "0",
                # Every run asks for the same buffers, so grow the arena by
                # exactly that rather than doubling it.
                "arena_extend_strategy": "kSameAsRequested",
                **{k: str(v) for k, v in provider_options.items()},
            }
            providers = [("CUDAExecutionProvider", cuda)]
            if tensorrt is not None:
                trt = {
                    "device_id": str(index),
                    "user_compute_stream": str(self.stream.cuda_stream),
                    "trt_cuda_graph_enable": str(cuda_graph),
                    **{k: str(v) for k, v in tensorrt.items()},
                }
                providers.insert(0, ("TensorrtExecutionProvider", trt))
        else:
            if cuda_graph:
                raise ValueError("cuda_graph needs a CUDA device")
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(path), options, providers=providers)
        if self.device.type == "cuda":
            wanted = [p if isinstance(p, str) else p[0] for p in providers]
            missing = set(wanted) - set(self.session.get_providers())
            if missing:
                raise RuntimeError(
                    f"onnxruntime couldn't load {sorted(missing)}; install onnxruntime-gpu "
                    "built for torch's CUDA version (and TensorRT's libraries for TensorRT)"
                )
        meta = {i.name: i.shape for i in self.session.get_inputs()}
        self.shapes = {name: tuple(meta[name]) for name in INPUTS}
        out_meta = {o.name: o.shape for o in self.session.get_outputs()}
        self.out_shapes = {name: tuple(out_meta[name]) for name in OUTPUTS}
        for shape in (*self.shapes.values(), *self.out_shapes.values()):
            if not all(isinstance(d, int) for d in shape):
                raise ValueError(f"{path} has dynamic shapes; export it with export()")

        if self.device.type == "cuda":
            self._bind_cuda()
            if cuda_graph:
                # ORT runs once plainly, to allocate, then captures on the
                # next run. Do both now, while nothing else on the device can
                # break the capture, rather than on the caller's first song.
                for _ in range(2):
                    self.session.run_with_iobinding(self.binding, self.run_options)
                self.stream.synchronize()

    @property
    def batch_size(self) -> int:
        return self.shapes["x"][0]

    def _bind_cuda(self) -> None:
        def empty(shape):
            return torch.empty(shape, device=self.device, dtype=torch.float32)

        self.outputs = {name: empty(self.out_shapes[name]) for name in OUTPUTS}
        self.binding = self.session.io_binding()
        for name, out in self.outputs.items():
            self._bind(self.binding.bind_output, name, out)
        if self.cuda_graph:
            self.inputs = {name: empty(self.shapes[name]).zero_() for name in INPUTS}
            for name, buf in self.inputs.items():
                self._bind(self.binding.bind_input, name, buf)
        import onnxruntime as ort

        self.run_options = ort.RunOptions()
        # Ordering against torch is by stream (see __call__); no need to block
        # the host at the end of every run.
        self.run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

    def _bind(self, bind, name: str, tensor: torch.Tensor) -> None:
        bind(name, "cuda", self.device.index, np.float32, tuple(tensor.shape), tensor.data_ptr())

    def __call__(self, x: torch.Tensor, xt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        given = {"x": x, "xt": xt}
        for name, t in given.items():
            if tuple(t.shape) != self.shapes[name]:
                raise ValueError(
                    f"{name} is {list(t.shape)} but the ONNX graph takes "
                    f"{list(self.shapes[name])} (use Config.static_batch with the export's "
                    "batch size)"
                )
        if self.device.type != "cuda":
            feed = {k: v.detach().to("cpu", torch.float32).contiguous().numpy()
                    for k, v in given.items()}  # fmt: skip
            x_out, xt_out = self.session.run(list(OUTPUTS), feed)
            return torch.from_numpy(x_out).to(x.device), torch.from_numpy(xt_out).to(x.device)

        given = {k: v.to(self.device, torch.float32).contiguous() for k, v in given.items()}
        caller = torch.cuda.current_stream(self.device)
        if self.cuda_graph:
            for name, t in given.items():
                self.inputs[name].copy_(t)
        else:
            for name, t in given.items():
                self._bind(self.binding.bind_input, name, t)
        self.stream.wait_stream(caller)
        self.session.run_with_iobinding(self.binding, self.run_options)
        caller.wait_stream(self.stream)
        if not self.cuda_graph:
            # The inputs were read on ORT's stream; keep the allocator from
            # handing their memory out again before that's done.
            for t in given.values():
                t.record_stream(self.stream)
        return self.outputs["x_out"], self.outputs["xt_out"]


__all__ = ["OnnxCore", "export"]
