"""The container image the benchmarks run in, shared by every bench app."""

import modal

TORCH = "2.11.0"

# Per-second Modal prices, from modal.com/pricing on 2026-10-08.
GPU_PRICE = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100": 0.000583,
    "H100": 0.001097,
}
CPU_CORE_PRICE = 0.0000131
MEMORY_GIB_PRICE = 0.00000222

volume = modal.Volume.from_name("stemsep-bench", create_if_missing=True)

_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .uv_pip_install(
        f"torch=={TORCH}",
        f"torchaudio=={TORCH}",
        "demucs==4.0.1",
        "lameenc==1.8.4",
        "numpy",
        "soundfile",
        "onnx",
        "onnxruntime-gpu",
    )
    .run_commands(
        "python -c \"from demucs.pretrained import get_model; "
        "get_model('htdemucs'); get_model('htdemucs_ft')\""
    )
)

# images too: every bench app imports it, in the container as well.
image = _base.add_local_python_source("stemsep", "images")

# ONNX Runtime's PyPI GPU wheels are built for CUDA 13 and cuDNN 9, as torch
# 2.11's are; the extras pull in those libraries, which torch already has.
# onnxscript is what torch.onnx's dynamo exporter writes the graph with.
_onnx = _base.uv_pip_install(
    "onnxruntime-gpu[cuda,cudnn]==1.30.0", "onnx==1.23.2", "onnxscript==0.7.2"
)
onnx_image = _onnx.add_local_python_source("stemsep", "images")

# The TensorRT that ORT 1.30's TensorRT provider is built against. Only the
# libraries: ORT has its own ONNX parser. They aren't where the loader looks,
# hence LD_LIBRARY_PATH.
TRT_LIBS = "/usr/local/lib/python3.11/site-packages/tensorrt_libs"
trt_image = (
    _onnx.uv_pip_install(
        "tensorrt-cu13-libs==10.14.1.48.post1", extra_index_url="https://pypi.nvidia.com"
    )
    .env({"LD_LIBRARY_PATH": TRT_LIBS})
    .add_local_python_source("stemsep", "images")
)
