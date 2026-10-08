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

image = (
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
    # images too: every bench app imports it, in the container as well.
    .add_local_python_source("stemsep", "images")
)
