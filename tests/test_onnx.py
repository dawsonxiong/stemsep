"""The ONNX backend against PyTorch, on the CPU. Slow: exports HTDemucs."""

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from stemsep import Config, Separator  # noqa: E402
from stemsep import model as htd  # noqa: E402
from stemsep.onnx_backend import OnnxCore, export  # noqa: E402
from stemsep.quality import sdr  # noqa: E402

pytestmark = pytest.mark.slow

BATCH = 2


@pytest.fixture(scope="module")
def eager():
    return Separator("htdemucs", Config(batch_size=BATCH), device="cpu")


@pytest.fixture(scope="module")
def onnx_path(eager, tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "htdemucs.onnx"
    return export(eager.bag.models[0], str(path), BATCH)


def test_core_matches_torch(eager, onnx_path):
    sub = eager.bag.models[0]
    torch.manual_seed(0)
    mix = torch.randn(BATCH, 2, htd.segment_length(sub)) * 0.1
    with torch.inference_mode():
        spec = htd.pre(sub, mix)
        expected = htd.Core(sub)(spec.x, spec.xt)
        got = OnnxCore(onnx_path, "cpu")(spec.x, spec.xt)
    # ORT's CPU kernels sum GroupNorm's millions of elements in one float32
    # pass, which loses a few more digits than PyTorch does (~2e-4 of the
    # peak; on CUDA cuDNN does the reductions). Hence a tolerance relative to
    # the peak rather than 1e-3 absolute.
    for name, a, b in zip(("x", "xt"), got, expected):
        assert a.shape == b.shape
        err = (a - b).abs().max() / b.abs().max()
        assert err < 1e-3, (name, err)


def test_core_rejects_other_batch_sizes(eager, onnx_path):
    x, xt = htd.example_inputs(eager.bag.models[0], BATCH + 1)
    with pytest.raises(ValueError, match="static_batch"):
        OnnxCore(onnx_path, "cpu")(x, xt)


def test_separator_matches_eager(eager, onnx_path):
    torch.manual_seed(1)
    wav = torch.randn(2, 44100 * 20) * 0.1
    config = Config(batch_size=BATCH, backend="onnx", onnx_paths=[onnx_path])
    ours = Separator("htdemucs", config, device="cpu").separate(wav)
    theirs = eager.separate(wav)
    for name in eager.sources:
        assert sdr(theirs[name], ours[name]) > 40, name
