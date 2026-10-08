"""Edge cases of the engine and audio I/O: odd lengths, silence, batching,
bag weights, autocast scope, and decoding/encoding against demucs.

Most engine tests swap the network for a pass-through, so every stem should
come back as the input mix and only the chunking, overlap-add and
normalization are under test. A few run the real model on one or two chunks.
"""

import shutil
import subprocess

import pytest
import torch

from stemsep import Config, Separator, plan_offsets
from stemsep import model as htd

SR = 44100


@pytest.fixture(scope="module")
def sep():
    return Separator("htdemucs", Config(batch_size=2), device="cpu")


@pytest.fixture(scope="module")
def seg(sep):
    return htd.segment_length(sep.bag.models[0])


# ---- plan_offsets -------------------------------------------------------


@pytest.mark.parametrize("overlap", [0.0, 0.1, 0.25, 0.5, 0.9])
def test_plan_offsets_edges(overlap):
    seg = 343_980
    stride = int((1 - overlap) * seg)
    for length in (1, 1023, seg - 1, seg, seg + 1, seg + stride, seg + stride + 1, 600 * SR):
        offsets = plan_offsets(length, seg, overlap)
        assert offsets[0] == 0
        if length <= seg:
            assert offsets == [0]
            continue
        assert offsets[-1] + seg == length
        gaps = [b - a for a, b in zip(offsets, offsets[1:])]
        # Strictly increasing, no wider than demucs's stride (so the overlap
        # is at least what was asked), and even to within a sample.
        assert min(gaps) > 0 and max(gaps) <= stride and max(gaps) - min(gaps) <= 1
        # No more chunks than needed.
        assert len(offsets) - 1 == -(-(length - seg) // stride)


# ---- the engine with a pass-through network -----------------------------


class PassThrough(torch.nn.Module):
    """Stands in for `Core`: hands `xt` back scaled by `factor`, and records
    batch sizes and whether autocast was on."""

    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor
        self.batches = []
        self.autocast = []

    def forward(self, x, xt):
        self.batches.append(xt.shape[0])
        self.autocast.append(torch.is_autocast_enabled("cpu"))
        return x, xt * self.factor


@pytest.fixture
def stub(sep, monkeypatch):
    """`sep` with the network replaced: every stem is chunk * factor."""
    autocast = {"pre": [], "post": []}

    def pre(model, mix):
        autocast["pre"].append(torch.is_autocast_enabled("cpu"))
        return htd.Spec(None, mix[:, :1, None, :1], mix, None, None, None, None)

    def post(model, spec, x, xt):
        autocast["post"].append(torch.is_autocast_enabled("cpu"))
        B, C, L = xt.shape
        return xt.float()[:, None].expand(B, len(model.sources), C, L)

    monkeypatch.setattr(htd, "pre", pre)
    monkeypatch.setattr(htd, "post", post)
    saved = (sep.config, sep._cores, sep.bag.models, sep.bag.weights)
    sep._cores = [PassThrough()]
    sep.autocast_log = autocast
    yield sep
    sep.config, sep._cores, sep.bag.models, sep.bag.weights = saved


@pytest.mark.parametrize("static_batch", [False, True])
def test_overlap_add_reconstructs(stub, seg, static_batch):
    stub.config = Config(batch_size=2, static_batch=static_batch)
    torch.manual_seed(0)
    # Under a hop, under a second, exactly a chunk, a sample over, and long
    # enough for an odd number of chunks (a padded last batch).
    for length in (2, 100, SR, seg, seg + 1, 3 * seg + 12_345):
        wav = torch.randn(2, length) * 0.1 + 0.01
        out = stub.separate(wav)
        for name, est in out.items():
            assert est.shape == wav.shape
            assert torch.allclose(est, wav, atol=1e-5), (length, name)
    batches = stub._cores[0].batches
    if static_batch:
        assert set(batches) == {2}
    else:
        assert 1 in batches  # the odd last batch ran unpadded


@pytest.mark.parametrize(
    "wav",
    [
        torch.zeros(2, 50_000),  # digital silence: demucs divides by 0
        torch.full((2, 50_000), 0.25),  # DC only: std 0, mean not
        torch.stack([torch.ones(50_000), -torch.ones(50_000)]) * 0.1,  # L = -R
        torch.tensor([[0.3], [-0.1]]),  # one sample: std is NaN
    ],
    ids=["silence", "dc", "cancelling", "one_sample"],
)
def test_degenerate_songs_round_trip(stub, wav):
    stub.config = Config(batch_size=2)
    for est in stub.separate(wav).values():
        assert torch.isfinite(est).all()
        assert torch.allclose(est, wav, atol=1e-5)


def test_rejects_bad_requests(stub):
    with pytest.raises(ValueError, match="no stems"):
        stub.separate(torch.zeros(2, 100), stems=[])
    with pytest.raises(ValueError, match="unknown"):
        stub.separate(torch.zeros(2, 100), stems=["kazoo"])
    with pytest.raises(ValueError, match="empty"):
        stub.separate(torch.zeros(2, 0))
    with pytest.raises(ValueError, match="expected"):
        stub.separate(torch.zeros(1, 100))


def test_progress_counts_chunks(stub, seg):
    stub.config = Config(batch_size=2)
    seen = []
    stub.separate(torch.randn(2, 3 * seg), progress=seen.append)
    n = len(plan_offsets(3 * seg, seg, 0.25))
    assert len(seen) == -(-n // 2)
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_autocast_only_around_the_network(stub):
    stub.config = Config(batch_size=2, precision="bf16")
    stub.separate(torch.randn(2, SR))
    assert stub._cores[0].autocast == [True]
    assert stub.autocast_log == {"pre": [False], "post": [False]}

    stub._cores = [PassThrough()]
    stub.config = Config(batch_size=2, precision="fp32")
    stub.separate(torch.randn(2, SR))
    assert stub._cores[0].autocast == [False]


def test_bag_weights_and_stem_subset(stub, seg):
    """A two-model bag, weighted per stem as demucs's BagOfModels is: each
    stem is the weighted mean of the models that have a say in it."""
    m = stub.bag.models[0]
    stub.bag.models = torch.nn.ModuleList([m, m])
    stub.bag.weights = [[1.0, 0.0, 2.0, 1.0], [3.0, 1.0, 0.0, 1.0]]
    a, b = PassThrough(2.0), PassThrough(-1.0)
    stub._cores = [a, b]
    stub.config = Config(batch_size=4)
    wav = torch.randn(2, seg + 5000) * 0.1
    ref = wav.mean(0)
    centred = wav - ref.mean()

    out = stub.separate(wav)
    for k, name in enumerate(stub.sources):
        wa, wb = stub.bag.weights[0][k], stub.bag.weights[1][k]
        factor = (2.0 * wa - 1.0 * wb) / (wa + wb)
        assert torch.allclose(out[name], centred * factor + ref.mean(), atol=1e-5), name

    # Only the second model has a say in "bass", so the first never runs.
    a.batches.clear(), b.batches.clear()
    out = stub.separate(wav, stems=["bass"])
    assert list(out) == ["bass"] and not a.batches and b.batches
    assert torch.allclose(out["bass"], -centred + ref.mean(), atol=1e-5)


# ---- the real model -----------------------------------------------------


def test_real_model_silence_and_tiny_input(sep):
    sep.config = Config(batch_size=2)
    out = sep.separate(torch.zeros(2, 4 * SR))
    for est in out.values():
        assert torch.isfinite(est).all() and est.abs().max() == 0
    # Shorter than a hop: the chunk is zero-padded, so `_spec`'s reflect
    # padding never sees fewer samples than it needs.
    wav = torch.randn(2, 300) * 0.1
    out = sep.separate(wav)
    for est in out.values():
        assert est.shape == wav.shape and torch.isfinite(est).all()


def test_real_model_padded_batch_matches(sep, seg):
    torch.manual_seed(3)
    wav = torch.randn(2, seg + SR) * 0.1  # two chunks
    sep.config = Config(batch_size=3, static_batch=True)  # one zero chunk added
    padded = sep.separate(wav)
    sep.config = Config(batch_size=1)
    plain = sep.separate(wav)
    sep.config = Config(batch_size=2)
    for name in sep.sources:
        assert torch.allclose(padded[name], plain[name], atol=1e-4), name


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_split_forward_on_mps(sep):
    sub = sep.bag.models[0]
    mix = torch.randn(1, 2, htd.segment_length(sub)) * 0.1
    with torch.inference_mode():
        expected = htd.forward(sub, htd.Core(sub), mix)
        sub.to("mps")
        try:
            got = htd.forward(sub, htd.Core(sub), mix.to("mps"))
        finally:
            sub.to("cpu")
    assert got.device.type == "mps"
    assert torch.allclose(got.cpu(), expected, atol=1e-3)


# ---- audio I/O ----------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _tone(path, channels, rate=48_000, seconds=1):
    graph = f"sine=frequency=440:sample_rate={rate}:duration={seconds}"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", graph,
         "-af", f"pan={channels}c|" + "|".join(f"c{i}=c0" for i in range(channels)),
         str(path)],
        check=True,
    )  # fmt: skip


@needs_ffmpeg
@pytest.mark.parametrize("channels", [1, 2, 6])
def test_decode_matches_demucs_audiofile(tmp_path, channels):
    """Mono is copied at full level and extra channels dropped, not ffmpeg's
    `-ac 2` downmix; the resampling is ffmpeg's in both."""
    from demucs.audio import AudioFile

    from stemsep.audio import decode

    path = tmp_path / f"tone{channels}.wav"
    _tone(path, channels)
    ours = decode(str(path), SR, 2)
    theirs = AudioFile(path).read(streams=0, samplerate=SR, channels=2)
    assert ours.shape == theirs.shape
    assert torch.equal(ours, theirs)


@needs_ffmpeg
def test_decode_errors(tmp_path):
    from stemsep.audio import DecodeError, decode

    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(DecodeError):
        decode(str(bad))


@needs_ffmpeg
def test_encode_mp3_rescales_and_keeps_shape(tmp_path):
    from stemsep.audio import decode, encode_mp3

    t = torch.arange(3 * SR) / SR
    left = 2.0 * torch.sin(2 * torch.pi * 440 * t)  # would clip
    wav = torch.stack([left, 0.5 * left])
    path = tmp_path / "out"  # no extension: the format is given explicitly
    encode_mp3(wav, str(path), SR)
    back = decode(str(path), SR, 2)
    assert abs(back.shape[-1] - wav.shape[-1]) < SR // 10
    # Scaled down as a whole, as demucs's clip="rescale" does: just under
    # full scale, channels in the same ratio.
    peaks = back.abs().amax(-1)
    assert 0.9 < peaks[0] < 1.05
    assert abs(peaks[1] / peaks[0] - 0.5) < 0.02


@needs_ffmpeg
def test_decode_paths_agree(tmp_path):
    """MP3 at 44.1 kHz is read by libsndfile; it matches ffmpeg's decode."""
    import subprocess

    from stemsep.audio import _read_ffmpeg, decode

    path = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=3",
         "-ac", "2", "-ar", str(SR), "-b:a", "192k", str(path)],
        check=True,
    )  # fmt: skip
    ours = decode(str(path), SR, 2)
    theirs = torch.from_numpy(_read_ffmpeg(str(path), SR).T.copy())
    assert ours.shape == theirs.shape
    assert (ours - theirs).abs().max() < 1e-4


@needs_ffmpeg
def test_decode_resamples_through_ffmpeg(tmp_path):
    import subprocess

    from stemsep.audio import decode

    path = tmp_path / "tone48k.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=2",
         "-ar", "48000", str(path)],
        check=True,
    )  # fmt: skip
    wav = decode(str(path), SR, 2)
    assert wav.shape[0] == 2
    assert abs(wav.shape[1] - 2 * SR) < 100
