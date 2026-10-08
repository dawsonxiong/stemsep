"""Stems handed to a sink as they're finished, and encoded as they arrive."""

import pytest
import torch

from stemsep import Config, Separator
from stemsep.audio import StreamingMP3, mp3_bytes

SR = 44100


@pytest.fixture(scope="module")
def sep():
    return Separator("htdemucs", Config(batch_size=2), device="cpu")


def test_sink_gets_the_result_in_order(sep):
    torch.manual_seed(0)
    wav = torch.randn(2, SR * 20) * 0.1
    parts = []
    result = sep.separate(wav, sink=parts.append)
    assert len(parts) > 1  # it streamed, rather than arriving whole
    for name in result:
        streamed = torch.cat([p[name] for p in parts], dim=-1)
        assert torch.equal(streamed, result[name])


def test_sink_on_a_short_song(sep):
    wav = torch.randn(2, SR) * 0.1
    parts = []
    result = sep.separate(wav, stems=["vocals"], sink=parts.append)
    assert list(parts[0]) == ["vocals"]
    assert torch.equal(torch.cat([p["vocals"] for p in parts], -1), result["vocals"])


def _chunks(wav, sizes):
    start = 0
    for n in sizes:
        yield wav[:, start : start + n]
        start += n
    if start < wav.shape[-1]:
        yield wav[:, start:]


def test_streamed_mp3_is_byte_identical():
    torch.manual_seed(1)
    quiet = torch.randn(2, SR * 3) * 0.2
    loud = torch.randn(2, SR * 3) * 0.6  # peaks past 1: needs the rescale
    stems = {"quiet": quiet, "loud": loud}
    enc = StreamingMP3(stems, SR, 2)
    for q, l in zip(_chunks(quiet, [1000, 50_000, 7, 30_000]), _chunks(loud, [1000, 50_000, 7, 30_000])):
        enc.feed({"quiet": q, "loud": l})
    out = enc.finish(stems)
    assert out["quiet"] == mp3_bytes(quiet, SR)
    assert out["loud"] == mp3_bytes(loud, SR)


def test_encoder_errors_surface():
    enc = StreamingMP3(["a"], SR, 2)
    enc.feed({"a": "not audio"})
    enc.feed({"a": torch.zeros(2, 10)})  # and nothing blocks after
    with pytest.raises(Exception):
        enc.finish({"a": torch.zeros(2, 20)})


def test_close_without_finishing():
    enc = StreamingMP3(["a", "b"], SR, 2)
    enc.feed({"a": torch.zeros(2, 100), "b": torch.zeros(2, 100)})
    enc.close()
    enc.close()
    assert not any(t.is_alive() for t in enc._threads)


@pytest.mark.parametrize("length", [1, 1152, 1153])
def test_streamed_mp3_of_a_tiny_stem(length):
    """Shorter than an MP3 frame, a frame exactly, and a sample over."""
    torch.manual_seed(2)
    wav = torch.randn(2, length) * 0.9
    enc = StreamingMP3(["a"], SR, 2)
    for part in _chunks(wav, [1, 5]):
        if part.shape[-1]:  # the engine never hands over an empty stretch
            enc.feed({"a": part})
    assert enc.finish({"a": wav})["a"] == mp3_bytes(wav, SR)


def test_separating_into_the_encoder(sep):
    torch.manual_seed(3)
    wav = torch.randn(2, SR * 10) * 0.1
    enc = StreamingMP3(sep.sources, SR, 2)
    try:
        result = sep.separate(wav, sink=enc.feed)
        out = enc.finish(result)
    finally:
        enc.close()
    for name in sep.sources:
        assert out[name] == mp3_bytes(result[name], SR)


def test_a_failed_separation_stops_the_encoder(sep):
    """As the app runs it: `close` in a finally, after separate raised
    mid-song with stretches already fed."""
    fed = []
    enc = StreamingMP3(sep.sources, SR, 2)

    def progress(done):
        if fed:
            raise RuntimeError("the GPU fell over")

    def sink(part):
        fed.append(part)
        enc.feed(part)

    with pytest.raises(RuntimeError, match="fell over"):
        try:
            sep.separate(torch.randn(2, SR * 20) * 0.1, progress=progress, sink=sink)
        finally:
            enc.close()
    assert fed and not any(t.is_alive() for t in enc._threads)


def test_a_failed_flush_doesnt_hang(monkeypatch):
    """The end of the stream was already taken when the flush failed, so
    there's nothing left to drain."""
    import threading

    from stemsep import audio

    class Broken:
        def __init__(self, real):
            self.real = real

        def encode(self, pcm):
            return self.real.encode(pcm)

        def flush(self):
            raise RuntimeError("flush failed")

    lame = audio._lame
    monkeypatch.setattr(audio, "_lame", lambda *a: Broken(lame(*a)))
    enc = StreamingMP3(["a"], SR, 2)
    enc.feed({"a": torch.zeros(2, 100)})
    raised = []

    def finish():
        try:
            enc.finish({"a": torch.zeros(2, 100)})
        except RuntimeError as e:
            raised.append(e)

    t = threading.Thread(target=finish, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive() and raised


def test_stored_gain_is_undone_exactly():
    """At gain 0.5 a stem peaking at 1.6 isn't rescaled; one at 2.4 is, and
    stored_gain says by how much, so 1 / it restores the level."""
    from stemsep.audio import decode, stored_gain

    torch.manual_seed(2)
    t = torch.arange(3 * SR) / SR
    tone = torch.sin(2 * torch.pi * 220 * t)
    stems = {"loud": torch.stack([1.6 * tone, 1.6 * tone]), "louder": torch.stack([2.4 * tone] * 2)}
    enc = StreamingMP3(stems, SR, 2, gain=0.5)
    for a in range(0, 3 * SR, SR):
        enc.feed({k: v[:, a : a + SR] for k, v in stems.items()})
    out = enc.finish(stems)
    assert out["loud"] == mp3_bytes(stems["loud"], SR, gain=0.5)
    assert out["louder"] == mp3_bytes(stems["louder"], SR, gain=0.5)
    assert enc.gains["loud"] == 0.5
    assert abs(enc.gains["louder"] - stored_gain(2.4, 0.5)) < 1e-4
    import tempfile, os
    for name, peak in (("loud", 1.6), ("louder", 2.4)):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(out[name])
        back = decode(f.name) / enc.gains[name]
        os.unlink(f.name)
        assert abs(back.abs().max().item() - peak) < 0.05 * peak
