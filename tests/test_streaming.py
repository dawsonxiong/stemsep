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
