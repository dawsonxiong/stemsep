"""The split forward and the batched separator against demucs itself."""

import pytest
import torch

from stemsep import Config, Separator, plan_offsets
from stemsep import model as htd


@pytest.fixture(scope="module")
def sep():
    torch.manual_seed(0)
    return Separator("htdemucs", Config(batch_size=3), device="cpu")


def test_split_forward_matches_htdemucs(sep):
    sub = sep.bag.models[0]
    mix = torch.randn(2, 2, htd.segment_length(sub)) * 0.1
    with torch.inference_mode():
        expected = sub(mix)
        got = htd.forward(sub, htd.Core(sub), mix)
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()


def test_plan_offsets_cover_the_song():
    seg = 1000
    for length in (1, 999, 1000, 1001, 5432, 10_000):
        offsets = plan_offsets(length, seg, 0.25)
        assert offsets[0] == 0
        assert offsets[-1] + seg >= length
        if length > seg:
            assert offsets[-1] + seg == length
            assert all(b - a <= 750 for a, b in zip(offsets, offsets[1:]))


def test_matches_apply_model(sep):
    from demucs.apply import apply_model

    torch.manual_seed(1)
    seconds = 20
    wav = torch.randn(2, 44100 * seconds) * 0.1
    ours = sep.separate(wav)

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    with torch.inference_mode():
        theirs = apply_model(
            sep.bag, ((wav - mean) / std)[None], shifts=0, split=True, overlap=0.25
        )[0] * std + mean

    # Both start with the same chunks; only where the last one sits differs
    # (ours ends flush with the song, demucs's overhangs into padding), and
    # that shifts the end. So compare the stretch both cut identically. Their
    # quality on real music is compared by SDR in the benchmarks instead.
    seg = htd.segment_length(sep.bag.models[0])
    # Up to the second chunk of either, only the first chunk counts.
    same = min(int(0.75 * seg), plan_offsets(wav.shape[-1], seg, 0.25)[1])
    for i, name in enumerate(sep.sources):
        a, b = ours[name][..., :same], theirs[i][..., :same]
        snr = 10 * torch.log10(b.pow(2).sum() / (a - b).pow(2).sum())
        assert snr > 40, (name, snr)


def test_short_input_and_stem_subset(sep):
    wav = torch.randn(2, 44100) * 0.1
    out = sep.separate(wav, stems=["vocals"])
    assert list(out) == ["vocals"]
    assert out["vocals"].shape == wav.shape
