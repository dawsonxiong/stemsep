"""SDR metrics and the test-set loader."""

import json
import math
import struct

import numpy as np
import pytest
import torch

from stemsep.quality import (
    chunked_sdr,
    load_fulltracks,
    load_testset,
    read_wav,
    report,
    sdr,
)

SR = 8000


def noise(shape, power, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float64) * math.sqrt(power)


def test_identical_signal_is_huge():
    ref = noise((2, SR * 3), 0.1, 0)
    assert sdr(ref, ref.clone()) > 100
    assert chunked_sdr(ref, ref.clone(), SR) > 100


def test_known_noise_level():
    ref = noise((2, SR * 10), 0.1, 0)
    err = noise((2, SR * 10), 0.001, 1)  # 20 dB below the signal
    expected = 10 * math.log10(float(ref.square().sum()) / float(err.square().sum()))
    assert sdr(ref, ref + err) == pytest.approx(expected, abs=1e-6)
    assert sdr(ref, ref + err) == pytest.approx(20, abs=0.1)
    assert chunked_sdr(ref, ref + err, SR) == pytest.approx(20, abs=0.3)


def test_scaled_estimate():
    ref = noise((2, SR), 0.1, 0)
    # error = 0.5 ref, so SDR = 10 log10(1 / 0.25)
    assert sdr(ref, 0.5 * ref) == pytest.approx(10 * math.log10(4), abs=1e-6)


def test_accepts_numpy_float32_and_mono():
    ref = noise((SR,), 0.1, 0)
    err = noise((SR,), 0.001, 1)
    a = sdr(ref, ref + err)
    b = sdr(ref.numpy().astype(np.float32), (ref + err).numpy().astype(np.float32))
    assert a == pytest.approx(b, abs=1e-3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sdr(torch.zeros(2, 10), torch.zeros(2, 11))


def test_chunked_skips_silent_windows():
    # Five 1 s windows; the reference is silent in the middle one, where the
    # estimate is pure noise as loud as the signal.
    ref = noise((2, SR * 5), 0.1, 0)
    ref[:, 2 * SR : 3 * SR] = 0
    est = ref + noise((2, SR * 5), 0.001, 1)
    est[:, 2 * SR : 3 * SR] = noise((2, SR), 0.1, 2)
    assert chunked_sdr(ref, est, SR) == pytest.approx(20, abs=0.3)
    # The global SDR does count the bad window.
    assert sdr(ref, est) < 10


def test_chunked_is_the_median_of_windows():
    ref = noise((2, SR * 3), 0.1, 0)
    est = ref.clone()
    for i, db in enumerate((10, 30, 20)):
        sl = slice(i * SR, (i + 1) * SR)
        e = noise((2, SR), 1.0, 10 + i)
        e *= math.sqrt(float(ref[:, sl].square().sum()) / float(e.square().sum()) / 10 ** (db / 10))
        est[:, sl] += e
    assert chunked_sdr(ref, est, SR) == pytest.approx(20, abs=1e-6)


def test_chunked_drops_partial_window_and_handles_all_silent():
    ref = noise((2, SR * 2 + SR // 2), 0.1, 0)
    est = ref.clone()
    est[:, 2 * SR :] = 0  # only the dropped partial window is wrong
    assert chunked_sdr(ref, est, SR) > 100
    assert math.isnan(chunked_sdr(torch.zeros(2, SR * 2), noise((2, SR * 2), 0.1, 1), SR))
    # Shorter than one window: one window.
    short = noise((2, SR // 4), 0.1, 3)
    assert chunked_sdr(short, short * 0.5, SR) == pytest.approx(10 * math.log10(4), abs=1e-6)


def test_report_means_and_silent_stems():
    ref = {
        "drums": noise((2, SR * 4), 0.1, 0),
        "bass": noise((2, SR * 4), 0.1, 1),
        "vocals": torch.zeros(2, SR * 4),
    }
    est = {
        "drums": ref["drums"] + noise((2, SR * 4), 0.001, 2),  # ~20 dB
        "bass": 0.5 * ref["bass"],  # 6.02 dB
        "vocals": noise((2, SR * 4), 0.01, 3),
    }
    out = report(ref, est, SR)
    assert set(out["stems"]) == {"drums", "bass", "vocals"}
    assert math.isnan(out["stems"]["vocals"]["sdr"])
    expected = (out["stems"]["drums"]["sdr"] + out["stems"]["bass"]["sdr"]) / 2
    assert out["mean"]["sdr"] == pytest.approx(expected)
    assert out["mean"]["sdr"] == pytest.approx((20 + 6.0206) / 2, abs=0.1)
    assert not math.isnan(out["mean"]["chunked_sdr"])
    with pytest.raises(KeyError):
        report(ref, {"drums": est["drums"]}, SR)


# ---- loader -------------------------------------------------------------------


def write_wav(path, wav: np.ndarray, rate: int, fmt: str = "float") -> None:
    """Minimal WAV writer for the tests: float32 or 16-bit PCM."""
    channels = wav.shape[0]
    if fmt == "float":
        tag, bits, payload = 3, 32, wav.T.astype("<f4").tobytes()
    else:
        tag, bits = 1, 16
        payload = (np.clip(wav.T, -1, 1 - 2**-15) * 2**15).astype("<i2").tobytes()
    block = channels * bits // 8
    fmt_chunk = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt_chunk
    body += b"data" + struct.pack("<I", len(payload)) + payload
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


@pytest.mark.parametrize("fmt", ["float", "pcm16"])
def test_read_wav_round_trip(tmp_path, fmt):
    wav = (np.random.default_rng(0).standard_normal((2, 1000)) * 0.1).astype(np.float32)
    write_wav(tmp_path / "a.wav", wav, 44100, fmt)
    got, rate = read_wav(str(tmp_path / "a.wav"))
    assert rate == 44100 and got.shape == (2, 1000) and got.dtype == np.float32
    np.testing.assert_allclose(got, wav, atol=0 if fmt == "float" else 2**-15)


def test_read_wav_matches_soundfile(tmp_path):
    sf = pytest.importorskip("soundfile")
    wav = (np.random.default_rng(1).standard_normal((1000, 2)) * 0.1).astype(np.float32)
    for subtype in ("FLOAT", "PCM_24", "PCM_16"):
        path = tmp_path / f"{subtype}.wav"
        sf.write(path, wav, 44100, subtype=subtype)
        got, _ = read_wav(str(path))
        expected, _ = sf.read(path, dtype="float32", always_2d=True)
        np.testing.assert_allclose(got, expected.T, atol=1e-7)


def test_load_testset(tmp_path):
    stems = ("drums", "bass", "other", "vocals")
    rng = np.random.default_rng(2)
    parts = {s: (rng.standard_normal((2, 3000)) * 0.1).astype(np.float32) for s in stems}
    clip_dir = tmp_path / "clips" / "a-track"
    clip_dir.mkdir(parents=True)
    for base, sl in ((tmp_path, slice(None)), (clip_dir, slice(1000, 2000))):
        for s in stems:
            write_wav(base / f"{s}.wav", parts[s][:, sl], 44100)
        write_wav(base / "mixture.wav", sum(parts.values())[:, sl], 44100)
    manifest = {
        "samplerate": 44100,
        "channels": 2,
        "stems": list(stems),
        "clips": [{"name": "A - Track", "dir": "clips/a-track", "offset": 1000, "length": 1000}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    mix, got, rate = load_testset(str(tmp_path))
    assert rate == 44100 and mix.shape == (2, 3000) and list(got) == list(stems)
    assert torch.allclose(mix, sum(got.values()), atol=1e-6)

    mix, got, _ = load_testset(str(tmp_path), clip="A - Track")
    assert mix.shape == (2, 1000)
    assert torch.equal(got["bass"], torch.from_numpy(parts["bass"][:, 1000:2000]))
    with pytest.raises(KeyError):
        load_testset(str(tmp_path), clip="nope")


def test_load_fulltracks(tmp_path):
    stems = ("drums", "bass", "other", "vocals")
    rng = np.random.default_rng(3)
    entries = []
    for name, length in (("A - One", 1500), ("B - Two", 2500)):
        d = tmp_path / name[0].lower()
        d.mkdir()
        parts = {s: (rng.standard_normal((2, length)) * 0.1).astype(np.float32) for s in stems}
        for s in stems:
            write_wav(d / f"{s}.wav", parts[s], 44100, "pcm16")
        # 16-bit rounding of the sum, as in MUSDB18-HQ.
        write_wav(d / "mixture.wav", sum(parts.values()), 44100, "pcm16")
        entries.append({"name": name, "dir": d.name, "length": length})
    manifest = {"samplerate": 44100, "channels": 2, "stems": list(stems), "tracks": entries}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    songs = list(load_fulltracks(str(tmp_path)))
    assert [s[0] for s in songs] == ["A - One", "B - Two"]
    name, mix, got, rate = songs[1]
    assert rate == 44100 and mix.shape == (2, 2500) and list(got) == list(stems)
    assert torch.allclose(mix, sum(got.values()), atol=4 * 2**-15)

    (_, mix, got, _), = load_fulltracks(str(tmp_path), tracks=["a"], mixture="sum")
    assert mix.shape == (2, 1500) and torch.equal(mix, sum(got.values()))
    with pytest.raises(KeyError):
        list(load_fulltracks(str(tmp_path), tracks=["nope"]))
    with pytest.raises(ValueError):
        list(load_fulltracks(str(tmp_path), mixture="mp3"))
