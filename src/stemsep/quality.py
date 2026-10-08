"""Separation quality: SDR metrics and the ground-truth test set.

Two SDRs, both plain energy ratios with no BSS Eval distortion filter:

- `sdr` is the global SDR of the MDX challenge and of the Demucs papers
  ("nSDR"/"uSDR"): signal energy over error energy, summed over every channel
  and sample of the song.
- `chunked_sdr` is the median of the same ratio over 1 s windows, skipping
  windows where the reference is silent. It approximates museval's
  median-over-frames "cSDR" (SiSEC 2018), without museval's 512-tap
  projection filter, so it reads a little lower than museval would.

`load_testset` reads the set that `bench/testset.py` writes to the
`stemsep-bench` Modal Volume (mounted at /data, so the set is at
/data/testset). It needs only numpy and torch, plus ffmpeg when the MP3
mixture is asked for. `load_fulltracks` reads the whole MUSDB18-HQ songs
that `bench/fulltracks.py` writes to /data/fulltracks.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Iterator, Mapping
from typing import Literal

import numpy as np
import torch

TESTSET_ROOT = "/data/testset"
FULLTRACKS_ROOT = "/data/fulltracks"
STEMS = ("drums", "bass", "other", "vocals")

ArrayLike = torch.Tensor | np.ndarray


# ---- metrics ----------------------------------------------------------------


def _pair(reference: ArrayLike, estimate: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
    ref = torch.as_tensor(reference).detach().to("cpu", torch.float64)
    est = torch.as_tensor(estimate).detach().to("cpu", torch.float64)
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch: reference {list(ref.shape)}, estimate {list(est.shape)}")
    if ref.dim() == 1:
        ref, est = ref[None], est[None]
    if ref.dim() != 2:
        raise ValueError(f"expected [channels, length] or [length], got {list(ref.shape)}")
    return ref, est


def sdr(reference: ArrayLike, estimate: ArrayLike, eps: float = 1e-8) -> float:
    """Global SDR in dB over all channels and samples:
    10 log10((Σ ref² + eps) / (Σ (ref - est)² + eps)).

    A silent reference gives a large negative number for any nonzero
    estimate; `report` turns that case into NaN."""
    ref, est = _pair(reference, estimate)
    num = ref.square().sum() + eps
    den = (ref - est).square().sum() + eps
    return float(10 * torch.log10(num / den))


def _window_energies(
    ref: torch.Tensor, est: torch.Tensor, win: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-window Σ ref², Σ err² and mean ref² (per sample per channel).

    Whole windows only, as museval does; a signal shorter than one window is
    one window."""
    length = ref.shape[-1]
    n = max(1, length // win)
    win = min(win, length)
    ref = ref[:, : n * win].reshape(ref.shape[0], n, win)
    err = est[:, : n * win].reshape(est.shape[0], n, win) - ref
    sig = ref.square().sum(dim=(0, 2))
    noise = err.square().sum(dim=(0, 2))
    return sig, noise, sig / (ref.shape[0] * win)


def chunked_sdr(
    reference: ArrayLike,
    estimate: ArrayLike,
    samplerate: int,
    window_s: float = 1.0,
    eps: float = 1e-8,
    silence_db: float = -80.0,
) -> float:
    """Median over `window_s` windows of the per-window SDR, skipping
    windows whose reference mean power is below `silence_db` dBFS (museval
    skips all-zero reference frames; a threshold also catches codec noise
    floors). NaN when every window is silent."""
    ref, est = _pair(reference, estimate)
    win = max(1, round(window_s * samplerate))
    sig, noise, power = _window_energies(ref, est, win)
    active = power > 10 ** (silence_db / 10)
    if not bool(active.any()):
        return math.nan
    scores = 10 * torch.log10((sig[active] + eps) / (noise[active] + eps))
    # torch.median takes the lower middle on even counts; museval uses
    # np.nanmedian, which averages the two.
    return float(np.median(scores.numpy()))


def is_silent(reference: ArrayLike, silence_db: float = -80.0) -> bool:
    """Whether the whole reference is below `silence_db` dBFS mean power."""
    ref = torch.as_tensor(reference).detach().to("cpu", torch.float64)
    return float(ref.square().mean()) <= 10 ** (silence_db / 10)


def report(
    references: Mapping[str, ArrayLike],
    estimates: Mapping[str, ArrayLike],
    samplerate: int,
    window_s: float = 1.0,
    silence_db: float = -80.0,
) -> dict:
    """SDR and chunked SDR per stem, and their means over stems.

    Returns {"stems": {name: {"sdr", "chunked_sdr"}}, "mean": {"sdr",
    "chunked_sdr"}}. A stem whose reference is silent throughout (a
    vocals-free excerpt, say) has no meaningful SDR: it gets NaN for both and
    is left out of the means."""
    missing = [name for name in references if name not in estimates]
    if missing:
        raise KeyError(f"estimates missing stems {missing}")
    stems: dict[str, dict[str, float]] = {}
    for name, ref in references.items():
        est = estimates[name]
        if is_silent(ref, silence_db):
            _pair(ref, est)  # still check shapes
            stems[name] = {"sdr": math.nan, "chunked_sdr": math.nan}
            continue
        stems[name] = {
            "sdr": sdr(ref, est),
            "chunked_sdr": chunked_sdr(ref, est, samplerate, window_s, silence_db=silence_db),
        }
    mean = {key: _nanmean([s[key] for s in stems.values()]) for key in ("sdr", "chunked_sdr")}
    return {"stems": stems, "mean": mean}


def _nanmean(values: list[float]) -> float:
    kept = [v for v in values if not math.isnan(v)]
    return sum(kept) / len(kept) if kept else math.nan


# ---- test set -----------------------------------------------------------------

_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_IEEE_FLOAT = 3
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """A PCM (16/24/32-bit) or IEEE float (32/64-bit) WAV file as a float32
    [channels, length] array and its sample rate. Enough of RIFF to read
    what libsndfile writes, without depending on it."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    fmt = None
    payload = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = pos + 8
        if cid == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[body : body + 16])
            if tag == _WAVE_FORMAT_EXTENSIBLE:
                (tag,) = struct.unpack("<H", data[body + 24 : body + 26])
            fmt = (tag, channels, rate, bits)
        elif cid == b"data":
            payload = data[body : min(body + size, len(data))]
            break
        pos = body + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, channels, rate, bits = fmt
    frame = channels * bits // 8
    payload = payload[: len(payload) - len(payload) % frame]

    if tag == _WAVE_FORMAT_IEEE_FLOAT and bits in (32, 64):
        samples = np.frombuffer(payload, dtype="<f4" if bits == 32 else "<f8").astype(np.float32)
    elif tag == _WAVE_FORMAT_PCM and bits == 16:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 2**15
    elif tag == _WAVE_FORMAT_PCM and bits == 32:
        samples = (np.frombuffer(payload, dtype="<i4") / 2**31).astype(np.float32)
    elif tag == _WAVE_FORMAT_PCM and bits == 24:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        ints = np.where(ints >= 2**23, ints - 2**24, ints)
        samples = (ints / 2**23).astype(np.float32)
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag}, {bits} bits")
    return np.ascontiguousarray(samples.reshape(-1, channels).T), rate


def testset_manifest(root: str = TESTSET_ROOT) -> dict:
    """The test set's manifest.json."""
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def load_testset(
    root: str = TESTSET_ROOT,
    *,
    clip: str | None = None,
    mixture: Literal["wav", "mp3"] = "wav",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """(mixture [2, L], {stem: [2, L]}, samplerate) from the test set at
    `root`, all float32 on the CPU.

    By default this is the whole concatenated song; `clip` picks one excerpt
    by its manifest `name` or `dir` instead. The WAV mixture is exactly the
    sum of the four stems. `mixture="mp3"` decodes the 192 kbps MP3 of the
    song (what a user would upload) with ffmpeg, trimmed or zero-padded to
    the stems' length; it only exists for the whole song.
    """
    manifest = testset_manifest(root)
    if clip is None:
        base = root
    else:
        match = [c for c in manifest["clips"] if clip in (c["name"], c["dir"])]
        if not match:
            raise KeyError(f"no clip {clip!r} in {root}")
        base = os.path.join(root, match[0]["dir"])

    stems = {}
    rate = manifest["samplerate"]
    for name in manifest["stems"]:
        wav, sr = read_wav(os.path.join(base, f"{name}.wav"))
        if sr != rate:
            raise ValueError(f"{name}.wav is {sr} Hz, manifest says {rate}")
        stems[name] = torch.from_numpy(wav)
    length = next(iter(stems.values())).shape[-1]

    if mixture == "wav":
        mix = torch.from_numpy(read_wav(os.path.join(base, "mixture.wav"))[0])
    elif mixture == "mp3":
        if clip is not None:
            raise ValueError("the MP3 mixture only exists for the whole song")
        from .audio import decode

        mix = decode(os.path.join(root, "mixture.mp3"), rate, manifest["channels"])
        if mix.shape[-1] >= length:
            mix = mix[:, :length]
        else:
            mix = torch.nn.functional.pad(mix, (0, length - mix.shape[-1]))
    else:
        raise ValueError(f"mixture must be 'wav' or 'mp3', not {mixture!r}")
    return mix, stems, rate


def fulltracks_manifest(root: str = FULLTRACKS_ROOT) -> dict:
    """The full-track set's manifest.json."""
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def load_fulltracks(
    root: str = FULLTRACKS_ROOT,
    *,
    tracks: list[str] | None = None,
    mixture: Literal["dataset", "sum"] = "dataset",
) -> Iterator[tuple[str, torch.Tensor, dict[str, torch.Tensor], int]]:
    """(name, mixture [2, L], {stem: [2, L]}, samplerate) for each song of
    the full-track set at `root`, in manifest order, all float32 on the CPU.

    A generator, one song at a time: the twelve songs hold about 1.4 GB as
    float32. `tracks` picks songs by manifest `name` or `dir`. The default
    mixture is the dataset's own, as MUSDB evaluations use; in MUSDB18-HQ it
    equals the sum of the stems up to 16-bit rounding (the manifest's
    `mixture_vs_sum_sdr`). `mixture="sum"` uses the exact sum instead.
    """
    if mixture not in ("dataset", "sum"):
        raise ValueError(f"mixture must be 'dataset' or 'sum', not {mixture!r}")
    manifest = fulltracks_manifest(root)
    rate = manifest["samplerate"]
    entries = manifest["tracks"]
    if tracks is not None:
        known = {key for t in entries for key in (t["name"], t["dir"])}
        unknown = [t for t in tracks if t not in known]
        if unknown:
            raise KeyError(f"no tracks {unknown} in {root}")
        entries = [t for t in entries if t["name"] in tracks or t["dir"] in tracks]
    for entry in entries:
        base = os.path.join(root, entry["dir"])
        wavs = {}
        for name in ("mixture", *manifest["stems"]):
            wav, sr = read_wav(os.path.join(base, f"{name}.wav"))
            if sr != rate:
                raise ValueError(f"{entry['dir']}/{name}.wav is {sr} Hz, manifest says {rate}")
            wavs[name] = torch.from_numpy(wav)
        mix = wavs.pop("mixture")
        if mixture == "sum":
            mix = sum(wavs.values())
        yield entry["name"], mix, wavs, rate


__all__ = [
    "FULLTRACKS_ROOT",
    "STEMS",
    "TESTSET_ROOT",
    "chunked_sdr",
    "fulltracks_manifest",
    "is_silent",
    "load_fulltracks",
    "load_testset",
    "read_wav",
    "report",
    "sdr",
    "testset_manifest",
]
