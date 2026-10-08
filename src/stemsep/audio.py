"""Decoding and encoding, with ffmpeg in subprocesses.

Subprocesses rather than torchaudio: no dependency on torchaudio's shifting
I/O backends, and encoders in separate processes really do run side by side.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import torch


class DecodeError(RuntimeError):
    """ffmpeg couldn't read the input as audio."""


def decode(path: str, samplerate: int = 44100, channels: int = 2) -> torch.Tensor:
    """The first audio stream of `path` as a float32 [channels, length]
    tensor, resampled by ffmpeg and remixed as demucs's AudioFile does: mono
    is copied to every channel and extra channels are dropped. ffmpeg's own
    `-ac` would upmix mono 3 dB quieter and fold surround channels in."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-map", "0:a:0",
        "-ar", str(samplerate),
        "-c:a", "pcm_f32le", "-f", "wav", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise DecodeError(proc.stderr.decode(errors="replace").strip()[-500:])
    src, data = _wav_samples(proc.stdout)
    frames = data[: data.size // src * src].reshape(-1, src)
    if frames.size == 0:
        raise DecodeError("no audio")
    if src == channels:
        pass
    elif channels == 1:
        frames = frames.mean(axis=1, keepdims=True)
    elif src == 1:
        frames = np.repeat(frames, channels, axis=1)
    elif src > channels:
        frames = frames[:, :channels]
    else:
        raise DecodeError(f"can't make {channels} channels from {src}")
    return torch.from_numpy(frames.T.copy())


def _wav_samples(data: bytes) -> tuple[int, np.ndarray]:
    """(channels, interleaved float32 samples) from ffmpeg's WAV on a pipe.
    WAV rather than raw f32le to learn the channel count; the size fields are
    placeholders, as ffmpeg can't seek back on a pipe to fill them in, so the
    samples run to the end of the output."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise DecodeError("no audio")
    pos, channels = 12, 0
    while pos + 8 <= len(data):
        tag, size = data[pos : pos + 4], int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = pos + 8
        if tag == b"fmt ":
            channels = int.from_bytes(data[body + 2 : body + 4], "little")
        elif tag == b"data" and channels:
            return channels, np.frombuffer(data, "<f4", (len(data) - body) // 4, body)
        pos = body + size + (size & 1)
    raise DecodeError("no audio")


def _rescale(wav: torch.Tensor) -> torch.Tensor:
    """demucs's clip="rescale": scale the whole stem down if it would clip."""
    peak = wav.abs().max().item()
    return wav / max(1.01 * peak, 1.0)


def encode_mp3(wav: torch.Tensor, path: str, samplerate: int, bitrate: int = 192) -> None:
    """Encode [channels, length] float audio to MP3 with ffmpeg's LAME.

    `-compression_level 2` is LAME's `-q 2`, the preset demucs's `--mp3`
    uses. LAME takes the floats as they are, without demucs's int16 step."""
    wav = _rescale(wav.float())
    # A byte view of the interleaved samples, not `tobytes()`: one copy of a
    # 10-minute stem (~200 MB) per encoder rather than three.
    pcm = memoryview(wav.T.contiguous().numpy()).cast("B")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(samplerate), "-ac", str(wav.shape[0]), "-i", "-",
        "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", "-compression_level", "2",
        "-f", "mp3", path,
    ]  # fmt: skip
    proc = subprocess.run(cmd, input=pcm, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip()[-500:])


def encode_mp3_lameenc(
    wav: torch.Tensor, path: str, samplerate: int, bitrate: int = 192
) -> None:
    """Encode as `demucs.audio.save_audio(..., bitrate, preset=2)` does, in
    process with lameenc. Kept to compare against `encode_mp3`."""
    import lameenc

    wav = _rescale(wav.float())
    pcm = (wav.clamp(-1, 1) * (2**15 - 1)).short().T.contiguous().numpy()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(wav.shape[0])
    encoder.set_quality(2)
    data = encoder.encode(pcm.tobytes()) + encoder.flush()
    with open(path, "wb") as f:
        f.write(data)


def encode_all(
    stems: dict[str, torch.Tensor],
    paths: dict[str, str],
    samplerate: int,
    bitrate: int = 192,
    encoder=encode_mp3,
) -> None:
    """Encode every stem at once, one thread (and with `encode_mp3`, one
    ffmpeg process) per stem."""
    names: Iterable[str] = list(stems)
    with ThreadPoolExecutor(max(1, len(stems))) as pool:
        futures = [
            pool.submit(encoder, stems[n], paths[n], samplerate, bitrate) for n in names
        ]
        for f in futures:
            f.result()
