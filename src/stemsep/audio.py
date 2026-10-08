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
    tensor, resampled and remixed by ffmpeg as demucs's AudioFile does."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-map", "0:a:0",
        "-ac", str(channels),
        "-ar", str(samplerate),
        "-f", "f32le", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise DecodeError(proc.stderr.decode(errors="replace").strip()[-500:])
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size == 0:
        raise DecodeError("no audio")
    return torch.from_numpy(data.reshape(-1, channels).T.copy())


def _rescale(wav: torch.Tensor) -> torch.Tensor:
    """demucs's clip="rescale": scale the whole stem down if it would clip."""
    peak = wav.abs().max().item()
    return wav / max(1.01 * peak, 1.0)


def encode_mp3(wav: torch.Tensor, path: str, samplerate: int, bitrate: int = 192) -> None:
    """Encode [channels, length] float audio to MP3 with ffmpeg's LAME.

    `-compression_level 2` is LAME's `-q 2`, the preset demucs's `--mp3`
    uses."""
    wav = _rescale(wav.float())
    pcm = wav.T.contiguous().numpy().astype(np.float32).tobytes()
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(samplerate), "-ac", str(wav.shape[0]), "-i", "-",
        "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", "-compression_level", "2",
        path,
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
    with ThreadPoolExecutor(len(stems)) as pool:
        futures = [
            pool.submit(encoder, stems[n], paths[n], samplerate, bitrate) for n in names
        ]
        for f in futures:
            f.result()
