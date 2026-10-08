"""Decoding and encoding, with ffmpeg in subprocesses.

Subprocesses rather than torchaudio: no dependency on torchaudio's shifting
I/O backends, and encoders in separate processes really do run side by side.
"""

from __future__ import annotations

import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator

import numpy as np
import torch


class DecodeError(RuntimeError):
    """ffmpeg couldn't read the input as audio."""


def decode(path: str, samplerate: int = 44100, channels: int = 2) -> torch.Tensor:
    """The first audio stream of `path` as a float32 [channels, length]
    tensor, remixed as demucs's AudioFile does: mono is copied to every
    channel and extra channels are dropped. ffmpeg's own `-ac` would upmix
    mono 3 dB quieter and fold surround channels in.

    MP3, WAV, FLAC and Ogg at `samplerate` are read in process by
    libsndfile, which for a 4-minute MP3 takes two thirds of the time of
    ffmpeg piping out 86 MB of samples, to within 3e-6 of the same samples.
    Anything else, or anything to resample, goes through ffmpeg."""
    frames = _read_soundfile(path, samplerate)
    if frames is None:
        frames = _read_ffmpeg(path, samplerate)
    if frames.size == 0:
        raise DecodeError("no audio")
    src = frames.shape[1]
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


class NotStreamable(Exception):
    """libsndfile can't read this file as `decode` would, so it has to be
    decoded whole, by `decode`. Raised before the first block if that's
    known up front, partway if a read fails, or after the last if the read
    came up short."""


def decode_blocks(
    path: str, samplerate: int = 44100, channels: int = 2, seconds: float = 2.0
) -> Iterator[torch.Tensor]:
    """`decode`, a block of `seconds` at a time, for `Separator.
    separate_stream`, so separating can start before decoding ends. Only
    what libsndfile reads as `decode` would (see `_read_soundfile`)."""
    import soundfile as sf

    try:
        raw = open(path, "rb")
        f = sf.SoundFile(reads := _Reads(raw))
    except Exception as e:
        raise NotStreamable(str(e)) from e
    with raw, f:
        if f.samplerate != samplerate or f.channels > 2:
            raise NotStreamable(f"{f.samplerate} Hz, {f.channels} channels")
        got, size = 0, int(seconds * samplerate)
        while True:
            # Not f.blocks(), which goes by the frame count in the header and
            # yields whole blocks even when a read comes up short, as one of
            # a truncated MP3 does: the rest of the block is left over from
            # the one before.
            try:
                block = f.read(size, dtype="float32", always_2d=True)
            except Exception as e:  # a read that fails partway: start again with decode
                raise NotStreamable(str(e)) from e
            if not len(block):
                break
            got += len(block)
            if f.channels == 1 and channels > 1:
                block = np.repeat(block, channels, axis=1)
            elif channels == 1 and f.channels > 1:
                block = block.mean(axis=1, keepdims=True)
            yield torch.from_numpy(block.T.copy())
        if not got:
            raise NotStreamable("no audio")
        if f.format == "MP3" and os.fstat(raw.fileno()).st_size - reads.end > 4096:
            raise NotStreamable("libsndfile stopped short of the end")


def _read_soundfile(path: str, samplerate: int) -> np.ndarray | None:
    """[length, channels] float32 frames, or None if libsndfile can't read
    the file at `samplerate` as it is, or would read it differently from
    ffmpeg."""
    try:
        import soundfile as sf

        with open(path, "rb") as raw, sf.SoundFile(reads := _Reads(raw)) as f:
            # libsndfile keeps Vorbis's channel order, left, centre, right,
            # where ffmpeg, and so demucs, puts right second.
            if f.samplerate != samplerate or f.channels > 2:
                return None
            frames = f.read(dtype="float32", always_2d=True)
            # An MP3's length is libsndfile's guess, from its Xing header or,
            # without one, from the first frame's bitrate, and it reads no
            # further: a VBR file with no header came back a quarter of its
            # length. A whole read reaches the end of the file, short of at
            # most a tag.
            if f.format == "MP3" and os.fstat(raw.fileno()).st_size - reads.end > 4096:
                return None
            # Nothing at all may be a WAV whose header never got its sizes
            # filled in, which ffmpeg reads to the end of the file.
            return frames if len(frames) else None
    except Exception:
        return None


class _Reads:
    """A file that remembers where its last read ended."""

    def __init__(self, f):
        self.f, self.end = f, 0

    def seek(self, *args):
        return self.f.seek(*args)

    def tell(self):
        return self.f.tell()

    def readinto(self, buf):
        n = self.f.readinto(buf)
        self.end = self.f.tell()
        return n


def _read_ffmpeg(path: str, samplerate: int) -> np.ndarray:
    import os
    import tempfile

    # To a file in memory rather than a pipe: for a 4-minute song that's
    # 86 MB Python no longer has to pull through a pipe, ~0.2 s faster.
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=shm) as out:
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", path,
            "-map", "0:a:0",
            "-ar", str(samplerate),
            "-c:a", "pcm_f32le", "-f", "wav", out.name,
        ]  # fmt: skip
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise DecodeError(proc.stderr.decode(errors="replace").strip()[-500:])
        with open(out.name, "rb") as f:
            src, data = _wav_samples(f.read())
    return data[: data.size // src * src].reshape(-1, src)


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


# LAME's quality preset (its `-q`). demucs's `--mp3` uses 2. At 192 kbps, 5
# measured the same SNR against the source (25.1 dB both) and encodes about
# 1.7x faster, which made it most of the difference end to end.
QUALITY = 5


def _to_int16(wav: torch.Tensor) -> np.ndarray:
    """Interleaved int16 samples. As demucs's i16_pcm does, give or take
    0.0003 dB: it clamps first and scales by 2**15 - 1."""
    return (wav * 2**15).clamp_(-(2**15), 2**15 - 1).short().T.contiguous().numpy()


def _lame(samplerate: int, channels: int, bitrate: int, quality: int):
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(channels)
    encoder.set_quality(quality)
    return encoder


def stored_gain(peak: float, gain: float = 1.0) -> float:
    """What a stem with this peak ends up scaled by, stored at `gain`: the
    gain itself, or less where `_rescale` turns it down further to keep it
    from clipping. Playing it back at 1 / this restores its level.

    A stem with a NaN in it, as from a float WAV that has one, reports
    `gain`: NaN isn't valid JSON, and a report carrying it never gets sent."""
    if math.isnan(peak):
        return gain
    return gain / max(1.01 * peak * gain, 1.0)


def mp3_bytes(
    wav: torch.Tensor,
    samplerate: int,
    bitrate: int = 192,
    quality: int = QUALITY,
    gain: float = 1.0,
) -> bytes:
    """[channels, length] float audio as CBR MP3, encoded in process with
    lameenc, as `demucs.audio.save_audio` does. lameenc releases the GIL, so
    stems encode in parallel on threads.

    `gain` scales it first: stored 6 dB down (0.5), a stem can peak at
    almost 2 before it needs `_rescale`, and loud songs' stems often do
    exceed 1. `stored_gain` says what it was scaled by in the end."""
    wav = _rescale(wav.float() * gain)
    encoder = _lame(samplerate, wav.shape[0], bitrate, quality)
    return bytes(encoder.encode(_to_int16(wav).tobytes()) + encoder.flush())


def encode_mp3(
    wav: torch.Tensor, path: str, samplerate: int, bitrate: int = 192, quality: int = QUALITY
) -> None:
    """`mp3_bytes`, written to `path`."""
    with open(path, "wb") as f:
        f.write(mp3_bytes(wav, samplerate, bitrate, quality))


class StreamingMP3:
    """Encodes stems while they're still being separated: pass `feed` as
    `Separator.separate`'s `sink`, then call `finish` with its result.

    The rescale that keeps a stem from clipping needs the stem's peak, which
    isn't known until the end. So each stem is encoded as it arrives as if it
    won't need one. A stem that turns out to need it stops there, and is
    encoded whole once the separation is done. No stem of 12 full MUSDB songs
    came close, but a loud commercial master's drums and bass both did.
    Either way the bytes are exactly `mp3_bytes`'s."""

    def __init__(
        self,
        names: Iterable[str],
        samplerate: int,
        channels: int,
        bitrate: int = 192,
        quality: int = QUALITY,
        gain: float = 1.0,
    ):
        import queue
        import threading

        self.samplerate, self.bitrate, self.quality = samplerate, bitrate, quality
        self.gain = gain
        # Each stem's stored_gain, once finish has run.
        self.gains: dict[str, float] = {}
        self._queues = {n: queue.Queue() for n in names}
        self._data: dict[str, bytearray] = {}
        self._peak = {n: 0.0 for n in self._queues}
        self._errors: list[BaseException] = []
        self._closed = False
        self._threads = [
            threading.Thread(target=self._run, args=(n, channels), daemon=True)
            for n in self._queues
        ]
        for t in self._threads:
            t.start()

    def feed(self, segments: dict[str, torch.Tensor]) -> None:
        for name, segment in segments.items():
            self._queues[name].put(segment)

    def _run(self, name: str, channels: int) -> None:
        segment = ...  # anything but None, until the end has been taken
        try:
            encoder = _lame(self.samplerate, channels, self.bitrate, self.quality)
            data = bytearray()
            while (segment := self._queues[name].get()) is not None:
                if self._needs_rescale(name):
                    continue  # this encoding will be thrown away; stop paying for it
                segment = segment.float() * self.gain
                self._peak[name] = max(self._peak[name], segment.abs().max().item())
                if not self._needs_rescale(name):
                    data += encoder.encode(_to_int16(segment).tobytes())
            if not self._needs_rescale(name):
                data += encoder.flush()
                self._data[name] = data
        except BaseException as e:  # surfaced by finish
            self._errors.append(e)
            # Drain to the end, freeing what's queued, so close's join
            # returns. If the flush failed, the end was already taken, and
            # waiting for another would hang it.
            while segment is not None:
                segment = self._queues[name].get()

    def _needs_rescale(self, name: str) -> bool:
        """Whether `_rescale` will scale this stem down, as far as it's been
        seen. Once true it stays true: the peak only grows."""
        return 1.01 * self._peak[name] > 1

    def close(self) -> None:
        """Stop the encoders without waiting for a result, as when the
        separation failed. Safe to call more than once."""
        if not self._closed:
            self._closed = True
            for q in self._queues.values():
                q.put(None)
        for t in self._threads:
            t.join()

    def finish(self, stems: dict[str, torch.Tensor]) -> dict[str, bytes]:
        """Each stem's MP3. `stems` is `separate`'s result, used to encode
        again any stem that turned out to need rescaling."""
        # Stems already known to clip start their real encoding now, beside
        # the encoders still finishing the rest. Loud masters often have one:
        # a commercial track's drums peaked at 1.64.
        def encode(name: str) -> bytes:
            return mp3_bytes(stems[name], self.samplerate, self.bitrate, self.quality, self.gain)

        with ThreadPoolExecutor(len(self._queues)) as pool:
            early = {n: pool.submit(encode, n) for n in self._queues if self._needs_rescale(n)}
            self.close()
            if self._errors:
                for f in early.values():
                    f.cancel()
                raise self._errors[0]
            # A stem whose peak crossed the line in its last stretch.
            late = {
                n: pool.submit(encode, n)
                for n in self._queues
                if self._needs_rescale(n) and n not in early
            }
            out = {n: bytes(d) for n, d in self._data.items() if not self._needs_rescale(n)}
            out.update({n: f.result() for n, f in {**early, **late}.items()})
        # A stem that stopped early stopped tracking its peak too.
        self.gains = {
            n: stored_gain(stems[n].abs().max().item() if n in early or n in late else
                           self._peak[n] / self.gain, self.gain)
            for n in self._queues
        }  # fmt: skip
        return out


def encode_mp3_ffmpeg(
    wav: torch.Tensor, path: str, samplerate: int, bitrate: int = 192, quality: int = QUALITY
) -> None:
    """The same with ffmpeg's LAME in a subprocess. It takes the floats
    without an int16 step, but pays for the pipe: ~1.5x slower than
    `encode_mp3` on Modal's CPUs."""
    wav = _rescale(wav.float())
    # A byte view of the interleaved samples, not `tobytes()`: one copy of a
    # 10-minute stem (~200 MB) per encoder rather than three.
    pcm = memoryview(wav.T.contiguous().numpy()).cast("B")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(samplerate), "-ac", str(wav.shape[0]), "-i", "-",
        "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", "-compression_level", str(quality),
        "-f", "mp3", path,
    ]  # fmt: skip
    proc = subprocess.run(cmd, input=pcm, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip()[-500:])


def encode_all(
    stems: dict[str, torch.Tensor],
    paths: dict[str, str],
    samplerate: int,
    bitrate: int = 192,
    quality: int = QUALITY,
    encoder=encode_mp3,
) -> None:
    """Encode every stem at once, one thread per stem."""
    names: Iterable[str] = list(stems)
    with ThreadPoolExecutor(max(1, len(stems))) as pool:
        futures = [
            pool.submit(encoder, stems[n], paths[n], samplerate, bitrate, quality)
            for n in names
        ]
        for f in futures:
            f.result()
