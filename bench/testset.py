"""Ground-truth test set for separation quality, built on Modal.

    modal run bench/testset.py            # build, skipping if already built
    modal run bench/testset.py --force    # rebuild

Source: the MUSDB18 7-second excerpts ("MUSDB18-7", Zenodo record 3270814,
the archive `musdb.DB(download=True)` fetches), test subset. Each excerpt is
a Native Instruments .stem.mp4 with five AAC streams: mixture, drums, bass,
other, vocals. They're decoded with ffmpeg.

The excerpts are concatenated, in sorted name order, until the song is at
least 240 s long. The mixture written here is the sum of the four decoded
stems, not the dataset's own mixture stream: the AAC-coded mixture differs
slightly from the sum of the AAC-coded stems, and with mixture = Σ stems the
ground truth is exact. (How far apart the two are is recorded per clip in the
manifest as `dataset_mix_vs_sum_sdr`.)

Layout, in the `stemsep-bench` Volume, which other functions mount at /data:

    /data/testset/manifest.json
    /data/testset/mixture.wav     float32 WAV, 44.1 kHz stereo
    /data/testset/mixture.mp3     192 kbps LAME, as a user would upload
    /data/testset/{drums,bass,other,vocals}.wav
    /data/testset/clips/<slug>/{mixture,drums,bass,other,vocals}.wav

Read it with `stemsep.quality.load_testset("/data/testset")`.
"""

from __future__ import annotations

import modal

VOLUME_NAME = "stemsep-bench"
MOUNT = "/data"
SUBDIR = "testset"

URL = "https://zenodo.org/records/3270814/files/MUSDB18-7-STEMS.zip?download=1"
MD5 = "dc2bdfac5b46ce742bd60009866758a5"
SAMPLERATE = 44100
CHANNELS = 2
STEMS = ("drums", "bass", "other", "vocals")
# Stream order inside a .stem.mp4.
STREAMS = ("mixture", *STEMS)
TARGET_S = 240.0
MP3_BITRATE = 192
# Peak the concatenated mixture may reach before everything is scaled down,
# so the MP3 encoder never clips and the WAV and MP3 stay comparable.
MAX_PEAK = 0.98

app = modal.App("stemsep-testset")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("numpy", "soundfile")
)


@app.function(image=image, volumes={MOUNT: volume}, timeout=30 * 60, cpu=4, memory=8192)
def prepare(force: bool = False) -> dict:
    import hashlib
    import json
    import os
    import re
    import shutil
    import subprocess
    import tempfile
    import time
    import urllib.request
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import soundfile as sf

    root = os.path.join(MOUNT, SUBDIR)
    manifest_path = os.path.join(root, "manifest.json")
    volume.reload()
    if os.path.exists(manifest_path) and not force:
        with open(manifest_path) as f:
            print(f"{manifest_path} exists; skipping (force=True rebuilds)")
            return json.load(f)

    work = tempfile.mkdtemp()
    archive = os.path.join(work, "musdb18-7.zip")
    t0 = time.time()
    print(f"downloading {URL}")
    md5 = hashlib.md5()
    with urllib.request.urlopen(URL) as resp, open(archive, "wb") as out:
        while chunk := resp.read(1 << 20):
            md5.update(chunk)
            out.write(chunk)
    if md5.hexdigest() != MD5:
        raise RuntimeError(f"md5 {md5.hexdigest()} != {MD5}")
    print(f"downloaded {os.path.getsize(archive) / 1e6:.0f} MB in {time.time() - t0:.0f} s")

    with zipfile.ZipFile(archive) as z:
        members = sorted(
            n
            for n in z.namelist()
            if n.endswith(".stem.mp4") and "test" in n.split("/")[:-1]
        )
        if not members:
            raise RuntimeError("no test/*.stem.mp4 in the archive")
        z.extractall(work, members)
    print(f"{len(members)} test excerpts")

    def decode(path: str, stream: int) -> np.ndarray:
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", path, "-map", f"0:a:{stream}",
            "-ac", str(CHANNELS), "-ar", str(SAMPLERATE), "-f", "f32le", "-",
        ]  # fmt: skip
        proc = subprocess.run(cmd, capture_output=True, check=True)
        return np.frombuffer(proc.stdout, np.float32).reshape(-1, CHANNELS).T

    def load(member: str) -> dict[str, np.ndarray]:
        path = os.path.join(work, member)
        streams = [decode(path, i) for i in range(len(STREAMS))]
        length = min(s.shape[-1] for s in streams)
        return {name: s[:, :length] for name, s in zip(STREAMS, streams)}

    def slug(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    def sdr(ref: np.ndarray, est: np.ndarray) -> float:
        ref, est = ref.astype(np.float64), est.astype(np.float64)
        num = np.square(ref).sum() + 1e-8
        den = np.square(ref - est).sum() + 1e-8
        return float(10 * np.log10(num / den))

    def rms_db(x: np.ndarray) -> float:
        return float(10 * np.log10(np.square(x.astype(np.float64)).mean() + 1e-20))

    # Decode in parallel; keep excerpts in sorted order until the target.
    clips = []
    total = 0
    with ThreadPoolExecutor(8) as pool:
        decoded = pool.map(load, members)
        for member, streams in zip(members, decoded):
            if total >= TARGET_S * SAMPLERATE:
                break
            name = os.path.basename(member)[: -len(".stem.mp4")]
            stems = {s: streams[s] for s in STEMS}
            mix = sum(stems.values())
            clips.append(
                {
                    "name": name,
                    "dir": f"clips/{slug(name)}",
                    "source_file": member,
                    "offset": total,
                    "length": mix.shape[-1],
                    "dataset_mix_vs_sum_sdr": sdr(streams["mixture"], mix),
                    "_stems": stems,
                }
            )
            total += mix.shape[-1]
    if len({c["dir"] for c in clips}) != len(clips):
        raise RuntimeError("clip slugs collide")

    song = {s: np.concatenate([c["_stems"][s] for c in clips], axis=1) for s in STEMS}
    mixture = sum(song.values())
    peak = float(np.abs(mixture).max())
    gain = min(1.0, MAX_PEAK / peak)
    if gain < 1:
        song = {s: w * gain for s, w in song.items()}
        mixture = mixture * gain
    print(f"song {total / SAMPLERATE:.1f} s from {len(clips)} clips; peak {peak:.3f}, gain {gain:.4f}")

    # The manifest is written last, so a build that dies halfway leaves no
    # manifest and the next run starts over.
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)

    def write(path: str, wav: np.ndarray) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, wav.T.astype(np.float32), SAMPLERATE, subtype="FLOAT")

    write(os.path.join(root, "mixture.wav"), mixture)
    for s in STEMS:
        write(os.path.join(root, f"{s}.wav"), song[s])
    for c in clips:
        sl = slice(c["offset"], c["offset"] + c["length"])
        base = os.path.join(root, c["dir"])
        write(os.path.join(base, "mixture.wav"), mixture[:, sl])
        for s in STEMS:
            write(os.path.join(base, f"{s}.wav"), song[s][:, sl])
        c["rms_db"] = {s: rms_db(song[s][:, sl]) for s in STEMS}
        c["offset_s"] = c["offset"] / SAMPLERATE
        c["duration_s"] = c["length"] / SAMPLERATE
        del c["_stems"]

    # MP3 of the song, encoded as stemsep.audio.encode_mp3 does.
    mp3 = os.path.join(root, "mixture.mp3")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", os.path.join(root, "mixture.wav"),
            "-c:a", "libmp3lame", "-b:a", f"{MP3_BITRATE}k", "-compression_level", "2",
            mp3,
        ],
        check=True,
    )  # fmt: skip
    # How the decoded MP3 lines up with the WAV: ffmpeg skips the encoder
    # delay using the LAME tag, so the lag should be 0.
    back = decode(mp3, 0)
    probe = slice(SAMPLERATE * 10, SAMPLERATE * 20)
    a, b = mixture[0, probe].astype(np.float64), back[0, :].astype(np.float64)
    lags = range(-4096, 4097)
    best = max(lags, key=lambda k: float(np.dot(a, b[probe.start + k : probe.stop + k])))
    n = min(back.shape[-1], mixture.shape[-1])
    mp3_info = {
        "bitrate_kbps": MP3_BITRATE,
        "encoder": "ffmpeg libmp3lame -compression_level 2 (LAME -q 2), from mixture.wav",
        "decoded_length": int(back.shape[-1]),
        "lag_samples": int(best),
        "sdr_vs_wav": sdr(mixture[:, :n], back[:, :n]),
    }

    manifest = {
        "version": 1,
        "name": "musdb18-7-test-concat",
        "samplerate": SAMPLERATE,
        "channels": CHANNELS,
        "stems": list(STEMS),
        "length": int(total),
        "duration_s": total / SAMPLERATE,
        "mixture": "sum of the four stems (the dataset's own mixture stream is not used)",
        "gain": gain,
        "files": {
            "mixture": "mixture.wav",
            "mixture_mp3": "mixture.mp3",
            **{s: f"{s}.wav" for s in STEMS},
        },
        "wav_format": "float32 WAV (IEEE float)",
        "mp3": mp3_info,
        "clips": clips,
        "source": {
            "dataset": "MUSDB18-7: the SiSEC18-MUS 7 s excerpts of MUSDB18",
            "subset": "test",
            "order": "sorted by file name, first clips totalling >= 240 s",
            "record": "https://zenodo.org/records/3270814",
            "url": URL,
            "md5": MD5,
            "decoder": "ffmpeg (native AAC), streams 0:a:1..4 = drums, bass, other, vocals",
        },
        "license": (
            "The Zenodo record is CC BY 4.0. The underlying MUSDB18 tracks keep their own "
            "licenses (mostly CC BY-NC-SA or 'restricted to research use'; see "
            "https://sigsep.github.io/datasets/musdb.html). Research and benchmarking "
            "only; don't redistribute."
        ),
    }
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    volume.commit()
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {root} in {time.time() - t0:.0f} s")
    return manifest


@app.local_entrypoint()
def main(force: bool = False):
    manifest = prepare.remote(force=force)
    print(
        f"{MOUNT}/{SUBDIR}: {manifest['duration_s']:.1f} s, {len(manifest['clips'])} clips, "
        f"{manifest['samplerate']} Hz; mp3 lag {manifest['mp3']['lag_samples']} samples, "
        f"mp3 vs wav SDR {manifest['mp3']['sdr_vs_wav']:.1f} dB"
    )
