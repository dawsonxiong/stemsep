"""Full-length ground-truth songs for separation quality, built on Modal.

    modal run bench/fulltracks.py            # build, skipping if already built
    modal run bench/fulltracks.py --force    # rebuild

Why another set: bench/testset.py's song is 36 seven-second excerpts glued
end to end, with an abrupt cut every 6.8 s. Real songs don't cut like that,
and where the cuts fall relative to the chunk grid moves SDR by tenths of a
dB, so that set can't tell chunk-grid luck from a real quality change.

Source: MUSDB18-HQ (Zenodo record 3338373, `musdb18hq.zip`, 22.7 GB, open
access), the uncompressed WAV release of MUSDB18. Only the members needed are
fetched, with HTTP range requests: a few for the zip's central directory,
then one ranged GET per WAV, inflated as it streams in and checked against
the zip's CRC-32. Tracks are the test split in sorted name order, the first
N_TRACKS no longer than MAX_S.

Layout, in the `stemsep-bench` Volume, which other functions mount at /data:

    /data/fulltracks/manifest.json
    /data/fulltracks/<slug>/{mixture,drums,bass,other,vocals}.wav

The WAVs are byte-for-byte the dataset's: 16-bit PCM, 44.1 kHz stereo.
Read them with `stemsep.quality.load_fulltracks("/data/fulltracks")`.
"""

from __future__ import annotations

import modal

VOLUME_NAME = "stemsep-bench"
MOUNT = "/data"
SUBDIR = "fulltracks"

RECORD = "https://zenodo.org/records/3338373"
URL = f"{RECORD}/files/musdb18hq.zip?download=1"
SAMPLERATE = 44100
CHANNELS = 2
STEMS = ("drums", "bass", "other", "vocals")
FILES = ("mixture", *STEMS)
N_TRACKS = 12
MAX_S = 360.0
WORKERS = 6

app = modal.App("stemsep-fulltracks")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


@app.function(image=image, volumes={MOUNT: volume}, timeout=60 * 60, cpu=4, memory=8192)
def prepare(force: bool = False) -> dict:
    import io
    import json
    import os
    import re
    import time
    import urllib.error
    import urllib.request
    import wave
    import zipfile
    import zlib
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    root = os.path.join(MOUNT, SUBDIR)
    manifest_path = os.path.join(root, "manifest.json")
    volume.reload()
    if os.path.exists(manifest_path) and not force:
        with open(manifest_path) as f:
            print(f"{manifest_path} exists; skipping (force=True rebuilds)")
            return json.load(f)

    def get(start: int, end: int):
        """A response for bytes [start, end], retrying Zenodo's rate limit
        and transient errors."""
        for attempt in range(8):
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            try:
                resp = urllib.request.urlopen(req, timeout=60)
            except OSError as e:  # URLError, HTTPError, resets, timeouts
                code = getattr(e, "code", None)
                if code is not None and code != 429 and code < 500:
                    raise
                retry = getattr(e, "headers", None) and e.headers.get("Retry-After")
                wait = float(retry) if retry else min(60.0, 2.0**attempt)
                print(f"range {start}-{end}: {e!r}; retrying in {wait:.0f} s")
                time.sleep(wait)
                continue
            if resp.status != 206:
                raise RuntimeError(f"expected 206 Partial Content, got {resp.status}")
            return resp
        raise RuntimeError(f"range {start}-{end}: gave up")

    class Remote(io.RawIOBase):
        """The zip as a seekable file, so `zipfile` can read its central
        directory (a few ranged reads at the end of the file)."""

        def __init__(self):
            with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD")) as r:
                self.size = int(r.headers["Content-Length"])
            self.pos = 0

        def readable(self):
            return True

        def seekable(self):
            return True

        def tell(self):
            return self.pos

        def seek(self, offset, whence=0):
            self.pos = (0, self.pos, self.size)[whence] + offset
            return self.pos

        def readinto(self, buf):
            if self.pos >= self.size:
                return 0
            end = min(self.pos + len(buf), self.size) - 1
            with get(self.pos, end) as resp:
                data = resp.read()
            buf[: len(data)] = data
            self.pos += len(data)
            return len(data)

    def fetch(info: zipfile.ZipInfo, dest: str) -> None:
        """Download and inflate one member to `dest` in a single ranged GET.

        The local header's name and extra fields can differ in length from
        the central directory's, so the range runs 64 KiB past the data to
        be safe; the connection is dropped once the data is read."""
        if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
            return
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise RuntimeError(f"{info.filename}: compression {info.compress_type}")
        start = info.header_offset
        for attempt in range(4):
            try:
                with get(start, start + 30 + (1 << 16) + info.compress_size) as resp:
                    head = resp.read(30)
                    if head[:4] != b"PK\x03\x04":
                        raise RuntimeError(f"{info.filename}: bad local header")
                    n, m = int.from_bytes(head[26:28], "little"), int.from_bytes(head[28:30], "little")
                    resp.read(n + m)
                    inflate = zlib.decompressobj(-15) if info.compress_type else None
                    crc, left = 0, info.compress_size
                    with open(dest + ".part", "wb") as out:
                        while left:
                            block = resp.read(min(left, 1 << 22))
                            if not block:
                                raise RuntimeError(f"{info.filename}: truncated")
                            left -= len(block)
                            if inflate:
                                block = inflate.decompress(block)
                            crc = zlib.crc32(block, crc)
                            out.write(block)
                        if inflate:
                            tail = inflate.flush()
                            crc = zlib.crc32(tail, crc)
                            out.write(tail)
                if crc != info.CRC or os.path.getsize(dest + ".part") != info.file_size:
                    raise RuntimeError(f"{info.filename}: CRC or size mismatch")
                os.replace(dest + ".part", dest)
                return
            except (OSError, RuntimeError, zlib.error) as e:
                print(f"{info.filename}: {e!r}; attempt {attempt + 1}")
                if attempt == 3:
                    raise

    def slug(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    def read(path: str) -> np.ndarray:
        with wave.open(path) as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLERATE, CHANNELS, 2):
                raise RuntimeError(f"{path}: not 16-bit {SAMPLERATE} Hz stereo")
            data = w.readframes(w.getnframes())
        return np.frombuffer(data, "<i2").reshape(-1, CHANNELS).T.astype(np.float64) / 2**15

    def sdr(ref: np.ndarray, est: np.ndarray) -> float:
        num = np.square(ref).sum() + 1e-8
        den = np.square(ref - est).sum() + 1e-8
        return float(10 * np.log10(num / den))

    def rms_db(x: np.ndarray) -> float:
        return float(10 * np.log10(np.square(x).mean() + 1e-20))

    t0 = time.time()
    remote = Remote()
    with zipfile.ZipFile(io.BufferedReader(remote, 1 << 20)) as z:
        members = {i.filename: i for i in z.infolist()}
    names = sorted({f.split("/")[1] for f in members if f.startswith("test/") and f.count("/") == 2})
    print(f"{len(members)} zip members, {len(names)} test tracks, {remote.size / 1e9:.1f} GB zip")

    chosen, skipped = [], []
    for name in names:
        info = members[f"test/{name}/mixture.wav"]
        # 44-byte header, 4 bytes a frame; close enough to filter on.
        seconds = (info.file_size - 44) / (4 * SAMPLERATE)
        (chosen if seconds <= MAX_S else skipped).append(name)
        if len(chosen) == N_TRACKS:
            break
    if len({slug(n) for n in chosen}) != len(chosen):
        raise RuntimeError("track slugs collide")

    jobs = []
    for name in chosen:
        os.makedirs(os.path.join(root, slug(name)), exist_ok=True)
        for f in FILES:
            jobs.append((members[f"test/{name}/{f}.wav"], os.path.join(root, slug(name), f"{f}.wav")))
    fetched = sum(info.compress_size for info, _ in jobs)
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda job: fetch(*job), jobs))
    print(f"fetched {len(jobs)} files, {fetched / 1e9:.2f} GB compressed, in {time.time() - t0:.0f} s")

    tracks = []
    for name in chosen:
        base = os.path.join(root, slug(name))
        audio = {f: read(os.path.join(base, f"{f}.wav")) for f in FILES}
        lengths = {a.shape[-1] for a in audio.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"{name}: stems differ in length {lengths}")
        length = lengths.pop()
        tracks.append(
            {
                "name": name,
                "dir": slug(name),
                "length": length,
                "duration_s": length / SAMPLERATE,
                # MUSDB18-HQ's mixture is the sum of its stems up to 16-bit
                # rounding; this says how close.
                "mixture_vs_sum_sdr": sdr(audio["mixture"], sum(audio[s] for s in STEMS)),
                "rms_db": {f: rms_db(a) for f, a in audio.items()},
                "crc32": {f: f"{members[f'test/{name}/{f}.wav'].CRC:08x}" for f in FILES},
            }
        )
        print(f"{name}: {length / SAMPLERATE:.1f} s, mix vs sum {tracks[-1]['mixture_vs_sum_sdr']:.1f} dB")

    manifest = {
        "version": 1,
        "name": "musdb18hq-test-full",
        "samplerate": SAMPLERATE,
        "channels": CHANNELS,
        "stems": list(STEMS),
        "files": {f: f"{f}.wav" for f in FILES},
        "wav_format": "16-bit PCM WAV, byte-identical to the dataset's",
        "mixture": "the dataset's own mixture.wav",
        "duration_s": sum(t["duration_s"] for t in tracks),
        "tracks": tracks,
        "source": {
            "dataset": "MUSDB18-HQ (uncompressed WAV release of MUSDB18)",
            "subset": "test",
            "selection": f"sorted by name, first {N_TRACKS} with a mixture of at most {MAX_S:.0f} s",
            "skipped_too_long": skipped,
            "record": RECORD,
            "url": URL,
            "zip_bytes": remote.size,
            "fetch": "HTTP range requests per zip member; CRC-32 checked against the zip",
        },
        "license": (
            "Zenodo lists MUSDB18-HQ as open access under a non-commercial licence; the "
            "underlying tracks keep their own licences (see "
            "https://sigsep.github.io/datasets/musdb.html). Research and benchmarking "
            "only; don't redistribute."
        ),
    }
    # Written last, so a build that dies halfway leaves no manifest; the next
    # run resumes, skipping files already at their full size.
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()
    print(f"wrote {root} in {time.time() - t0:.0f} s")
    return manifest


@app.local_entrypoint()
def main(force: bool = False):
    manifest = prepare.remote(force=force)
    print(
        f"{MOUNT}/{SUBDIR}: {len(manifest['tracks'])} tracks, "
        f"{manifest['duration_s'] / 60:.1f} min, {manifest['samplerate']} Hz"
    )
    for t in manifest["tracks"]:
        print(f"  {t['name']:<50} {t['duration_s']:6.1f} s  mix vs sum {t['mixture_vs_sum_sdr']:.1f} dB")
