"""lameenc's LAME against Debian's and against LAME 3.100 built with -O3,
x86-64-v3 and fast-math, through ctypes: one stem, 192 kbps, -q 5.

    modal run bench/lamebuild.py
"""

import json, time
from pathlib import Path
import modal
from images import _base, volume
app = modal.App("stemsep-lamebuild")

LAME_URL = "https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz"
def build(flags, prefix):
    return (f"cd /tmp && rm -rf lame-3.100 && tar xzf lame.tgz && cd lame-3.100 && "
            f"CFLAGS='{flags}' ./configure --prefix={prefix} --disable-frontend --disable-static --enable-nasm=no -q "
            f"&& make -j8 -s && make install -s")
img = (_base.apt_install("build-essential", "curl", "libmp3lame0")
       .run_commands(f"curl -sL -o /tmp/lame.tgz {LAME_URL}",
                     build("-O2", "/opt/lame-o2"),
                     build("-O3 -march=x86-64-v3 -mtune=generic", "/opt/lame-v3"),
                     build("-O3 -march=x86-64-v3 -ffast-math -funroll-loops", "/opt/lame-fast"))
       .add_local_python_source("stemsep", "images"))

@app.function(image=img, volumes={"/data": volume}, cpu=4.0, timeout=1200)
def run() -> dict:
    import ctypes, numpy as np, torch
    from stemsep.audio import _to_int16, mp3_bytes
    from stemsep.quality import load_testset
    _, stems, sr = load_testset("/data/testset")
    wav = stems["drums"]
    pcm = _to_int16(wav)
    n = pcm.shape[0]
    out = {}
    def ctypes_encode(path):
        L = ctypes.CDLL(path)
        L.lame_init.restype = ctypes.c_void_p
        g = ctypes.c_void_p(L.lame_init())
        L.lame_set_in_samplerate(g, sr); L.lame_set_num_channels(g, 2)
        L.lame_set_brate(g, 192); L.lame_set_quality(g, 5); L.lame_set_bWriteVbrTag(g, 0)
        assert L.lame_init_params(g) == 0
        buf = ctypes.create_string_buffer(int(1.25 * n + 7200))
        data = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        k = L.lame_encode_buffer_interleaved(g, data, n, buf, len(buf))
        tail = ctypes.create_string_buffer(7200)
        k2 = L.lame_encode_flush(g, tail, len(tail))
        L.lame_close(g)
        return buf.raw[:k] + tail.raw[:k2]
    def t(label, fn, reps=3):
        ts = []
        for _ in range(reps):
            s = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - s)
        out[label] = round(min(ts), 3)
        return r
    ref = t("lameenc", lambda: mp3_bytes(wav, sr, 192, 5))
    for name, path in [("debian libmp3lame", "/usr/lib/x86_64-linux-gnu/libmp3lame.so.0"),
                       ("built -O2", "/opt/lame-o2/lib/libmp3lame.so"),
                       ("built -O3 x86-64-v3", "/opt/lame-v3/lib/libmp3lame.so"),
                       ("built -O3 v3 fast-math", "/opt/lame-fast/lib/libmp3lame.so")]:
        r = t(name, lambda: ctypes_encode(path))
        out[name + " same bytes as lameenc"] = r == ref
        out[name + " size"] = len(r)
    out["lameenc size"] = len(ref)
    return out

@app.local_entrypoint()
def main():
    result = run.remote()
    out = Path(__file__).resolve().parent.parent / "results" / "lame-build.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))
