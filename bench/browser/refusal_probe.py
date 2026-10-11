"""Which ways around YouTube's bot check work from Modal's Google Cloud
containers? Each container tries several popular videos with:
  pinned   yt-dlp 2026.8.19, as production (IPv4)
  nightly  the newest yt-dlp pre-release (IPv4)
  ipv6     pinned, forced over IPv6 (only if the container has IPv6)
and reports its addresses. Metadata only (no download), so it's quick.

modal run probe.py
"""
import json
import subprocess
import time

import modal

app = modal.App("stemplayer-refusal-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("yt-dlp[default]==2026.8.19", "deno==2.9.7")
    .run_commands(
        "pip install --pre --target /nightly 'yt-dlp[default]' && python -c \"import sys; sys.path.insert(0, '/nightly'); import yt_dlp; print(yt_dlp.version.__version__)\""
    )
)

# Popular videos: the first five were refused somewhere in earlier runs
# (Bohemian Rhapsody, Gangnam Style, Uptown Funk, Despacito, Shape of You);
# Rick Astley never was.
VIDEOS = ["fJ9rUzIMcZQ", "9bZkp7q19f0", "OPf0YbXqDm0", "kJQP7kiw5Fk", "JGwWNGJdvx8", "dQw4w9WgXcQ"]

SCRIPT = r"""
import json, sys, time
if sys.argv[1] == "nightly":
    sys.path.insert(0, "/nightly")
import yt_dlp
opts = {"quiet": True, "no_warnings": True, "format": "bestaudio[protocol^=http]/bestaudio/best"}
if sys.argv[2] == "6":
    opts["source_address"] = "::"
out = {"version": yt_dlp.version.__version__}
for v in sys.argv[3:]:
    t = time.time()
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(f"https://www.youtube.com/watch?v={v}", download=False)
        fmt = (info.get("requested_downloads") or [info])[0]
        out[v] = "ok"
    except Exception as e:
        s = str(e)
        out[v] = "REFUSED" if "not a bot" in s else ("403" if "403" in s else "err:" + s[-80:])
    out[v + "_s"] = round(time.time() - t, 1)
print(json.dumps(out))
"""


def attempt() -> dict:
    def addr(flag):
        r = subprocess.run(["curl", flag, "-s", "-m", "5", "https://api64.ipify.org"], capture_output=True, text=True)
        return r.stdout.strip() or None

    res = {"ipv4": addr("-4"), "ipv6": addr("-6")}
    runs = [("pinned", "4"), ("nightly", "4")] + ([("pinned", "6")] if res["ipv6"] else [])
    for which, fam in runs:
        r = subprocess.run(["python", "-c", SCRIPT, which, fam, *VIDEOS], capture_output=True, text=True, timeout=240)
        try:
            res[f"{which}-v{fam}"] = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            res[f"{which}-v{fam}"] = {"error": (r.stderr or r.stdout)[-200:]}
    return res


# One function per region, defined at module level so Modal can import it.
@app.function(image=image, cloud="gcp", region="us-central", cpu=1.0, timeout=600)
def probe_us_central(): return attempt()
@app.function(image=image, cloud="gcp", region="us-east", cpu=1.0, timeout=600)
def probe_us_east(): return attempt()
@app.function(image=image, cloud="gcp", region="us-west", cpu=1.0, timeout=600)
def probe_us_west(): return attempt()
@app.function(image=image, cloud="gcp", region="eu-west", cpu=1.0, timeout=600)
def probe_eu_west(): return attempt()
@app.function(image=image, cloud="gcp", region="ap-northeast", cpu=1.0, timeout=600)
def probe_ap_northeast(): return attempt()
@app.function(image=image, cloud="gcp", region="ap-southeast", cpu=1.0, timeout=600)
def probe_ap_southeast(): return attempt()

FNS = {
    "us-central": probe_us_central, "us-east": probe_us_east, "us-west": probe_us_west,
    "eu-west": probe_eu_west, "ap-northeast": probe_ap_northeast, "ap-southeast": probe_ap_southeast,
}


@app.local_entrypoint()
def main(per_region: int = 4):
    calls = [(r, fn.spawn()) for r, fn in FNS.items() for _ in range(per_region)]
    with open("results.jsonl", "a") as f:
        for region, call in calls:
            try:
                res = call.get(timeout=600)
            except Exception as e:
                res = {"failed": str(e)[:200]}
            res["region"] = region
            res["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.write(json.dumps(res) + "\n")
            print(json.dumps(res))
