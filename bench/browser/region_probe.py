import modal, time
app = modal.App("stemplayer-region-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("yt-dlp[default]==2026.8.19", "deno==2.9.7")
VIDEOS = ["fJ9rUzIMcZQ", "9bZkp7q19f0", "dQw4w9WgXcQ"]

def attempt():
    import yt_dlp, requests
    ip = requests.get("https://api.ipify.org", timeout=5).text
    out = {"ip": ip}
    for v in VIDEOS:
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "bestaudio[protocol^=http]/bestaudio/best"}) as y:
                y.extract_info(f"https://www.youtube.com/watch?v={v}", download=False)
            out[v] = "ok"
        except Exception as e:
            out[v] = "REFUSED" if "not a bot" in str(e) else "err"
    return out

REGIONS = ['us-central', 'us-west', 'us-east', 'eu-west', 'eu-north']
fns = {}

@app.function(image=image, cloud="gcp", region="us-central", cpu=1.0)
def probe_us_central():
    return attempt()
fns["us-central"] = probe_us_central

@app.function(image=image, cloud="gcp", region="us-west", cpu=1.0)
def probe_us_west():
    return attempt()
fns["us-west"] = probe_us_west

@app.function(image=image, cloud="gcp", region="us-east", cpu=1.0)
def probe_us_east():
    return attempt()
fns["us-east"] = probe_us_east

@app.function(image=image, cloud="gcp", region="eu-west", cpu=1.0)
def probe_eu_west():
    return attempt()
fns["eu-west"] = probe_eu_west

@app.function(image=image, cloud="gcp", region="eu-north", cpu=1.0)
def probe_eu_north():
    return attempt()
fns["eu-north"] = probe_eu_north

@app.local_entrypoint()
def main():
    calls = [(r, fns[r].spawn()) for r in REGIONS for _ in range(3)]
    for r, c in calls:
        try:
            print(r, c.get(timeout=240))
        except Exception as e:
            print(r, "failed:", str(e)[:120])
