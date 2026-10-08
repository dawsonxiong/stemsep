"""Separation quality of production demucs against stemsep configs.

    cd bench && uv run modal run quality.py
    cd bench && uv run modal run quality.py --sets fulltracks

On the concatenated test set (bench/testset.py), stemsep at 25% overlap
scored ~0.2 dB SDR below production demucs. That song cuts to a new excerpt
every 6.8 s, so where the chunk grid falls relative to the cuts could move
SDR on its own. This scores both on whole MUSDB18-HQ songs
(bench/fulltracks.py) as well as on the old set, and runs stemsep again on
inputs delayed by k samples, which moves only the chunk grid, to measure how
much SDR that alone is worth. Two diagnostics go further: demucs on delayed
inputs, and stemsep cutting its chunks at demucs's stride instead of spreading
them evenly, which separates "where the grid falls" from "how it's spaced".

Everything runs in one L4 container. Results land in
results/quality/<set>.json, summarized in results/quality/README.md.
"""

import json
import math
import statistics
import time
from pathlib import Path

import modal
from images import image, volume

app = modal.App("stemsep-quality")

SETS = ("fulltracks", "testset")
SR = 44100
MAX_SHIFT = SR // 2  # demucs's shifts=1 draws its offset from [0, 0.5 s]
SEEDS = (0, 1, 2)
# Up to demucs's random shift, and two beyond it: the old set's cuts repeat
# every 6.8 s, so a grid only escapes an unlucky phase by moving seconds.
PADS_S = (0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 3.0)
DEMUCS_PADS_S = (1.5, 3.0)

S0 = "demucs shifts=0"
PROD = [f"demucs shifts=1 seed {s}" for s in SEEDS]
REF = "stemsep fp32 ov.25"
GRID = f"{REF} demucs stride"


def configs() -> list[dict]:
    return [
        {"name": S0, "kind": "demucs", "shifts": 0},
        *({"name": n, "kind": "demucs", "shifts": 1, "seed": s} for n, s in zip(PROD, SEEDS)),
        {"name": REF, "overlap": 0.25},
        {"name": "stemsep fp32 ov.10", "overlap": 0.10},
        {"name": "stemsep fp32 ov.05", "overlap": 0.05},
        {"name": "stemsep fp16 ov.10", "overlap": 0.10, "precision": "fp16"},
        {"name": "stemsep bf16 ov.10", "overlap": 0.10, "precision": "bf16"},
        *(
            {"name": f"{REF} delay {p:.1f}s", "overlap": 0.25, "pad": round(p * SR)}
            for p in PADS_S
        ),
        *(
            {"name": f"{S0} delay {p:.1f}s", "kind": "demucs", "shifts": 0, "pad": round(p * SR)}
            for p in DEMUCS_PADS_S
        ),
        {"name": GRID, "overlap": 0.25, "grid": "demucs"},
    ]


@app.function(image=image, gpu="L4", volumes={"/data": volume}, cpu=4, memory=32 * 1024,
              timeout=2 * 3600)  # fmt: skip
def evaluate(sets: list[str]) -> dict:
    import random

    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    from stemsep import Config, Separator
    from stemsep.quality import load_fulltracks, load_testset, report, sdr

    demucs = get_model("htdemucs").eval().to("cuda")
    separators = {}
    for cfg in configs():
        key = (cfg.get("overlap"), cfg.get("precision", "fp32"), cfg.get("grid"))
        if "overlap" in cfg and key not in separators:
            separators[key] = Separator(
                "htdemucs", Config(overlap=key[0], precision=key[1], batch_size=8), "cuda"
            )
    # A diagnostic, not an engine option: chunk starts at demucs's stride,
    # range(0, ..., int(0.75 * seg)), with the last chunk still flush with the
    # song's end, in place of plan_offsets's even spread.
    grid_sep = separators[(0.25, "fp32", "demucs")]

    def demucs_offsets(length, sub):
        seg = int(sub.segment * sub.samplerate)
        if length <= seg:
            return [0]
        return sorted({*range(0, length - seg + 1, int(0.75 * seg)), length - seg})

    grid_sep._offsets = demucs_offsets

    def run_demucs(wav, shifts):
        """As stemplayer's worker does: normalize the song as a whole, then
        apply_model at 25% overlap."""
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std()
        with torch.inference_mode():
            out = apply_model(demucs, ((wav - mean) / std)[None].to("cuda"),
                              shifts=shifts, split=True, overlap=0.25)[0]  # fmt: skip
        out = (out * std.to(out.device) + mean.to(out.device)).cpu()
        return dict(zip(demucs.sources, out))

    def run(cfg, wav, track_index):
        extra = {}
        pad = cfg.get("pad", 0)
        if pad:
            # Silence in front moves every chunk boundary by `pad` samples.
            extra["delay_samples"] = pad
            out, _ = run({k: v for k, v in cfg.items() if k != "pad"},
                         torch.nn.functional.pad(wav, (pad, 0)), track_index)  # fmt: skip
            return {s: w[:, pad:] for s, w in out.items()}, extra
        if cfg.get("kind") == "demucs":
            if cfg["shifts"]:
                # Seeded per song, so each song gets its own offset as in
                # production, yet reruns repeat. demucs pads the song with
                # MAX_SHIFT zeros and starts `offset` into the padding, which
                # is a delay of MAX_SHIFT - offset.
                seed = cfg["seed"] * 1000 + track_index
                random.seed(seed)
                extra["delay_samples"] = MAX_SHIFT - random.Random(seed).randint(0, MAX_SHIFT)
            return run_demucs(wav, cfg["shifts"]), extra
        sep = separators[(cfg["overlap"], cfg.get("precision", "fp32"), cfg.get("grid"))]
        return sep.separate(wav), extra

    def tracks(name):
        if name == "testset":
            mix, stems, sr = load_testset("/data/testset")
            yield "musdb18-7 test, concatenated", mix, stems, sr
        else:
            yield from load_fulltracks("/data/fulltracks")

    props = torch.cuda.get_device_properties(0)
    out = {"env": {"device": props.name, "torch": torch.__version__, "cuda": torch.version.cuda}}
    for set_name in sets:
        rows = []
        t_set = time.time()
        for index, (name, mix, stems, sr) in enumerate(tracks(set_name)):
            assert sr == SR, sr
            results, baseline = {}, None
            for cfg in configs():
                try:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    est, extra = run(cfg, mix, index)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - t0
                except Exception as e:  # keep the rest of the run
                    results[cfg["name"]] = {"error": repr(e)}
                    continue
                row = report(stems, est, sr)
                row.update(extra, seconds=round(seconds, 3))
                if cfg["name"] == S0:
                    baseline = est
                elif baseline is not None:
                    # How far each output is from deterministic demucs: the
                    # same model, so anything below ~100 dB is chunking.
                    row["vs_demucs_s0"] = statistics.mean(sdr(baseline[s], est[s]) for s in est)
                results[cfg["name"]] = row
            rows.append({"name": name, "duration_s": mix.shape[-1] / sr, "results": results})
            print(f"[{set_name}] {name}: {_fmt(results)}", flush=True)
        out[set_name] = {"tracks": rows, "wall_s": round(time.time() - t_set, 1)}
    return out


def _fmt(results: dict) -> str:
    keys = (S0, PROD[0], REF, "stemsep fp16 ov.10")
    return ", ".join(f"{k} {results[k]['mean']['sdr']:.2f}" for k in keys if "mean" in results[k])


# ---- summary (runs locally) ----------------------------------------------------


def _sdr(track: dict, name: str, key: str = "sdr", stem: str | None = None) -> float:
    r = track["results"].get(name, {})
    if "mean" not in r:
        return math.nan
    return r["stems"][stem][key] if stem else r["mean"][key]


def _prod(track: dict, key: str = "sdr", stem: str | None = None) -> float:
    """Production demucs's expected score on a song: the mean over seeds."""
    return statistics.mean(_sdr(track, n, key, stem) for n in PROD)


def _stats(values: list[float]) -> dict:
    values = [v for v in values if not math.isnan(v)]
    n = len(values)
    if not n:
        return {"n": 0, "median": math.nan, "mean": math.nan, "se": math.nan}
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else math.nan
    return {"n": n, "median": statistics.median(values), "mean": statistics.mean(values), "se": se}


def summarize(result: dict) -> dict:
    """Per config: median and mean over tracks of the per-track mean SDR,
    per-stem medians, and the paired per-track difference against
    production demucs (mean over its seeds) and deterministic demucs."""
    tracks = result["tracks"]
    names = [c["name"] for c in configs()]
    summary = {}
    for name in names:
        sdrs = [_sdr(t, name) for t in tracks]
        csdrs = [_sdr(t, name, "chunked_sdr") for t in tracks]
        agree = [t["results"].get(name, {}).get("vs_demucs_s0", math.nan) for t in tracks]
        summary[name] = {
            "sdr": _stats(sdrs),
            "chunked_sdr": _stats(csdrs),
            # Silent stems score NaN and are left out, as in `report`.
            "stems": {
                s: _stats([_sdr(t, name, stem=s) for t in tracks])["median"]
                for s in ("drums", "bass", "other", "vocals")
            },
            "minus_prod": _stats([v - _prod(t) for v, t in zip(sdrs, tracks)]),
            "minus_prod_csdr": _stats([v - _prod(t, "chunked_sdr") for v, t in zip(csdrs, tracks)]),
            "minus_s0": _stats([v - _sdr(t, S0) for v, t in zip(sdrs, tracks)]),
            "vs_demucs_s0_median": _stats(agree)["median"],
            "seconds": sum(t["results"].get(name, {}).get("seconds", 0) for t in tracks),
            "errors": [t["name"] for t in tracks if "error" in t["results"].get(name, {})],
        }
    return {"configs": summary, "grid": {k: _spread(tracks, v) for k, v in grid_families().items()}}


def grid_families() -> dict[str, list[str]]:
    """Runs that differ only in where the chunk grid falls, grouped by its
    spacing: stemsep's even spread, and demucs's fixed stride."""
    names = [c["name"] for c in configs()]
    return {
        "stemsep spacing": [REF] + [n for n in names if n.startswith(f"{REF} delay")],
        "demucs spacing": [S0, *PROD, GRID] + [n for n in names if n.startswith(f"{S0} delay")],
    }


def _spread(tracks: list[dict], names: list[str]) -> dict:
    """Per track, the spread of mean SDR over `names`; then its mean over
    tracks."""
    per_track = []
    for t in tracks:
        values = [v for v in (_sdr(t, n) for n in names) if not math.isnan(v)]
        per_track.append({"name": t["name"], "mean": statistics.mean(values),
                          "std": statistics.stdev(values), "range": max(values) - min(values)})  # fmt: skip
    return {
        "configs": names,
        "per_track": per_track,
        "mean_std": statistics.mean(p["std"] for p in per_track),
        "mean_range": statistics.mean(p["range"] for p in per_track),
        "mean": statistics.mean(p["mean"] for p in per_track),
    }


def _f(x: float, signed: bool = False) -> str:
    if math.isnan(x):
        return "–"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def readme(results: dict, summaries: dict) -> str:
    lines = [
        "# Separation quality: production demucs vs stemsep",
        "",
        "`cd bench && uv run modal run quality.py` (NVIDIA L4, htdemucs). SDR is the global",
        "energy-ratio SDR (\"nSDR\") and cSDR the median over 1 s windows, from",
        "`stemsep.quality.report`; a track's score is the mean over its four stems.",
        "\"prod\" is production demucs (`apply_model(shifts=1, overlap=0.25)`), whose one random",
        "shift makes it a different chunk grid per run; its per-track score here is the mean",
        "of three seeds. Δ columns are paired per-track differences, mean ± standard error.",
        "\"delay k s\" rows run on the input with k s of silence prepended (then trimmed), which",
        "moves only the chunk grid. \"demucs stride\" is stemsep with its chunks cut every",
        "`int(0.75 * segment)` samples, as demucs does, instead of spread evenly (last chunk",
        "still flush with the end): a diagnostic, not an engine option.",
        "",
    ]
    titles = {
        "fulltracks": "Full songs: MUSDB18-HQ test, first 12 by name (bench/fulltracks.py)",
        "testset": "Old set: 36 MUSDB18-7 test excerpts concatenated into one 245 s song",
    }
    for set_name, summary in summaries.items():
        tracks = results[set_name]["tracks"]
        minutes = sum(t["duration_s"] for t in tracks) / 60
        lines += [f"## {titles[set_name]}", "",
                  f"{len(tracks)} track(s), {minutes:.1f} min of audio.", ""]  # fmt: skip
        many = len(tracks) > 1
        head = ["config", "median SDR" if many else "SDR"]
        if many:
            head.append("mean SDR")
        head += ["Δ vs prod", "Δ vs demucs s0", "cSDR", "Δ cSDR vs prod",
                 "drums", "bass", "other", "vocals", "vs demucs s0 (dB)"]  # fmt: skip
        lines += ["| " + " | ".join(head) + " |", "|---|" + "--:|" * (len(head) - 1)]
        for name, s in summary["configs"].items():
            if s["sdr"]["n"] == 0:
                lines.append(f"| {name} | error: {', '.join(s['errors'])} |")
                continue
            def delta(d, many=many):
                se = f" ± {d['se']:.2f}" if many and not math.isnan(d["se"]) else ""
                return _f(d["mean"], True) + se

            row = [name, _f(s["sdr"]["median"])]
            if many:
                row.append(_f(s["sdr"]["mean"]))
            row += [
                delta(s["minus_prod"]),
                "–" if name == S0 else delta(s["minus_s0"]),
                _f(s["chunked_sdr"]["median"]),
                delta(s["minus_prod_csdr"]),
                *(_f(s["stems"][k]) for k in ("drums", "bass", "other", "vocals")),
                "–" if name == S0 else f"{s['vs_demucs_s0_median']:.1f}",
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", ("Chunk-grid noise: runs that differ only in where the grid falls, grouped "
                       "by chunk spacing. Per track, the spread of mean SDR across the group, "
                       "averaged over tracks."), "",
                  "| spacing | runs | mean SDR | std within track | range within track |",
                  "|---|--:|--:|--:|--:|"]  # fmt: skip
        for family, g in summary["grid"].items():
            lines.append(f"| {family} | {len(g['configs'])} | {g['mean']:.3f} | "
                         f"{g['mean_std']:.3f} | {g['mean_range']:.3f} |")  # fmt: skip
        lines.append("")
        if many:
            lines += ["Per track, mean SDR over stems:", "",
                      ("| track | s | demucs s0 | prod (3 seeds) | stemsep fp32 ov.25 "
                       "| delays min–max | demucs stride | fp32 ov.10 | fp16 ov.10 | bf16 ov.10 |"),
                      "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]  # fmt: skip
            delays = [n for n in summary["configs"] if n.startswith(f"{REF} delay")]
            for t in tracks:
                d = [_sdr(t, n) for n in delays]
                lines.append(
                    f"| {t['name']} | {t['duration_s']:.0f} | {_f(_sdr(t, S0))} | {_f(_prod(t))} | "
                    f"{_f(_sdr(t, REF))} | {_f(min(d))}–{_f(max(d))} | {_f(_sdr(t, GRID))} | "
                    f"{_f(_sdr(t, 'stemsep fp32 ov.10'))} | {_f(_sdr(t, 'stemsep fp16 ov.10'))} | "
                    f"{_f(_sdr(t, 'stemsep bf16 ov.10'))} |"
                )
            lines.append("")
    return "\n".join(lines)


@app.local_entrypoint()
def main(sets: str = ",".join(SETS), from_json: bool = False):
    """`--from-json` rebuilds the README from results already on disk."""
    out_dir = Path(__file__).resolve().parent.parent / "results" / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [s for s in sets.split(",") if s]
    if not from_json:
        result = evaluate.remote(names)
        for name in names:
            data = {"env": result["env"], "configs": configs(), **result[name]}
            (out_dir / f"{name}.json").write_text(json.dumps(data, indent=2))
            print(f"wrote {out_dir / name}.json")
    results = {}
    for name in SETS:
        path = out_dir / f"{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text())
    summaries = {name: summarize(r) for name, r in results.items()}
    for name, s in summaries.items():
        results[name]["summary"] = s
        (out_dir / f"{name}.json").write_text(json.dumps(results[name], indent=2))
    (out_dir / "README.md").write_text(readme(results, summaries))
    print(f"wrote {out_dir / 'README.md'}")
