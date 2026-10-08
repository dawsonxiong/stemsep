"""Markdown tables from results/<sweep>/*.json.

    python bench/report.py results/full > results/full/README.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(v, digits=2):
    return "–" if v is None else f"{v:.{digits}f}"


def table(result: dict) -> str:
    env = result["env"]
    rows = result["rows"]
    base = next((r for r in rows if r["name"].startswith("demucs")), None)
    lines = [
        f"### {env['gpu']} ({env['device']}, {env['vram_gib']} GiB) — torch {env['torch']}, "
        f"CUDA {env['cuda']}, {env['audio_seconds']:.0f} s song",
        "",
        "| config | separate (s) | speedup | × realtime | peak VRAM (GiB) | ¢ per audio min "
        "| SDR (dB) | cSDR (dB) | vs fp32 (dB) | first call (s) |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | error: {r['error'][:80]} |" + " |" * 9)
            continue
        q = r["quality"]["mean"]
        speedup = base["separate_s"] / r["separate_s"] if base and "separate_s" in base else None
        lines.append(
            f"| {r['name']} | {fmt(r['separate_s'])} | {fmt(speedup)}× | {fmt(r['realtime_x'], 1)} "
            f"| {fmt(r['peak_reserved_gib'])} | {fmt(r['inference_cost_per_audio_min'] * 100, 3)} "
            f"| {fmt(q['sdr'])} | {fmt(q['chunked_sdr'])} | {fmt(r.get('sdr_vs_fp32'), 1)} "
            f"| {fmt(r['first_call_s'])} |"
        )
    e2e = result.get("e2e") or {}
    if e2e:
        lines += ["", "End to end (decode MP3 → separate → encode 4 MP3s), fp16 ov.10 b8:", ""]
        for enc, v in e2e.items():
            stages = ", ".join(f"{k} {s:.2f}s" for k, s in v["stages_s"].items())
            lines.append(f"- {enc} encoder: **{v['total_s']:.2f} s** ({stages})")
    return "\n".join(lines)


def main(folder: str) -> None:
    order = ["T4", "L4", "A10", "L40S", "A100", "H100"]
    files = sorted(Path(folder).glob("*.json"), key=lambda p: (order + [p.stem]).index(p.stem))
    print("\n\n".join(table(json.loads(f.read_text())) for f in files))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/full")
