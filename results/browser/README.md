# Real splits in a browser (2026-10-09 to 2026-10-11)

Each row is a real split through a deployed copy of stemplayer, run by Playwright in Chromium on a Mac in Toronto (scripts in `bench/browser/`). The time is measured from pressing Split to all four stems playing.

**How a run worked:**
- **Song:** each run picks a copy of the song with a unique tag (`ffmpeg -c copy -metadata comment=<uuid>`), so the stem cache never hits.
- **Network:** throttled with CDP `Network.emulateNetworkConditions`, and the CPU with `Emulation.setCPUThrottlingRate`.
- **"Playable":** the fourth `canplay` from wrapped `Audio` elements, which is when `AudioEngine.loadStems` resolves.
- **First sound, with live playback** (`live.mjs`): RMS above 1e-4 on an analyser tapped off the limiter.

The bench Workers and Modal apps these ran against have since been deleted, so the URLs in the scripts no longer resolve. To run them again, deploy a bench copy as stemplayer's `docs/handoff.md` describes.

| File | What |
|---|---|
| `matrix.jsonl` | Uploads under 11 simulated conditions, 3 each, on the A10 pinned to `region="us"`. Times are seconds after Split; a negative `put_end` means the upload finished before Split. |
| `youtube-retry.jsonl` | YouTube links with refusals retried in other regions (stemplayer #14). In the first `retry-regions` batch, two runs failed: one from a bug since fixed (a stand-in's error counted as a win), and one from a GPU container still on the previous version. `retry-regions-2` was after both. |
| `region-probe.txt` | YouTube refusals for two gated videos and one ungated video, 3 containers in each of 5 Google Cloud regions (`region_probe.py`). eu-north had no capacity. |
| `live-chromium.jsonl` | Live playback against today's path (`live` 1 or 0). `audible` is ms to first sound, `ready` ms to the finished files. |
| `production.jsonl` | Production after the second pass, before live playback was switched on. |

**Not kept as rows.** These numbers come from the session's notes; their raw rows were lost with a scratch directory:
- **Old stack (stemplayer 387164c) against current `main`, Split → playable:**

  | | Old | New |
  |---|---|---|
  | Warm uploads (8 each) | 60.6 s | 10.4 s |
  | Cold uploads (3 each) | 59.9 s | 9.3 s |
  | Production, warm (n=2) | | 7.4–7.7 s |

  - The new bench's A10 was in Frankfurt; the old stack's L4 was in IAD and EWR.
  - Four links: old 47.6–60.8 s; new 10.5, 39.3, 14.1 and 56.7 s.
- **`region="us"`:** 8.4 s warm, against 10.4 s with the A10 in Frankfurt (n=8 each).
- **Upload on pick, at 20 Mbps up, Split 2 s after the pick:** 10.9 s → 8.8 s (n=8 each).
- **Snapshots after a deploy:** 3 cold splits after a 6-way warm took 3.5, 6.2 and 16.7 s to their first progress report. The last one landed on a machine type with no snapshot yet.
