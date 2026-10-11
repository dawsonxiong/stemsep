// One real split through a deployed stemplayer, timed from pressing Split to
// all four stems being playable (the 4th "canplay", when loadStems resolves).
//
// node run.mjs <old|new|prod> <song file | link> [label]
// Conditions, from the environment:
//   UP_MBPS / DOWN_MBPS / RTT_MS  network (CDP emulation), e.g. 5 / 20 / 80
//   CPU_X                         CPU slowdown (CDP), e.g. 4
//   PICK_MS                       pick → Split delay (default 2000)
//   MOBILE=1                      phone viewport, touch and UA
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { mkdtempSync, appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, basename } from "node:path";
import { randomUUID } from "node:crypto";

const SITES = {
  old: "https://stemplayer-bench-old.dawsonxiong.workers.dev",
  new: "https://stemplayer-bench-new.dawsonxiong.workers.dev",
  prod: "https://stemplayer.dawsonxiong.workers.dev",
};
const [site, song, label = ""] = process.argv.slice(2);
const base = SITES[site];
if (!base || !song) throw new Error("usage: run.mjs <old|new|prod> <song|link> [label]");
const env = process.env;
const pickMs = Number(env.PICK_MS ?? 2000);

// A file is picked as a copy with a unique tag, so its hash never hits the
// stem cache. A link is pasted.
const isLink = /^https?:/.test(song);
const file = isLink ? null : join(mkdtempSync(join(tmpdir(), "split-")), basename(song));
if (file)
  execFileSync("ffmpeg", ["-v", "error", "-y", "-i", song, "-c", "copy",
    "-metadata", `comment=${randomUUID()}`, file]);

const browser = await chromium.launch({
  executablePath: `${env.HOME}/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
  args: ["--autoplay-policy=no-user-gesture-required"],
});
const context = await browser.newContext(
  env.MOBILE
    ? {
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 3,
        isMobile: true,
        hasTouch: true,
        userAgent:
          "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36",
      }
    : {},
);
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
if (env.UP_MBPS || env.DOWN_MBPS || env.RTT_MS) {
  await cdp.send("Network.enable");
  await cdp.send("Network.emulateNetworkConditions", {
    offline: false,
    latency: Number(env.RTT_MS ?? 20),
    uploadThroughput: env.UP_MBPS ? (Number(env.UP_MBPS) * 1e6) / 8 : -1,
    downloadThroughput: env.DOWN_MBPS ? (Number(env.DOWN_MBPS) * 1e6) / 8 : -1,
  });
}
if (env.CPU_X) await cdp.send("Emulation.setCPUThrottlingRate", { rate: Number(env.CPU_X) });

await page.addInitScript(() => {
  window.__bench = { canplay: [] };
  const Real = window.Audio;
  window.Audio = function (...args) {
    const a = new Real(...args);
    a.addEventListener("canplay", () => window.__bench.canplay.push(Date.now()), { once: true });
    return a;
  };
  window.Audio.prototype = Real.prototype;
});

const events = [];
const kind = (url) => {
  const u = new URL(url);
  if (u.hostname.endsWith("r2.cloudflarestorage.com"))
    return u.pathname.includes("/uploads/") ? "r2_put" : "r2_stem";
  const m = u.pathname.match(/^\/api\/(\w+)/);
  return m ? m[1] : null;
};
page.on("request", (r) => {
  const k = kind(r.url());
  if (k) events.push({ k, e: "start", t: Date.now() });
});
page.on("requestfinished", async (r) => {
  const k = kind(r.url());
  if (!k) return;
  const ev = { k, e: "end", t: Date.now() };
  if (k === "status") {
    try {
      const body = await (await r.response()).json();
      ev.status = body.status;
      ev.progress = body.progress;
    } catch {}
  }
  events.push(ev);
});

const loadStart = Date.now();
await page.goto(base, { waitUntil: "load" });
const pageLoad = Date.now() - loadStart;
// Someone finding their song after the page loads.
await page.waitForTimeout(3000);

if (file) await page.setInputFiles('input[type="file"]', file);
else await page.fill('input[type="text"]', song);
const picked = Date.now();
await page.waitForTimeout(pickMs);
const split = Date.now();
await page.getByRole("button", { name: "Split", exact: true }).click();

let result = "ok";
try {
  await page.waitForFunction(
    (t) => window.__bench.canplay.filter((c) => c >= t).length >= 4,
    split,
    { timeout: Number(env.WAIT_MS ?? 240_000), polling: 50 },
  );
} catch {
  result = (await page.getByRole("alert").textContent().catch(() => null)) ?? "timeout";
}
const bench = await page.evaluate(() => window.__bench);
await browser.close();

const rel = (t) => (t == null ? null : +((t - split) / 1000).toFixed(2));
const first = (k, e, f = () => true) => events.find((x) => x.k === k && x.e === e && f(x))?.t;
const canplay = bench.canplay.filter((c) => c >= split).sort();
const conditions = ["UP_MBPS", "DOWN_MBPS", "RTT_MS", "CPU_X", "MOBILE"]
  .filter((k) => env[k])
  .map((k) => `${k}=${env[k]}`)
  .join(" ");

const row = {
  site, song: basename(song), label, conditions, pick_ms: pickMs, result,
  at: new Date(split).toISOString(),
  page_load: +(pageLoad / 1000).toFixed(2),
  // seconds after Split (negative: before it, while the song waited):
  put_start: rel(first("r2_put", "start")),
  put_end: rel(first("r2_put", "end")),
  separate_ok: rel(first("separate", "end")),
  first_progress: rel(first("status", "end", (x) => x.progress > 0)),
  ready_seen: rel(first("status", "end", (x) => x.status === "ready")),
  first_canplay: rel(canplay[0]),
  playable: rel(canplay[3]),
};
console.log(JSON.stringify(row));
appendFileSync(new URL("./results.jsonl", `file://${process.env.WORK ?? process.cwd()}/`), JSON.stringify(row) + "\n");
