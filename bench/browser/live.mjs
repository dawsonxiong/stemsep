// Time from Split to the first audible sample at the limiter, live (?live=1)
// against today's path (?live=0). Usage: node measure.mjs <chromium|webkit> <runs>
import { chromium, webkit } from "playwright";
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";

const S = process.env.WORK ?? ".";
const BASE = process.env.BASE ?? "https://stemplayer-bench-live.dawsonxiong.workers.dev";
const which = process.argv[2] ?? "chromium";
const runs = Number(process.argv[3] ?? 5);
const modes = (process.env.MODES ?? "1,0").split(",");

const browser =
  which === "webkit"
    ? await webkit.launch()
    : await chromium.launch({
        executablePath: `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
        args: ["--autoplay-policy=no-user-gesture-required", "--disable-audio-output"],
      });

const tap = () => {
  const AWN = window.AudioWorkletNode;
  window.AudioWorkletNode = class extends AWN {
    constructor(...args) {
      super(...args);
      window.__node = this;
      window.__nodeMsgs = [];
      this.addEventListener("processorerror", (e) => window.__nodeMsgs.push(["processorerror", String(e.message ?? e.error ?? e.type)]));
      this.port.addEventListener("message", (e) => {
        window.__lastState = e.data;
        if (e.data.type !== "state" || window.__nodeMsgs.length < 40) window.__nodeMsgs.push([Math.round(performance.now()), JSON.stringify(e.data)]);
      });
    }
  };
  const AC = window.AudioContext;
  const orig = AC.prototype.createDynamicsCompressor;
  window.__m = { marks: {} };
  AC.prototype.createDynamicsCompressor = function () {
    const node = orig.call(this);
    if (window.__tapped) return node;
    window.__tapped = true;
    const analyser = this.createAnalyser();
    analyser.fftSize = 512;
    node.connect(analyser);
    window.__tap = analyser;
    const buf = new Float32Array(analyser.fftSize);
    const id = setInterval(() => {
      analyser.getFloatTimeDomainData(buf);
      let e = 0;
      for (const x of buf) e += x * x;
      if (Math.sqrt(e / buf.length) > 1e-4) {
        window.__m.audible = performance.now();
        clearInterval(id);
      }
    }, 2);
    return node;
  };
  document.addEventListener(
    "click",
    (e) => {
      if (e.target instanceof Element && e.target.closest("button[type=submit]")) window.__m.split = performance.now();
    },
    true,
  );
  new MutationObserver(() => {
    if (!window.__m.transport && document.querySelector('[role=slider][aria-label=Seek]')) window.__m.transport = performance.now();
  }).observe(document, { subtree: true, childList: true });
};

const results = [];
for (let i = 0; i < runs * modes.length; i++) {
  const mode = modes[i % modes.length];
  const file = `${S}/audio/run-${randomUUID()}${(process.env.SONG ?? ".mp3").slice(-4)}`;
  execFileSync("ffmpeg", ["-v", "error", "-y", "-i", process.env.SONG ?? `${S}/audio/mixture.mp3`, "-c", "copy", "-metadata", `comment=${randomUUID()}`, file]);
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on("console", (m) => { if (process.env.DEBUG) console.log("console", m.type(), m.text()); if (m.type() === "error" || m.type() === "warning") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.addInitScript(tap);
  // An older browser: no AudioWorklet, so live playback can't run.
  if (process.env.NO_WORKLET) await page.addInitScript(() => { window.AudioWorkletNode = undefined; });
  await page.goto(`${BASE}/?live=${mode}`);
  if (process.env.LINK) {
    await page.fill('input[type="text"]', process.env.LINK);
  } else {
    const put = page.waitForResponse((r) => r.request().method() === "PUT" && r.url().includes("r2.cloudflarestorage.com"), { timeout: 60000 });
    await page.setInputFiles("input[type=file]", file);
    await put;
  }
  // Give the GPU the warm-up it would get while someone reaches for Split.
  await page.waitForTimeout(Number(process.env.WAIT ?? 4000));
  await page.click("button[type=submit]");
  await page.waitForFunction(() => window.__m.audible, null, { timeout: 120000 }).catch(() => {});
  if (process.env.EXPECT_FAIL) {
    // The job fails after its stream started playing: it should stop and say so.
    const heard = await page.evaluate(() => (window.__m.audible ? Math.round(window.__m.audible - window.__m.split) : null));
    await page.waitForSelector("[role=alert]", { timeout: 90000 }).catch(() => {});
    await page.waitForTimeout(1500);
    console.log("failure:", JSON.stringify(await page.evaluate(async () => ({
      alert: document.querySelector("[role=alert]")?.textContent,
      retry: [...document.querySelectorAll("button")].some((b) => b.textContent === "Try again"),
      worklet: window.__lastState,
      // Loudest output over the half second after the error showed.
      rmsAfter: await new Promise((done) => {
        const buf = new Float32Array(512);
        let peak = 0;
        const id = setInterval(() => {
          window.__tap.getFloatTimeDomainData(buf);
          let e = 0;
          for (const x of buf) e += x * x;
          peak = Math.max(peak, Math.sqrt(e / buf.length));
        }, 10);
        setTimeout(() => { clearInterval(id); done(peak); }, 500);
      }),
    })) ), "heard at", heard);
    await context.close();
    continue;
  }
  // Wait for the real ready (the URL gets ?t=) and a little playback after.
  await page.waitForFunction(() => new URL(location.href).searchParams.has("t"), null, { timeout: 120000 }).catch(() => {});
  const readyAt = await page.evaluate(() => performance.now());
  await page.waitForTimeout(3000);
  if (process.env.SEEK_STEPS) {
    // Past the loud part, where a rescaled stem's new encoding takes over.
    await page.focus("[role=slider][aria-label=Seek]");
    for (let k = 0; k < Number(process.env.SEEK_STEPS); k++) await page.keyboard.press("ArrowRight");
    await page.waitForTimeout(4000);
    console.log("after seek:", JSON.stringify(await page.evaluate(() => ({ pos: document.querySelector("[role=slider][aria-label=Seek]")?.getAttribute("aria-valuetext"), worklet: window.__lastState }))));
  }
  const m = await page.evaluate(() => {
    const marks = Object.fromEntries(performance.getEntriesByType("mark").map((e) => [e.name, e.startTime]));
    const t = document.querySelector("[role=slider][aria-label=Seek]");
    return { ...window.__m, marks, now: t?.getAttribute("aria-valuetext"), nodeMsgs: window.__nodeMsgs, ctx: window.__node?.context.state, rate: window.__node?.context.sampleRate };
  });
  const rel = (v) => (v == null ? null : Math.round(v - m.split));
  const row = {
    browser: which,
    live: mode,
    audible: rel(m.audible),
    transport: rel(m.transport),
    ready: rel(readyAt),
    open: rel(m.marks["live:open"]),
    header: rel(m.marks["live:header"]),
    firstData: rel(m.marks["live:first-data"]),
    start: rel(m.marks["live:start"]),
    position: m.now,
    task: await page.evaluate(() => new URL(location.href).searchParams.get("t")),
    errors: errors.slice(0, 3),
  };
  if (process.env.DEBUG) console.log(JSON.stringify({ nodeMsgs: m.nodeMsgs, ctx: m.ctx, rate: m.rate }));
  console.log(JSON.stringify(row)); (await import("node:fs")).appendFileSync(`rows-${which}.jsonl`, JSON.stringify(row) + "\n");
  results.push(row);
  await context.close();
}
await browser.close();
for (const mode of modes) {
  const xs = results.filter((r) => r.live === mode && r.audible != null).map((r) => r.audible).sort((a, b) => a - b);
  const med = xs.length ? xs[Math.floor(xs.length / 2)] : null;
  console.log(`${which} live=${mode}: n=${xs.length} median=${med} range=${xs[0]}–${xs.at(-1)} all=${xs.join(",")}`);
}
