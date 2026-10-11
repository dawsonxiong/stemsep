// Median (min–max) of each stage per label, from results.jsonl.
// node summary.mjs [label prefix]
import { readFileSync } from "node:fs";

const rows = readFileSync(new URL("./results.jsonl", `file://${process.env.WORK ?? process.cwd()}/`), "utf8")
  .trim().split("\n").map((l) => JSON.parse(l))
  .filter((r) => !process.argv[2] || r.label.startsWith(process.argv[2]));
const COLS = ["put_end", "separate_ok", "first_progress", "ready_seen", "playable"];
const fmt = (xs) => {
  xs = xs.filter((x) => x != null).sort((a, b) => a - b);
  if (!xs.length) return "–";
  const m = xs.length % 2 ? xs[(xs.length - 1) / 2] : (xs[xs.length / 2 - 1] + xs[xs.length / 2]) / 2;
  return xs.length > 1 ? `${m.toFixed(1)} (${xs[0].toFixed(1)}–${xs.at(-1).toFixed(1)})` : m.toFixed(1);
};
console.log(["label", "n", "ok", ...COLS].join("\t"));
for (const [label, rs] of Map.groupBy(rows, (r) => `${r.site} ${r.label}`)) {
  console.log([label, rs.length, rs.filter((r) => r.result === "ok").length,
    ...COLS.map((c) => fmt(rs.map((r) => r[c])))].join("\t"));
}
