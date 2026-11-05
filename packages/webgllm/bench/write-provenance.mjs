/** Record reproducible source/build identities without reading configuration secrets. */
import fs from "node:fs"
import path from "node:path"
import { createHash } from "node:crypto"
import { execFileSync } from "node:child_process"
const [destination, ...summaries] = process.argv.slice(2)
if (!destination || !summaries.length)
  throw Error(
    "Usage: node bench/write-provenance.mjs OUTPUT.json SUMMARY.json ..."
  )
const root = execFileSync("git", ["rev-parse", "--show-toplevel"], {
  encoding: "utf8"
}).trim()
const hash = (file) =>
  createHash("sha256").update(fs.readFileSync(file)).digest("hex")
const changed = [
  ...execFileSync("git", ["diff", "2c10f9e", "--name-only", "--", "packages/webgl"], {
    encoding: "utf8"
  })
    .trim()
    .split("\n"),
  ...execFileSync(
    "git",
    ["ls-files", "--others", "--exclude-standard", "packages/webgl"],
    { encoding: "utf8" }
  )
    .trim()
    .split("\n")
].filter((file) => /\.(ts|mjs|html|py)$/.test(file))
const manifest = {
  baselineCommit: execFileSync("git", ["rev-parse", "2c10f9e"], {
    encoding: "utf8"
  }).trim(),
  branch: execFileSync("git", ["branch", "--show-current"], {
    encoding: "utf8"
  }).trim(),
  engineCommit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
  clientCadenceGuardCommit: "6a6e641",
  clientCadenceGuardCoverage: "Rounded-millisecond 60/120 Hz paths were unit-tested after timing runs; measured fractional-ms estimator unchanged.",
  recordedAt: new Date().toISOString(),
  node: process.version,
  modelMeasurements:
    "Physical Pixel and iPhone, followed by the user-authorized M3 Ultra Chrome benchmark.",
  baselineWorkerSha256: hash("packages/webgl/bench/dist/baseline-worker.js"),
  phoneVerifiedCommit: "70d3ea0",
  phoneVerifiedEngineCode: 446196241,
  phoneVerifiedWorkerSha256: hash("packages/webgl/bench/dist/phone-verified-worker.js"),
  priorPixelMacCommit: "f9ffe1d",
  priorPixelMacEngineCode: 2693625472,
  priorPixelMacWorkerSha256: hash("packages/webgl/bench/dist/pre-load-shortcut-worker.js"),
  proofScreenshotSha256: hash("packages/webgl/bench/results/mac-final-proof.jpg"),
  presentationClipsSha256: Object.fromEntries(["before", "after"].map(mode => [mode, hash(`packages/webgl/bench/results/presentation-${mode}.mp4`)])),
  selectedResultsSha256: hash("packages/webgl/bench/results/webgl-performance-2026-11-03.jsonl"),
  presentationMotionSummarySha256: hash("packages/webgl/bench/results/presentation-motion-summary.json"),
  presentationFramesSha256: hash("packages/webgl/bench/results/presentation-motion-frames.jsonl"),
  reportSha256: hash("packages/webgl/bench/WEBGL-PERFORMANCE.md"),
  optimizedWorkerSha256: hash("packages/webgl/bench/dist/current-worker.js"),
  buildCommand:
    "node packages/webgl/bench/build-workers.mjs CHECKOUT current-worker.js",
  sources: Object.fromEntries(
    [...new Set(changed)]
      .sort()
      .map((file) => [file, hash(path.join(root, file))])
  ),
  results: summaries.map((file) => ({
    file: path.basename(file),
    sha256: hash(file),
    engineCode: JSON.parse(fs.readFileSync(file)).optimizedEngineCode
  }))
}
fs.writeFileSync(destination, JSON.stringify(manifest, null, 2) + "\n")
