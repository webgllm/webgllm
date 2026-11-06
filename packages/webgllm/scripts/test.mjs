// Runs src/**/*.test.ts with Node's test runner: each file is bundled with
// esbuild first (the sources import ./x.js for x.ts, which Node's own type
// stripping doesn't resolve).
import { readdirSync } from "node:fs"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, relative } from "node:path"
import { spawnSync } from "node:child_process"
import { fileURLToPath } from "node:url"
import { build } from "esbuild"

const root = fileURLToPath(new URL("..", import.meta.url))
const src = join(root, "src")
const tests = readdirSync(src, { recursive: true })
  .map(String)
  .filter((f) => f.endsWith(".test.ts"))
  .map((f) => join(src, f))

const out = await mkdtemp(join(tmpdir(), "webgl-tests-"))
try {
  const files = []
  for (const file of tests) {
    const outfile = join(out, relative(src, file).replace(/\.ts$/, ".mjs"))
    await build({
      entryPoints: [file],
      outfile,
      bundle: true,
      format: "esm",
      platform: "node",
      conditions: ["webgllm/source"],
      logLevel: "error"
    })
    files.push(outfile)
  }
  const run = spawnSync(process.execPath, ["--test", ...files], { stdio: "inherit" })
  process.exitCode = run.status ?? 1
} finally {
  await rm(out, { recursive: true, force: true })
}
