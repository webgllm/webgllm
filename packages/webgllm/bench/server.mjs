/**
 * Serves the bench pages, cross-origin isolated (COOP same-origin, COEP
 * require-corp) so wllama gets threads, and prints what the pages post.
 * `/bundle/<path>.js` is the package's `<path>.ts` bundled with esbuild on
 * request, so pages and workers can run the engine's TypeScript.
 *
 *   node bench/server.mjs [port]     MODELS_DIR=… for model files, REF_DIR=… for reference tensors
 *
 * On a phone over USB: `adb reverse tcp:8787 tcp:8787`, then open
 * http://localhost:8787/?run=gpu (localhost counts as a secure context).
 */
import { createReadStream, appendFileSync, statSync } from "node:fs"
import { createServer } from "node:http"
import { dirname, extname, join, normalize } from "node:path"
import { fileURLToPath } from "node:url"

import { build } from "esbuild"

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, "..")
const repo = join(here, "../../..")
const port = Number(process.argv[2] ?? 8787)
const resultsFile = process.env.RESULTS_FILE ?? join(here, "results.jsonl")

const roots = {
  "/wllama/": join(
    repo,
    "node_modules/.pnpm/@wllama+wllama@3.8.1/node_modules/@wllama/wllama/esm/"
  ),
  "/wasm/": join(repo, "packages/wllama/wasm/"),
  "/models/": process.env.MODELS_DIR ?? join(here, "models/"),
  "/ref/": process.env.REF_DIR ?? join(here, "ref/"),
  "/ref-onnx/": process.env.REF_ONNX_DIR ?? join(here, "ref-onnx/"),
  "/onnx-web/": process.env.ONNX_DIR ?? join(here, "onnx-web/"),
  "/onnx-community/":
    process.env.ONNX_COMMUNITY_DIR ?? join(here, "onnx-community/"),
  "/hf/": process.env.HF_DIR ?? join(here, "hf/"),
  "/": here
}

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".gguf": "application/octet-stream"
}

const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Cache-Control": "no-store"
}

function resolve(url) {
  const path = decodeURIComponent(new URL(url, "http://x").pathname)
  for (const [prefix, root] of Object.entries(roots)) {
    if (!path.startsWith(prefix)) continue
    const file = normalize(
      join(root, path.slice(prefix.length) || "index.html")
    )
    if (!file.startsWith(normalize(root))) return null
    return file
  }
  return null
}

/** `/bundle/src/worker.js` → packages/webgl/src/worker.ts, bundled. */
async function bundle(path, res) {
  const entry = normalize(
    join(pkg, path.replace(/^\/bundle\//, "").replace(/\.js$/, ".ts"))
  )
  if (!entry.startsWith(pkg)) return res.writeHead(404, isolation).end()
  try {
    const out = await build({
      entryPoints: [entry],
      bundle: true,
      format: "esm",
      platform: "browser",
      target: "es2022",
      write: false,
      sourcemap: "inline",
      logLevel: "silent"
    })
    res.writeHead(200, { ...isolation, "Content-Type": types[".js"] })
    res.end(out.outputFiles[0].text)
  } catch (error) {
    console.error(`[bundle] ${path}: ${error.message}`)
    res.writeHead(500, isolation).end(String(error.message))
  }
}

createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url ?? "/", "http://x").pathname)
  if (req.method === "GET" && path.startsWith("/bundle/"))
    return void bundle(path, res)
  if (req.method === "POST") {
    let body = ""
    req.on("data", (chunk) => (body += chunk))
    req.on("end", () => {
      if (req.url === "/log") console.log(`[page] ${body}`)
      if (req.url === "/result") appendFileSync(resultsFile, `${body}\n`)
      res.writeHead(204, isolation).end()
    })
    return
  }
  const file = resolve(req.url ?? "/")
  let stat
  try {
    stat = file && statSync(file)
  } catch {
    stat = null
  }
  if (!file || !stat?.isFile()) {
    res.writeHead(404, isolation).end("not found")
    return
  }
  const headers = {
    ...isolation,
    "Content-Type": types[extname(file)] ?? "application/octet-stream",
    "Accept-Ranges": "bytes"
  }
  const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? "")
  if (range) {
    const start = range[1] ? Number(range[1]) : 0
    const end = range[2]
      ? Math.min(Number(range[2]), stat.size - 1)
      : stat.size - 1
    res.writeHead(206, {
      ...headers,
      "Content-Range": `bytes ${start}-${end}/${stat.size}`,
      "Content-Length": end - start + 1
    })
    if (req.method === "HEAD") return res.end()
    createReadStream(file, { start, end }).pipe(res)
    return
  }
  res.writeHead(200, { ...headers, "Content-Length": stat.size })
  if (req.method === "HEAD") return res.end()
  createReadStream(file).pipe(res)
}).listen(port, () => console.log(`http://localhost:${port}/`))
