/**
 * Serves the bench pages, cross-origin isolated (COOP same-origin, COEP
 * require-corp) so SharedArrayBuffer and precise timers are available, and
 * prints what the pages post (`[page] …`). `/bundle/<path>.js` is this
 * package's `<path>.ts` bundled with esbuild on request, so pages and
 * workers can run the engine's TypeScript.
 *
 *   node packages/webgllm/bench/server.mjs [port]
 *     MODELS_DIR=… model files (and the phone driver's job.json)
 *     HF_DIR=…     tokenizer files for GGUF checks
 *     REF_DIR=… / REF_ONNX_DIR=…  reference tensors (llama.cpp / onnxruntime)
 *     ONNX_COMMUNITY_DIR=… / ONNX_DIR=…  ONNX builds for check.html
 *
 * Listens on localhost only (HOST=0.0.0.0 to serve the network; phones reach
 * it through `adb reverse` or a tunnel to localhost).
 *
 * On a phone over USB: `adb reverse tcp:8787 tcp:8787`, then open
 * http://localhost:8787/gl-info.html (localhost counts as a secure context).
 */
import { createReadStream, appendFileSync, statSync } from "node:fs"
import { createServer } from "node:http"
import {
  dirname,
  extname,
  isAbsolute,
  join,
  relative,
  resolve as resolvePath
} from "node:path"
import { fileURLToPath } from "node:url"

import { build } from "esbuild"

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, "..")
const port = Number(process.argv[2] ?? 8787)
const resultsFile = process.env.RESULTS_FILE ?? join(here, "results.jsonl")

const roots = {
  "/models/": process.env.MODELS_DIR ?? join(here, "models/"),
  "/ref/": process.env.REF_DIR ?? join(here, "ref/"),
  "/ref-onnx/": process.env.REF_ONNX_DIR ?? join(here, "ref-onnx/"),
  "/onnx/": process.env.ONNX_DIR ?? join(here, "onnx/"),
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

/**
 * Whether `file` is `root` or inside it: a path boundary, not a string
 * prefix (`/a/bench-private` starts with `/a/bench` but isn't in it).
 */
function inside(root, file) {
  const rel = relative(resolvePath(root), resolvePath(file))
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel))
}

function resolve(url) {
  const path = decodeURIComponent(new URL(url, "http://x").pathname)
  for (const [prefix, root] of Object.entries(roots)) {
    if (!path.startsWith(prefix)) continue
    const file = resolvePath(root, `./${path.slice(prefix.length) || "index.html"}`)
    return inside(root, file) ? file : null
  }
  return null
}

/** `/bundle/src/worker.js` → packages/webgllm/src/worker.ts, bundled. */
async function bundle(path, res) {
  const entry = resolvePath(
    pkg,
    `./${path.replace(/^\/bundle\//, "").replace(/\.js$/, ".ts")}`
  )
  if (!inside(pkg, entry)) return res.writeHead(404, isolation).end()
  try {
    const out = await build({
      entryPoints: [entry],
      bundle: true,
      format: "esm",
      platform: "browser",
      target: "es2022",
      // The package's own source entry, if a page imports "webgllm"; its
      // dependencies resolve from packages/webgllm/node_modules.
      conditions: ["webgllm/source"],
      nodePaths: [join(pkg, "node_modules")],
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
  if (
    req.method === "GET" &&
    path.endsWith(".js") &&
    path.startsWith("/bundle/")
  )
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
}).listen(port, process.env.HOST ?? "127.0.0.1", () =>
  console.log(`http://localhost:${port}/`)
)
