/**
 * The vision encoder (src/vision.ts) against reference features: loads
 * `/models/<dir>/vision_encoder_q4f16.onnx` (+ data), encodes the patches in
 * `/models/<image>.pixels` and compares with `/models/<image>.features`
 * (onnxruntime's, from the same patches). `?dir=vision08&image=circle-256&grid=16x16&rounds=3`.
 */
import { blobSource } from "../src/byte-source.js"
import { Gpu } from "../src/gpu/gl.js"
import { loadVision } from "../src/load-vision.js"
import { VisionEncoder } from "../src/vision.js"

const out = document.getElementById("out") as HTMLPreElement
const log = (text: string) => {
  out.textContent += `${text}\n`
  void fetch("/log", { method: "POST", body: text }).catch(() => undefined)
}
const params = new URLSearchParams(location.search)
const dir = params.get("dir") ?? "vision08"
const image = params.get("image") ?? "circle-256"
const [gridH, gridW] = (params.get("grid") ?? "16x16").split("x").map(Number)
const rounds = Number(params.get("rounds") ?? 3)

async function bytes(path: string) {
  const response = await fetch(path)
  if (!response.ok) throw new Error(`${path}: ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
}

try {
  log(navigator.userAgent)
  const gpu = new Gpu(document.getElementById("gl") as HTMLCanvasElement)
  const graph = await bytes(`/models/${dir}/vision_encoder_q4f16.onnx`)
  const data = blobSource(await (await fetch(`/models/${dir}/vision_encoder_q4f16.onnx_data`)).blob())
  const encoder = new VisionEncoder(gpu)
  let t = performance.now()
  await encoder.load(loadVision({ graph, data }))
  await gpu.finish()
  log(`loaded in ${Math.round(performance.now() - t)} ms: ${(encoder.bytes / 2 ** 20).toFixed(1)} MiB of textures; ${JSON.stringify(encoder.config)}`)
  const pixels = new Float32Array((await bytes(`/models/${image}.pixels`)).buffer)
  const ref = new Float32Array((await bytes(`/models/${image}.features`)).buffer)
  for (let r = 0; r < rounds; r++) {
    t = performance.now()
    const got = await encoder.encode({ pixels, gridH, gridW })
    const ms = performance.now() - t
    let maxAbs = 0, num = 0, den = 0, worstCos = 1
    const cosines: Array<[number, number]> = []
    const W = encoder.config.outWidth
    for (let row = 0; row < got.length / W; row++) {
      let dot = 0, a2 = 0, b2 = 0
      for (let i = row * W; i < (row + 1) * W; i++) {
        const d = got[i] - ref[i]
        maxAbs = Math.max(maxAbs, Math.abs(d))
        num += d * d
        den += ref[i] * ref[i]
        dot += got[i] * ref[i]
        a2 += got[i] * got[i]
        b2 += ref[i] * ref[i]
      }
      worstCos = Math.min(worstCos, dot / Math.sqrt(a2 * b2))
      cosines.push([row, dot / Math.sqrt(a2 * b2)])
    }
    log(`round ${r}: ${got.length / W} tokens in ${ms.toFixed(0)} ms; max abs ${maxAbs.toFixed(4)}, relative L2 ${Math.sqrt(num / den).toFixed(5)}, worst row cosine ${worstCos.toFixed(6)}; first ${Array.from(got.slice(0, 4), (v) => v.toFixed(4)).join(" ")} vs ${Array.from(ref.slice(0, 4), (v) => v.toFixed(4)).join(" ")}; worst rows ${cosines.sort((a, b) => a[1] - b[1]).slice(0, 6).map(([r, c]) => `${r}:${c.toFixed(5)}`).join(" ")}`)
  }
  encoder.dispose()
  log("ALL DONE")
} catch (error) {
  log(`ERROR: ${error instanceof Error ? `${error.message}\n${error.stack}` : String(error)}`)
}
