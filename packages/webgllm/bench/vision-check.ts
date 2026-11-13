/**
 * The vision encoder (src/vision.ts) against reference features: loads an
 * encoder under `/models` (an ONNX graph with its data file beside it, or a
 * GGUF projector), encodes the patches in `/models/<image>.pixels` (merge
 * order, as the engine cuts them) and compares with
 * `/models/<image>.features` (onnxruntime's, from the same patches).
 * `?encoder=vision08/vision_encoder_q4f16.onnx&image=circle-256&grid=16x16&rounds=3`
 * (`?dir=vision08` is that encoder path).
 */
import { blobSource } from "../src/byte-source.js"
import { Gpu } from "../src/gpu/gl.js"
import { loadVision, planVision } from "../src/load-vision.js"
import { loadVisionGguf } from "../src/load-vision-gguf.js"
import { VisionEncoder } from "../src/vision.js"

const out = document.getElementById("out") as HTMLPreElement
const log = (text: string) => {
  out.textContent += `${text}\n`
  void fetch("/log", { method: "POST", body: text }).catch(() => undefined)
}
const params = new URLSearchParams(location.search)
const encoderPath = params.get("encoder") ?? `${params.get("dir") ?? "vision08"}/vision_encoder_q4f16.onnx`
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
  const encoder = new VisionEncoder(gpu)
  // The encoder's pieces, from its file each time they're asked for.
  let pieces: () => ReturnType<typeof loadVision>
  if (/\.gguf$/i.test(encoderPath)) {
    const source = blobSource(await (await fetch(`/models/${encoderPath}`)).blob())
    pieces = () => loadVisionGguf(source)
  } else {
    const graph = await bytes(`/models/${encoderPath}`)
    const dataFile = planVision(graph).reads.find((r) => r.tensor.external)?.tensor.external?.location
    const folder = encoderPath.slice(0, encoderPath.lastIndexOf("/") + 1)
    const data = blobSource(await (await fetch(`/models/${folder}${dataFile}`)).blob())
    pieces = () => loadVision({ graph, data })
  }
  let t = performance.now()
  await encoder.load(pieces())
  await gpu.finish()
  log(`loaded in ${Math.round(performance.now() - t)} ms: ${(encoder.bytes / 2 ** 20).toFixed(1)} MiB of textures; ${JSON.stringify(encoder.config)}`)
  const pixels = new Float32Array((await bytes(`/models/${image}.pixels`)).buffer)
  const ref = new Float32Array((await bytes(`/models/${image}.features`)).buffer)
  const tokens = (gridH * gridW) / (encoder.config.merge * encoder.config.merge)
  if (pixels.length !== gridH * gridW * encoder.config.patchValues)
    throw new Error(`${image}.pixels has ${pixels.length} values; a ${gridH}×${gridW} grid needs ${gridH * gridW * encoder.config.patchValues}`)
  if (ref.length !== tokens * encoder.config.outWidth)
    throw new Error(`${image}.features has ${ref.length} values; ${tokens} tokens need ${tokens * encoder.config.outWidth}`)
  let whole: Float32Array | null = null
  for (let r = 0; r < rounds; r++) {
    t = performance.now()
    const got = await encoder.encode({ pixels, gridH, gridW })
    whole = got
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
    if (!(Number.isFinite(maxAbs) && Number.isFinite(num) && Number.isFinite(worstCos)))
      throw new Error(`round ${r}: the features aren't finite (max abs ${maxAbs}, worst cosine ${worstCos})`)
    log(`round ${r}: ${got.length / W} tokens in ${ms.toFixed(0)} ms; max abs ${maxAbs.toFixed(4)}, relative L2 ${Math.sqrt(num / den).toFixed(5)}, worst row cosine ${worstCos.toFixed(6)}; first ${Array.from(got.slice(0, 4), (v) => v.toFixed(4)).join(" ")} vs ${Array.from(ref.slice(0, 4), (v) => v.toFixed(4)).join(" ")}; worst rows ${cosines.sort((a, b) => a[1] - b[1]).slice(0, 6).map(([r, c]) => `${r}:${c.toFixed(5)}`).join(" ")}`)
  }
  encoder.dispose()
  // The same view as a request encodes it: the encoder's weights a piece
  // at a time (encodeStream), against the loaded encoder's rows, and
  // against loading and encoding one after the other for time.
  for (let r = 0; r < rounds; r++) {
    t = performance.now()
    const loaded = new VisionEncoder(gpu)
    await loaded.load(pieces())
    await loaded.encode({ pixels, gridH, gridW })
    loaded.dispose()
    const wholeMs = performance.now() - t
    const before = gpu.bytes
    t = performance.now()
    const [streamed] = await new VisionEncoder(gpu).encodeStream(pieces(), [{ pixels, gridH, gridW }])
    const streamMs = performance.now() - t
    let differ = 0
    for (let i = 0; i < streamed.length; i++) if (streamed[i] !== whole![i]) differ++
    log(`streamed ${r}: ${streamMs.toFixed(0)} ms (load then encode: ${wholeMs.toFixed(0)} ms); ${differ} of ${streamed.length} values differ from the loaded encoder's; textures back to ${gpu.bytes === before ? "where they were" : `${gpu.bytes - before} bytes more`}`)
    if (differ) throw new Error("The streamed encode's rows aren't the loaded encoder's.")
  }
  log("ALL DONE")
} catch (error) {
  log(`ERROR: ${error instanceof Error ? `${error.message}\n${error.stack}` : String(error)}`)
}
