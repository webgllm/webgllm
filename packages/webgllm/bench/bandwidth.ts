/**
 * How fast this GPU streams a large texture through texelFetch, by texel
 * size: the same bytes read as RGBA4 (2 bytes a fetch), RGBA8UI (4),
 * RGBA16UI (8) and RGBA32UI (16). Each fragment sums a run of texels along
 * its row, as a split few-token matmul reads weights. Every draw reads the
 * previous draw's output, so no GPU can skip one (PowerVR drops draws
 * whose output is overwritten unread).
 */
import { Gpu } from "../src/gpu/gl.js"

const log = (text: string) => postMessage({ type: "log", text })

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
`

type Format = { name: string; internal: number; format: number; type: number; bytes: number; float: boolean }

/**
 * "strip": fragment (s, y) reads row y; "rows": fragment (y, s), so
 * neighbours in x read neighbouring rows, as the engine's matmuls do.
 */
type Mode = "strip" | "rows"

function kernel(f: Format, unroll: number, mode: Mode) {
  const sampler = f.float ? "sampler2D" : "usampler2D"
  const acc = f.float ? "vec4" : "uvec4"
  const reads = Array.from(
    { length: unroll },
    (_, i) => `    acc += texelFetch(uW, ivec2(x0 + i + ${i}, y), 0);`
  ).join("\n")
  return `${HEADER}
uniform ${sampler} uW;
uniform sampler2D uPrev;
uniform int uRun;
out vec4 o;
void main() {
  int s = int(gl_FragCoord.${mode === "rows" ? "y" : "x"});
  int y = int(gl_FragCoord.${mode === "rows" ? "x" : "y"});
  int x0 = s * uRun;
  ${acc} acc = ${acc}(0);
  for (int i = 0; i < uRun; i += ${unroll}) {
${reads}
  }
  o = vec4(acc) + texelFetch(uPrev, ivec2(0, 0), 0) * 1e-30;
}`
}

async function main() {
  const canvas = new OffscreenCanvas(1, 1)
  const gpu = new Gpu(canvas)
  const gl = WebGL2RenderingContext
  log(`GPU: ${gpu.limits.renderer}`)
  const formats: Array<Format> = [
    { name: "RGBA4", internal: gl.RGBA4, format: gl.RGBA, type: gl.UNSIGNED_SHORT_4_4_4_4, bytes: 2, float: true },
    { name: "RGBA8", internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, bytes: 4, float: true },
    { name: "RGBA8UI", internal: gl.RGBA8UI, format: gl.RGBA_INTEGER, type: gl.UNSIGNED_BYTE, bytes: 4, float: false },
    { name: "RGBA16UI", internal: gl.RGBA16UI, format: gl.RGBA_INTEGER, type: gl.UNSIGNED_SHORT, bytes: 8, float: false },
    { name: "RGBA32UI", internal: gl.RGBA32UI, format: gl.RGBA_INTEGER, type: gl.UNSIGNED_INT, bytes: 16, float: false }
  ]
  // 64 MiB per texture: past any cache, about one big layer's weights.
  const total = 64 * 2 ** 20
  const width = 4096
  const outs = {
    strip: [0, 1].map(() => gpu.texture(gl.RGBA32F, 128, 8192)),
    rows: [0, 1].map(() => gpu.texture(gl.RGBA32F, 8192, 128))
  }
  for (const f of formats) {
    const rows = total / f.bytes / width
    const pixels =
      f.type === gl.UNSIGNED_BYTE
        ? new Uint8Array(width * rows * 4)
        : f.type === gl.UNSIGNED_INT
          ? new Uint32Array(width * rows * 4)
          : new Uint16Array(width * rows * (f.bytes === 2 ? 1 : 4))
    // Random contents: some GPUs compress constant textures.
    const words = new Uint32Array(pixels.buffer)
    let seed = 1
    for (let i = 0; i < words.length; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0
      words[i] = seed
    }
    const w = gpu.texture(f.internal, width, rows, { format: f.format, type: f.type, pixels })
    const cases: Array<[Mode, number]> = [["strip", 512], ["rows", 32], ["rows", 128], ["rows", 512]]
    for (const [mode, run] of cases) {
      for (const unroll of [8]) {
        const program = gpu.program(kernel(f, unroll, mode))
        const draw = (i: number) =>
          gpu.run({
            name: "bw",
            program,
            inputs: { uW: w, uPrev: outs[mode][(i + 1) % 2] },
            uniforms: { uRun: { int: run } },
            output: outs[mode][i % 2],
            rect:
              mode === "rows"
                ? { x: 0, y: 0, width: rows, height: width / run }
                : { x: 0, y: 0, width: width / run, height: rows }
          })
        draw(0)
        draw(1)
        await gpu.finish()
        let best = Infinity
        for (let round = 0; round < 4; round++) {
          const n = 10
          const t = performance.now()
          for (let i = 0; i < n; i++) draw(i)
          await gpu.finish()
          best = Math.min(best, (performance.now() - t) / n)
        }
        const fetches = width * rows
        log(
          `BW ${f.name.padEnd(8)} ${mode.padEnd(5)} run ${String(run).padStart(3)} (${(fetches / run / 1000).toFixed(0)}k fragments): ${best.toFixed(2)} ms, ${(total / best / 1e6).toFixed(1)} GB/s, ${(fetches / best / 1e6).toFixed(2)} G fetches/s`
        )
      }
    }
    gpu.delete(w)
  }
}

main().then(
  () => postMessage({ type: "done" }),
  (e: unknown) => postMessage({ type: "error", message: String(e) })
)
