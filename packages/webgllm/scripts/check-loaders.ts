/**
 * Checks the GGUF and ONNX weight loaders on the real files:
 *
 * - every piece has the shapes weights.ts asks for, and layer types match
 *   the config;
 * - GGUF: sampled rows of every matrix dequantize to exactly the raw Q4_0
 *   blocks' values (through the q/k head reordering of full-attention
 *   layers); the Q6_K embedding's 8-bit error, over the whole matrix and
 *   against an independent Q6_K dequantization;
 * - ONNX: sampled rows dequantize to exactly (q - zp) * scale computed from
 *   the raw MatMulNBits / GatherBlockQuantized bytes;
 * - cross-format: norms nearly equal, and dequantized rows of each matrix
 *   kind highly correlated (with shifted-row controls), embedding rows
 *   matched through the two vocabularies' token strings;
 * - time and peak memory of each full load.
 *
 * The src files import each other with .js specifiers, so bundle first:
 *
 *   packages/webgl/node_modules/.bin/esbuild packages/webgl/scripts/check-loaders.ts \
 *     --bundle --platform=node --format=esm --outfile=/tmp/check-loaders.mjs
 *   node --expose-gc /tmp/check-loaders.mjs GGUF_FILE ONNX_DIR
 *
 * ONNX_DIR holds onnx/model_q4f16.onnx(+_data), config.json and
 * tokenizer.json. Exits 1 if a check fails.
 */

import { execFileSync } from "node:child_process"
import { open, readFile } from "node:fs/promises"
import { join } from "node:path"
import {
  readGgufArray,
  readGgufHeader,
  type GgufHeader,
  type GgufLazyArray
} from "../src/gguf.js"
import { headPermutation, loadGguf } from "../src/load-gguf.js"
import { loadOnnx } from "../src/load-onnx.js"
import { parseOnnxModel, type OnnxGraph } from "../src/onnx.js"
import type { RequantStats } from "../src/repack.js"
import type {
  ByteSource,
  LayerWeights,
  ModelConfig,
  QuantMatrix,
  WeightPiece
} from "../src/weights.js"

const timeOnly = process.argv.indexOf("--time")
const timeFormat = timeOnly > 0 ? process.argv[timeOnly + 1] : undefined
const [ggufPath, onnxDir] = process.argv
  .slice(2)
  .filter((a, i, all) => a !== "--time" && all[i - 1] !== "--time")
if (!ggufPath || !onnxDir) {
  console.error("usage: check-loaders GGUF_FILE ONNX_DIR")
  process.exit(2)
}
const onnxGraphPath = join(onnxDir, "onnx/model_q4f16.onnx")
const onnxDataPath = join(onnxDir, "onnx/model_q4f16.onnx_data")

let failures = 0
function fail(message: string): void {
  failures++
  console.log(`  FAIL ${message}`)
}
function expect(condition: boolean, message: string): void {
  if (!condition) fail(message)
}

// ---------------------------------------------------------------- sources

async function fileSource(
  path: string
): Promise<ByteSource & { close: () => Promise<void> }> {
  const handle = await open(path, "r")
  const { size } = await handle.stat()
  return {
    size,
    read: async (offset, length) => {
      const out = new Uint8Array(length)
      let done = 0
      while (done < length) {
        const { bytesRead } = await handle.read(
          out,
          done,
          length - done,
          offset + done
        )
        if (bytesRead === 0) throw new Error(`short read at ${offset + done}`)
        done += bytesRead
      }
      return out
    },
    close: () => handle.close()
  }
}

// --------------------------------------------- independent decoders (not src)

function half(h: number): number {
  const s = h & 0x8000 ? -1 : 1
  const e = (h >> 10) & 31
  const m = h & 1023
  if (e === 0) return s * m * 2 ** -24
  if (e === 31) return m ? NaN : s * Infinity
  return s * (1 + m / 1024) * 2 ** (e - 15)
}

/** A QuantMatrix row, decoded from the layout described in weights.ts. */
function decodeRow(m: QuantMatrix, n: number): Float64Array {
  const blocks = m.K / 32
  const out = new Float64Array(m.K)
  const g = n >> 2
  const r = n & 3
  for (let b = 0; b < blocks; b++) {
    const idx = (g * blocks + b) * 4 + r
    const scale = half(m.scales[idx])
    if (m.bits === 4) {
      const zero = m.zeros ? m.zeros[idx] : 8
      for (let w = 0; w < 4; w++) {
        const word = m.quants[idx * 4 + w]
        for (let k = 0; k < 4; k++) {
          const byte = (word >>> (8 * k)) & 255
          const j = w * 4 + k
          out[b * 32 + j] = scale * ((byte & 15) - zero)
          out[b * 32 + j + 16] = scale * ((byte >> 4) - zero)
        }
      }
    } else {
      for (let w = 0; w < 8; w++) {
        const word = m.quants[idx * 8 + w]
        for (let k = 0; k < 4; k++)
          out[b * 32 + w * 4 + k] = scale * (((word >>> (8 * k)) << 24) >> 24)
      }
    }
  }
  return out
}

/** A raw Q4_0 row (blocks of fp16 d + 16 bytes: w[j] = d(lo - 8), w[j + 16] = d(hi - 8)). */
function q4_0Row(raw: Uint8Array, n: number, K: number): Float64Array {
  const blocks = K / 32
  const out = new Float64Array(K)
  for (let b = 0; b < blocks; b++) {
    const p = (n * blocks + b) * 18
    const d = half(raw[p] | (raw[p + 1] << 8))
    for (let j = 0; j < 16; j++) {
      const q = raw[p + 2 + j]
      out[b * 32 + j] = d * ((q & 15) - 8)
      out[b * 32 + j + 16] = d * ((q >> 4) - 8)
    }
  }
  return out
}

/** A raw MatMulNBits row: element e of a block is nibble e & 1 (low first) of byte e >> 1. */
function nbitsRow(
  quant: Uint8Array,
  scales: Uint8Array,
  zp: Uint8Array | null,
  n: number,
  K: number
): Float64Array {
  const blocks = K / 32
  const zpStride = Math.ceil(blocks / 2)
  const out = new Float64Array(K)
  for (let b = 0; b < blocks; b++) {
    const s = half(
      scales[(n * blocks + b) * 2] | (scales[(n * blocks + b) * 2 + 1] << 8)
    )
    const z = zp ? (zp[n * zpStride + (b >> 1)] >> ((b & 1) * 4)) & 15 : 8
    for (let e = 0; e < 32; e++) {
      const byte = quant[(n * blocks + b) * 16 + (e >> 1)]
      const q = e & 1 ? byte >> 4 : byte & 15
      out[b * 32 + e] = (q - z) * s
    }
  }
  return out
}

/** ggml dequantize_row_q6_K, in float32 arithmetic. */
function q6kRow(raw: Uint8Array, n: number, K: number): Float64Array {
  const out = new Float64Array(K)
  const f = Math.fround
  for (let s = 0; s < K / 256; s++) {
    const base = (n * (K / 256) + s) * 210
    const d = half(raw[base + 208] | (raw[base + 209] << 8))
    for (let half128 = 0; half128 < 2; half128++) {
      const ql = base + 64 * half128
      const qh = base + 128 + 32 * half128
      const sc = base + 192 + 8 * half128
      const y = s * 256 + 128 * half128
      const scale = (i: number) => (raw[sc + i] << 24) >> 24
      for (let l = 0; l < 32; l++) {
        const is = Math.floor(l / 16)
        const q1 = ((raw[ql + l] & 0xf) | (((raw[qh + l] >> 0) & 3) << 4)) - 32
        const q2 =
          ((raw[ql + l + 32] & 0xf) | (((raw[qh + l] >> 2) & 3) << 4)) - 32
        const q3 = ((raw[ql + l] >> 4) | (((raw[qh + l] >> 4) & 3) << 4)) - 32
        const q4 =
          ((raw[ql + l + 32] >> 4) | (((raw[qh + l] >> 6) & 3) << 4)) - 32
        out[y + l] = f(f(d * scale(is)) * q1)
        out[y + l + 32] = f(f(d * scale(is + 2)) * q2)
        out[y + l + 64] = f(f(d * scale(is + 4)) * q3)
        out[y + l + 96] = f(f(d * scale(is + 6)) * q4)
      }
    }
  }
  return out
}

function f16Array(bytes: Uint8Array): Float64Array {
  const out = new Float64Array(bytes.length / 2)
  for (let i = 0; i < out.length; i++)
    out[i] = half(bytes[2 * i] | (bytes[2 * i + 1] << 8))
  return out
}

function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length
  let ma = 0
  let mb = 0
  for (let i = 0; i < n; i++) {
    ma += a[i]
    mb += b[i]
  }
  ma /= n
  mb /= n
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma
    const y = b[i] - mb
    sab += x * y
    saa += x * x
    sbb += y * y
  }
  return sab / Math.sqrt(saa * sbb)
}

function sampleRows(N: number, count = 48): Array<number> {
  const rows = new Set<number>([0, 1, 2, 3, N - 4, N - 3, N - 2, N - 1])
  const stride = Math.max(1, Math.floor(N / count))
  for (let r = 5; r < N; r += stride) rows.add(r)
  return [...rows].filter((r) => r >= 0 && r < N).sort((x, y) => x - y)
}

// ------------------------------------------------------------ shape checks

function checkMatrix(
  label: string,
  m: QuantMatrix,
  N: number,
  K: number,
  bits: 4 | 8
): void {
  const blocks = (N * K) / 32
  expect(
    m.N === N && m.K === K && m.bits === bits,
    `${label}: ${m.N}x${m.K} ${m.bits}-bit, expected ${N}x${K} ${bits}-bit`
  )
  expect(
    m.quants.length === blocks * (bits === 4 ? 4 : 8),
    `${label}: quants length ${m.quants.length}`
  )
  expect(
    m.scales.length === blocks,
    `${label}: scales length ${m.scales.length}`
  )
  if (m.zeros) {
    expect(
      bits === 4 && m.zeros.length === blocks,
      `${label}: zeros length ${m.zeros.length}`
    )
    expect(
      m.zeros.every((z) => z <= 15),
      `${label}: zero point above 15`
    )
  }
  let bad = 0
  for (let i = 0; i < m.scales.length; i++)
    if (((m.scales[i] >> 10) & 31) === 31) bad++
  expect(bad === 0, `${label}: ${bad} scales are Inf/NaN`)
}

function checkVector(label: string, v: Float32Array, length: number): void {
  expect(
    v instanceof Float32Array && v.length === length,
    `${label}: length ${v.length}, expected ${length}`
  )
  expect(v.every(Number.isFinite), `${label}: non-finite values`)
}

function checkLayer(c: ModelConfig, i: number, l: LayerWeights): void {
  const {
    hiddenSize: D,
    intermediateSize: F,
    numHeads,
    numKvHeads,
    headDim
  } = c
  const at = `layer ${i}`
  expect(
    l.type === c.layerTypes[i],
    `${at}: type ${l.type}, config says ${c.layerTypes[i]}`
  )
  for (const k of [
    "operatorNorm",
    "postOperatorNorm",
    "ffnNorm",
    "postFfnNorm"
  ] as const)
    checkVector(`${at} ${k}`, l[k], D)
  checkMatrix(`${at} gateUp`, l.gateUp, 2 * F, D, 4)
  checkMatrix(`${at} down`, l.down, D, F, 4)
  if (l.type === "conv") {
    expect(
      !!l.conv && !l.attention,
      `${at}: conv layer without conv weights (or with attention)`
    )
    if (!l.conv) return
    checkMatrix(`${at} inProj`, l.conv.inProj, 3 * D, D, 4)
    checkMatrix(`${at} outProj`, l.conv.outProj, D, D, 4)
    checkVector(`${at} kernel`, l.conv.kernel, D * c.convKernel)
  } else {
    expect(
      !!l.attention && !l.conv,
      `${at}: attention layer without attention weights (or with conv)`
    )
    if (!l.attention) return
    checkMatrix(
      `${at} qkvg`,
      l.attention.qkvg,
      numHeads * headDim + 2 * numKvHeads * headDim + numHeads,
      D,
      4
    )
    checkMatrix(`${at} o`, l.attention.o, D, numHeads * headDim, 4)
    checkVector(`${at} qNorm`, l.attention.qNorm, headDim)
    checkVector(`${at} kNorm`, l.attention.kNorm, headDim)
  }
}

function checkConfig(c: ModelConfig): void {
  for (const [key, value] of Object.entries(c)) {
    if (key === "format" || key === "layerTypes") continue
    expect(
      typeof value === "number" && Number.isFinite(value),
      `config.${key} = ${String(value)}`
    )
  }
  expect(c.layerTypes.length === c.numLayers, "config.layerTypes length")
}

// ------------------------------------------------------- timing and memory

function mb(bytes: number): string {
  return `${(bytes / 2 ** 20).toFixed(0)} MB`
}

async function timedLoad(
  label: string,
  pieces: () => AsyncGenerator<WeightPiece>
): Promise<void> {
  const gc = (globalThis as { gc?: () => void }).gc
  gc?.()
  const base = process.memoryUsage()
  let peakRss = base.rss
  let peakBuffers = base.arrayBuffers
  let peakHeap = base.heapUsed
  const t0 = performance.now()
  let tLayers = 0
  let tEmbeddingStart = 0
  let count = 0
  const sample = () => {
    const m = process.memoryUsage()
    peakRss = Math.max(peakRss, m.rss)
    peakBuffers = Math.max(peakBuffers, m.arrayBuffers)
    peakHeap = Math.max(peakHeap, m.heapUsed)
  }
  // Also between reads, while a piece is being built.
  const timer = setInterval(sample, 5)
  let layersPeak = 0
  // GC_EACH=1: collect garbage at every piece, so the peak is the live set
  // (the loader's own footprint) rather than what the collector left behind.
  const gcEach = process.env.GC_EACH === "1"
  for await (const piece of pieces()) {
    count++
    if (gcEach) gc?.()
    sample()
    if (piece.kind === "layer" && piece.layer === 0) tLayers = performance.now()
    if (piece.kind === "finalNorm") {
      tEmbeddingStart = performance.now()
      layersPeak = peakBuffers
    }
  }
  const t1 = performance.now()
  clearInterval(timer)
  console.log(
    `${label}: ${count} pieces in ${((t1 - t0) / 1000).toFixed(2)} s ` +
      `(header+config+layer 0: ${((tLayers - t0) / 1000).toFixed(2)} s, embedding: ${((t1 - tEmbeddingStart) / 1000).toFixed(2)} s); ` +
      `peak over baseline: ArrayBuffers +${mb(peakBuffers - base.arrayBuffers)} (up to the embedding: +${mb(layersPeak - base.arrayBuffers)}), heap +${mb(peakHeap - base.heapUsed)}, RSS +${mb(peakRss - base.rss)} (RSS peak ${mb(peakRss)})`
  )
}

// ------------------------------------------------------------------- main

/** Rows that are all zero in both files (dead channels) are counted apart. */
type Corr = {
  sum: number
  min: number
  count: number
  control: number
  controlCount: number
  bothZero: number
  oneZero: number
}
const corr = new Map<string, Corr>()
function addCorr(
  kind: string,
  value: number,
  control: number,
  zeroA = false,
  zeroB = false
): void {
  const c = corr.get(kind) ?? {
    sum: 0,
    min: Infinity,
    count: 0,
    control: 0,
    controlCount: 0,
    bothZero: 0,
    oneZero: 0
  }
  corr.set(kind, c)
  if (zeroA && zeroB) c.bothZero++
  else if (zeroA || zeroB) c.oneZero++
  else {
    c.sum += value
    c.min = Math.min(c.min, value)
    c.count++
  }
  if (Number.isFinite(control)) {
    c.control += control
    c.controlCount++
  }
}
const normDiff = new Map<
  string,
  { maxAbs: number; maxRel: number; corr: number; n: number }
>()
function addNorm(kind: string, a: Float32Array, b: Float32Array): void {
  let maxAbs = 0
  let maxRel = 0
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i])
    maxAbs = Math.max(maxAbs, d)
    maxRel = Math.max(maxRel, d / Math.max(Math.abs(a[i]), 1e-3))
  }
  const s = normDiff.get(kind) ?? { maxAbs: 0, maxRel: 0, corr: 0, n: 0 }
  s.maxAbs = Math.max(s.maxAbs, maxAbs)
  s.maxRel = Math.max(s.maxRel, maxRel)
  s.corr += pearson(a, b)
  s.n++
  normDiff.set(kind, s)
}

/** All zero, or nearly: dead channels hold ±4.5e-36 in the GGUF, which f16 flushes to 0. */
function isZero(v: ArrayLike<number>): boolean {
  for (let i = 0; i < v.length; i++) if (Math.abs(v[i]) > 1e-30) return false
  return true
}

/** Correlates sampled rows of two matrices' row ranges (and a shifted-row control). */
function compareRows(
  kind: string,
  a: QuantMatrix,
  aStart: number,
  b: QuantMatrix,
  bStart: number,
  rows: number
): void {
  for (const r of sampleRows(rows, 24)) {
    const x = decodeRow(a, aStart + r)
    const y = decodeRow(b, bStart + r)
    const shifted = decodeRow(b, bStart + ((r + 1) % rows))
    addCorr(kind, pearson(x, y), pearson(x, shifted), isZero(x), isZero(y))
  }
}

async function main(): Promise<void> {
  console.log(`node ${process.version}`)
  const gguf = await fileSource(ggufPath!)
  const onnxData = await fileSource(onnxDataPath)
  const onnxGraphBytes = new Uint8Array(await readFile(onnxGraphPath))
  const onnxConfigJson = JSON.parse(
    await readFile(join(onnxDir!, "config.json"), "utf8")
  ) as unknown

  // 1. Timing and memory: each loader alone in a fresh process, pieces dropped as they come.
  if (timeFormat === "gguf") return timedLoad("GGUF", () => loadGguf(gguf))
  if (timeFormat === "onnx") {
    return timedLoad("ONNX", () =>
      loadOnnx({
        graph: onnxGraphBytes,
        data: onnxData,
        config: onnxConfigJson
      })
    )
  }
  console.log(
    "\n== Full loads (each in its own process; pieces dropped as they arrive)"
  )
  for (const format of ["gguf", "onnx"]) {
    const args = [
      ...process.execArgv,
      process.argv[1],
      ggufPath!,
      onnxDir!,
      "--time",
      format
    ]
    process.stdout.write(
      execFileSync(process.execPath, args, { encoding: "utf8" })
        .split("\n")
        .slice(1)
        .join("\n")
    )
  }
  const tHeader = performance.now()
  const header = await readGgufHeader(gguf)
  console.log(
    `GGUF header alone: ${(performance.now() - tHeader).toFixed(0)} ms, ${header.headerLength} bytes`
  )

  // 2. Both loaders in lockstep, with per-format raw checks and cross-format comparisons.
  console.log("\n== Checks")
  const graph = parseOnnxModel(onnxGraphBytes)
  const stats: RequantStats = {
    count: 0,
    sumSquaredError: 0,
    sumSquared: 0,
    maxError: 0
  }
  const a = loadGguf(gguf, { header, requantStats: stats })
  const b = loadOnnx({
    graph: onnxGraphBytes,
    data: onnxData,
    config: onnxConfigJson
  })
  let configG: ModelConfig | undefined
  let configO: ModelConfig | undefined
  const order: Array<string> = []
  for (;;) {
    const [pa, pb] = await Promise.all([a.next(), b.next()])
    if (pa.done || pb.done) {
      expect(
        !!pa.done && !!pb.done,
        "loaders yield different numbers of pieces"
      )
      break
    }
    const g = pa.value
    const o = pb.value
    order.push(g.kind === "layer" ? `layer ${g.layer}` : g.kind)
    expect(g.kind === o.kind, `piece kinds differ: ${g.kind} / ${o.kind}`)
    if (g.kind === "config" && o.kind === "config") {
      configG = g.config
      configO = o.config
      checkConfig(configG)
      checkConfig(configO)
      console.log(
        "GGUF config:",
        JSON.stringify({ ...configG, layerTypes: undefined })
      )
      console.log(
        "ONNX config:",
        JSON.stringify({ ...configO, layerTypes: undefined })
      )
      console.log("layer types:", configG.layerTypes.map((t) => t[0]).join(""))
      for (const key of Object.keys(configG) as Array<keyof ModelConfig>) {
        if (
          [
            "format",
            "vocabSize",
            "bosTokenId",
            "eosTokenId",
            "padTokenId"
          ].includes(key)
        )
          continue
        expect(
          JSON.stringify(configG[key]) === JSON.stringify(configO[key]),
          `config.${key} differs: ${JSON.stringify(configG[key])} / ${JSON.stringify(configO[key])}`
        )
      }
    } else if (g.kind === "layer" && o.kind === "layer") {
      expect(g.layer === o.layer, `layer order: ${g.layer} / ${o.layer}`)
      checkLayer(configG!, g.layer, g.value)
      checkLayer(configO!, o.layer, o.value)
      await rawCheckGguf(gguf, header, configG!, g.layer, g.value)
      await rawCheckOnnx(onnxData, graph, configO!, o.layer, o.value)
      crossLayer(configG!, g.value, o.value)
    } else if (g.kind === "finalNorm" && o.kind === "finalNorm") {
      checkVector("GGUF finalNorm", g.value, configG!.hiddenSize)
      checkVector("ONNX finalNorm", o.value, configO!.hiddenSize)
      addNorm("finalNorm", g.value, o.value)
    } else if (g.kind === "embedding" && o.kind === "embedding") {
      checkMatrix(
        "GGUF embedding",
        g.value,
        configG!.vocabSize,
        configG!.hiddenSize,
        8
      )
      checkMatrix(
        "ONNX embedding",
        o.value,
        configO!.vocabSize,
        configO!.hiddenSize,
        4
      )
      await checkEmbeddings(
        gguf,
        header,
        graph,
        onnxData,
        g.value,
        o.value,
        stats
      )
    }
  }
  const expectedOrder = [
    "config",
    ...configG!.layerTypes.map((_, i) => `layer ${i}`),
    "finalNorm",
    "embedding"
  ]
  expect(
    JSON.stringify(order) === JSON.stringify(expectedOrder),
    `piece order: ${order.join(", ")}`
  )

  console.log("\n== Cross-format: norms (GGUF f32 vs ONNX f16)")
  for (const [kind, s] of normDiff) {
    console.log(
      `${kind.padEnd(18)} max |diff| ${s.maxAbs.toExponential(2)}  max rel ${s.maxRel.toExponential(2)}  mean corr ${(s.corr / s.n).toFixed(6)}`
    )
    expect(s.corr / s.n > 0.99, `${kind} norms differ`)
  }
  console.log(
    "\n== Cross-format: dequantized row correlation (sampled rows of every layer)"
  )
  console.log(
    "kind                      rows   mean corr   min corr   shifted control   zero in both / one"
  )
  for (const [kind, c] of corr) {
    const mean = c.sum / c.count
    console.log(
      `${kind.padEnd(24)} ${String(c.count).padStart(5)}   ${mean.toFixed(4)}      ${c.min.toFixed(4)}     ${c.controlCount ? (c.control / c.controlCount).toFixed(4).padStart(7) : "    n/a"}           ${c.bothZero} / ${c.oneZero}`
    )
    if (kind.includes("control"))
      expect(mean < 0.5, `${kind}: control correlates`)
    else
      expect(
        mean > 0.9 && c.min > 0.5 && c.oneZero === 0,
        `${kind}: low correlation, a layout mismatch`
      )
  }

  await gguf.close()
  await onnxData.close()
  console.log(
    failures ? `\n${failures} check(s) FAILED` : "\nall checks passed"
  )
  process.exitCode = failures ? 1 : 0
}

// ------------------------------------------------------------- raw checks

async function readGgufTensor(
  source: ByteSource,
  header: GgufHeader,
  name: string
): Promise<Uint8Array> {
  const info = header.tensors.get(name)!
  return source.read(info.offset, info.byteLength)
}

let ggufRowsChecked = 0
let onnxRowsChecked = 0

function exactRows(
  label: string,
  m: QuantMatrix,
  row0: number,
  rows: number,
  raw: (r: number) => Float64Array,
  srcRow = (r: number) => r
): number {
  let checked = 0
  for (const r of sampleRows(rows)) {
    const got = decodeRow(m, row0 + r)
    const want = raw(srcRow(r))
    let bad = -1
    for (let k = 0; k < got.length; k++) {
      if (got[k] !== want[k]) {
        bad = k
        break
      }
    }
    if (bad >= 0) {
      fail(`${label} row ${r} col ${bad}: ${got[bad]} != ${want[bad]}`)
      return checked
    }
    checked++
  }
  return checked
}

async function rawCheckGguf(
  source: ByteSource,
  header: GgufHeader,
  c: ModelConfig,
  i: number,
  l: LayerWeights
): Promise<void> {
  const blk = `blk.${i}.`
  const D = c.hiddenSize
  const parts: Array<
    [string, QuantMatrix, number, number, number, ((r: number) => number)?]
  > = []
  const full = l.type === "full_attention"
  if (l.conv) {
    parts.push([`${blk}shortconv.in_proj.weight`, l.conv.inProj, 0, 3 * D, D])
    parts.push([`${blk}shortconv.out_proj.weight`, l.conv.outProj, 0, D, D])
  }
  if (l.attention) {
    const q = c.numHeads * c.headDim
    const kv = c.numKvHeads * c.headDim
    const qPerm = headPermutation(c.numHeads, c.headDim, c.ropeDimsFull)
    const kPerm = headPermutation(c.numKvHeads, c.headDim, c.ropeDimsFull)
    parts.push([
      `${blk}attn_q.weight`,
      l.attention.qkvg,
      0,
      q,
      D,
      full ? (r) => qPerm[r] : undefined
    ])
    parts.push([
      `${blk}attn_k.weight`,
      l.attention.qkvg,
      q,
      kv,
      D,
      full ? (r) => kPerm[r] : undefined
    ])
    parts.push([`${blk}attn_v.weight`, l.attention.qkvg, q + kv, kv, D])
    parts.push([
      `${blk}attn_gate.weight`,
      l.attention.qkvg,
      q + 2 * kv,
      c.numHeads,
      D
    ])
    parts.push([`${blk}attn_output.weight`, l.attention.o, 0, D, q])
  }
  parts.push([`${blk}ffn_gate.weight`, l.gateUp, 0, c.intermediateSize, D])
  parts.push([
    `${blk}ffn_up.weight`,
    l.gateUp,
    c.intermediateSize,
    c.intermediateSize,
    D
  ])
  parts.push([`${blk}ffn_down.weight`, l.down, 0, D, c.intermediateSize])
  for (const [name, m, row0, rows, K, srcRow] of parts) {
    const raw = await readGgufTensor(source, header, name)
    ggufRowsChecked += exactRows(
      `GGUF ${name}`,
      m,
      row0,
      rows,
      (r) => q4_0Row(raw, r, K),
      srcRow
    )
  }
  // Norms and the conv kernel: F32 copied as is (q/k norms reordered like their rows).
  const f32 = async (name: string) => {
    const raw = await readGgufTensor(source, header, name)
    return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4)
  }
  const same = (
    label: string,
    got: Float32Array,
    want: ArrayLike<number>,
    perm?: Int32Array
  ) => {
    expect(
      got.every((v, k) => v === want[perm ? perm[k] : k]),
      `GGUF ${label} values`
    )
  }
  same("attn_norm", l.operatorNorm, await f32(`${blk}attn_norm.weight`))
  same(
    "post_attention_norm",
    l.postOperatorNorm,
    await f32(`${blk}post_attention_norm.weight`)
  )
  same("ffn_norm", l.ffnNorm, await f32(`${blk}ffn_norm.weight`))
  same("post_ffw_norm", l.postFfnNorm, await f32(`${blk}post_ffw_norm.weight`))
  if (l.conv)
    same(
      "shortconv.conv",
      l.conv.kernel,
      await f32(`${blk}shortconv.conv.weight`)
    )
  if (l.attention) {
    const perm = full
      ? headPermutation(1, c.headDim, c.ropeDimsFull)
      : undefined
    same(
      "attn_q_norm",
      l.attention.qNorm,
      await f32(`${blk}attn_q_norm.weight`),
      perm
    )
    same(
      "attn_k_norm",
      l.attention.kNorm,
      await f32(`${blk}attn_k_norm.weight`),
      perm
    )
  }
}

async function readOnnxInit(
  data: ByteSource,
  graph: OnnxGraph,
  name: string
): Promise<Uint8Array> {
  const t = graph.initializers.get(name)
  if (!t) throw new Error(`no initializer ${name}`)
  if (t.raw) return t.raw
  return data.read(t.external!.offset, t.external!.length)
}

async function rawCheckOnnx(
  data: ByteSource,
  graph: OnnxGraph,
  c: ModelConfig,
  i: number,
  l: LayerWeights
): Promise<void> {
  const D = c.hiddenSize
  const parts: Array<[string, QuantMatrix, number, number, number, number]> = [] // prefix, matrix, row0, rows, K, src row0
  if (l.conv) {
    parts.push([`conv_in_proj`, l.conv.inProj, 0, 3 * D, D, 0])
    parts.push([`conv_out_proj`, l.conv.outProj, 0, D, D, 0])
  }
  if (l.attention) {
    const q = c.numHeads * c.headDim
    const kv = c.numKvHeads * c.headDim
    parts.push([`attn_q_proj`, l.attention.qkvg, 0, q, D, 0])
    parts.push([`attn_k_proj`, l.attention.qkvg, q, kv, D, 0])
    parts.push([`attn_v_proj`, l.attention.qkvg, q + kv, kv, D, 0])
    // The gate: q_proj rows q.. (the Split's second output, read by the Sigmoid).
    parts.push([`attn_q_proj`, l.attention.qkvg, q + 2 * kv, c.numHeads, D, q])
    parts.push([`attn_o_proj`, l.attention.o, 0, D, q, 0])
  }
  parts.push([`mlp_gate_proj`, l.gateUp, 0, c.intermediateSize, D, 0])
  parts.push([
    `mlp_up_proj`,
    l.gateUp,
    c.intermediateSize,
    c.intermediateSize,
    D,
    0
  ])
  parts.push([`mlp_down_proj`, l.down, 0, D, c.intermediateSize, 0])
  for (const [part, m, row0, rows, K, src0] of parts) {
    const prefix = `model_layers_${i}_${part}_MatMul_weight_`
    const quant = await readOnnxInit(data, graph, `${prefix}quant`)
    const scales = await readOnnxInit(data, graph, `${prefix}scales`)
    const zp = graph.initializers.has(`${prefix}zp`)
      ? await readOnnxInit(data, graph, `${prefix}zp`)
      : null
    onnxRowsChecked += exactRows(
      `ONNX ${prefix} (+${src0})`,
      m,
      row0,
      rows,
      (r) => nbitsRow(quant, scales, zp, r, K),
      (r) => r + src0
    )
  }
  const p = `model.layers.${i}.`
  const f16 = async (name: string) =>
    f16Array(await readOnnxInit(data, graph, name))
  const same = (label: string, got: Float32Array, want: Float64Array) => {
    expect(
      got.every((v, k) => v === want[k]),
      `ONNX ${label} values`
    )
  }
  same(
    "operator_layernorm",
    l.operatorNorm,
    await f16(`${p}operator_layernorm.weight`)
  )
  same(
    "post_operator_layernorm",
    l.postOperatorNorm,
    await f16(`${p}post_operator_layernorm.weight`)
  )
  same("ffn_layernorm", l.ffnNorm, await f16(`${p}ffn_layernorm.weight`))
  same(
    "post_ffn_layernorm",
    l.postFfnNorm,
    await f16(`${p}post_ffn_layernorm.weight`)
  )
  if (l.conv)
    same("conv.conv", l.conv.kernel, await f16(`${p}conv.conv.weight`))
  if (l.attention) {
    same(
      "q_norm",
      l.attention.qNorm,
      await f16(`${p}attn.q_norm.layernorm.weight`)
    )
    same(
      "k_norm",
      l.attention.kNorm,
      await f16(`${p}attn.k_norm.layernorm.weight`)
    )
  }
}

function crossLayer(c: ModelConfig, g: LayerWeights, o: LayerWeights): void {
  const D = c.hiddenSize
  const F = c.intermediateSize
  addNorm("operatorNorm", g.operatorNorm, o.operatorNorm)
  addNorm("postOperatorNorm", g.postOperatorNorm, o.postOperatorNorm)
  addNorm("ffnNorm", g.ffnNorm, o.ffnNorm)
  addNorm("postFfnNorm", g.postFfnNorm, o.postFfnNorm)
  compareRows("w1 (gate)", g.gateUp, 0, o.gateUp, 0, F)
  compareRows("w3 (up)", g.gateUp, F, o.gateUp, F, F)
  compareRows("w2 (down)", g.down, 0, o.down, 0, D)
  if (g.conv && o.conv) {
    compareRows("in_proj B", g.conv.inProj, 0, o.conv.inProj, 0, D)
    compareRows("in_proj C", g.conv.inProj, D, o.conv.inProj, D, D)
    compareRows("in_proj x", g.conv.inProj, 2 * D, o.conv.inProj, 2 * D, D)
    compareRows("control B vs x", g.conv.inProj, 0, o.conv.inProj, 2 * D, D)
    compareRows("out_proj", g.conv.outProj, 0, o.conv.outProj, 0, D)
    const W = c.convKernel
    for (let ch = 0; ch < D; ch += 3) {
      const x = g.conv.kernel.subarray(ch * W, ch * W + W)
      const y = o.conv.kernel.subarray(ch * W, ch * W + W)
      const next = o.conv.kernel.subarray(
        ((ch + 1) % D) * W,
        ((ch + 1) % D) * W + W
      )
      addCorr(
        "conv kernel channel",
        pearson(x, y),
        pearson(x, next),
        isZero(x),
        isZero(y)
      )
    }
    const shifted = new Float32Array(o.conv.kernel.length)
    shifted.set(o.conv.kernel.subarray(W))
    shifted.set(o.conv.kernel.subarray(0, W), shifted.length - W)
    addCorr(
      "conv kernel (whole)",
      pearson(g.conv.kernel, o.conv.kernel),
      pearson(g.conv.kernel, shifted)
    )
  }
  if (g.attention && o.attention) {
    const q = c.numHeads * c.headDim
    const kv = c.numKvHeads * c.headDim
    const ga = g.attention
    const oa = o.attention
    const kind = g.type === "full_attention" ? "full" : "swa"
    compareRows(`q (${kind})`, ga.qkvg, 0, oa.qkvg, 0, q)
    compareRows(`k (${kind})`, ga.qkvg, q, oa.qkvg, q, kv)
    compareRows("v", ga.qkvg, q + kv, oa.qkvg, q + kv, kv)
    compareRows("gate", ga.qkvg, q + 2 * kv, oa.qkvg, q + 2 * kv, c.numHeads)
    compareRows("o", ga.o, 0, oa.o, 0, D)
    // Per quarter of the head dim (rotated dims first after reordering).
    const H = c.headDim
    for (let quarter = 0; quarter < 4; quarter++) {
      for (let h = 0; h < c.numHeads; h += 5) {
        for (let d = quarter * (H / 4); d < (quarter + 1) * (H / 4); d += 7) {
          const r = h * H + d
          const x = decodeRow(ga.qkvg, r)
          // Control: the ONNX row the HF (unreordered) layout would put here.
          const full = g.type === "full_attention"
          const hf = headPermutation(1, H, c.ropeDimsFull)[d]
          addCorr(
            `q ${kind} dims ${quarter * (H / 4)}-${(quarter + 1) * (H / 4) - 1}`,
            pearson(x, decodeRow(oa.qkvg, r)),
            full ? pearson(x, decodeRow(oa.qkvg, h * H + hf)) : Number.NaN
          )
        }
      }
    }
    addNorm(`qNorm (${kind})`, ga.qNorm, oa.qNorm)
    addNorm(`kNorm (${kind})`, ga.kNorm, oa.kNorm)
  }
}

async function checkEmbeddings(
  gguf: ByteSource,
  header: GgufHeader,
  graph: OnnxGraph,
  data: ByteSource,
  g: QuantMatrix,
  o: QuantMatrix,
  stats: RequantStats
): Promise<void> {
  const D = g.K
  console.log("\n== GGUF embedding: Q6_K → 8-bit")
  console.log(
    `whole matrix (${stats.count} weights): max |err| ${stats.maxError.toExponential(3)}, ` +
      `RMS err ${Math.sqrt(stats.sumSquaredError / stats.count).toExponential(3)}, ` +
      `RMS err / RMS weight ${Math.sqrt(stats.sumSquaredError / stats.sumSquared).toExponential(3)}`
  )
  // Independently: dequantize sampled raw Q6_K rows and compare with the 8-bit rows.
  const info = header.tensors.get("token_embd.weight")!
  const rowBytes = info.byteLength / g.N
  let maxErr = 0
  let maxRelBlock = 0
  let sse = 0
  let ss = 0
  let n = 0
  for (const r of sampleRows(g.N, 4000)) {
    const raw = await gguf.read(info.offset + r * rowBytes, rowBytes)
    const want = q6kRow(raw, 0, D)
    const got = decodeRow(g, r)
    for (let b = 0; b < D / 32; b++) {
      let max = 0
      for (let j = 0; j < 32; j++)
        max = Math.max(max, Math.abs(want[b * 32 + j]))
      for (let j = 0; j < 32; j++) {
        const e = Math.abs(got[b * 32 + j] - want[b * 32 + j])
        maxErr = Math.max(maxErr, e)
        if (max > 0) maxRelBlock = Math.max(maxRelBlock, e / max)
        sse += e * e
        ss += want[b * 32 + j] ** 2
        n++
      }
    }
  }
  console.log(
    `independent check, ${n / D} sampled rows: max |err| ${maxErr.toExponential(3)}, RMS err ${Math.sqrt(sse / n).toExponential(3)}, ` +
      `RMS err / RMS weight ${Math.sqrt(sse / ss).toExponential(3)}, max err / block max|w| ${maxRelBlock.toExponential(3)} (0.5/127 = ${(0.5 / 127).toExponential(3)})`
  )
  expect(maxRelBlock < 0.5 / 127 + 2e-3, "Q6_K → 8-bit error above half a step")
  expect(
    Math.abs(maxErr - stats.maxError) / stats.maxError < 1 ||
      maxErr <= stats.maxError,
    "sampled error exceeds the loader's whole-matrix max"
  )

  // ONNX embedding against the raw GatherBlockQuantized bytes.
  const gather = graph.nodes.find((x) => x.opType === "GatherBlockQuantized")!
  const qT = graph.initializers.get(gather.inputs[0])!
  const sT = graph.initializers.get(gather.inputs[2])!
  const zT = graph.initializers.get(gather.inputs[3])!
  let onnxEmbRows = 0
  for (const r of sampleRows(o.N, 500)) {
    const quant = await data.read(qT.external!.offset + r * (D / 2), D / 2)
    const scales = await data.read(sT.external!.offset + r * (D / 16), D / 16)
    const zp = await data.read(zT.external!.offset + r * (D / 64), D / 64)
    onnxEmbRows += exactRows(`ONNX embedding row ${r}`, o, r, 1, () =>
      nbitsRow(quant, scales, zp, 0, D)
    )
  }
  console.log(
    `exact rows: GGUF ${ggufRowsChecked} matrix rows, ONNX ${onnxRowsChecked} matrix rows + ${onnxEmbRows} embedding rows`
  )

  // Cross-format, through the token strings.
  const tokens = await readGgufArray(
    gguf,
    header.metadata.get("tokenizer.ggml.tokens") as GgufLazyArray
  )
  const ggufId = new Map<string, number>()
  tokens.forEach((t, id) => ggufId.set(String(t), id))
  const tok = JSON.parse(
    await readFile(join(onnxDir!, "tokenizer.json"), "utf8")
  ) as {
    model: { vocab: Record<string, number> }
    added_tokens: Array<{ id: number; content: string }>
  }
  const pairs: Array<[number, number]> = []
  for (const [text, id] of Object.entries(tok.model.vocab)) {
    const gid = ggufId.get(text)
    if (gid !== undefined) pairs.push([gid, id])
  }
  for (const { id, content } of tok.added_tokens) {
    const gid = ggufId.get(content)
    if (gid !== undefined) pairs.push([gid, id])
  }
  console.log(`token strings in both vocabularies: ${pairs.length} of ${o.N}`)
  pairs.sort((x, y) => x[1] - y[1])
  let sum = 0
  let min = Infinity
  let control = 0
  let count = 0
  const step = Math.max(1, Math.floor(pairs.length / 3000))
  for (let k = 0; k < pairs.length; k += step) {
    const [gid, oid] = pairs[k]
    const x = decodeRow(g, gid)
    const v = pearson(x, decodeRow(o, oid))
    sum += v
    min = Math.min(min, v)
    control += pearson(x, decodeRow(o, pairs[(k + 1) % pairs.length][1]))
    count++
  }
  console.log(
    `embedding rows matched by token string: ${count} rows, mean corr ${(sum / count).toFixed(4)}, min ${min.toFixed(4)}, control ${(control / count).toFixed(4)}`
  )
  expect(sum / count > 0.9, "embedding rows don't match across formats")
}

main().catch((error: unknown) => {
  console.error(error)
  process.exit(1)
})
