// TesseraGpu's work textures on a recording GPU (no shaders run): the
// bytes it claims for a batch are the bytes it makes, a batch too tall for
// the GPU's textures runs in parts, and a texture the GPU can't make
// leaves the ones it has in place.
import assert from "node:assert/strict"
import { test } from "node:test"

import type { Pass, Texture } from "./gpu/types.js"
import { TesseraGpu } from "./tessera.js"

type FakeTexture = Texture & { deleted: boolean }

/** A GPU that makes and deletes textures and records passes, refusing textures past its limit or while `refuse` is set. */
function recordingGpu(maxTextureSize: number) {
  let ids = 0
  const gpu = {
    refuse: false,
    used: [] as Array<FakeTexture>,
    limits: { maxTextureSize, maxDrawBuffers: 8, maxTextureUnits: 16, renderer: "recording" },
    texture(internal: number, width: number, height: number): FakeTexture {
      if (width > maxTextureSize || height > maxTextureSize || gpu.refuse) throw new Error(`no ${width} × ${height} texture`)
      return { id: ++ids, tex: null, width, height, internal, bytes: width * height * 16, deleted: false }
    },
    delete(t: FakeTexture) {
      t.deleted = true
    },
    write(t: FakeTexture) {
      gpu.used.push(t)
    },
    zero(t: FakeTexture) {
      gpu.used.push(t)
    },
    async read(t: FakeTexture, rect: { width: number; height: number }) {
      gpu.used.push(t)
      return new Float32Array(rect.width * rect.height * 4)
    },
    program: () => ({ samplers: [] }),
    async runAsync(pass: Pass) {
      for (const t of [...Object.values(pass.inputs), ...[pass.output].flat()]) gpu.used.push(t as FakeTexture)
    },
    async pace() {}
  }
  return gpu
}

const tex = (width: number, height: number): FakeTexture => ({ id: -1, tex: null, width, height, internal: 0, bytes: 0, deleted: false })
const linear = (N: number, K: number) => ({ N, K, axis: "out" as const, w: tex(K / 16, N), s: tex(N / 4, 1), bias: null })
const block = () => ({ qkv: linear(1536, 512), qNorm: tex(16, 1), kNorm: tex(16, 1), proj: linear(512, 512), w12: linear(3072, 512), w3: linear(512, 1536), kernel: tex(384, 9), convBias: tex(384, 1), modBias: tex(768, 1) })

/** A TesseraGpu whose weights are stand-ins (the shapes are what count), on `gpu`. */
function tessera(gpu: ReturnType<typeof recordingGpu>) {
  const t = new TesseraGpu(gpu as never)
  const norm = { weight: tex(128, 1), bias: tex(128, 1) }
  Object.assign(t as object, {
    den: {
      embed: tex(128, 513), pos: tex(128, 256), scProj: linear(512, 512), textNorm: tex(128, 1), textProj: linear(512, 512), nullText: tex(128, 24),
      poolNorm: tex(128, 1), pool1: linear(512, 512), pool3: linear(512, 512), nullCond: new Float32Array(512), ratio0: linear(512, 16),
      ratio2: linear(512, 512), registers: tex(128, 4), tBlock: linear(3072, 512), blocksIn: [block(), block()], downNorm: tex(512, 1),
      down: linear(512, 2048), blocksCore: [block()], up: linear(2048, 512), blocksOut: [block()], finalMod: linear(1024, 512), head: linear(512, 512),
      ropeIn: tex(16, 284), ropeCore: tex(16, 92)
    },
    clip: {
      layers: [{ ln1: norm, q: linear(512, 512), k: linear(512, 512), v: linear(512, 512), out: linear(512, 512), ln2: norm, fc1: linear(2048, 512), fc2: linear(512, 2048) }],
      final: norm,
      tokens: { q: new Int8Array(4 * 512), scale: new Float32Array(4), vocab: 4 },
      positions: new Float32Array(24 * 512),
      width: 512,
      eps: 1e-5
    }
  })
  return t
}

const canvases = (batch: number) => ({ tokens: new Int32Array(batch * 256).fill(512), ratio: new Float32Array(batch).fill(1), sc: new Float32Array(batch * 256 * 512) })

test("the work bytes claimed for a batch are the work textures it makes", async () => {
  for (const batch of [2, 8]) {
    const gpu = recordingGpu(16384)
    const t = tessera(gpu)
    const text = await t.encodeText(new Array(24).fill(0))
    await t.setText(text.states, text.pooled)
    const c = canvases(batch)
    await t.denoise(c.tokens, c.ratio, c.sc, batch, batch / 2)
    assert.equal(t.bytes.work, t.workBytesFor(batch), `batch ${batch}`)
  }
})

test("a batch too tall for the GPU's textures runs in parts, each within its limit", async () => {
  const gpu = recordingGpu(4096)
  const t = tessera(gpu)
  assert.equal(t.maxBatch, 14)
  await t.setText(new Float32Array(24 * 512), new Float32Array(512))
  const c = canvases(16)
  const logits = await t.denoise(c.tokens, c.ratio, c.sc, 16, 8)
  assert.equal(logits.length, 16 * 256 * 512)
  assert.equal(t.workBytesFor(16), t.workBytesFor(14))
})

test("a texture the GPU can't make leaves the old one in place: the next, smaller batch runs on textures that exist", async () => {
  const gpu = recordingGpu(16384)
  const t = tessera(gpu)
  await t.setText(new Float32Array(24 * 512), new Float32Array(512))
  const small = canvases(2)
  await t.denoise(small.tokens, small.ratio, small.sc, 2, 1)
  // The GPU refuses what a larger batch needs.
  gpu.refuse = true
  const large = canvases(8)
  await assert.rejects(t.denoise(large.tokens, large.ratio, large.sc, 8, 4), /no .* texture/)
  gpu.refuse = false
  gpu.used.length = 0
  await t.denoise(small.tokens, small.ratio, small.sc, 2, 1)
  assert.ok(gpu.used.length > 0)
  assert.deepEqual(gpu.used.filter((u) => u.deleted).map((u) => `${u.width} × ${u.height}`), [], "a pass used a deleted texture")
})
