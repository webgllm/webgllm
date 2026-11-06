import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import { attentionError } from "./attention-check.js"

function fake(linear = false) {
  return {
    config: {
      vocabSize: 128,
      headDim: 1,
      numHeads: 1,
      numKvHeads: 1,
      ...(linear && { linear: {} })
    },
    layers: [{ type: "full_attention" }, { type: "full_attention" }],
    contextLength: 8192,
    work: null,
    position: 0,
    forwards: 0,
    chunkErrors: null as number[] | null,
    debug: null,
    kvDebug: null,
    reset() {
      this.position = 0
    },
    async forward(tokens: number[]) {
      this.forwards++
      this.position += tokens.length
      this.chunkErrors?.push(0.1)
      return new Float32Array(1)
    },
    async recurrenceError() {
      return 0.7
    },
    async panelError(_tokens: number[], reuse: boolean) {
      if (!linear) {
        assert.equal(reuse, true)
        assert.equal(this.position, 40)
      }
      this.position = 0
      return 0.2
    },
    gpu: {
      delete() {
        return undefined
      }
    }
  }
}
test("startup checks reuse real reference inputs and leave no model cache", async () => {
  const state = fake()
  const error = await Engine.prototype.selfCheck.call(
    state as unknown as Engine
  )
  assert.equal(error, 0.2)
  assert.equal(state.forwards, 1)
  assert.equal(state.position, 0)
  assert.equal(state.chunkErrors, null)
})
test("recurrent models retain their existing numerical rejection result", async () => {
  const state = fake(true)
  assert.equal(
    await Engine.prototype.selfCheck.call(state as unknown as Engine),
    0.7
  )
})
test("a failed reference forward always clears its cache even when reuse was requested", async () => {
  const state = fake()
  state.forward = async () => {
    state.position = 40
    throw new Error("GPU read failed")
  }
  await assert.rejects(
    attentionError(state as unknown as Engine, [1, 2, 3], true),
    /GPU read failed/
  )
  assert.equal(state.position, 0)
})

test("the fused startup check conservatively converts GPU-relative error to a CPU-relative bound", async () => {
  const state = fake()
  state.panelError = async () => 0
  const error = await Engine.prototype.selfCheck.call(
    state as unknown as Engine
  )
  assert.ok(Math.abs(error - 0.1 / 0.9) < 1e-12)
  assert.equal(state.position, 0)
})

test("dense startup checks retain every attention layout and restore layer indexing on failure", async () => {
  const state = fake()
  const layers = [
    { type: "full_attention" },
    { type: "full_attention" },
    { type: "sliding_attention" },
    { type: "sliding_attention", attention: { ring: true } },
    { type: "full_attention" }
  ]
  Object.assign(state, { layers })
  const config = state.config
  state.forward = async () => {
    assert.equal(state.layers.length, 4)
    assert.deepEqual(Reflect.get(state.config, "layerTypes"), [
      "full_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention"
    ])
    throw new Error("reference read failed")
  }
  await assert.rejects(
    Engine.prototype.selfCheck.call(state as unknown as Engine),
    /reference read failed/
  )
  assert.equal(state.layers, layers)
  assert.equal(state.config, config)
  assert.equal(state.position, 0)
})

/**
 * selfCheck's state with the real panelError, whose forwards fail past
 * `contextLength` as the engine's do.
 */
function bounded(contextLength: number, linear = false) {
  const state = fake(linear)
  const panelError = Reflect.get(Engine.prototype, "panelError") as (...args: unknown[]) => Promise<number>
  const forward = async (tokens: number[]) => {
    assert.ok(tokens.length > 0, "a forward reads at least a token")
    if (state.position + tokens.length > contextLength)
      throw new Error(`The cache holds ${contextLength} tokens; this would need ${state.position + tokens.length}.`)
    state.position += tokens.length
    state.chunkErrors?.push(0.1)
    Reflect.get(state, "panelErrors")?.push(0.2)
    return new Float32Array(1)
  }
  return Object.assign(state, {
    contextLength,
    mrt: { chunks: true },
    tuned: { flashDecode: null },
    panelErrors: null,
    forward,
    forwardHidden: forward,
    panelError,
    gpu: { ...state.gpu, finish: async () => undefined }
  })
}

for (const linear of [false, true]) {
  test(`startup checks fit a small requested context${linear ? " (recurrent model)" : ""}`, async () => {
    for (const contextLength of [3, 8, 40, 41, 64]) {
      const state = bounded(contextLength, linear)
      const error = await Engine.prototype.selfCheck.call(state as unknown as Engine)
      assert.ok(Number.isFinite(error), `context ${contextLength}`)
      assert.equal(state.position, 0)
    }
  })
}

test("the warm-up fits a small requested context", async () => {
  for (const contextLength of [3, 9, 13, 64, 65, 8192]) {
    let position = 0
    const sizes: number[] = []
    const state = {
      contextLength,
      get position() { return position },
      capacity: 576,
      canDraft: true,
      quadMatmuls: false,
      layers: [{ type: "full_attention", attention: {} }],
      work: null,
      warmCounts: {},
      gpu: { compiled: { count: 0 } },
      warmUpLayers: Reflect.get(Engine.prototype, "warmUpLayers"),
      f16Trim() {},
      reset() { position = 0 },
      async forwardHidden(tokens: number[]) {
        assert.ok(tokens.length > 0, "a forward reads at least a token")
        if (position + tokens.length > contextLength)
          throw new Error(`The cache holds ${contextLength} tokens; this would need ${position + tokens.length}.`)
        position += tokens.length
        sizes.push(tokens.length)
        return tokens.length - 1
      },
      async candidates() { return { ids: new Int32Array(1), logits: new Float32Array(1) } },
      async logits() { return new Float32Array(1) },
      async argmax() { return new Int32Array(1) },
      dropGuess() {},
      accept(kept: number) { position -= 2 - kept },
      snapshot() {}
    }
    await Engine.prototype.warmUp.call(state as unknown as Engine)
    assert.equal(position, 0)
    // A big enough context reads every batch shape the warm-up has.
    if (contextLength === 8192) assert.deepEqual(sizes, [9, 64, 2, 1, 12, 1])
  }
})

test("a context too small for the startup checks fails before the GPU is touched", () => {
  for (const contextLength of [0, 2, 64.5])
    assert.throws(() => new Engine({} as OffscreenCanvas, { contextLength }), /at least 3/)
})
