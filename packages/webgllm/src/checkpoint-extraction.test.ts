import assert from "node:assert/strict"
import test from "node:test"
import { Engine, TUNING } from "./engine.js"
import { GL } from "./gpu/gl.js"
import * as K from "./gpu/kernels.js"
import type { Pass, Program, Texture } from "./gpu/gl.js"

type Snapshot = { conv: Texture[]; windows: Texture[]; linear: Texture[]; hidden?: Texture }
type CaptureState = {
  snapshots: Map<number, Snapshot>
  saving: Texture[] | null
  prefixSaving: { row: number; state: Snapshot } | null
  layers: { conv?: { state: Texture[] }; linear?: { convState: Texture[]; state: Texture[] } }[]
}
type Scenario = { name: string; kind: "conv" | "window" | "linear"; chunked?: boolean; prefixAt: number; saveAt?: number }
type Fault = { allocation: number } | { draw: number } | { stateWrite: true }
const texturesOf = (saved: Snapshot) => [...saved.conv, ...saved.windows, ...saved.linear, ...(saved.hidden ? [saved.hidden] : [])]

/** Real forwardBatch/convolve/linearAttend/recurrenceChunks/copyPositions run here.
 * Only weight arithmetic is stubbed: the GPU records ownership and checks every
 * input/output remains live. This is an allocation test, not a numerical GPU test.
 */
function fixture(scenario: Scenario) {
  const engine = Object.create(Engine.prototype) as Engine
  const state = engine as unknown as CaptureState
  const live = new Set<Texture>(), allocated: Texture[] = [], draws: Pass[] = []
  const checkpoint = new Set<Texture>(), zeroed = new Set<Texture>()
  const programs = new Map<Program, string>()
  let id = 0, recording = false, fault: Fault | undefined, allocationCount = 0
  const faultError = new Error("injected checkpoint failure")
  const bounds = () => assert.ok(state.snapshots.size + Number(state.saving !== null) + Number(state.prefixSaving !== null) <= engine.snapshotLimit)
  const gpu = {
    readsPerSlice: 0,
    decodeLayers: 1,
    setPhase: () => undefined,
    decodeWork: () => undefined,
    configurePacing: () => undefined,
    texture(internal: number, width: number, height: number): Texture {
      if (recording) {
        bounds()
        allocationCount++
        if (fault && "allocation" in fault && fault.allocation === allocationCount) throw faultError
      }
      const tex = { id: id++, tex: {}, internal, width, height, bytes: width * height * 16 } as Texture
      live.add(tex)
      if (recording) { allocated.push(tex); checkpoint.add(tex) }
      return tex
    },
    program(source: string): Program {
      const program = { program: {}, uniforms: new Map(), samplers: [] } as Program
      programs.set(program, source)
      return program
    },
    run(pass: Pass) {
      for (const tex of [...Object.values(pass.inputs), ...[pass.output].flat()]) assert.ok(live.has(tex), `use of missing/freed texture ${tex?.id}`)
      if (recording && fault && "stateWrite" in fault && programs.get(pass.program) === K.linConvState(4) &&
        (pass.uniforms?.uM as { int: number } | undefined)?.int === 12) throw faultError
      if ([pass.output].flat().some(tex => checkpoint.has(tex))) {
        draws.push(pass)
        if (fault && "draw" in fault && fault.draw === draws.length) throw faultError
      }
    },
    async runAsync(pass: Pass) { this.run(pass) },
    write(tex: Texture, _rect: unknown, data: { pixels: ArrayLike<number> }) {
      assert.ok(live.has(tex))
      if (Array.from(data.pixels).every(value => value === 0)) zeroed.add(tex)
      else zeroed.delete(tex)
    },
    delete(tex: Texture) { assert.ok(live.delete(tex), `double deletion of texture ${tex.id}`) },
    pace: async () => undefined,
    flush: () => undefined
  }
  const tex = (width = 4, height = 12) => gpu.texture(GL.RGBA32F, width, height)
  const matrix = { N: 16, K: 16, bits: 4, blocks: 1, rowsPerColumn: 1, rowTexels: 8, quants: tex(), scales: tex() }
  const norm = tex(), kernel = tex(), makeLayer = () => ({ type: "conv", operatorNorm: norm, ffnNorm: norm, gateUp: matrix, down: matrix })
  const conv = () => ({ ...makeLayer(), conv: { inProj: matrix, outProj: matrix, kernel, state: [tex(), tex()], prefixes: tex() } })
  const attention = () => ({ ...makeLayer(), type: "attention", attention: { qkvg: matrix, o: matrix, qNorm: norm, kNorm: norm, kv: tex(4, 16), ring: 16 } })
  const linear = () => ({ ...makeLayer(), type: "linear_attention", linear: {
    inProj: matrix, ba: matrix, outProj: matrix, kernel, convState: [tex(12, 3), tex(12, 3)], state: [tex(4, 4), tex(4, 4)], a: tex(), dtBias: tex(), norm
  } })
  const lin = Object.fromEntries(["ba", "gates", "inProj", "qkv", "qk", "delta", "o", "out"].map(name => [name, tex()]))
  const chunks = Object.fromEntries(["cum", "a", "t", "p", "u", "w", "oi", "qp"].map(name => [name, tex()]))
  const residual = [tex(), tex()]
  Object.assign(engine, {
    tuned: {},
    gpu, position: 0, contextLength: 4096, capacity: 12, batchSize: 12, kvPositions: 256, kvShift: 8,
    guessAt: null, draft: null, images: [], snapshots: new Map(), saving: null, prefixSaving: null, work: null, unpaced: 0,
    keepPrefixes: false, slicing: new Map(), quads: new Map(), mrt: {}, chunkedRecurrence: !!scenario.chunked,
    config: { hiddenSize: 16, intermediateSize: 16, convKernel: 4, slidingWindow: 2, normEps: 1e-5, numHeads: 4, numKvHeads: 4, headDim: 4,
      ...(scenario.kind === "linear" && { linear: { numKeyHeads: 4, numValueHeads: 4, keyHeadDim: 4, valueHeadDim: 4, convKernel: 4 } }) },
    layers: scenario.kind === "conv" ? [conv(), conv()] : scenario.kind === "window" ? [conv(), attention(), attention()] : [linear(), linear()],
    embedding: matrix,
    buffers: { ids: tex(), residual, residualSmall: residual, residualLarge: residual, normed: tex(), rms: tex(), bcx: tex(), convOut: tex(), opOut: tex(), gateUp: tex(), act: tex(), ffnOut: tex(), qkvg: tex(), q: tex(), attnOut: tex(), lin },
    linChunks: { chunk: 4, textures: chunks },
    // Matrix multiplication and attention computation are unrelated to checkpoint ownership.
    project: () => 0, matmul: () => undefined, attend: async () => 0
  })
  const permanent = new Set(live)
  const tokens = Array.from({ length: 12 }, (_, i) => i + 1)
  const options = { prefixAt: scenario.prefixAt, ...(scenario.saveAt !== undefined && { saveAt: scenario.saveAt }) }
  const noLeaks = () => {
    const ids = (textures: Iterable<Texture>) => [...textures].map(tex => tex.id).sort((a, b) => a - b)
    assert.deepEqual(ids(live), ids(new Set([...permanent, ...[...state.snapshots.values()].flatMap(texturesOf)])))
  }
  return { engine, state, allocated, draws, programs, faultError, tokens, options, noLeaks,
    assertReset() {
      assert.equal(engine.position, 0)
      for (const layer of state.layers) for (const tex of [...(layer.conv?.state ?? []), ...(layer.linear?.convState ?? []), ...(layer.linear?.state ?? [])])
        assert.ok(zeroed.has(tex), "recovery zeroes both sides of mutable state before another cold forward")
    },
    async seed() {
      for (let i = 0; i <= engine.snapshotLimit; i++) {
        await engine.forwardHidden(tokens)
        engine.snapshot(undefined, i === 0 ? "prefix" : true)
        assert.ok(state.snapshots.size <= engine.snapshotLimit)
      }
      noLeaks()
    },
    begin(nextFault?: Fault) { fault = nextFault; recording = true },
    stop() { recording = false; fault = undefined; checkpoint.clear() }
  }
}

const scenarios: Scenario[] = [
  { name: "LFM convolution", kind: "conv", prefixAt: 3 },
  { name: "sliding-window KV, including wrapped copy", kind: "window", prefixAt: 3 },
  { name: "Qwen token recurrence", kind: "linear", prefixAt: 3 },
  { name: "Qwen inside-chunk recurrence", kind: "linear", chunked: true, prefixAt: 3 },
  { name: "Qwen chunk-boundary recurrence", kind: "linear", chunked: true, prefixAt: 4 },
  { name: "Qwen token recurrence with independent anchor", kind: "linear", prefixAt: 3, saveAt: 6 },
  { name: "Qwen chunk recurrence with independent anchor", kind: "linear", chunked: true, prefixAt: 3, saveAt: 6 },
  { name: "Qwen shared prefix and anchor", kind: "linear", chunked: true, prefixAt: 6, saveAt: 6 }
]
for (const scenario of scenarios) test(`real checkpoint extraction owns partial allocations: ${scenario.name}`, async () => {
  const prior = { ...TUNING }
  Object.assign(TUNING, { linChunk: 4, linChunkMin: 4, lastRowOnly: 0, foldNorm: 0, fuseConv: 0 })
  try {
    const control = fixture(scenario)
    await control.seed()
    control.begin()
    await control.engine.forwardHidden(control.tokens, control.options)
    const allocationCount = control.allocated.length, drawCount = control.draws.length
    assert.ok(allocationCount > 1 && drawCount >= allocationCount, "test reaches actual multi-layer capture allocations/draws")
    const capturePrograms = control.draws.map(pass => control.programs.get(pass.program))
    if (scenario.kind !== "linear") assert.ok(capturePrograms.includes(K.convState(4)), "real LFM conv capture ran")
    if (scenario.kind === "window") assert.ok(capturePrograms.includes(K.COPY_POSITIONS), "real KV ring extraction ran")
    if (scenario.kind === "linear") {
      assert.ok(capturePrograms.includes(K.linConvState(4)), "real Qwen conv capture ran")
      assert.ok(capturePrograms.some(source => source === K.COPY || source === K.linChunkState({ keyHeadDim: 4, valueHeadDim: 4 })), "real Qwen recurrent capture ran")
    }
    if (scenario.kind === "window") assert.ok(drawCount > allocationCount, "wrapped KV copies exercise more than one draw per allocation")
    control.noLeaks()
    control.stop(); control.engine.reset(); control.noLeaks()
    for (const fault of [
      ...Array.from({ length: allocationCount }, (_, i) => ({ allocation: i + 1 })),
      ...Array.from({ length: drawCount }, (_, i) => ({ draw: i + 1 })),
      ...(scenario.saveAt !== undefined ? [{ stateWrite: true as const }] : [])
    ]) {
      const h = fixture(scenario)
      await h.seed()
      const start = h.engine.position, before = new Set(h.state.snapshots.values())
      h.begin(fault)
      await assert.rejects(h.engine.forwardHidden(h.tokens, h.options), error => error === h.faultError, JSON.stringify(fault))
      assert.equal(h.state.saving, null)
      assert.equal(h.state.prefixSaving, null)
      assert.ok(h.state.snapshots.size <= h.engine.snapshotLimit)
      assert.ok([...h.state.snapshots.values()].every(saved => before.has(saved)), "a failed forward cannot publish a partial checkpoint")
      h.noLeaks()
      h.stop()
      assert.ok(h.engine.rewind(start + scenario.prefixAt) <= start, "recovery cannot select the failed prefix")
      h.engine.reset(); h.noLeaks()
      h.assertReset()
      await h.engine.forwardHidden(h.tokens, h.options)
      assert.equal(h.engine.snapshotPosition("prefix"), scenario.prefixAt)
      assert.equal(h.engine.rewind(scenario.prefixAt), scenario.prefixAt)
      assert.equal(h.state.saving, null)
      assert.equal(h.state.prefixSaving, null)
      h.noLeaks()
      h.engine.reset(); h.noLeaks()
    }
  } finally { Object.assign(TUNING, prior) }
})
