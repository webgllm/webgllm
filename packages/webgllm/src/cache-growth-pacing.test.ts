import test from "node:test"
import assert from "node:assert/strict"
import { Engine, TUNING } from "./engine.js"

test("cache growth bounds overlap and drains before returning without changing capacity", async () => {
  let pending = 0,
    copies = 0,
    fences = 0
  const state = {
    kvPositions: 1024,
    kvShift: 12,
    contextLength: 4096,
    layers: Array.from({ length: 3 }, () => ({
      attention: { kv: { width: 256, height: 1024 }, ring: 0 }
    })),
    kvTexture: (positions: number) => {
      assert.ok(pending <= 1)
      return { width: 256, height: positions }
    },
    gpu: {
      readsPerSlice: 1,
      activePhase: "prompt",
      setPhase(phase: string) {
        this.activePhase = phase
      },
      program: () => ({}),
      run: () => {
        pending++
        copies++
      },
      delete: () => undefined,
      pace: async (ahead: number) => {
        assert.ok(ahead === 0 || ahead === 1)
        assert.equal(state.gpu.activePhase, "load")
        await Promise.resolve()
        pending = Math.min(pending, ahead)
        fences++
      },
      outOfMemory: () => false
    }
  }
  const reserve = Reflect.get(Engine.prototype, "reserve") as (
    end: number
  ) => Promise<void>
  await reserve.call(state, 1536)
  assert.equal(pending, 0)
  assert.equal(copies, 3)
  assert.equal(fences, 4)
  assert.equal(state.kvPositions, 1536)
  assert.equal(state.gpu.activePhase, "prompt")
})

for (const end of [512, 548, 2085]) {
  test(`known ${end}-token prompt reserves only its final required capacity`, async () => {
    const ends: number[] = []
    const state = {
      position: 0,
      contextLength: 4096,
      batchSize: 512,
      reserve: async (value: number) => {
        ends.push(value)
      }
    }
    await Engine.prototype.preparePrompt.call(state as unknown as Engine, end)
    assert.deepEqual(ends, end > 512 ? [end] : [])
  })
}

test("invalid prompt reservations fail before allocating memory", async () => {
  const state = {
    position: 512,
    contextLength: 4096,
    batchSize: 512,
    reserve: async () => assert.fail("invalid allocation")
  }
  for (const end of [511, 4097, NaN, 1024.5])
    await assert.rejects(
      Engine.prototype.preparePrompt.call(state as unknown as Engine, end),
      /Invalid prompt/
    )
})


test("fast calibrated allocation overlap retains the original cache capacity", async () => {
  let copies = 0, fences = 0
  const state = {
    kvPositions: 512, kvShift: 12, contextLength: 4096,
    layers: Array.from({ length: 3 }, () => ({ attention: { kv: { width: 256, height: 512 }, ring: 0 } })),
    kvTexture: (positions: number) => ({ width: 256, height: positions }),
    gpu: {
      readsPerSlice: TUNING.submissionReads * 8, activePhase: "prompt",
      setPhase(phase: string) { this.activePhase = phase },
      program: () => ({}), run: () => { copies++ }, delete: () => undefined,
      async pace(ahead: number) { assert.equal(ahead, 0); assert.equal(copies, 3); fences++ },
      outOfMemory: () => false
    }
  }
  await Reflect.get(Engine.prototype, "reserve").call(state, 1536)
  assert.equal(fences, 1)
  assert.equal(state.kvPositions, 1536)
  assert.equal(state.gpu.activePhase, "prompt")
})
