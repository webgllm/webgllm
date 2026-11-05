import assert from "node:assert/strict"
import test from "node:test"
import { Gpu, type Pass } from "./gpu/gl.js"

function harness(readsPerSlice: number) {
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    readsPerSlice,
    submittedWork: 0,
    sliceReads: 0,
    sliceMatrixReads: 0,
    sliceCalibrates: true
  })
  const passes: Pass[] = []
  let finishes = 0
  gpu.run = (pass) => {
    passes.push(pass)
  }
  gpu.pace = async () => {
    finishes++
  }
  return { gpu, passes, finishes: () => finishes }
}
const pass = {
  program: {},
  inputs: {},
  output: { width: 20, height: 20 },
  rect: { x: 3, y: 7, width: 8, height: 5 },
  work: 4
} as Pass

test("paced bands cover exactly the original rectangle, using absolute coordinates", async () => {
  const h = harness(64)
  await h.gpu.runAsync(pass)
  assert.deepEqual(
    h.passes.map((p) => p.rect),
    [
      { x: 3, y: 7, width: 8, height: 2 },
      { x: 3, y: 9, width: 8, height: 2 },
      { x: 3, y: 11, width: 8, height: 1 }
    ]
  )
  assert.equal(h.finishes(), 2)
  assert.ok(
    h.passes.every(
      (p) => p.output === pass.output && p.program === pass.program
    )
  )
})
test("disabled pacing preserves a single submission without a fence", async () => {
  const h = harness(0)
  await h.gpu.runAsync(pass)
  assert.deepEqual(h.passes, [pass])
  assert.equal(h.finishes(), 0)
})
test("a row larger than the budget still progresses, and MRT uses the full target", async () => {
  const h = harness(1)
  await h.gpu.runAsync({
    ...pass,
    rect: undefined,
    output: [pass.output, pass.output]
  } as Pass)
  assert.equal(h.passes.length, 20)
  assert.deepEqual(h.passes[19].rect, { x: 0, y: 19, width: 20, height: 1 })
})

test("short output heads are sliced even below the read budget", async () => {
  const h = harness(100000)
  await h.gpu.runAsync(pass, 2)
  assert.deepEqual(
    h.passes.map((p) => p.rect?.height),
    [2, 2, 1]
  )
  assert.equal(h.finishes(), 3)
})

test("paced matrix bands align to raster tiles without consuming the leftover sliver", async () => {
  const h = harness(68 * 128 * 1024)
  await h.gpu.runAsync({
    ...pass,
    name: "matmul",
    work: 1024,
    rect: { x: 0, y: 0, width: 128, height: 128 }
  })
  assert.deepEqual(
    h.passes.map((p) => p.rect),
    [
      { x: 0, y: 0, width: 128, height: 64 },
      { x: 0, y: 64, width: 128, height: 64 }
    ]
  )
  assert.equal(h.finishes(), 1)
})

test("a delayed load-frame window cannot reduce the new prompt's budget", () => {
  let observed = 0,
    heartbeats = 0
  const budget = {
    reads: 256,
    heartbeat() {
      heartbeats++
    },
    frame() {
      observed++
    }
  }
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    phaseAt: 1000,
    budgets: { load: budget, prompt: budget, decode: budget },
    submissionBudget: budget
  })
  gpu.observeFrames(150, 16.7, 24, 900)
  assert.equal(observed, 0)
  assert.equal(heartbeats, 3)
  gpu.observeFrames(16.7, 16.7, 24, 1001)
  assert.equal(observed, 1)
})


test("a small projection starts a new chunk instead of splitting across its boundary", async () => {
  const h = harness(1024 * 128 * 1024)
  Object.assign(h.gpu, { sliceReads: 1000 * 128 * 1024 })
  await h.gpu.runAsync({ ...pass, name: "matmul", work: 1024,
    rect: { x: 0, y: 0, width: 128, height: 64 } })
  assert.deepEqual(h.passes.map(p => p.rect), [{ x: 0, y: 0, width: 128, height: 64 }])
  assert.equal(h.finishes(), 1)
})
