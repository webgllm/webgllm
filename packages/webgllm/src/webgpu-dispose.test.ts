import assert from "node:assert/strict"
import test from "node:test"
import { WebGpu } from "./gpu/webgpu.js"

test("disposing destroys every texture and buffer at once, though commands were still being recorded", () => {
  const destroyed: Array<string> = []
  const resource = (name: string) => ({ destroy: () => destroyed.push(name) })
  const gpu = Object.create(WebGpu.prototype)
  const live = new Set([
    { id: 1, tex: resource("weights"), bytes: 100 },
    { id: 2, tex: resource("storage"), bytes: 50 }
  ])
  Object.assign(gpu, {
    live,
    bytes: 150,
    // An encoder still open: a deletion then waits for its submission.
    encoder: {},
    computePass: {},
    queued: [],
    doomed: [resource("deleted earlier")],
    transient: [resource("staging")],
    touched: new Map(),
    writers: new Map(),
    groupsByTexture: new Map(),
    bindGroups: new Map(),
    programs: new Map(),
    paced: [],
    uniformBuffer: resource("uniforms"),
    device: { destroy: () => destroyed.push("device") }
  })
  gpu.dispose()
  assert.deepEqual(destroyed.sort(), ["deleted earlier", "device", "staging", "storage", "uniforms", "weights"])
  assert.equal(gpu.bytes, 0)
  assert.equal(gpu.doomed.length, 0)
})
