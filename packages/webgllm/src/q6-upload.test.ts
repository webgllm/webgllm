import test from "node:test"
import assert from "node:assert/strict"
import { Engine } from "./engine.js"
import { newQuantMatrix } from "./repack.js"
import { expandQ6 } from "./q6-expanded.js"

test("expanded head uploads stay within 128 KiB and preserve all row data across chunks", () => {
  const m = newQuantMatrix(512, 2048, 6, false)
  for (let i = 0; i < m.quants.length; i++)
    m.quants[i] = Math.imul(i + 7, 1664525)
  const expected = expandQ6(m, 0, 128).quants
  const uploaded = new Uint8Array(expected.length)
  let writes = 0
  const quants = {},
    scales = {}
  const state = {
    expandedUpload: new Uint8Array(0),
    gpu: {
      write: (
        target: object,
        rect: { y: number; height: number },
        data: { pixels: Uint8Array }
      ) => {
        if (target !== quants) return
        assert.ok(data.pixels.byteLength <= 128 * 1024)
        uploaded.set(data.pixels, rect.y * 2048 * 4)
        writes++
      }
    }
  }
  const matrix = {
    q6Expanded: true,
    blocks: 8,
    rowTexels: 2048,
    rowsPerColumn: 128,
    quants,
    scales
  }
  const write = Reflect.get(Engine.prototype, "writeMatrix") as (
    gpu: object,
    cpu: object,
    row: number
  ) => void
  write.call(state, matrix, m, 0)
  assert.equal(writes, 8)
  assert.ok(state.expandedUpload.byteLength <= 128 * 1024)
  assert.deepEqual(uploaded, expected)
})

test("GPU upload borrows existing storage and routes chunk data across panels and columns", () => {
  const m = newQuantMatrix(32, 256, 6, false)
  for (let i = 0; i < m.quants.length; i++) m.quants[i] = i * 7919
  const stage = { internal: 0x8d70, width: 52, height: 3 }
  const panels = [{ height: 2 }, { height: 2 }, { height: 1 }]
  const payloads: number[] = [],
    regions: object[] = []
  const state = {
    buffers: { act: stage },
    expandedUpload: new Uint8Array(0),
    gpu: {
      texture() {
        throw new Error("upload allocated more GPU memory")
      },
      program() {
        return {}
      },
      write(
        target: object,
        rect: { width: number; height: number },
        data: { pixels: Uint32Array }
      ) {
        if (target !== stage) return
        assert.ok(rect.width <= stage.width && rect.height <= stage.height)
        payloads.push(...data.pixels)
      },
      run(pass: {
        inputs: { uNative: object }
        rect: { x: number; y: number; width: number; height: number }
        uniforms: { uX0: { int: number }; uY0: { int: number } }
        output: { height: number }
      }) {
        assert.equal(pass.inputs.uNative, stage)
        assert.equal(pass.uniforms.uX0.int, pass.rect.x)
        assert.equal(pass.uniforms.uY0.int, pass.rect.y)
        assert.ok(pass.rect.y + pass.rect.height <= pass.output.height)
        regions.push(pass.rect)
      }
    }
  }
  const matrix = {
    q6Expanded: true,
    blocks: 1,
    rowTexels: 256,
    rowsPerColumn: 5,
    quantPanels: panels,
    quants: panels[0],
    scales: {}
  }
  const write = Reflect.get(Engine.prototype, "writeMatrix") as (
    gpu: object,
    cpu: object,
    row: number
  ) => void
  write.call(state, matrix, m, 4)
  assert.deepEqual(payloads, [...m.quants])
  assert.deepEqual(regions, [
    { x: 0, y: 1, width: 256, height: 1 },
    { x: 0, y: 0, width: 256, height: 2 },
    { x: 0, y: 0, width: 256, height: 1 },
    { x: 256, y: 0, width: 256, height: 2 },
    { x: 256, y: 0, width: 256, height: 2 }
  ])
  assert.equal(state.expandedUpload.byteLength, 0)
})
