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

test('embedding panels allocate incrementally without increasing final storage',()=>{
 const made:Array<{internal:number;width:number;height:number}>=[]
 const state={bytes:true,headChoice:{expanded:true,split:2},buffers:{act:{internal:0x8d70,width:512,height:1}},gpu:{limits:{maxTextureSize:16384,maxTextureUnits:16},readsPerSlice:2**28,texture:(internal:number,width:number,height:number)=>{const t={internal,width,height};made.push(t);return t},write:()=>undefined,program:()=>({}),run:()=>undefined}}
 const matrix=Reflect.get(Engine.prototype,'allocateMatrix').call(state,32768,2048,6,false)
 assert.equal(matrix.quantPanels.length,4);assert.equal(matrix.quantPanels.filter(Boolean).length,1);assert.equal(made.length,2)
 const m=newQuantMatrix(4,2048,6,false)
 for(const row of[8192,16384,24576])Reflect.get(Engine.prototype,'writeMatrix').call(state,matrix,m,row)
 assert.equal(matrix.quantPanels.filter(Boolean).length,4)
 const bytes=matrix.quantPanels.reduce((n:number,t:{width:number;height:number})=>n+t.width*t.height*4,0)
 assert.equal(bytes,32768*2048)
})
