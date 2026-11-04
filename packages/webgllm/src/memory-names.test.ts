import assert from "node:assert/strict"
import { test } from "node:test"
import { names } from "../bench/memory-names.js"
import type { Engine } from "./engine.js"
import type { Texture } from "./gpu/gl.js"
const texture=(id:number)=>({id,tex:{}} as Texture)
function engine(f16:unknown){return {layers:[],buffers:{},quads:new Map(),chunkBufs:new Map(),snapshots:new Map(),f16} as unknown as Engine}
test("packed f16 checkpoints allow missing activation scratch and name wh2",()=>{
  const packed=texture(1),result=names(engine({wh2:packed}))
  assert.equal(result.get(packed),"scratch f16 packed weights")
  assert.equal(result.size,1)
})
test("legacy and mixed f16 scratch remain named without requiring every allocation",()=>{
  const weights=texture(1),packed=texture(2),a=texture(3),b=texture(4)
  for(const scratch of [{w:weights,a:[a,b]},{wh2:packed,a:[a,b]},{w:weights,wh2:packed,a:[a,b]}]){
    const result=names(engine(scratch));assert.equal(result.get(a),"scratch f16 activations");assert.equal(result.get(b),"scratch f16 activations")
    if("w" in scratch)assert.equal(result.get(weights),"scratch f16 weights")
    if("wh2" in scratch)assert.equal(result.get(packed),"scratch f16 packed weights")
  }
  for(const scratch of [null,false,{}, {a:[]}])assert.equal(names(engine(scratch)).size,0)
})
