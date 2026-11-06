import assert from "node:assert/strict"
import { mock, test } from "node:test"
import { Engine, TUNING } from "./engine.js"

type Layer = { gateUp: { N: number; K: number }; down: { N: number; K: number } }
type Probe = {
  layers: Layer[]; work: number[] | null; unpaced: number
  layerWork: (index:number)=>number; charge:(work:number)=>Promise<void>
  tune: Engine["tune"]
}
for(const mode of ["success","stopped","throw"] as const)for(const warm of [false,true]){
  test(`subset tuning restores work estimates: ${mode}, ${warm?"populated":"empty"} cache`,async()=>{
    const prior={...TUNING};Object.assign(TUNING,{tuneLayers:8,paceLayers:1,quad:1,f16Matmul:1})
    let clock=0;const timer=mock.method(performance,"now",()=>clock+=100)
    const layers=Array.from({length:24},(_,i)=>({gateUp:{N:i+1,K:10},down:{N:4,K:6}}))
    const saved=warm?layers.map(l=>l.gateUp.N*l.gateUp.K+l.down.N*l.down.K):null
    const probe=Object.create(Engine.prototype) as Probe
    let paced=0,forwarded=0,stopped=false
    Object.assign(probe,{
      layers,work:saved,unpaced:0,batchSize:512,capacity:576,contextLength:8192,
      buffers:{},mrt:{chunks:false},tuneReport:{},tuned:{shortSplit:null,f16Matmul:null,f16Pack:null,flashDecode:null},
      gpu:{finish:async()=>undefined,pace:async()=>{paced++}},
      matmul:()=>undefined,reset:()=>{probe.unpaced=0},f16Mats:()=>[{rgba4:true}],f16Sized:()=>true,usesF16:()=>false,f16Trim:()=>undefined,
      forwardHidden:async()=>{
        forwarded++;assert.equal(probe.layers.length,8)
        probe.layerWork(0);assert.equal(probe.work?.length,8,"subset gets its own estimates")
        if(mode==="throw")throw new Error("interrupted tuning forward")
        if(mode==="stopped")stopped=true
        return 0
      }
    })
    Object.defineProperty(probe,"quadMatmuls",{value:true})
    try{
      if(mode==="throw")await assert.rejects(probe.tune(()=>stopped),/interrupted tuning forward/)
      else {const picked=await probe.tune(()=>stopped);assert.equal(picked===null,mode==="stopped")}
      assert.ok(forwarded>0);assert.equal(probe.layers,layers)
      assert.equal(probe.work,saved,"restore estimates for the original layer list, including an empty cache")
      await probe.charge(0);assert.equal(paced,0,"zero queued work must not pace after tuning")
      assert.ok(Number.isFinite(probe.layerWork(8)));assert.equal(probe.work?.length,24)
    }finally{timer.mock.restore();Object.assign(TUNING,prior)}
  })
}
