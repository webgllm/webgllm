import assert from "node:assert/strict"
import test from "node:test"
import { loadGguf } from "./load-gguf.js"
import { GgmlType } from "./gguf.js"
import type { GgufHeader, GgufTensorInfo, GgufValue } from "./gguf.js"
import { modelMeta, WEBGL_TEXT_MODELS, WEBGL_MODEL_REASONING } from "./model-meta.js"
import { WebGlTextAdapter } from "./adapter.js"
import type { Sampling } from "./sampler.js"
import type { WeightPiece } from "./weights.js"

function fixture(arch = "lfm2", sliding = 0) {
  const tensors = new Map<string,GgufTensorInfo>(), chunks:Uint8Array[]=[]
  let offset=0
  const add=(name:string,dims:number[],type:number=GgmlType.Q4_0)=>{
    const count=dims.reduce((a,b)=>a*b,1),bytes=new Uint8Array(type===GgmlType.F32?count*4:count/32*18)
    if(type===GgmlType.F32){const view=new DataView(bytes.buffer);for(let i=0;i<count;i++)view.setFloat32(i*4,1.25,true)}
    else for(let i=0;i<bytes.length;i+=18){bytes[i]=0;bytes[i+1]=60;bytes.fill(136,i+2,i+18)}
    tensors.set(name,{name,dims,type,offset,byteLength:bytes.length});offset+=bytes.length;chunks.push(bytes)
  }
  const lfm2=arch==="lfm2"
  for(let i=0;i<2;i++){
    const prefix=`blk.${i}.`
    for(const norm of lfm2?["attn_norm","ffn_norm"]:["attn_norm","post_attention_norm","ffn_norm","post_ffw_norm"])add(prefix+norm+".weight",[32],GgmlType.F32)
    if(i===0){add(prefix+"shortconv.in_proj.weight",[32,96]);add(prefix+"shortconv.out_proj.weight",[32,32]);add(prefix+"shortconv.conv.weight",[3,32],GgmlType.F32)}
    else{
      for(const [name,n] of [["attn_q",32],["attn_k",16],["attn_v",16],["attn_output",32],...(!lfm2?[["attn_gate",8]]:[])] as [string,number][])add(prefix+name+".weight",[32,n])
      add(prefix+"attn_q_norm.weight",[4],GgmlType.F32);add(prefix+"attn_k_norm.weight",[4],GgmlType.F32)
    }
    for(const name of ["ffn_gate","ffn_up","ffn_down"])add(prefix+name+".weight",[32,32])
  }
  add(lfm2?"token_embd_norm.weight":"output_norm.weight",[32],GgmlType.F32)
  add("token_embd.weight",[32,8])
  const metadata=new Map<string,GgufValue>([
    ["general.architecture",arch],[`${arch}.block_count`,2],[`${arch}.embedding_length`,32],[`${arch}.attention.head_count`,8],
    [`${arch}.attention.head_count_kv`,[0,4]],[`${arch}.feed_forward_length`,32],[`${arch}.context_length`,128000],
    [`${arch}.rope.freq_base`,1000000],[`${arch}.attention.layer_norm_rms_epsilon`,1e-5],[`${arch}.shortconv.l_cache`,3],
    [`${arch}.attention.sliding_window`,sliding]
  ])
  const bytes=new Uint8Array(offset);let at=0;for(const chunk of chunks){bytes.set(chunk,at);at+=chunk.length}
  const header:GgufHeader={version:3,metadata,tensors,dataOffset:0,headerLength:0}
  return {header,source:{size:bytes.length,async read(offset:number,length:number){return bytes.slice(offset,offset+length)}}}
}
const collect=async(f:ReturnType<typeof fixture>)=>{const pieces:WeightPiece[]=[];for await(const p of loadGguf(f.source,{header:f.header,embeddingChunkRows:4}))pieces.push(p);return pieces}

test("LFM2 GGUF uses ungated attention, no sandwich norms and token_embd_norm",async()=>{
  const pieces=await collect(fixture()),config=pieces.find(p=>p.kind==="config")
  assert.equal(config?.kind,"config");if(config?.kind!=="config")return
  assert.equal(config.config.attentionGate,false);assert.equal(config.config.contextLength,128000)
  assert.deepEqual(config.config.layerTypes,["conv","full_attention"])
  const layers=pieces.filter(p=>p.kind==="layer")
  assert.equal(layers.length,2)
  for(const p of layers){assert.equal(p.value.postOperatorNorm,undefined);assert.equal(p.value.postFfnNorm,undefined);assert.equal(p.value.operatorNorm[0],1.25)}
  assert.equal(layers[0].value.conv?.inProj.N,96)
  assert.equal(layers[1].value.attention?.qkvg.N,64)
  const final=pieces.find(p=>p.kind==="finalNorm");assert.ok(final?.kind==="finalNorm");assert.equal(final.value[0],1.25)
  assert.equal(pieces.filter(p=>p.kind==="embedding").reduce((n,p)=>n+p.value.N,0),8)
})
test("LFM2 sliding attention and LFM3's existing gated/sandwich layout stay distinct",async()=>{
  const sliding=await collect(fixture("lfm2",4)),old=await collect(fixture("lfm3"))
  const cfg=sliding.find(p=>p.kind==="config");assert.ok(cfg?.kind==="config");assert.deepEqual(cfg.config.layerTypes,["conv","sliding_attention"])
  const legacy=old.find(p=>p.kind==="config");assert.ok(legacy?.kind==="config");assert.equal(legacy.config.attentionGate,true)
  const layers=old.filter(p=>p.kind==="layer");assert.ok(layers.every(p=>p.value.postOperatorNorm&&p.value.postFfnNorm));assert.equal(layers[1].value.attention?.qkvg.N,72)
})
test("unsupported architectures, untied LFM2 heads and incomplete LFM2 files fail explicitly",async()=>{
  await assert.rejects(collect(fixture("unknown")),/unsupported architecture/)
  const untied=fixture();untied.header.tensors.set("output.weight",untied.header.tensors.get("token_embd.weight") as GgufTensorInfo)
  await assert.rejects(collect(untied),/separate output head/)
  const missing=fixture();missing.header.tensors.delete("token_embd_norm.weight")
  await assert.rejects(collect(missing),/missing tensor token_embd_norm/)
})
test("three official GGUF entries use recommended sampling with caller overrides and Thinking-only reasoning",()=>{
  const names=["LiquidAI/LFM2.5-350M-GGUF","LiquidAI/LFM2.5-1.2B-Instruct-GGUF","LiquidAI/LFM2.5-1.2B-Thinking-GGUF"] as const
  for(const name of names){
    assert.ok(WEBGL_TEXT_MODELS.includes(name));const meta=modelMeta(name),thinking=name.includes("Thinking")
    assert.equal(meta.format,"gguf");assert.equal(meta.toolFormat,"lfm2");assert.equal(meta.files.kind,"gguf")
    assert.deepEqual(meta.sampling,{temperature:thinking?0.05:0.1,top_k:50,repetition_penalty:1.05})
    const adapter=new WebGlTextAdapter(name)
    const sample=Reflect.get(adapter,"sampling").bind(adapter) as (options:Partial<Sampling>|undefined,thinking:boolean)=>Sampling
    assert.deepEqual(sample(undefined,thinking),meta.sampling)
    assert.deepEqual(sample({temperature:0.3,top_k:9,top_p:0.8,min_p:0.1,repetition_penalty:1.2},thinking),{temperature:0.3,top_k:9,top_p:0.8,min_p:0.1,repetition_penalty:1.2})
    assert.equal(meta.sampling.top_p??1,1);assert.equal(meta.sampling.min_p??0,0)
    if(thinking){assert.deepEqual(WEBGL_MODEL_REASONING[name],{levels:{high:"true"},budget:true});assert.ok(meta.supports.capabilities.includes("thinking"))}
    else{assert.equal(WEBGL_MODEL_REASONING[name],undefined);assert.equal(meta.reasoning,undefined)}
    assert.equal(meta.files.kind==="gguf"&&meta.files.model.path.includes("QAD"),!thinking)
  }
})
