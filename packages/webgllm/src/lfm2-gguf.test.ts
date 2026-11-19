import assert from "node:assert/strict"
import test from "node:test"
import { loadGguf, packRows, rowsToFloats } from "./load-gguf.js"
import { GgmlType } from "./gguf.js"
import type { GgufHeader, GgufTensorInfo, GgufValue } from "./gguf.js"
import { modelMeta, WEBGL_TEXT_MODELS } from "./model-meta.js"
import type { WeightPiece } from "./weights.js"
import { toHalf } from "./half.js"
import { matrixRow, newQuantMatrix } from "./repack.js"

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
test("three official GGUF entries use recommended sampling with Thinking-only reasoning",()=>{
  const names=["LiquidAI/LFM2.5-350M-GGUF","LiquidAI/LFM2.5-1.2B-Instruct-GGUF","LiquidAI/LFM2.5-1.2B-Thinking-GGUF"] as const
  for(const name of names){
    assert.ok(WEBGL_TEXT_MODELS.includes(name));const meta=modelMeta(name),thinking=name.includes("Thinking")
    assert.equal(meta.format,"gguf");assert.equal(meta.toolFormat,"lfm2");assert.equal(meta.files.kind,"gguf")
    assert.deepEqual(meta.sampling,{temperature:thinking?0.05:0.1,top_k:50,repetition_penalty:1.05})
    assert.equal(meta.sampling.top_p??1,1);assert.equal(meta.sampling.min_p??0,0)
    if(thinking){assert.deepEqual(modelMeta(name).reasoning,{levels:{high:"high"},budget:true});assert.ok(meta.supports.capabilities.includes("thinking"))}
    else{assert.equal(modelMeta(name).reasoning,undefined);assert.equal(meta.reasoning,undefined)}
    assert.equal(meta.files.kind==="gguf"&&meta.files.model.path.includes("QAD"),!thinking)
  }
})

/** A two-layer Qwen3.5 GGUF: Gated DeltaNet, then full attention with partial RoPE (8 of 16 dims). */
function qwenFixture() {
  const tensors = new Map<string, GgufTensorInfo>(), chunks: Uint8Array[] = []
  let offset = 0
  const add = (name: string, dims: number[], type: number = GgmlType.Q4_0) => {
    const count = dims.reduce((a, b) => a * b, 1), bytes = new Uint8Array(type === GgmlType.F32 ? count * 4 : (count / 32) * 18)
    if (type === GgmlType.F32) { const view = new DataView(bytes.buffer); for (let i = 0; i < count; i++) view.setFloat32(i * 4, 1.25, true) }
    else for (let i = 0; i < bytes.length; i += 18) { bytes[i] = 0; bytes[i + 1] = 60; bytes.fill(136, i + 2, i + 18) }
    tensors.set(name, { name, dims, type, offset, byteLength: bytes.length }); offset += bytes.length; chunks.push(bytes)
  }
  const F32 = GgmlType.F32
  for (let i = 0; i < 2; i++) {
    const p = `blk.${i}.`
    add(p + "attn_norm.weight", [32], F32); add(p + "post_attention_norm.weight", [32], F32)
    if (i === 0) {
      add(p + "attn_qkv.weight", [32, 96]); add(p + "attn_gate.weight", [32, 32])
      add(p + "ssm_beta.weight", [32, 2]); add(p + "ssm_alpha.weight", [32, 2]); add(p + "ssm_out.weight", [32, 32])
      add(p + "ssm_conv1d.weight", [4, 96], F32); add(p + "ssm_a", [2], F32); add(p + "ssm_dt.bias", [2], F32); add(p + "ssm_norm.weight", [16], F32)
    } else {
      add(p + "attn_q.weight", [32, 64]); add(p + "attn_k.weight", [32, 16]); add(p + "attn_v.weight", [32, 16]); add(p + "attn_output.weight", [32, 32])
      add(p + "attn_q_norm.weight", [16], F32); add(p + "attn_k_norm.weight", [16], F32)
    }
    for (const name of ["ffn_gate", "ffn_up", "ffn_down"]) add(p + name + ".weight", [32, 32])
  }
  add("output_norm.weight", [32], F32)
  add("token_embd.weight", [32, 8])
  const metadata = new Map<string, GgufValue>([
    ["general.architecture", "qwen35"], ["qwen35.block_count", 2], ["qwen35.embedding_length", 32], ["qwen35.attention.head_count", 2],
    ["qwen35.attention.head_count_kv", 1], ["qwen35.attention.key_length", 16], ["qwen35.rope.dimension_count", 8], ["qwen35.feed_forward_length", 32],
    ["qwen35.context_length", 4096], ["qwen35.rope.freq_base", 1e7], ["qwen35.attention.layer_norm_rms_epsilon", 1e-6], ["qwen35.full_attention_interval", 2],
    ["qwen35.ssm.group_count", 2], ["qwen35.ssm.time_step_rank", 2], ["qwen35.ssm.state_size", 16], ["qwen35.ssm.inner_size", 32], ["qwen35.ssm.conv_kernel", 4]
  ])
  const bytes = new Uint8Array(offset); let at = 0; for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length }
  const header: GgufHeader = { version: 3, metadata, tensors, dataOffset: 0, headerLength: 0 }
  return { header, source: { size: bytes.length, async read(offset: number, length: number) { return bytes.slice(offset, offset + length) } } }
}

test("only the layers whose q/k rows the loader reorders say so: LFM's full attention, not its conv, sliding or Qwen's", async () => {
  const flags = (pieces: WeightPiece[]) => pieces.flatMap((p) => (p.kind === "layer" ? [[p.value.attention ? "attention" : p.value.linear ? "linear" : "conv", !!p.headsReordered]] : []))
  assert.deepEqual(flags(await collect(fixture())), [["conv", false], ["attention", true]])
  assert.deepEqual(flags(await collect(fixture("lfm2", 4))), [["conv", false], ["attention", false]], "sliding attention keeps its layout")
  const qwen = await collect(qwenFixture())
  const cfg = qwen.find((p) => p.kind === "config"); assert.ok(cfg?.kind === "config"); assert.deepEqual(cfg.config.layerTypes, ["linear_attention", "full_attention"])
  assert.ok(cfg.config.ropeDimsFull < cfg.config.headDim, "partial RoPE, where a reorder would change the rows")
  assert.deepEqual(flags(qwen), [["linear", false], ["attention", false]], "Qwen's q/k keep their layout")
})

test("a Qwen3.5 attention matrix of mixed tensor types takes every head's query and gate rows from where the tensor has them", async () => {
  // attn_q as F32, a value a row (per head: its query rows, then its gate rows), beside Q4_0 k and v: the parts go through floats.
  const f = qwenFixture()
  const q = f.header.tensors.get("blk.1.attn_q.weight")
  assert.ok(q)
  const K = 32, tensorRows = 64, headDim = 16
  const rowValue = (r: number) => 0.015625 * (r + 1)
  const floats = new Float32Array(K * tensorRows)
  for (let r = 0; r < tensorRows; r++) floats.fill(rowValue(r), r * K, (r + 1) * K)
  const eight = (name: string, rows: number, value: number) => new Float32Array(K * rows).fill(value)
  const extra = [
    { name: "blk.1.attn_q.weight", data: floats, rows: tensorRows },
    { name: "blk.1.attn_k.weight", data: eight("k", 16, 0.25), rows: 16 },
    { name: "blk.1.attn_v.weight", data: eight("v", 16, 0.5), rows: 16 }
  ]
  const old = new Uint8Array(await f.source.read(0, f.source.size))
  let size = old.length
  const chunks = [old]
  for (const e of extra) {
    const bytes = new Uint8Array(e.data.buffer)
    f.header.tensors.set(e.name, { name: e.name, dims: [K, e.rows], type: GgmlType.F32, offset: size, byteLength: bytes.length })
    chunks.push(bytes)
    size += bytes.length
  }
  const all = new Uint8Array(size)
  let at = 0
  for (const c of chunks) { all.set(c, at); at += c.length }
  const pieces: WeightPiece[] = []
  for await (const p of loadGguf({ size, async read(offset: number, length: number) { return all.slice(offset, offset + length) } }, { header: f.header, embeddingChunkRows: 4 })) pieces.push(p)
  const layer = pieces.find((p) => p.kind === "layer" && p.value.attention)
  assert.ok(layer?.kind === "layer" && layer.value.attention)
  const m = layer.value.attention.qkvg
  assert.equal(m.bits, 8)
  const near = (got: number, want: number, what: string) => assert.ok(Math.abs(got - want) <= want / 100, `${what}: ${got}, not ${want}`)
  for (let r = 0; r < 32; r++) {
    const head = Math.floor(r / headDim), d = r % headDim
    const query = matrixRow(m, r), gate = matrixRow(m, 32 + 16 + 16 + r)
    assert.ok(query && gate)
    near(query[0], rowValue(head * 2 * headDim + d), `head ${head}'s query row ${d}`)
    near(gate[K - 1], rowValue(head * 2 * headDim + headDim + d), `head ${head}'s gate row ${d}`)
  }
})

test("rows picked from beyond a part's own count are the tensor's rows, for a type packed through floats", () => {
  // Q4_1: 32 weights a block (a half scale, a half minimum, 16 bytes of nibbles), so a row of 32 is a block.
  const K = 32, tensorRows = 8, rows = 4
  const bytes = new Uint8Array(tensorRows * 20)
  const view = new DataView(bytes.buffer)
  for (let r = 0; r < tensorRows; r++) {
    view.setUint16(r * 20, toHalf(0.01 * (r + 1)), true)
    view.setUint16(r * 20 + 2, toHalf(-0.05 * (r + 1)), true)
    for (let i = 0; i < 16; i++) bytes[r * 20 + 4 + i] = ((r * 7 + i * 3) & 15) | (((r + i * 5) & 15) << 4)
  }
  const info = { name: "w", dims: [K, tensorRows], type: GgmlType.Q4_1, offset: 0, byteLength: bytes.length }
  const want = new Float32Array(tensorRows * K)
  rowsToFloats(info, bytes, tensorRows, want)
  const perm = Int32Array.of(4, 5, 6, 7)
  const m = newQuantMatrix(rows, K, 8, false)
  packRows(m, 0, info, bytes, rows, perm)
  for (let i = 0; i < rows; i++) {
    const got = matrixRow(m, i)
    assert.ok(got)
    let peak = 0
    for (let k = 0; k < K; k++) peak = Math.max(peak, Math.abs(want[perm[i] * K + k]))
    for (let k = 0; k < K; k++) assert.ok(Math.abs(got[k] - want[perm[i] * K + k]) <= peak / 100, `row ${i}, weight ${k}`)
  }
  assert.throws(() => packRows(m, 0, info, bytes, rows, Int32Array.of(0, 1, 2, 8)), /no row 8/)
})
