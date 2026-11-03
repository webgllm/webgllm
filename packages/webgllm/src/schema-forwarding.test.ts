import assert from "node:assert/strict"
import { test } from "node:test"
import { chat } from "@tanstack/ai"
import type { PromptCacheOptions } from "@tanstack/ai"
import { resolveDebugOption } from "@tanstack/ai/adapter-internals"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"

class SchemaWorker {
  static requests: WorkerRequest[] = []
  static failure: string | undefined
  onmessage: ((event: { data: WorkerResponse }) => void) | null = null
  postMessage(message: WorkerRequest) {
    SchemaWorker.requests.push(message)
    if (message.type === "load") this.onmessage?.({ data: { type: "ready", model: message.model } })
    if (message.type === "generate") {
      if (SchemaWorker.failure) this.onmessage?.({ data: { type: "error", id: message.id, message: SchemaWorker.failure } })
      else {
        this.onmessage?.({ data: { type: "token", id: message.id, text: '{"moveId":"b6-a5"}' } })
        this.onmessage?.({ data: { type: "done", id: message.id, stats: { inputTokens: 10, cachedTokens: 0, outputTokens: 8, firstTokenMs: 1, promptMs: 1, decodeMs: 1 } } })
      }
    }
  }
  terminate() { this.onmessage = null }
}
;(globalThis as { Worker?: unknown }).Worker = SchemaWorker

test("structured output forwards the actual schema and limits to worker generation", async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel()
  const adapter = webglText("LiquidAI/LFM3-1.7B-GGUF")
  const schema = { type: "object", properties: { moveId: { type: "string", enum: ["b6-a5"] } }, required: ["moveId"], additionalProperties: false }
  const chatOptions = { model: adapter.model, threadId: "schema-forwarding", logger: resolveDebugOption(false), messages: [{ role: "user" as const, content: "Choose" }], modelOptions: { maxTokens: 8192 }, reasoning: { level: "high" as const, summary: true, budgetTokens: 512 } }
  const result = await adapter.structuredOutput({ chatOptions, outputSchema: schema })
  assert.deepEqual(result.data, { moveId: "b6-a5" })
  const request = SchemaWorker.requests.find(r => r.type === "generate")
  assert.ok(request && request.type === "generate")
  assert.deepEqual(request.outputSchema, schema)
  assert.equal(request.maxTokens, 8192)
  assert.equal(request.thinkingBudgetTokens, 512)
  SchemaWorker.failure = "Structured output was truncated before the JSON Schema was satisfied."
  await assert.rejects(adapter.structuredOutput({ chatOptions, outputSchema: schema }), /truncated/)
  unloadModel()
})


test("standard promptCache forwards none and key policy in both public structured and streaming chat",async()=>{
  const {webglText}=await import("./adapter.js")
  const {unloadModel}=await import("./client.js")
  unloadModel();SchemaWorker.failure=undefined
  const schema={type:"object",properties:{moveId:{enum:["b6-a5"]}},required:["moveId"],additionalProperties:false}
  const cases:Array<{option?:PromptCacheOptions;scope:string;none:boolean}>=[
    {scope:"public-cache",none:false},
    {option:"none",scope:"public-cache",none:true},
    {option:"short",scope:"public-cache",none:false},
    {option:"long",scope:"public-cache",none:false},
    {option:{retention:"long",key:"explicit-cache-key"},scope:"explicit-cache-key",none:false},
    {option:{retention:"none",key:"none-key"},scope:"none-key",none:true},
    {option:{retention:"short",key:""},scope:"public-cache",none:false}
  ]
  try{
    for(const entry of cases)for(const structured of [false,true]){
      SchemaWorker.requests=[]
      await chat({adapter:webglText("LiquidAI/LFM3-1.7B-GGUF"),threadId:"public-cache",messages:[{role:"user",content:"Choose a move."}],...(entry.option!==undefined&&{promptCache:entry.option}),...(structured&&{outputSchema:schema}),stream:false})
      const request=SchemaWorker.requests.find(message=>message.type==="generate")
      assert.ok(request&&request.type==="generate")
      assert.equal("cacheScope" in request,false)
      assert.ok(!JSON.stringify(request.chat).includes(entry.scope),"cache key is never rendered as model instructions")
      assert.equal(request.cachePrompt,entry.none?false:undefined)
      assert.equal("cachePrompt" in request,entry.none,"no automatic positive override for defaults, short or long")
      assert.equal(request.outputSchema!==undefined,structured)
      if(structured)assert.deepEqual(request.outputSchema,schema)
    }
  }finally{unloadModel()}
})


test("default structured serialization is byte-identical to the original system/schema contract",async()=>{
  const {webglText}=await import("./adapter.js"),{unloadModel}=await import("./client.js")
  unloadModel();SchemaWorker.failure=undefined;SchemaWorker.requests=[]
  const adapter=webglText("LiquidAI/LFM3-1.7B-GGUF")
  const messages=[{role:"user" as const,content:"Keep this exact original question."},{role:"assistant" as const,content:'{"moveId":'}]
  const original=JSON.stringify(messages)
  const schemas=[{type:"object",properties:{moveId:{enum:["b6-a5","d6-e5"]}},required:["moveId"]},{type:"object",properties:{moveId:{enum:["d6-e5","b6-a5"]}},required:["moveId"]}]
  try{
    for(const outputSchema of schemas)await adapter.structuredOutput({chatOptions:{model:adapter.model,threadId:"byte-parity",logger:resolveDebugOption(false),systemPrompts:["Original caller instructions"],messages},outputSchema})
    const sent=SchemaWorker.requests.filter(request=>request.type==="generate")
    assert.equal(sent.length,2)
    for(const[index,request]of sent.entries()){
      assert.deepEqual(request.chat,[{role:"system",content:"Original caller instructions\n\nAnswer with only a JSON value matching this JSON Schema:\n"+JSON.stringify(schemas[index])},...messages])
      assert.equal(request.outputSchema,schemas[index]);assert.equal(JSON.stringify(messages),original)
      for(const key of ["cacheAnchorMessages","cacheAnchorContentLength","cacheScope","cachePrefixContentLength","preserveThinking"])assert.equal(key in request,false)
    }
  }finally{unloadModel()}
})

test("current tool thinking and authoritative tool results use baseline roles without an extra user message",async()=>{
  const {webglText}=await import("./adapter.js"),{unloadModel}=await import("./client.js")
  unloadModel();SchemaWorker.failure=undefined;SchemaWorker.requests=[]
  const adapter=webglText("LiquidAI/LFM3-1.7B-GGUF")
  const messages=[{role:"user" as const,content:"Old query"},{role:"assistant" as const,content:"Old answer",thinking:[{content:"OLD_THOUGHT"}]},{role:"user" as const,content:"Current query"},{role:"assistant" as const,content:"",thinking:[{content:"CURRENT_THOUGHT"}],toolCalls:[{id:"current",type:"function" as const,function:{name:"lookup",arguments:'{"id":"current"}'}}]},{role:"tool" as const,content:'{"value":"current"}',toolCallId:"current"}]
  const original=JSON.stringify(messages)
  try{
    await adapter.structuredOutput({chatOptions:{model:adapter.model,threadId:"tool-parity",logger:resolveDebugOption(false),messages,reasoning:{level:"off",summary:true}},outputSchema:{type:"object"}})
    const request=SchemaWorker.requests.find(message=>message.type==="generate");assert.ok(request&&request.type==="generate")
    assert.deepEqual(request.chat.map(message=>message.role),["system","user","assistant","user","assistant","tool"])
    const assistants=request.chat.filter(message=>message.role==="assistant")
    assert.equal(assistants[0].thinking,undefined);assert.equal(assistants[1].thinking,"CURRENT_THOUGHT")
    assert.equal(request.chat.at(-1)?.content,'{"value":"current"}');assert.equal(request.thinking,false)
    assert.ok(!JSON.stringify(request.chat).includes("OLD_THOUGHT"));assert.equal(JSON.stringify(messages),original)
  }finally{unloadModel()}
})
