import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { anchorFor, naturalPromptPlan, promptPlan, readPrompt, sharedPrefix, systemPrefixFor, unsharedTail } from "./prompt.js"
import { Engine } from "./engine.js"
import { checkpointsToDrop } from "./checkpoint-policy.js"
import type { ChatMessage, ChatTemplateOptions, Tokenizer } from "./tokenizer.js"

// Token ids for a toy chat template: words map to ids, specials below 10.
const IM_START = 1
const IM_END = 2
const THINK = 3
const END_THINK = 4
const NL = 5
const words = new Map<string, number>()
const word = (w: string) => {
  if (!words.has(w)) words.set(w, 100 + words.size)
  return words.get(w) as number
}
const text = (s: string) => s.split(" ").filter(Boolean).map(word)

/**
 * A tokenizer whose template writes each message as <|im_start|> role \n
 * content <|im_end|>. Qwen-like (`rewrites`): the generation prompt opens
 * <think> \n </think> that a past reply loses; LFM3-like: the generation
 * prompt is <think> </think> and past replies keep it.
 */
function toyTokenizer(rewrites: boolean): Tokenizer {
  const encodeChat = (messages: Array<ChatMessage>, options?: { addGenerationPrompt?: boolean }) => {
    const out: Array<number> = []
    for (const m of messages) {
      out.push(IM_START, word(m.role), NL)
      if (m.role === "assistant" && !rewrites) out.push(THINK, END_THINK)
      out.push(...text(typeof m.content === "string" ? m.content : ""), IM_END, NL)
    }
    if (options?.addGenerationPrompt) {
      out.push(IM_START, word("assistant"), NL)
      out.push(...(rewrites ? [THINK, NL, END_THINK, NL] : [THINK, END_THINK]))
    }
    return out
  }
  return {
    encodeChat,
    specialTokens: { imStart: IM_START }
  } as unknown as Tokenizer
}

describe("sharedPrefix", () => {
  test("counts the leading items both have", () => {
    assert.equal(sharedPrefix([1, 2, 3], [1, 2, 4, 5]), 2)
    assert.equal(sharedPrefix([1, 2], [1, 2, 3]), 2)
    assert.equal(sharedPrefix([], [1]), 0)
  })
})

describe("unsharedTail and anchorFor", () => {
  test("a template that rewrites past replies: the generation prompt's tail, anchored at its <|im_start|>", () => {
    const tok = toyTokenizer(true)
    assert.equal(unsharedTail(tok, false), 4)
    const chat: Array<ChatMessage> = [{ role: "user", content: "a long question" }]
    const prompt = tok.encodeChat(chat, { addGenerationPrompt: true })
    const anchor = anchorFor(tok, prompt, chat, false)
    assert.equal(prompt[anchor as number], IM_START)
    assert.equal(anchor, prompt.length - 7)
    // The next question's prompt shares everything before the anchor.
    const next = tok.encodeChat(
      [...chat, { role: "assistant", content: "an answer" }, { role: "user", content: "more" }],
      { addGenerationPrompt: true }
    )
    assert.ok(sharedPrefix(prompt, next) >= (anchor as number))
  })

  test("a template that writes replies back as asked: the anchor at the prompt's end", () => {
    const tok = toyTokenizer(false)
    assert.equal(unsharedTail(tok, false), 0)
    const chat: Array<ChatMessage> = [{ role: "user", content: "a question" }]
    const prompt = tok.encodeChat(chat, { addGenerationPrompt: true })
    assert.equal(anchorFor(tok, prompt, chat, false), prompt.length)
  })

  test("no anchor after a tool's result: the last question's stays", () => {
    const tok = toyTokenizer(true)
    const chat: Array<ChatMessage> = [
      { role: "user", content: "weather" },
      { role: "assistant", content: "call" },
      { role: "tool", content: "sunny" }
    ]
    const prompt = tok.encodeChat(chat, { addGenerationPrompt: true })
    assert.equal(anchorFor(tok, prompt, chat, false), undefined)
  })
})

describe("promptPlan", () => {
  const sizes = (n: number) => {
    const out: Array<number> = []
    for (; n > 576; n -= 512) out.push(512)
    if (n > 0) out.push(n)
    return out
  }

  test("a batch ends at the anchor, the rest read after it", () => {
    assert.deepEqual(promptPlan(sizes, 560, 0, 553, false), [
      { size: 553, snapshot: "anchor" },
      { size: 7, snapshot: "end" }
    ])
    assert.deepEqual(promptPlan(sizes, 600, 0, 593, false), [
      { size: 512, snapshot: "batch" },
      { size: 81, snapshot: "anchor" },
      { size: 7, snapshot: "end" }
    ])
  })

  test("with few snapshots kept, only the anchor's", () => {
    assert.deepEqual(promptPlan(sizes, 1300, 0, 1290, true), [
      { size: 512, snapshot: null },
      { size: 512, snapshot: null },
      { size: 266, snapshot: "anchor" },
      { size: 10, snapshot: null }
    ])
  })

  test("an anchor saved from inside its batch splits nothing", () => {
    assert.deepEqual(promptPlan(sizes, 560, 0, 553, true, true), [
      { size: 560, snapshot: null, saveAt: 553 }
    ])
    assert.deepEqual(promptPlan(sizes, 1300, 100, 1290, false, true), [
      { size: 512, snapshot: "batch" },
      { size: 512, snapshot: "batch" },
      { size: 176, snapshot: "end", saveAt: 166 }
    ])
    // At a batch's end: a snapshot after it, as without `inside`.
    assert.deepEqual(promptPlan(sizes, 1100, 0, 512, true, true), [
      { size: 512, snapshot: "anchor" },
      { size: 512, snapshot: null },
      { size: 76, snapshot: null }
    ])
  })

  test("an anchor at the end, before the start or missing splits nothing", () => {
    assert.deepEqual(promptPlan(sizes, 300, 0, 300, true), [{ size: 300, snapshot: "anchor" }])
    assert.deepEqual(promptPlan(sizes, 300, 100, 50, false), [{ size: 200, snapshot: "end" }])
    assert.deepEqual(promptPlan(sizes, 1100, 0, undefined, false), [
      { size: 512, snapshot: "batch" },
      { size: 512, snapshot: "batch" },
      { size: 76, snapshot: "end" }
    ])
  })
})


describe("token-verified stable prefix", () => {
  test("leading system messages stay reusable across independent requests", () => {
    const tok = toyTokenizer(true)
    const system: ChatMessage = { role: "system", content: "Stable instructions and schema contract" }
    const a: ChatMessage[] = [system, { role: "user", content: "First task and schema" }]
    const b: ChatMessage[] = [system, { role: "user", content: "Different task and schema" }]
    const pa = tok.encodeChat(a, { addGenerationPrompt: true })
    const pb = tok.encodeChat(b, { addGenerationPrompt: true })
    const boundary = systemPrefixFor(tok, pa, a, false)
    assert.ok(boundary)
    assert.equal(boundary, tok.encodeChat([system], { addGenerationPrompt: false }).length)
    assert.equal(systemPrefixFor(tok, pb, b, false), boundary)
    assert.ok(sharedPrefix(pa, pb) >= boundary)
  })

  test("actual token equality handles template joins and rejects stale system prefixes", () => {
    const ordinary = toyTokenizer(false)
    const tok = { ...ordinary, encodeChat(chat: ChatMessage[], options?: { addGenerationPrompt?: boolean }) {
      const ids = ordinary.encodeChat(chat, options)
      if (chat.length === 1 && chat[0].role === "system") ids[ids.length - 1] = word("system-only-join")
      return ids
    } } as Tokenizer
    const chat: ChatMessage[] = [{ role: "system", content: "Original system" }, { role: "user", content: "Task" }]
    const prompt = tok.encodeChat(chat, { addGenerationPrompt: true })
    const boundary = systemPrefixFor(tok, prompt, chat, false)
    assert.equal(boundary, tok.encodeChat([chat[0]], { addGenerationPrompt: false }).length - 1)
    const edited = [{ ...chat[0], content: "Edited system" }, chat[1]]
    const newer = tok.encodeChat(edited, { addGenerationPrompt: true })
    const newBoundary = systemPrefixFor(tok, newer, edited, false)
    assert.ok(newBoundary)
    assert.ok(sharedPrefix(prompt, newer) < newBoundary)
  })

  test("user-required templates use verified tokenize-only probes; prompts without stable system use existing behavior", () => {
    const ordinary = toyTokenizer(false)
    const chat: ChatMessage[] = [{ role: "user", content: "Task" }]
    assert.equal(systemPrefixFor(ordinary, ordinary.encodeChat(chat), chat, false), undefined)
    const rejecting = { ...ordinary, encodeChat(chat: ChatMessage[]) {
      if (chat.every(message => message.role === "system")) throw new Error("Needs a user turn")
      return ordinary.encodeChat(chat)
    } } as Tokenizer
    const withSystem: ChatMessage[] = [{ role:"system", content:"System" }, ...chat]
    assert.ok(systemPrefixFor(rejecting, rejecting.encodeChat(withSystem), withSystem, false))
  })

  test("tools and thinking settings are part of prefix compatibility", () => {
    let encodes = 0
    const ordinary = toyTokenizer(false)
    const tok = { ...ordinary, encodeChat(chat: ChatMessage[], options?: ChatTemplateOptions) {
      encodes++
      return [word(JSON.stringify({tools:options?.tools,thinking:options?.enableThinking})), ...ordinary.encodeChat(chat, options)]
    } } as Tokenizer
    const chat: ChatMessage[] = [{role:"system",content:"System"},{role:"user",content:"Task"}]
    const tools = [{type:"function" as const,function:{name:"lookup",parameters:{type:"object"}}}]
    const prompt = tok.encodeChat(chat,{tools,enableThinking:false})
    assert.ok(systemPrefixFor(tok,prompt,chat,false,tools))
    const before = encodes
    systemPrefixFor(tok,prompt,chat,false,tools)
    assert.equal(encodes,before,"same prefix tokenization is reused")
    assert.equal(systemPrefixFor(tok,prompt,chat,true,tools),undefined,"different template state cannot reuse tokens")
  })
})

type SavedPoint = { anchor?: boolean; prefix?: boolean }
function memoryEngine(limit: number) {
  // Use the real engine snapshot/rewind policy with no GPU layers. State-copy
  // equivalence is additionally exercised by the native browser cache check.
  const engine = Object.create(Engine.prototype) as Engine
  const snapshots = new Map<number, SavedPoint>()
  Object.defineProperty(engine,"snapshotLimit",{value:limit})
  Object.assign(engine,{position:0,guessAt:null,draft:null,layers:[],config:{slidingWindow:0},snapshots,gpu:{delete(){ return undefined }}})
  Reflect.set(engine,"copyHidden",()=>({width:1,height:1,bytes:0}))
  Reflect.set(engine,"reset",()=>{engine.position=0;snapshots.clear()})
  Reflect.set(engine,"batchSizes",(count:number)=>{const sizes=[];for(;count>128;count-=128)sizes.push(128);if(count)sizes.push(count);return sizes})
  Reflect.set(engine,"forwardHidden",async(tokens:number[],options?:{saveAt?:number})=>{
    const from=engine.position
    if(options?.saveAt!==undefined){engine.position=from+options.saveAt;engine.snapshot(undefined,true)}
    engine.position=from+tokens.length
    return 0
  })
  return {engine,snapshots}
}

describe("stable prefix retention within the existing snapshot allowance",()=>{
  test("baseline end-only anchors lose independent prefixes; protected prefixes survive long dynamic tails",async()=>{
    for(const limit of [2,8]) for(const prefix of [63,320]){
      const before=memoryEngine(limit)
      await readPrompt(before.engine,Array(1800).fill(1),0,[],{anchor:1790})
      assert.equal(before.engine.rewind(prefix),0,"early batches either miss the boundary or were evicted by the long tail")
      const after=memoryEngine(limit)
      await readPrompt(after.engine,Array(1800).fill(1),0,[],{anchor:1790,prefix})
      const retained=Math.floor(prefix/128)*128
      assert.equal(after.engine.snapshotPosition("prefix"),retained||undefined)
      assert.ok(after.snapshots.get(1790)?.anchor)
      assert.ok(after.snapshots.size<=limit)
      assert.equal(after.engine.rewind(prefix),retained)
    }
  })

  test("conversational anchors still survive alongside the prefix and after generation snapshots",async()=>{
    const {engine,snapshots}=memoryEngine(2)
    await readPrompt(engine,Array(1800).fill(1),0,[],{anchor:1790,prefix:320})
    engine.position+=12;engine.snapshot()
    assert.equal(snapshots.size,2)
    assert.equal(engine.rewind(1790),1790)
    assert.ok(snapshots.get(256)?.prefix)
    await readPrompt(engine,Array(2100).fill(1),1790,[],{anchor:2090,prefix:320})
    assert.equal(engine.rewind(2090),2090)
    assert.ok(snapshots.get(256)?.prefix)
    assert.ok(snapshots.size<=2)
  })

  test("a one-snapshot configuration preserves its previous conversational policy",async()=>{
    const {engine,snapshots}=memoryEngine(1)
    await readPrompt(engine,Array(1800).fill(1),0,[],{anchor:1790,prefix:63})
    assert.equal(snapshots.size,1)
    assert.ok(snapshots.get(1790)?.anchor)
    assert.ok(!snapshots.has(63))
  })

  test("changed instructions and cancellation cannot restore an invalid suffix",async()=>{
    const {engine,snapshots}=memoryEngine(2)
    const cached:number[]=[]
    await readPrompt(engine,Array(1800).fill(1),0,cached,{anchor:1790,prefix:320,stop:()=>cached.length>=256})
    assert.equal(cached.length,256)
    assert.equal(engine.rewind(256),256,"completed prefix from a cancelled turn is usable")
    assert.equal(engine.rewind(12),0,"an earlier changed token invalidates it")
    assert.equal(snapshots.size,0)
  })

  test("an existing point can become a prefix and carry both roles without role coercion",()=>{
    const {engine,snapshots}=memoryEngine(2)
    engine.position=63;engine.snapshot()
    engine.snapshot(undefined,"prefix")
    assert.equal(snapshots.get(63)?.prefix,true)
    assert.equal(snapshots.get(63)?.anchor,undefined)
    engine.snapshot(undefined,true)
    assert.equal(snapshots.get(63)?.prefix,true)
    assert.equal(snapshots.get(63)?.anchor,true)
    engine.position=120;engine.snapshot(undefined,true)
    assert.equal(snapshots.get(63)?.prefix,true)
    assert.equal(snapshots.get(63)?.anchor,false)
    assert.equal(snapshots.get(120)?.anchor,true)
    engine.position=180;engine.snapshot(undefined,"prefix")
    assert.equal(snapshots.get(180)?.prefix,true)
    assert.equal(snapshots.get(120)?.anchor,true)
    assert.equal(snapshots.size,2)
  })

  test("checkpoint eviction prioritizes a new role without exceeding the count",()=>{
    const points=[{position:10,prefix:true},{position:20},{position:30,anchor:true},{position:40}]
    assert.deepEqual(checkpointsToDrop(points,2),[20,40])
    assert.deepEqual(checkpointsToDrop(points,3,"anchor"),[30])
    assert.deepEqual(checkpointsToDrop(points,3,"prefix"),[10])
    assert.deepEqual(checkpointsToDrop(points,0).sort((a,b)=>a-b),[10,20,30,40])
  })


})




describe("stable prefixes for templates requiring a user query",()=>{
  test("A/B probes are cached and verified against real tokens without changing request messages",()=>{
    const ordinary=toyTokenizer(true)
    const calls:Array<{chat:ChatMessage[];options?:ChatTemplateOptions}>=[]
    const tok={...ordinary,encodeChat(chat:ChatMessage[],options?:ChatTemplateOptions){
      calls.push({chat:structuredClone(chat),options})
      if(!chat.some(message=>message.role==="user"))throw new Error("No user query found in messages.")
      return ordinary.encodeChat(chat,options)
    }} as Tokenizer
    const chat:ChatMessage[]=[{role:"system",content:"Stable instructions"},{role:"user",content:"The actual request"}]
    const original=structuredClone(chat),prompt=tok.encodeChat(chat,{addGenerationPrompt:true})
    const prefix=systemPrefixFor(tok,prompt,chat,false)
    assert.ok(prefix)
    assert.deepEqual(calls.slice(-2).map(call=>call.chat.at(-1)?.content),["A","B"])
    assert.ok(calls.slice(-2).every(call=>call.options?.addGenerationPrompt===false))
    assert.deepEqual(chat,original,"probes never become real request content")
    const before=calls.length
    assert.equal(systemPrefixFor(tok,prompt,chat,false),prefix)
    assert.equal(calls.length,before,"cached fallback avoids repeated probing")
    const mismatch=[word("not-this-template"),...prompt.slice(1)]
    assert.equal(systemPrefixFor(tok,mismatch,chat,false),undefined,"cached probe prefix must still match actual tokens")
    assert.equal(calls.length,before)
  })

  test("fallback compatibility includes tools/thinking, and unusable probes are safely cached",()=>{
    const ordinary=toyTokenizer(false)
    let calls=0
    const tok={...ordinary,encodeChat(chat:ChatMessage[],options?:ChatTemplateOptions){
      calls++
      if(!chat.some(message=>message.role==="user"))throw new Error("Needs a user query")
      return [word(JSON.stringify({tools:options?.tools,thinking:options?.enableThinking})),...ordinary.encodeChat(chat,options)]
    }} as Tokenizer
    const chat:ChatMessage[]=[{role:"system",content:"System"},{role:"user",content:"Real request"}]
    const tools=[{type:"function" as const,function:{name:"lookup",parameters:{type:"object"}}}]
    const prompt=tok.encodeChat(chat,{tools,enableThinking:false,addGenerationPrompt:true})
    assert.ok(systemPrefixFor(tok,prompt,chat,false,tools))
    const before=calls
    assert.ok(systemPrefixFor(tok,prompt,chat,false,tools));assert.equal(calls,before)
    assert.equal(systemPrefixFor(tok,prompt,chat,true,tools),undefined)
    const changedTools=[{type:"function" as const,function:{name:"other",parameters:{type:"object"}}}]
    assert.equal(systemPrefixFor(tok,prompt,chat,false,changedTools),undefined)
    let failures=0
    const unsupported={...ordinary,encodeChat(){failures++;throw new Error("Template unavailable")}} as Tokenizer
    assert.equal(systemPrefixFor(unsupported,prompt,chat,false),undefined)
    assert.equal(failures,2,"one system-only attempt and an early probe failure")
    const failedBefore=failures
    assert.equal(systemPrefixFor(unsupported,prompt,chat,false),undefined)
    assert.equal(failures,failedBefore)
  })
})




describe("cost-neutral system checkpoints with legacy conversational anchors",()=>{
  const batches=(n:number)=>{const out=[];while(n>576){out.push(512);n-=512}if(n>0)out.push(n);return out}
  test("short requests keep the original single model pass",()=>{
    assert.deepEqual(naturalPromptPlan(batches,100,0,100,false,false,20),[{size:100,snapshot:"anchor"}])
    assert.deepEqual(naturalPromptPlan(batches,100,0,93,true,true,20),[{size:100,snapshot:null,saveAt:93}])
  })
  test("static prefix tagging never changes legacy batch sizes, including capacity tails",()=>{
    const plan=naturalPromptPlan(batches,1144,0,1144,false,false,1080)
    assert.deepEqual(plan.map(step=>step.size),[512,512,120])
    let at=0;assert.ok(plan.some(step=>{at+=step.size;return at===1024&&step.snapshot==="prefix"}))
    const wide=(n:number)=>{const out=[];while(n>1024){out.push(512);n-=512}if(n>0)out.push(n);return out}
    assert.deepEqual(naturalPromptPlan(wide,1153,0,1153,false,false,1023).map(step=>step.size),wide(1153))
    assert.deepEqual(naturalPromptPlan(wide,1117,512,1117,false,false,1023),[{size:605,snapshot:"anchor"}])
  })
  test("a changing schema after the natural1024 checkpoint can reuse it rather than replay from512",async()=>{
    const {engine,snapshots}=memoryEngine(2)
    Reflect.set(engine,"batchSizes",batches)
    Object.defineProperty(engine,"canSaveInside",{value:false})
    await readPrompt(engine,Array(1144).fill(1),0,[],{prefix:1080,anchor:1144})
    assert.equal(engine.snapshotPosition("prefix"),1024)
    assert.equal(engine.snapshotPosition("anchor"),1144)
    assert.equal(engine.rewind(1057),1024)
    assert.ok(snapshots.size<=2)
  })
  test("ordinary conversation checkpoints retain exact original positions",()=>{
    const ordinary=promptPlan(batches,600,0,593,false,false)
    const plan=naturalPromptPlan(batches,600,0,593,false,false,131)
    assert.ok(plan.length<=ordinary.length)
    let at=0;assert.ok(plan.some(step=>{at+=step.size;return at===593&&step.snapshot==="anchor"}))
    assert.deepEqual(naturalPromptPlan(batches,560,0,553,true,true,131),[{size:560,snapshot:null,saveAt:553}])
  })
  test("protected roles remain within one/two/eight existing snapshot allowances",async()=>{
    for(const limit of [1,2,8]){
      const {engine,snapshots}=memoryEngine(limit)
      await readPrompt(engine,Array(1800).fill(1),0,[],{prefix:320,anchor:1790})
      assert.ok(snapshots.size<=limit)
      assert.equal(engine.snapshotPosition("anchor"),1790)
      if(limit>1)assert.equal(engine.snapshotPosition("prefix"),256)
      else assert.equal(engine.snapshotPosition("prefix"),undefined)
    }
  })
  test("none skips optional checkpoint work but keeps in-call token accounting",async()=>{
    const {engine,snapshots}=memoryEngine(2),calls:Array<{length:number;saveAt?:number}>=[]
    Reflect.set(engine,"forwardHidden",async(tokens:number[],options?:{saveAt?:number})=>{calls.push({length:tokens.length,...options});engine.position+=tokens.length;return 0})
    const cached:number[]=[]
    await readPrompt(engine,Array(300).fill(1),0,cached,{prefix:20,anchor:70,checkpoints:false})
    assert.deepEqual(calls.map(call=>call.length),[128,128,44]);assert.ok(calls.every(call=>call.saveAt===undefined))
    assert.equal(snapshots.size,0);assert.equal(cached.length,300)
  })
})
