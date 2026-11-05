import assert from "node:assert/strict"
import { test } from "node:test"
import { withPromptCachePolicy } from "./prompt-cache-policy.js"

function state() {
  const weights = { uploaded: true }
  const state = { cached:[1,2,3],
    engine:{position:3,weights,resets:0,reset(){this.resets++;this.position=0}} }
  return state
}

test("cache none clears cross-call state while run-local KV and uploaded weights remain usable",async()=>{
  const model=state(),weights=model.engine.weights
  const answer=await withPromptCachePolicy(model,false,async()=>{
    assert.equal(model.cached.length,0)
    assert.equal(model.engine.position,0)
    for(const token of [8,9,10]){model.cached.push(token);model.engine.position++}
    assert.deepEqual(model.cached,[8,9,10],"autoregressive steps retain their own KV during generation")
    assert.equal(model.engine.position,3)
    return "answer"
  })
  assert.equal(answer,"answer")
  assert.equal(model.cached.length,0)
  assert.equal(model.engine.resets,2)
  assert.equal(model.engine.position,0)
  assert.equal(model.engine.weights,weights,"no model upload/unload operation")
})

test("none releases retained prompt state after generation errors and cancellation",async()=>{
  for(const abort of [false,true]){
    const model=state(),controller=new AbortController()
    await assert.rejects(withPromptCachePolicy(model,false,async()=>{
      model.cached.push(8);model.engine.position++
      if(abort){controller.abort(new Error("cancel generation"));controller.signal.throwIfAborted()}
      throw new Error("schema setup failed")
    }),abort?/cancel generation/:/schema setup failed/)
    assert.equal(model.cached.length,0)
    assert.equal(model.engine.position,0)
    assert.equal(model.engine.resets,2)
  }
})

test("omitted and positive native cache policy preserve existing reuse without new retention limits",async()=>{
  for(const option of [undefined,true]){
    const model=state()
    await withPromptCachePolicy(model,option,async()=>{
      assert.deepEqual(model.cached,[1,2,3])
      model.cached.push(4);model.engine.position++
    })
    assert.equal(model.engine.resets,0)
    assert.deepEqual(model.cached,[1,2,3,4])
  }
})
