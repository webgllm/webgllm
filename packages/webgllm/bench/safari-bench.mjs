/** Reuse one explicitly selected iPhone benchmark tab through WebKit Inspector.
 * node bench/safari-bench.mjs WS_URL eval EXPRESSION
 * node bench/safari-bench.mjs WS_URL run URL OUTPUT.json [background-tune]
 * Opening or closing tabs is deliberately outside this helper.
 */
import fs from "node:fs"
const [socketUrl, action, inputArgument, output, background] =
  process.argv.slice(2)
let argument = inputArgument
if (
  action === "run" &&
  background !== "no-memory" &&
  background !== "background-tune"
) {
  const url = new URL(argument)
  if (!url.searchParams.has("idleMs")) url.searchParams.set("idleMs", "3000")
  argument = url.href
}
if (!socketUrl || !["eval", "run"].includes(action))
  throw Error("Expected WebKit socket, eval|run, argument")
const ws = new WebSocket(socketUrl),
  pending = new Map(),
  workers = new Set(),
  workerAgents = new Set(),
  workerErrors = [],
  memoryPressure = [],
  samples = [],
  errors = []
let sequence = 0,
  target,
  phase = "loading",
  complete = false,
  tracking = false
const pause = (ms) => new Promise((r) => setTimeout(r, ms))
const send = (method, params = {}, targetId = target) =>
  new Promise((resolve, reject) => {
    const id = ++sequence,
      timer = setTimeout(() => {
        pending.delete(id)
        reject(Error(`Inspector timeout: ${method}`))
      }, 20000)
    pending.set(id, (m) => {
      clearTimeout(timer)
      resolve(m)
    })
    const message = JSON.stringify({ id, method, params })
    ws.send(
      targetId
        ? JSON.stringify({
            id: ++sequence,
            method: "Target.sendMessageToTarget",
            params: { targetId, message }
          })
        : message
    )
  })
const sendWorker = (method, params, workerId) =>
  new Promise((resolve, reject) => {
    const id = ++sequence
    const timer = setTimeout(() => {
      pending.delete(id)
      reject(Error(`Worker inspector timeout: ${method}`))
    }, 20000)
    pending.set(id, (message) => {
      clearTimeout(timer)
      resolve(message)
    })
    void send("Worker.sendMessageToWorker", {
      workerId,
      message: JSON.stringify({ id, method, params })
    }).then((response) => {
      if (response.error) {
        clearTimeout(timer)
        pending.delete(id)
        resolve(response)
      }
    }, reject)
  })
ws.onmessage = ({ data }) => {
  const outer = JSON.parse(data)
  if (outer.method === "Target.targetCreated") {
    const info = outer.params.targetInfo
    if (info.type === "page") target = info.targetId
    else if (info.type === "worker") workers.add(info.targetId)
  }
  if (outer.method === "Target.targetDestroyed")
    workers.delete(outer.params.targetId)
  let m =
    outer.method === "Target.dispatchMessageFromTarget"
      ? JSON.parse(outer.params.message)
      : outer
  if (m.method === "Worker.workerCreated") {
    workerAgents.add(m.params.workerId)
    void send("Worker.initialized", { workerId: m.params.workerId })
  }
  if (m.method === "Worker.workerTerminated")
    workerAgents.delete(m.params.workerId)
  if (m.method === "Worker.dispatchMessageFromWorker")
    m = JSON.parse(m.params.message)
  if (m.id) {
    pending.get(m.id)?.(m)
    pending.delete(m.id)
  }
  if (m.method === "Memory.trackingUpdate")
    samples.push({ at: Date.now(), phase, ...m.params.event })
  if (m.method === "Memory.memoryPressure")
    memoryPressure.push({
      at: Date.now(),
      phase,
      method: m.method,
      ...m.params
    })
}
await new Promise((resolve, reject) => {
  ws.onopen = resolve
  ws.onerror = reject
})
await pause(500)
const evaluate = async (expression) => {
  let r
  for (let attempt = 0; attempt < 20; attempt++) {
    r = await send("Runtime.evaluate", { expression, returnByValue: true })
    if (r.error?.code !== -32601) break
    await pause(100)
  }
  if (r.error || r.result?.wasThrown) throw Error(JSON.stringify(r))
  return r.result?.result?.value
}
try {
  if (action === "eval") {
    await evaluate(
      `globalThis.__safariBenchResult=undefined;(async()=>(${argument}))().then(value=>globalThis.__safariBenchResult=JSON.stringify({value}),error=>globalThis.__safariBenchResult=JSON.stringify({error:String(error.stack||error.message||error)}))`
    )
    for (let i = 0; i < 600; i++) {
      const v = await evaluate("globalThis.__safariBenchResult")
      if (v !== undefined) {
        console.log(v)
        break
      }
      await pause(250)
    }
  } else {
    const existing = await evaluate("location.href")
    if (
      !existing ||
      !["/idle.html", "/webgl-perf.html"].includes(
        new URL(existing).pathname
      ) ||
      new URL(existing).origin !== new URL(argument).origin
    )
      throw Error("Refusing to navigate a tab outside this benchmark origin")
    // Release this tab's previous task worker before another model is loaded.
    await evaluate(
      "globalThis.__benchPreShutdown=false;Promise.resolve(globalThis.__benchShutdown?.()).then(()=>globalThis.__benchPreShutdown=true);true"
    )
    for (let attempt = 0; ; attempt++) {
      if (await evaluate("globalThis.__benchPreShutdown")) break
      if (attempt >= 200)
        throw Error("Previous benchmark worker did not shut down")
      await pause(50)
    }
    if (background !== "no-memory" && background !== "background-tune") {
      const workerEnabled = await send("Worker.enable")
      if (workerEnabled.error) workerErrors.push(workerEnabled.error)
      const enabled = await send("Memory.enable")
      if (enabled.error) errors.push(enabled.error)
      else {
        const started = await send("Memory.startTracking")
        tracking = !started.error
        if (started.error) errors.push(started.error)
      }
    }
    await evaluate(
      `setTimeout(()=>location.assign(${JSON.stringify(argument)}),0);true`
    )
    for (let i = 0; i < 2400; i++) {
      await pause(250)
      const state = await evaluate(
        `({url:location.href,phase:document.documentElement.dataset.phase,text:document.querySelector('pre')?.textContent??'',visible:document.visibilityState})`
      )
      if (state?.url !== argument) continue
      phase = state.phase ?? "loading"
      if (phase === "error") throw Error(state.text.slice(-2000))
      if (phase === "complete") {
        if (
          background === "background-tune" &&
          !state.text.includes("tuned in the background:")
        )
          continue
        complete = true
        break
      }
    }
    if (!complete) throw Error("Safari benchmark timed out")
    phase = "retained"
    const gc = await send("Heap.gc")
    if (gc.error) errors.push(gc.error)
    for (const worker of workers) {
      const r = await send("Heap.gc", {}, worker)
      if (r.error) workerErrors.push(r.error)
    }
    let workerGpu
    for (const workerId of workerAgents) {
      const collected = await sendWorker("Heap.gc", {}, workerId)
      if (collected.error) workerErrors.push(collected.error)
      const memory = await sendWorker(
        "Runtime.evaluate",
        { expression: "globalThis.__benchMemory?.()", returnByValue: true },
        workerId
      )
      if (memory.error) workerErrors.push(memory.error)
      else workerGpu = memory.result?.result?.value ?? workerGpu
    }
    await pause(1800)
    const workerTargetCount = workerAgents.size
    if (tracking) {
      phase = "unloaded"
      await evaluate(
        "globalThis.__benchMemoryShutdown=false;Promise.resolve(globalThis.__benchShutdown?.()).then(()=>globalThis.__benchMemoryShutdown=true);true"
      )
      for (let attempt = 0; ; attempt++) {
        if (await evaluate("globalThis.__benchMemoryShutdown")) break
        if (attempt >= 200) throw Error("Memory worker did not shut down")
        await pause(50)
      }
      const pageGc = await send("Heap.gc")
      if (pageGc.error) errors.push(pageGc.error)
      await pause(1800)
    }
    if (tracking) {
      await send("Memory.stopTracking")
      const disabled = await send("Memory.disable")
      if (disabled.error) errors.push(disabled.error)
      const detached = await send("Worker.disable")
      if (detached.error) workerErrors.push(detached.error)
    }
    const state = await evaluate(
      `({url:location.href,phase:document.documentElement.dataset.phase,text:document.querySelector('pre')?.textContent??'',visible:document.visibilityState,userAgent:navigator.userAgent})`
    )
    const result = {
      url: argument,
      complete,
      tracking,
      samples,
      errors,
      memoryPressure,
      state,
      workerTargets: workerTargetCount,
      workerErrors,
      workerGpu,
      gcCollected: !gc.error && !workerErrors.length
    }
    if (output) fs.writeFileSync(output, JSON.stringify(result, null, 2) + "\n")
    console.log(
      JSON.stringify({
        url: argument,
        complete,
        tracking,
        samples: samples.length,
        errors,
        visible: state.visible,
        workerTargets: workerTargetCount,
        workerErrors
      })
    )
    await evaluate("globalThis.__benchShutdown?.()")
  }
} catch (error) {
  if (output)
    fs.writeFileSync(
      output,
      JSON.stringify(
        {
          url: argument,
          complete: false,
          tracking,
          samples,
          errors,
          error: String(error.stack || error.message || error)
        },
        null,
        2
      ) + "\n"
    )
  throw error
} finally {
  ws.close()
}
