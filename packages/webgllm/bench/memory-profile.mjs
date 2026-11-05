import fs from "node:fs"
const cdpOrigin = process.argv[4] ?? "http://127.0.0.1:9223"
const targets = await fetch(`${cdpOrigin}/json/list`).then((r) => r.json())
const tab = targets.find(
  (x) => x.type === "page" && x.url.startsWith("http://localhost:8788/")
)
if (!tab) throw new Error("No existing benchmark tab")
const ws = new WebSocket(tab.webSocketDebuggerUrl)
await new Promise((r) => (ws.onopen = r))
let id = 0,
  session
const pending = new Map()
ws.onmessage = ({ data }) => {
  const m = JSON.parse(data)
  if (m.id) {
    pending.get(m.id)?.(m)
    pending.delete(m.id)
  } else if (
    m.method === "Target.attachedToTarget" &&
    m.params.targetInfo.type === "worker"
  ) {
    session = m.params.sessionId
    console.log("Attached to benchmark worker")
  }
}
const call = (method, params = {}, sid) =>
  new Promise((r) => {
    pending.set(++id, r)
    ws.send(
      JSON.stringify({ id, method, params, ...(sid ? { sessionId: sid } : {}) })
    )
  })
// Dispose the prior task worker before attaching to and measuring the next one.
if (process.argv[2]) {
  const released = await call("Runtime.evaluate", {
    expression: "globalThis.__benchShutdown?.()", awaitPromise: true
  })
  if (released.error || released.result?.exceptionDetails)
    throw Error("Could not release the previous benchmark worker")
}
console.log(
  JSON.stringify(
    await call("Target.setAutoAttach", {
      autoAttach: true,
      waitForDebuggerOnStart: false,
      flatten: true
    })
  )
)
await call("Page.bringToFront")
if (process.argv[2]) await call("Page.navigate", { url: process.argv[2] })
const samples = []
let complete = false
let benchmarkError = false
for (let i = 0; i < 1200; i++) {
  if (session) {
    const r = await call("Runtime.getHeapUsage", {}, session)
    if (r.result) samples.push({ at: Date.now(), ...r.result })
    else if (samples.length) break
    else console.log(JSON.stringify(r))
  }
  if (i % 5 === 0) {
    const r = await call("Runtime.evaluate", {
      expression: "document.documentElement.dataset.phase",
      returnByValue: true
    })
    if (r.result?.result?.value === "error") {
      benchmarkError = true
      break
    }
    if (samples.length && r.result?.result?.value === "complete") {
      complete = true
      break
    }
  }
  await new Promise((r) => setTimeout(r, 100))
}
const peak = {
  usedSize: 0,
  totalSize: 0,
  embedderHeapUsedSize: 0,
  backingStorageSize: 0,
  combinedUsedAndBacking: 0
}
for (const s of samples) {
  for (const k of Object.keys(peak)) {
    if (k === "combinedUsedAndBacking")
      peak[k] = Math.max(peak[k], s.usedSize + (s.backingStorageSize ?? 0))
    else peak[k] = Math.max(peak[k], s[k] ?? 0)
  }
}
let retainedAfterGc,
  gpu,
  gcCollected = false
if (session) {
  await new Promise((r) => setTimeout(r, 1500))
  const gc = await call("HeapProfiler.collectGarbage", {}, session)
  gcCollected = !gc.error
  await new Promise((r) => setTimeout(r, 150))
  retainedAfterGc = (await call("Runtime.getHeapUsage", {}, session)).result
  gpu = (
    await call(
      "Runtime.evaluate",
      { expression: "globalThis.__benchMemory?.()", returnByValue: true },
      session
    )
  ).result?.result?.value
}
const phase = (
  await call("Runtime.evaluate", {
    expression: "document.documentElement.dataset.phase",
    returnByValue: true
  })
).result?.result?.value
complete = phase === "complete"
const result = {
  url: process.argv[2],
  samples,
  peak,
  complete,
  benchmarkError,
  retainedAfterGc,
  gpu,
  gcCollected
}
if (process.argv[3]) fs.writeFileSync(process.argv[3], JSON.stringify(result))
console.log(
  JSON.stringify({
    samples: samples.length,
    peak,
    complete,
    retainedAfterGc,
    gpu,
    gcCollected
  })
)
await call("Runtime.evaluate", {
  expression: "globalThis.__benchShutdown?.()",
  awaitPromise: true
})
ws.close()
if (!complete || benchmarkError) process.exitCode = 1
