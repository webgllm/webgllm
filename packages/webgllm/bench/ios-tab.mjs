// Runs JavaScript in an iPhone Safari tab through ios_webkit_debug_proxy and
// prints the tab's console while it waits. WebKit ignores awaitPromise, so the
// result is kept on the page (window.__iosResult) and read back by polling.
//
//   ios_webkit_debug_proxy -c <udid>:9333 --no-frontend   # not 9222 (the Pixel's adb forward)
//   curl -s http://127.0.0.1:9333/json                     # the tabs and their webSocketDebuggerUrl
//   node packages/webgl/bench/ios-tab.mjs <ws url> '<async function body>' [watch seconds]
//
// The body runs inside an async function: `return` a JSON-able value.
const [url, expression, watchSeconds = "0"] = process.argv.slice(2)
if (!url || !expression) {
  console.log("usage: node ios-tab.mjs <ws url> '<async function body>' [watch seconds]")
  process.exit(1)
}
const ws = new WebSocket(url)
let id = 0
let target = null
const pending = new Map()
const send = (method, params = {}) =>
  new Promise((resolve) => {
    const inner = ++id
    pending.set(inner, resolve)
    const message = JSON.stringify({ id: inner, method, params })
    if (target) {
      ws.send(JSON.stringify({ id: ++id, method: "Target.sendMessageToTarget", params: { targetId: target, message } }))
    } else ws.send(message)
  })
ws.onmessage = ({ data }) => {
  const msg = JSON.parse(data)
  if (msg.method === "Target.targetCreated") target = msg.params.targetInfo.targetId
  const inner = msg.method === "Target.dispatchMessageFromTarget" ? JSON.parse(msg.params.message) : msg
  if (inner.id && pending.has(inner.id)) {
    pending.get(inner.id)(inner)
    pending.delete(inner.id)
  } else if (inner.method === "Console.messageAdded") {
    const m = inner.params.message
    console.log(`[console ${m.level}] ${m.text}`.slice(0, 400))
  }
}
ws.onopen = async () => {
  await new Promise((r) => setTimeout(r, 500)) // the target arrives first
  await send("Console.enable")
  await send("Runtime.evaluate", {
    expression: `window.__iosResult = undefined; (async () => { ${expression} })().then((v) => { window.__iosResult = JSON.stringify(v ?? null) }, (e) => { window.__iosResult = JSON.stringify({ error: String(e && e.stack || e) }) })`,
    returnByValue: true
  })
  for (let i = 0; i < 400; i++) {
    const res = await send("Runtime.evaluate", { expression: "window.__iosResult", returnByValue: true })
    const value = res.result?.result?.value
    if (value !== undefined) {
      console.log("RESULT", String(value).slice(0, 3000))
      break
    }
    await new Promise((r) => setTimeout(r, 250))
  }
  setTimeout(() => process.exit(0), Number(watchSeconds) * 1000)
}
ws.onerror = (e) => {
  console.log("ws error", e.message)
  process.exit(1)
}
