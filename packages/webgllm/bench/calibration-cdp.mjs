/** Save/restore only derived GPU calibration notes in the one WebGL benchmark tab. */
import fs from "node:fs"
const [action, file, cdpOrigin = "http://127.0.0.1:9223"] =
  process.argv.slice(2)
if (!["save", "restore"].includes(action) || !file)
  throw new Error("Usage: node bench/calibration-cdp.mjs save|restore FILE")
const targets = await fetch(`${cdpOrigin}/json/list`).then((r) => r.json())
const pages = targets.filter(
  (t) => t.type === "page" && t.url.startsWith("http://localhost:8788/")
)
if (pages.length !== 1)
  throw new Error(`Expected one benchmark tab; found ${pages.length}`)
const ws = new WebSocket(pages[0].webSocketDebuggerUrl)
await new Promise((r) => (ws.onopen = r))
const content =
  action === "restore" ? JSON.parse(fs.readFileSync(file, "utf8")) : null
const expression =
  action === "save"
    ? `(async()=>{const d=await(await navigator.storage.getDirectory()).getDirectoryHandle('webgl-models');return JSON.parse(await(await(await d.getFileHandle('gpu-notes.json')).getFile()).text())})()`
    : `(async()=>{const d=await(await navigator.storage.getDirectory()).getDirectoryHandle('webgl-models');const w=await(await d.getFileHandle('gpu-notes.json',{create:true})).createWritable();await w.write(${JSON.stringify(JSON.stringify(content.notes ?? content))});await w.close();return true})()`
ws.onmessage = ({ data }) => {
  const m = JSON.parse(data)
  if (m.id) {
    if (m.error || m.result?.exceptionDetails)
      throw new Error(JSON.stringify(m))
    const value = m.result?.result?.value
    if (action === "save")
      fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n")
    console.log(
      action === "save"
        ? `Saved calibration to ${file}`
        : "Restored derived calibration notes"
    )
    ws.close()
  }
}
ws.send(
  JSON.stringify({
    id: 1,
    method: "Runtime.evaluate",
    params: { expression, returnByValue: true, awaitPromise: true }
  })
)
