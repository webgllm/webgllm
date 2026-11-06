/** Save/restore only the benchmark origin's derived GPU notes in its one tab. */
import fs from "node:fs"
import { execFileSync } from "node:child_process"
const [socket, action, file] = process.argv.slice(2)
if (!socket || !["save", "restore"].includes(action) || !file)
  throw Error("Expected WebKit socket, save|restore, file")
const notes =
  action === "restore" ? JSON.parse(fs.readFileSync(file, "utf8")) : null
const value = notes?.notes ?? notes?.value ?? notes
const expr =
  action === "save"
    ? `(async()=>{const d=await(await navigator.storage.getDirectory()).getDirectoryHandle('webgl-models');return JSON.parse(await(await(await d.getFileHandle('gpu-notes.json')).getFile()).text())})()`
    : `(async()=>{const d=await(await navigator.storage.getDirectory()).getDirectoryHandle('webgl-models');const w=await(await d.getFileHandle('gpu-notes.json',{create:true})).createWritable();await w.write(${JSON.stringify(JSON.stringify(value))});await w.close();return true})()`
const output = JSON.parse(
  execFileSync(
    process.execPath,
    [
      new URL("./safari-bench.mjs", import.meta.url).pathname,
      socket,
      "eval",
      expr
    ],
    { encoding: "utf8" }
  )
)
if (output.error) throw Error(output.error)
if (action === "save")
  fs.writeFileSync(file, JSON.stringify(output.value, null, 2) + "\n")
console.log(
  action === "save"
    ? `Saved calibration to ${file}`
    : "Restored derived calibration notes"
)
