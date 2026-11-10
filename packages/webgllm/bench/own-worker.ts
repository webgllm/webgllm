/** A worker of the caller's own, as the README writes it (own-worker.html), with a message of its own beside webgllm's. */
import { WebGLLMWorkerHandler } from "../src/index.js"

const handler = new WebGLLMWorkerHandler()
self.onmessage = (event) => handler.onmessage(event)

// The page's own traffic: answered here, ignored by the handler.
self.addEventListener("message", ({ data }) => {
  if (data?.type === "ping") postMessage({ type: "pong", model: data.model })
})
