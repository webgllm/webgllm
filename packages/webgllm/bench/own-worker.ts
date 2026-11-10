/** A worker of the caller's own, as the README writes it (own-worker.html). */
import { WebGLLMWorkerHandler } from "../src/index.js"

const handler = new WebGLLMWorkerHandler()
self.onmessage = (event) => handler.onmessage(event)
