/**
 * The worker webgllm starts for each model: the handler (handler.ts) behind
 * the worker's messages.
 */
import { WebGLLMWorkerHandler } from "./handler.js"

const handler = new WebGLLMWorkerHandler()
self.onmessage = (event) => handler.onmessage(event)
