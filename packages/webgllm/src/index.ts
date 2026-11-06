/** WebGLLM: LLM inference in the browser on WebGL 2. The engine moves here from tanstack-ai-browser. */
export const version = "0.0.1"

/** Whether this browser can create a WebGL 2 context. */
export function hasWebGl2(): boolean {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(1, 1).getContext("webgl2") !== null
  if (typeof document !== "undefined") return document.createElement("canvas").getContext("webgl2") !== null
  return false
}
