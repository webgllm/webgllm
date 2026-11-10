/** The Hugging Face Hub as recorded, for the resolver's tests. */
import { WebGLLMError } from "./errors.js"
import { hub } from "./resolve-model.js"
import hubFixture from "./test-fixtures/hub.json"

export type Answer = { status: number; body: unknown }

/**
 * The Hub as recorded (test-fixtures/hub.json: API answers trimmed to the
 * fields the resolver reads, and the chat templates and generation configs
 * it fetched), plus `extra` answers; anything else is a 404. Counts calls.
 * ONNX graphs weren't recorded (they're 0.2 to 1 MB each): unless `extra`
 * has one, a graph answers with its own name, a config.json with {}, and
 * the graph checks (decoder and vision encoder) say the graph's weights
 * are in `<graph>_data`, as the exports' are. Tests of the real check set it back.
 */
export function fakeHub(extra: Record<string, Answer | (() => Answer)> = {}) {
  const answers: Record<string, Answer | (() => Answer)> = { ...recorded, ...extra }
  const calls: Array<string> = []
  hub.checkGraph = (bytes: Uint8Array) => ({ dataFiles: [`${new TextDecoder().decode(bytes)}_data`] })
  hub.checkVisionGraph = (bytes: Uint8Array) => [`${new TextDecoder().decode(bytes)}_data`]
  hub.fetch = async (url: string) => {
    calls.push(url)
    const found = answers[url]
    const answer = typeof found === "function" ? found() : found
    if (!answer) {
      if (url.endsWith(".onnx")) return new Response(url.split("/").pop())
      if (url.endsWith("/resolve/main/config.json")) return new Response("{}")
      return new Response("Entry not found", { status: 404, headers: { "X-Error-Code": "EntryNotFound" } })
    }
    const body = answer.body instanceof Uint8Array ? answer.body : typeof answer.body === "string" ? answer.body : JSON.stringify(answer.body)
    return new Response(body as BodyInit, { status: answer.status })
  }
  return calls
}

/** Every recorded answer of repository `from`, as repository `to`'s. */
export function asRepo(from: string, to: string): Record<string, Answer> {
  const out: Record<string, Answer> = {}
  for (const [url, answer] of Object.entries(recorded)) {
    if (url.includes(`/${from}/`) || url.endsWith(`/${from}`)) out[url.replace(from, to)] = answer
  }
  return out
}

export const recorded = hubFixture as Record<string, Answer>

export const api = (repo: string) => `https://huggingface.co/api/models/${repo}`

/** The code a call rejects with ("resolved" if it doesn't). */
export const code = async (promise: Promise<unknown>) => {
  try {
    await promise
  } catch (error) {
    return error instanceof WebGLLMError ? error.code : (error as Error).name
  }
  return "resolved"
}
