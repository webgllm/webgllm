/**
 * logits.html: the token- and logit-level API on a real model. Log
 * probabilities and a logit bias on chat completions, a raw prompt through
 * completions.create, then WebLLM's forwardTokensAndSample, resetChat and
 * LogitProcessor in a worker of the page's own (logits-worker.ts).
 */
import { fileUrl, WEBGL_MODEL_META } from "../src/model-meta.js"
import type { WebGlTextModel } from "../src/model-meta.js"
import { loadTokenizer } from "../src/tokenizer.js"

// The package's entry from its own bundle (not this one), so its `new URL("./worker.js", import.meta.url)` is /bundle/src/worker.js.
const entry = "/bundle/src/index.js"
const { WebGLLM } = (await import(entry)) as typeof import("../src/index.js")

const out = document.getElementById("out") as HTMLPreElement
const log = (...parts: Array<unknown>) => {
  const line = parts.map((p) => (typeof p === "string" ? p : JSON.stringify(p))).join(" ")
  out.textContent += `${line}\n`
  void fetch("/log", { method: "POST", body: line }).catch(() => undefined)
}
let failed = 0
const check = (name: string, ok: boolean, detail?: unknown) => {
  if (!ok) failed++
  log(`${ok ? "PASS" : "FAIL"} ${name}`, detail ?? "")
}
const model = (new URLSearchParams(location.search).get("model") ?? "LiquidAI/LFM2.5-350M-GGUF") as WebGlTextModel
const PROMPT = "The capital of France is"
const messages = [{ role: "user" as const, content: "What is the capital of France? Answer in one short sentence." }]

try {
  // The model's tokenizer, on the page: token ids for the bias and the forward pass.
  const files = WEBGL_MODEL_META[model].tokenizer
  const json = (url: string) => fetch(url).then((r) => r.json())
  const tokenizer = loadTokenizer({
    tokenizerJson: await json(fileUrl(files.tokenizer)),
    tokenizerConfig: await json(fileUrl(files.config)),
    ...("chatTemplate" in files && files.chatTemplate && { chatTemplate: await fetch(fileUrl(files.chatTemplate)).then((r) => r.text()) })
  })

  const webgllm = new WebGLLM()
  const plain = await webgllm.chat.completions.create({ model, messages, temperature: 0, max_tokens: 24 })
  check("no logprobs unless asked for", plain.choices[0].logprobs === null)

  const scored = await webgllm.chat.completions.create({ model, messages, temperature: 0, max_tokens: 24, logprobs: true, top_logprobs: 3 })
  const content = scored.choices[0].logprobs?.content ?? []
  check("logprobs: the same reply as without them", scored.choices[0].message.content === plain.choices[0].message.content, scored.choices[0].message.content)
  check(
    "logprobs: one entry a token, each a log probability with its 3 likeliest",
    content.length === scored.usage.completion_tokens - (scored.choices[0].finish_reason === "stop" ? 1 : 0) &&
      content.every((t) => t.logprob <= 0 && Number.isFinite(t.logprob) && t.top_logprobs.length === 3 && t.top_logprobs[0].logprob >= t.top_logprobs[1].logprob),
    content.slice(0, 4).map((t) => [t.token, +t.logprob.toFixed(3), t.top_logprobs.map((a) => a.token)])
  )
  check("logprobs: greedy writes the likeliest token", content.every((t) => t.top_logprobs[0].token === t.token))
  const textOf = (entries: typeof content) => new TextDecoder().decode(Uint8Array.from(entries.flatMap((t) => t.bytes)))
  check("logprobs: the tokens' bytes are the reply", textOf(content).includes(scored.choices[0].message.content ?? "\0"))

  // A character split across tokens: each token has its own bytes, and together they're the character.
  const emoji = await webgllm.chat.completions.create({
    model, messages: [{ role: "user", content: "Repeat exactly: 🧪🧪🧪" }], temperature: 0, max_tokens: 16, logprobs: true, top_logprobs: 2
  })
  const emojiTokens = emoji.choices[0].logprobs?.content ?? []
  check(
    "logprobs: bytes of tokens that split a character decode to the reply, with no replacement characters",
    /🧪/.test(emoji.choices[0].message.content ?? "") && textOf(emojiTokens).includes(emoji.choices[0].message.content ?? "\0") && !textOf(emojiTokens).includes("\ufffd"),
    [emoji.choices[0].message.content, emojiTokens.map((t) => t.bytes)]
  )

  // A stop sequence: the reply ends before it, and so do the tokens reported.
  const cut = await webgllm.chat.completions.create({ model, messages, temperature: 0, max_tokens: 24, logprobs: true, stop: [" Paris"] })
  const cutTokens = cut.choices[0].logprobs?.content ?? []
  check(
    "logprobs: no entry for text a stop sequence cut",
    !/Paris/.test(cut.choices[0].message.content ?? "Paris") && !/Paris/.test(textOf(cutTokens)) && textOf(cutTokens) === cut.choices[0].message.content,
    [cut.choices[0].message.content, cutTokens.map((t) => t.token)]
  )

  // A stop sequence inside a token: the token's text before it is written, and the token has no entry.
  const inside = await webgllm.chat.completions.create({ model, messages, temperature: 0, max_tokens: 24, logprobs: true, stop: ["aris"] })
  const insideTokens = inside.choices[0].logprobs?.content ?? []
  const insideText = inside.choices[0].message.content ?? ""
  check(
    "logprobs: a token a stop sequence cuts into has no entry",
    insideText.endsWith(" P") && insideText.startsWith(textOf(insideTokens)) && !/P$/.test(textOf(insideTokens)),
    [insideText, insideTokens.map((t) => t.token)]
  )

  const streamed = await webgllm.chat.completions.create({ model, messages, temperature: 0, max_tokens: 24, logprobs: true, stream: true })
  let streamedTokens = 0
  for await (const chunk of streamed) streamedTokens += chunk.choices[0]?.logprobs?.content?.length ?? 0
  check("logprobs: streamed chunks carry them", streamedTokens === content.length, streamedTokens)

  // The reply's first token, banned: another is written in its place.
  const first = tokenizer.encode(content[0].token, { addSpecialTokens: false })
  const biased = await webgllm.chat.completions.create({
    model, messages, temperature: 0, max_tokens: 24, logprobs: true,
    logit_bias: Object.fromEntries(first.map((t) => [t, -100]))
  })
  const biasedFirst = biased.choices[0].logprobs?.content?.[0]?.token
  check("logit_bias: a banned token isn't written", first.length === 1 && biasedFirst !== undefined && biasedFirst !== content[0].token, [content[0].token, biasedFirst])

  const text = await webgllm.completions.create({ model, prompt: PROMPT, temperature: 0, max_tokens: 6, echo: true })
  check("completions.create: a raw prompt, continued", text.choices[0].text.startsWith(PROMPT) && /Paris/.test(text.choices[0].text), text.choices[0].text)
  check("completions.create: the prompt's tokens are counted as they are, with no chat template", text.usage?.prompt_tokens === tokenizer.encode(PROMPT).length, text.usage)
  let refused = ""
  await webgllm.completions.create({ model, prompt: "" }).catch((error: Error) => (refused = error.message))
  check("completions.create: an empty prompt is refused", /empty/.test(refused), refused)
  await webgllm.unloadModel()

  // WebLLM's token-level API, in a worker whose handler has a logit processor.
  const worker = new Worker(`/bundle/bench/logits-worker.js?model=${encodeURIComponent(model)}`, { type: "module", name: "logits" })
  const seen = { logits: [] as Array<{ length: number; argmax: number }>, tokens: [] as Array<number>, resets: 0 }
  worker.addEventListener("message", ({ data }) => {
    if (data?.type === "bench-logits") seen.logits.push(data)
    else if (data?.type === "bench-token") seen.tokens.push(data.token)
    else if (data?.type === "bench-reset") seen.resets++
  })
  const own = new WebGLLM({ worker })
  const ids = tokenizer.encode(PROMPT)
  const next = await own.forwardTokensAndSample(ids, true, model)
  check("forwardTokensAndSample: the next token after the prompt's ids", /Paris/.test(tokenizer.decode([next])), [next, tokenizer.decode([next])])
  check("LogitProcessor: given every logit, and its choice is the token returned", seen.logits.length === 1 && seen.logits[0].length === tokenizer.vocabSize && seen.logits[0].argmax === next && seen.tokens.at(-1) === next, seen.logits)
  // The token passed back in continues the sequence.
  const after = await own.forwardTokensAndSample([next], false)
  await own.resetChat()
  check("resetChat: the processor is reset", seen.resets === 1, seen.resets)
  const again = await own.forwardTokensAndSample(ids, true)
  const afterAgain = await own.forwardTokensAndSample([next], false)
  check("resetChat: the same ids give the same tokens again", again === next && afterAgain === after, [tokenizer.decode([next, after]), tokenizer.decode([again, afterAgain])])
  // The same prompt through completions.create writes those tokens.
  seen.tokens.length = 0
  const same = await own.completions.create({ model, prompt: PROMPT, max_tokens: 2 })
  check("LogitProcessor: completions sample from its logits too", seen.tokens[0] === next && seen.tokens[1] === after, [same.choices[0].text, seen.tokens])
  await own.unloadModel()
  log(failed ? `ERROR ${failed} check${failed === 1 ? "" : "s"} failed` : "ALL DONE")
} catch (error) {
  log("ERROR", String((error as Error)?.stack ?? error))
}
