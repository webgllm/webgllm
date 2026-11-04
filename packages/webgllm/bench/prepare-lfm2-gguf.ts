/** Prepare exact HF tokenizer/chat inputs for the local GGUF/llama.cpp bench. */
import { mkdir, readFile, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { fileUrl, modelMeta } from "../src/model-meta.js"
import { loadTokenizer } from "../src/tokenizer.js"
const directory = process.argv[2]
if (!directory) throw Error("Usage: prepare-lfm2-gguf <models-directory>")
for (const [key, name] of [["350m","LiquidAI/LFM2.5-350M-GGUF"],["instruct","LiquidAI/LFM2.5-1.2B-Instruct-GGUF"],["thinking","LiquidAI/LFM2.5-1.2B-Thinking-GGUF"]] as const) {
  const meta = modelMeta(name), folder = join(directory,key)
  await mkdir(folder,{recursive:true})
  const files: Record<string,string> = {}
  for (const [kind,file] of Object.entries(meta.tokenizer)) {
    const local = kind === "tokenizer" ? "tokenizer.json" : kind === "config" ? "tokenizer_config.json" : "chat_template.jinja"
    let text: string
    try { text = await readFile(join(folder,local),"utf8") }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error
      const response = await fetch(fileUrl(file))
      if (!response.ok) throw Error(`${file.path}: HTTP ${response.status}`)
      text = await response.text(); await writeFile(join(folder,local),text)
    }
    files[kind] = text
  }
  const tokenizer = loadTokenizer({tokenizerJson:JSON.parse(files.tokenizer),tokenizerConfig:JSON.parse(files.config),chatTemplate:files.chatTemplate})
  const prompts = ["What is the capital of France?","Count from one to three."].map(content => {
    const messages = [{role:"system" as const,content:"You are a helpful assistant."},{role:"user" as const,content}]
    return {messages,ids:tokenizer.encodeChat(messages,{addGenerationPrompt:true}),rendered:tokenizer.applyChatTemplate(messages,{addGenerationPrompt:true})}
  })
  await writeFile(join(folder,"fixtures.json"),JSON.stringify({key,vocab:tokenizer.vocabSize,prompts},null,2))
  console.log(key,prompts.map(prompt=>prompt.ids.length))
}
