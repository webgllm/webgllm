/**
 * OpenAI's tool_choice in the worker: which tools the chat template renders,
 * and for a forced call, the JSON Schema its answer is constrained to.
 */
import type { ToolChoice } from "./protocol.js"
import type { ChatTool } from "./tokenizer.js"
import { CodedError } from "./worker-errors.js"

export type ToolCall = { name: string; arguments: Record<string, unknown> }

export type ToolPlan =
  | { mode: "auto" | "none"; tools: Array<ChatTool> }
  | {
      mode: "forced"
      tools: Array<ChatTool>
      /** The answer's JSON Schema: the tool's arguments, or `{ name, arguments }` for one of several. */
      schema: object
      /** The call the schema-valid answer stands for. */
      read: (value: unknown) => ToolCall
    }

const isObject = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value)

/**
 * Objects that don't say otherwise take no other properties, as in OpenAI's
 * strict mode. An open object lets a constrained model keep inventing keys
 * and never close it, running to the output limit.
 */
export function closeObjects(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(closeObjects)
  if (!isObject(schema)) return schema
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(schema)) {
    if (key === "properties" || key === "$defs" || key === "definitions" || key === "patternProperties") {
      out[key] = isObject(value) ? Object.fromEntries(Object.entries(value).map(([k, v]) => [k, closeObjects(v)])) : value
    } else if (key === "items" || key === "prefixItems" || key === "anyOf" || key === "oneOf" || key === "allOf" || key === "not") {
      out[key] = closeObjects(value)
    } else out[key] = value
  }
  const isObjectType = out.type === "object" || (Array.isArray(out.type) && out.type.includes("object"))
  if (isObjectType && out.additionalProperties === undefined && out.patternProperties === undefined) out.additionalProperties = false
  return out
}

/** A tool's parameters as an object schema (no parameters: an empty object). */
function argumentSchema(tool: ChatTool): Record<string, unknown> {
  const parameters = tool.function.parameters ?? { type: "object", properties: {}, additionalProperties: false }
  if (!isObject(parameters))
    throw new CodedError(`Tool ${tool.function.name}'s parameters aren't a JSON Schema object.`, "invalid-schema")
  if (parameters.type !== undefined && parameters.type !== "object")
    throw new CodedError(`Tool ${tool.function.name}'s parameters must describe an object.`, "invalid-schema")
  return closeObjects({ type: "object", ...parameters }) as Record<string, unknown>
}

/** Keywords whose values are data, not schemas: a "$ref" there isn't a reference. */
const DATA_KEYWORDS = new Set(["const", "enum", "default", "examples"])

/** `schema` with its local references ("#..." pointers) moved beneath `base`. */
function scopeRefs(schema: unknown, base: string): unknown {
  if (Array.isArray(schema)) return schema.map((s) => scopeRefs(s, base))
  if (!isObject(schema)) return schema
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(schema)) {
    if (key === "$ref" && typeof value === "string" && value.startsWith("#")) out[key] = base + value.slice(1)
    else out[key] = DATA_KEYWORDS.has(key) ? value : scopeRefs(value, base)
  }
  return out
}

const argumentsOf = (value: unknown, name: string) => {
  if (!isObject(value)) throw new Error(`Tool ${name}'s arguments aren't a JSON object.`)
  return value
}

export function planToolChoice(tools: Array<ChatTool>, choice: ToolChoice = "auto"): ToolPlan {
  if (choice === "auto") return { mode: "auto", tools }
  // "none" renders no tools, as if none were offered: the template then
  // neither describes them nor asks for calls, so the model answers in text
  // (past calls in the chat still render). Keeping the tools and banning the
  // call markup instead would need each format's markers in the worker, and
  // a model that meant to call would write the call some other way.
  if (choice === "none") return { mode: "none", tools: [] }
  // Forced calls keep every tool in the prompt, exactly as "auto" renders
  // it: the model reads the tools in its own template's words, and the
  // prompt stays the same for the cache across auto and forced turns.
  let chosen: Array<ChatTool>
  if (choice === "required") {
    if (!tools.length) throw new Error('toolChoice "required" needs at least one tool.')
    chosen = tools
  } else {
    const named = tools.find((t) => t.function.name === choice.name)
    if (!named) throw new Error(`toolChoice names ${choice.name}, which isn't one of the tools.`)
    chosen = [named]
  }
  if (chosen.length === 1) {
    // One possible tool: its arguments alone, as for a named tool.
    const name = chosen[0].function.name
    return { mode: "forced", tools, schema: argumentSchema(chosen[0]), read: (value) => ({ name, arguments: argumentsOf(value, name) }) }
  }
  // Several: the name first (each branch fixes it), then its arguments.
  // Each tool's arguments sit in the root's $defs, and its local references
  // point beneath them: nested under anyOf, "#/$defs/x" would resolve
  // against the union's root instead of the tool's own schema.
  const $defs: Record<string, unknown> = {}
  const schema = {
    anyOf: chosen.map((tool, i) => {
      const key = `tool${i}`
      $defs[key] = scopeRefs(argumentSchema(tool), `#/$defs/${key}`)
      return {
        type: "object",
        properties: { name: { const: tool.function.name }, arguments: { $ref: `#/$defs/${key}` } },
        required: ["name", "arguments"],
        additionalProperties: false
      }
    }),
    $defs
  }
  return {
    mode: "forced",
    tools,
    schema,
    read: (value) => {
      if (!isObject(value) || typeof value.name !== "string") throw new Error("The forced tool call has no tool name.")
      return { name: value.name, arguments: argumentsOf(value.arguments, value.name) }
    }
  }
}
