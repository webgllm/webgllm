/**
 * OpenAI's tool_choice in the worker: which tools the chat template renders,
 * and which ones a call may name (see tool-call-grammar.ts). Also how a
 * schema's objects are closed, as llama.cpp's grammar reads them.
 */
import type { ToolChoice } from "./protocol.js"
import type { ChatTool } from "./tokenizer.js"
import { CodedError } from "./worker-errors.js"

export type ToolPlan = {
  mode: "auto" | "none" | "required"
  /** The tools the template renders. */
  tools: Array<ChatTool>
  /** The tools a call may name, where fewer than all (a named choice). */
  callable?: Array<ChatTool>
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value)

// Where subschemas sit. In maps (properties, $defs...) the keys are names,
// not keywords: a property may be called "default" or "$ref".
const SCHEMA_MAPS = ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"]
const SCHEMA_VALUES = ["items", "additionalItems", "additionalProperties", "unevaluatedItems", "unevaluatedProperties", "propertyNames", "contains", "not", "if", "then", "else"]
const SCHEMA_LISTS = ["prefixItems", "allOf", "anyOf", "oneOf"]

/** Calls `visit` on each direct subschema of `node`, with its path segments. */
function eachSubschema(node: Record<string, unknown>, visit: (child: unknown, path: Array<string>, keyword: string) => void) {
  for (const [key, value] of Object.entries(node)) {
    if (SCHEMA_MAPS.includes(key) && isObject(value)) for (const [name, child] of Object.entries(value)) visit(child, [key, name], key)
    else if (SCHEMA_VALUES.includes(key)) {
      if (Array.isArray(value)) value.forEach((child, i) => visit(child, [key, String(i)], key))
      else visit(value, [key], key)
    } else if (SCHEMA_LISTS.includes(key) && Array.isArray(value)) value.forEach((child, i) => visit(child, [key, String(i)], key))
    else if (key === "dependencies" && isObject(value))
      for (const [name, child] of Object.entries(value)) if (!Array.isArray(child)) visit(child, [key, name], "dependentSchemas")
  }
}

/** `node` with each direct subschema replaced by `map(child, path)`. */
function mapSubschemas(node: Record<string, unknown>, map: (child: unknown, path: Array<string>) => unknown) {
  const out: Record<string, unknown> = { ...node }
  eachSubschema(node, (child, path) => {
    const mapped = map(child, path)
    if (path.length === 1) out[path[0]] = mapped
    else {
      const container = out[path[0]]
      if (Array.isArray(container)) {
        if (container === node[path[0]]) out[path[0]] = [...container]
        ;(out[path[0]] as Array<unknown>)[Number(path[1])] = mapped
      } else {
        if (container === node[path[0]]) out[path[0]] = { ...(container as object) }
        ;(out[path[0]] as Record<string, unknown>)[path[1]] = mapped
      }
    }
  })
  return out
}

const pointer = (path: Array<string>) => "#" + path.map((s) => "/" + s.replace(/~/g, "~0").replace(/\//g, "~1")).join("")

// Keywords that let an object's properties come from somewhere other than
// its own "properties", or pin its value.
const PROPERTY_SOURCES = ["patternProperties", "additionalProperties", "unevaluatedProperties", "allOf", "anyOf", "oneOf", "$ref", "if", "then", "else", "dependentSchemas", "dependencies", "const", "enum"]
// Schemas where narrowing can widen what the whole accepts.
const UNSAFE = new Set(["not", "if", "oneOf", "contains"])
// How far closing reaches into a node: it may close, it keeps its own node
// open (its properties come from elsewhere too), or nothing in it closes.
const CLOSE = 0, OPEN = 1, FROZEN = 2

/**
 * Objects that list their properties take no others unless they say so, as
 * llama.cpp's grammar reads a schema (and OpenAI's strict mode); an object
 * that lists none (`{"type": "object"}`) stays any object, as there. An
 * open object lets a constrained model keep inventing keys, and each token
 * of a free-form key costs llguidance 0.2.1 about a second to mask on a real
 * vocabulary (https://github.com/nico-martin/llguidance-js/issues/5).
 *
 * Only where that just narrows the schema and every declared property stays
 * allowed: an object closes when its own "properties" are all it may have,
 * and nothing under not, if, oneOf or contains closes (narrowing there can
 * accept more). allOf branches, a $ref beside other keywords and anyOf
 * branches beside the parent's own properties stay open.
 *
 * An object made of allOf branches closes over the keys they declare, only
 * when every branch's keys are known exactly and every required key is among
 * them (otherwise it stays open). Branches with nothing but properties and
 * required merge into the object, which accepts the same values: llguidance
 * masks an allOf about 30 times slower than one object (10 s against 0.3 s
 * for a two-key extraction on LFM3). Other branches stay, and each key is
 * listed in the object's own "properties" (as `{}`: the branches still check
 * the value) before additionalProperties: false, which only sees keys
 * declared beside it.
 */
export function closeObjects(schema: unknown): unknown {
  if (!isObject(schema)) return schema
  // References this can't follow to a place in the document: leave it be.
  let resolvable = true
  const check = (node: unknown) => {
    if (!isObject(node)) return
    if ("$id" in node && node !== schema) resolvable = false
    if ("$anchor" in node || "$dynamicRef" in node || "$dynamicAnchor" in node || "$recursiveRef" in node) resolvable = false
    if ("$ref" in node && !(typeof node.$ref === "string" && node.$ref.startsWith("#"))) resolvable = false
    eachSubschema(node, check)
  }
  check(schema)
  if (!resolvable) return schema
  const nodes = new Map<string, Record<string, unknown>>()
  const index = (node: unknown, path: Array<string>) => {
    if (!isObject(node)) return
    nodes.set(pointer(path), node)
    eachSubschema(node, (child, sub) => index(child, [...path, ...sub]))
  }
  index(schema, [])
  // The most a node is held back by any place it's reached from, references included.
  const level = new Map<string, number>()
  const reach = (node: unknown, at: string, l: number): boolean => {
    if (!isObject(node)) return true
    if ((level.get(at) ?? -1) >= l) return true
    level.set(at, l)
    // The keywords other than "properties" that the instance's keys may come from.
    const sources = PROPERTY_SOURCES.filter((k) => k in node)
    const only = (...keys: Array<string>) => !("properties" in node) && sources.every((k) => keys.includes(k))
    let ok = true
    if ("$ref" in node) {
      const ref = decodeURIComponent(node.$ref as string)
      const target = nodes.get(ref === "#" ? "#" : ref)
      if (!target) return false
      ok = reach(target, ref, only("$ref") ? l : Math.max(l, OPEN))
    }
    eachSubschema(node, (child, path, keyword) => {
      let next: number
      if (l === FROZEN || UNSAFE.has(keyword)) next = FROZEN
      else if (keyword === "$defs" || keyword === "definitions") next = CLOSE
      else if (keyword === "anyOf") next = only("anyOf") ? l : OPEN
      else if (keyword === "then" || keyword === "else") next = only("if", "then", "else") ? l : OPEN
      else if (keyword === "allOf" || keyword === "dependentSchemas") next = OPEN
      else next = CLOSE
      if (!reach(child, at === "#" ? pointer(path) : at + pointer(path).slice(1), next)) ok = false
    })
    return ok
  }
  if (!reach(schema, "#", CLOSE)) return schema
  // An allOf branch's declared and required keys, when "properties" are all it may have.
  const branchKeys = (branch: unknown, seen: Set<string>): { keys: Array<string>; required: Array<string> } | null => {
    if (!isObject(branch)) return null
    if ("$ref" in branch) {
      const ref = decodeURIComponent(branch.$ref as string)
      if (Object.keys(branch).length !== 1 || seen.has(ref)) return null
      seen.add(ref)
      return branchKeys(nodes.get(ref), seen)
    }
    if (PROPERTY_SOURCES.some((k) => k in branch)) return null
    if (branch.type !== undefined && branch.type !== "object") return null
    const required = Array.isArray(branch.required) ? (branch.required as Array<string>) : []
    return { keys: isObject(branch.properties) ? Object.keys(branch.properties) : [], required }
  }
  // Every key an object made of allOf branches may have, or null when that can't be known.
  const allOfKeys = (node: Record<string, unknown>): Array<string> | null => {
    if (node.type !== undefined && node.type !== "object") return null
    if (!Array.isArray(node.allOf) || PROPERTY_SOURCES.some((k) => k !== "allOf" && k in node)) return null
    const keys = new Set(isObject(node.properties) ? Object.keys(node.properties) : [])
    const required = new Set(Array.isArray(node.required) ? (node.required as Array<string>) : [])
    for (const branch of node.allOf) {
      const found = branchKeys(branch, new Set())
      if (!found) return null
      found.keys.forEach((k) => keys.add(k))
      found.required.forEach((k) => required.add(k))
    }
    return [...required].every((k) => keys.has(k)) ? [...keys] : null
  }
  // What a branch may hold to merge into its object: nothing that checks the object as a whole.
  const MERGEABLE = new Set(["type", "properties", "required", "title", "description", "$comment"])
  // `node`'s allOf branches merged, closed, into `out` (node with its subschemas closed), or null when one can't merge.
  const merge = (node: Record<string, unknown>, out: Record<string, unknown>, at: string): Record<string, unknown> | null => {
    const branches = (node.allOf as Array<unknown>).map((branch, i) => {
      if (isObject(branch) && "$ref" in branch) {
        const ref = decodeURIComponent(branch.$ref as string)
        return close(nodes.get(ref), ref)
      }
      return close(branch, (at === "#" ? "#" : at) + `/allOf/${i}`)
    })
    if (!branches.every((b) => isObject(b) && Object.keys(b).every((k) => MERGEABLE.has(k)))) return null
    const schemas = new Map<string, Array<unknown>>()
    const required = new Set<string>(Array.isArray(out.required) ? (out.required as Array<string>) : [])
    for (const source of [out, ...(branches as Array<Record<string, unknown>>)]) {
      if (isObject(source.properties)) for (const [key, schema] of Object.entries(source.properties)) schemas.set(key, [...(schemas.get(key) ?? []), schema])
      if (source !== out && Array.isArray(source.required)) (source.required as Array<string>).forEach((k) => required.add(k))
    }
    const rest = { ...out }
    delete rest.allOf
    return {
      ...rest,
      type: "object",
      properties: Object.fromEntries([...schemas].map(([key, list]) => [key, list.length === 1 ? list[0] : { allOf: list }])),
      ...(required.size > 0 && { required: [...required] }),
      additionalProperties: false
    }
  }
  const close = (node: unknown, at: string): unknown => {
    if (!isObject(node)) return node
    const out = mapSubschemas(node, (child, path) => close(child, at === "#" ? pointer(path) : at + pointer(path).slice(1)))
    if (level.get(at) !== CLOSE) return out
    const isObjectType = out.type === "object" || (Array.isArray(out.type) && out.type.includes("object"))
    // As llama.cpp reads a schema: an object that lists its properties takes no
    // others (a schema with "properties" and no type is an object there too);
    // one that lists none, `{"type": "object"}`, is any object.
    if ((isObjectType || out.type === undefined) && "properties" in out && !PROPERTY_SOURCES.some((k) => k in out)) out.additionalProperties = false
    else {
      const keys = allOfKeys(node)
      const merged = keys && merge(node, out, at)
      if (merged) return merged
      if (keys) {
        const own = isObject(out.properties) ? out.properties : {}
        out.properties = { ...Object.fromEntries(keys.map((k) => [k, {}])), ...own }
        out.additionalProperties = false
      }
    }
    return out
  }
  return close(schema, "#")
}

/** A tool a call is held to must describe its parameters as an object. */
function checkParameters(tool: ChatTool) {
  const parameters = tool.function.parameters
  if (parameters === undefined || parameters === null) return
  if (!isObject(parameters))
    throw new CodedError(`Tool ${tool.function.name}'s parameters aren't a JSON Schema object.`, "invalid-schema")
  if (parameters.type !== undefined && parameters.type !== "object")
    throw new CodedError(`Tool ${tool.function.name}'s parameters must describe an object.`, "invalid-schema")
}

export function planToolChoice(tools: Array<ChatTool>, choice: ToolChoice = "auto"): ToolPlan {
  if (choice === "auto") return { mode: "auto", tools }
  // "none" as llama.cpp has it: the template renders the tools as for
  // "auto" (the prompt, and its cache, stay the same), and nothing about a
  // call is constrained or read: markup the model writes stays text.
  if (choice === "none") return { mode: "none", tools }
  // "required" as llama.cpp has it: the model writes its call in its own
  // template's format, as with "auto" (text may come first), held to the
  // tools as it's written (see tool-call-grammar.ts), and the reply can't
  // end before one (see the handler).
  if (choice === "required") {
    if (!tools.length) throw new Error('toolChoice "required" needs at least one tool.')
    tools.forEach(checkParameters)
    return { mode: "required", tools }
  }
  // A named tool (and "any", which repairs use) is "required" with only
  // those tools callable. llama.cpp has no named choice (it reads one as
  // "auto"); OpenAI's is kept, its call written as the model writes any.
  // Every tool stays in the prompt, as "auto" renders it: the model reads
  // them in its own template's words, and the prompt's cache holds.
  if (choice === "any") {
    if (!tools.length) throw new Error('toolChoice "any" needs at least one tool.')
    tools.forEach(checkParameters)
    return { mode: "required", tools }
  }
  const named = tools.find((t) => t.function.name === choice.name)
  if (!named) throw new Error(`toolChoice names ${choice.name}, which isn't one of the tools.`)
  checkParameters(named)
  return { mode: "required", tools, callable: [named] }
}
