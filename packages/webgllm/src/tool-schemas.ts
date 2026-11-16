/**
 * A tool's parameters as its calls are held to and read by them (see
 * tool-call-grammar.ts and tool-calls.ts): the object they describe (a root
 * `$ref` followed to it), each parameter's schema on its own with the
 * references it makes resolved against the whole tool, and whether a value
 * may be a string (written raw in the XML call format, and read back as the
 * text it is, as llama.cpp reads it).
 */
import type { ChatTool } from "./tokenizer.js"
import { mapSubschemas } from "./tool-choice.js"

const isObject = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value)

/** The node a local reference ("#/a/b") points at in `root`, or undefined. */
function resolve(root: Record<string, unknown>, ref: string): unknown {
  if (ref === "#") return root
  if (!ref.startsWith("#/")) return undefined
  let node: unknown = root
  for (const raw of ref.slice(2).split("/")) {
    const key = decodeURIComponent(raw).replace(/~1/g, "/").replace(/~0/g, "~")
    if (!isObject(node) && !Array.isArray(node)) return undefined
    if (!Object.hasOwn(node as object, key)) return undefined
    node = (node as Record<string, unknown>)[key]
  }
  return node
}

/** `schema` with chained `$ref`s (a reference and nothing else of its own) followed, at most 16 deep. */
function followed(root: Record<string, unknown>, schema: unknown): unknown {
  let node = schema
  for (let i = 0; i < 16 && isObject(node) && typeof node.$ref === "string" && !("properties" in node); i++) {
    const target = resolve(root, node.$ref)
    if (target === undefined) return node
    node = target
  }
  return node
}

export type ToolSchema = {
  name: string
  /** The tool's parameters schema as given: what every reference resolves against. */
  root: Record<string, unknown>
  /** The parameters, as the object the schema describes declares them. */
  properties: Array<[string, unknown]>
  required: Array<string>
}

export function toolSchema(tool: ChatTool): ToolSchema {
  const root = isObject(tool.function.parameters) ? tool.function.parameters : {}
  const object = followed(root, root)
  const declared = isObject(object) && isObject(object.properties) ? Object.entries(object.properties) : []
  const names = new Set(declared.map(([name]) => name))
  const required = isObject(object) && Array.isArray(object.required)
    ? (object.required as Array<unknown>).filter((r): r is string => typeof r === "string" && names.has(r))
    : []
  return { name: tool.function.name, root, properties: declared, required }
}

const DEFINITION = /^#\/(\$defs|definitions)\/[^/]+$/

/**
 * A parameter's schema on its own, for the JSON constraint: references into
 * the tool's definitions stay (with the definitions beside them, so a
 * recursive one still works); any other local reference, such as
 * "#/properties/a", is replaced by what it points at.
 */
export function valueSchema(tool: ToolSchema, schema: unknown): object {
  const { root } = tool
  const inline = (node: unknown, through: Set<string>): unknown => {
    if (!isObject(node)) return node
    if (typeof node.$ref === "string" && !DEFINITION.test(node.$ref) && !through.has(node.$ref)) {
      const target = resolve(root, node.$ref)
      if (target !== undefined) {
        const { $ref, ...rest } = node
        const resolved = inline(target, new Set([...through, $ref as string]))
        return Object.keys(rest).length && isObject(resolved) ? { ...resolved, ...rest } : resolved
      }
    }
    return mapSubschemas(node, (child) => inline(child, through))
  }
  const own = inline(isObject(schema) ? schema : {}, new Set())
  const defs = {
    ...(isObject(root.$defs) && { $defs: inline(root.$defs, new Set()) }),
    ...(isObject(root.definitions) && { definitions: inline(root.definitions, new Set()) })
  }
  return isObject(own) ? { ...defs, ...own } : own === true ? {} : { not: {} }
}

/** Whether a value held to `schema` may be a string, its references followed (llama.cpp's may_be_string). */
export function mayBeString(tool: ToolSchema, schema: unknown, seen = new Set<unknown>()): boolean {
  const node = followed(tool.root, schema)
  if (!isObject(node) || seen.has(node)) return node !== false
  seen.add(node)
  const type = node.type
  if (type === "string") return true
  if (Array.isArray(type)) return type.includes("string")
  if (type !== undefined) return false
  for (const key of ["anyOf", "oneOf"])
    if (Array.isArray(node[key]) && (node[key] as Array<unknown>).some((branch) => mayBeString(tool, branch, seen))) return true
  if (Array.isArray(node.enum)) return node.enum.some((v) => typeof v === "string")
  if ("const" in node) return typeof node.const === "string"
  return !["properties", "items", "prefixItems", "anyOf", "oneOf", "allOf", "$ref", "minimum", "maximum"].some((key) => key in node)
}

/** Whether a parameter is declared a string, its references followed (llama.cpp's Python-style check: quoted only then). */
export function isString(tool: ToolSchema, schema: unknown): boolean {
  const node = followed(tool.root, schema)
  return isObject(node) && node.type === "string"
}
