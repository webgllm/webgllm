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
  // Required keys from every schema on the way to the object: a root `$ref`'s siblings require too.
  const wanted = new Set<string>()
  for (let node: unknown = root, i = 0; isObject(node) && i < 17; i++) {
    if (Array.isArray(node.required)) for (const r of node.required) if (typeof r === "string") wanted.add(r)
    if (node === object || typeof node.$ref !== "string") break
    node = resolve(root, node.$ref)
  }
  return { name: tool.function.name, root, properties: declared, required: declared.map(([name]) => name).filter((name) => wanted.has(name) && names.has(name)) }
}

const DEFINITION = /^#\/(\$defs|definitions)\/[^/]+$/
/** Keywords that only describe a schema: beside a `$ref`, they're kept as they are. */
const ANNOTATIONS = new Set(["title", "description", "$comment", "examples", "default", "deprecated", "readOnly", "writeOnly"])

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
        const constraints = Object.keys(rest).filter((key) => !ANNOTATIONS.has(key))
        if (!constraints.length) return Object.keys(rest).length && isObject(resolved) ? { ...resolved, ...rest } : resolved
        // Both hold, as beside a reference: the siblings narrow what it points at, never replace it.
        return { allOf: [resolved, inline(rest, through)] }
      }
    }
    return mapSubschemas(node, (child) => inline(child, through))
  }
  const own = inline(isObject(schema) ? schema : {}, new Set())
  // Each definition on its own, its references resolved against the tool too.
  const definitions = (map: unknown) =>
    Object.fromEntries(Object.entries(map as Record<string, unknown>).map(([name, sub]) => [name, inline(sub, new Set())]))
  const defs = {
    ...(isObject(root.$defs) && { $defs: definitions(root.$defs) }),
    ...(isObject(root.definitions) && { definitions: definitions(root.definitions) })
  }
  return isObject(own) ? { ...defs, ...own } : own === true ? {} : { not: {} }
}

/**
 * Whether a value held to `schema` may be a string (llama.cpp's
 * may_be_string): what a `$ref` points at and the keywords beside it both
 * allow one. `path` holds the schemas on the way here, for references that
 * come round again (each branch has its own).
 */
export function mayBeString(tool: ToolSchema, schema: unknown, path: ReadonlySet<unknown> = new Set()): boolean {
  if (schema === false) return false
  if (!isObject(schema) || path.has(schema)) return true
  const within = new Set(path).add(schema)
  if (typeof schema.$ref === "string") {
    const { $ref, ...rest } = schema
    const target = resolve(tool.root, $ref as string)
    return (target === undefined || mayBeString(tool, target, within)) && mayBeString(tool, rest, within)
  }
  const type = schema.type
  if (type !== undefined && type !== "string" && !(Array.isArray(type) && type.includes("string"))) return false
  if (Array.isArray(schema.enum) && !schema.enum.some((v) => typeof v === "string")) return false
  if ("const" in schema && typeof schema.const !== "string") return false
  if (Array.isArray(schema.allOf) && !schema.allOf.every((branch) => mayBeString(tool, branch, within))) return false
  for (const key of ["anyOf", "oneOf"])
    if (Array.isArray(schema[key]) && !(schema[key] as Array<unknown>).some((branch) => mayBeString(tool, branch, within))) return false
  // No type: keywords that only an object, an array or a number has say it isn't one.
  if (type === undefined && ["properties", "items", "prefixItems", "minimum", "maximum"].some((key) => key in schema)) return false
  return true
}

/** Whether a parameter is declared a string, its references followed (llama.cpp's Python-style check: quoted only then). */
export function isString(tool: ToolSchema, schema: unknown): boolean {
  const node = followed(tool.root, schema)
  return isObject(node) && node.type === "string"
}
