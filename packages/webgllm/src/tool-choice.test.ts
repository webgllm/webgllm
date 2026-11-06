import assert from "node:assert/strict"
import { test } from "node:test"
import { planToolChoice } from "./tool-choice.js"
import { validateStructuredValue } from "./constraint/index.js"
import { errorCode } from "./worker-errors.js"
import type { ChatTool } from "./tokenizer.js"

const tool = (name: string, parameters?: unknown): ChatTool => ({ type: "function", function: { name, description: name, parameters } })
const weather = tool("get_weather", { type: "object", properties: { city: { type: "string" } }, required: ["city"], additionalProperties: false })
const time = tool("get_time", { type: "object", properties: { zone: { type: "string", enum: ["UTC", "PST"] } }, required: ["zone"], additionalProperties: false })

test("auto renders the tools and constrains nothing", () => {
  assert.deepEqual(planToolChoice([weather]), { mode: "auto", tools: [weather] })
  assert.deepEqual(planToolChoice([weather], "auto"), { mode: "auto", tools: [weather] })
})

test("none renders no tools", () => {
  assert.deepEqual(planToolChoice([weather, time], "none"), { mode: "none", tools: [] })
})

test("a named tool: its arguments alone, every tool still rendered", async () => {
  const plan = planToolChoice([weather, time], { name: "get_time" })
  assert.ok(plan.mode === "forced")
  assert.deepEqual(plan.tools, [weather, time])
  assert.deepEqual(plan.schema, time.function.parameters)
  assert.deepEqual(plan.read({ zone: "UTC" }), { name: "get_time", arguments: { zone: "UTC" } })
  assert.equal(await validateStructuredValue(plan.schema, { zone: "UTC" }), true)
  assert.equal(await validateStructuredValue(plan.schema, { city: "Paris" }), false)
})

test("required with one tool is that tool", () => {
  const plan = planToolChoice([weather], "required")
  assert.ok(plan.mode === "forced")
  assert.deepEqual(plan.read({ city: "Oslo" }), { name: "get_weather", arguments: { city: "Oslo" } })
})

test("required with several: a name and its own tool's arguments", async () => {
  const plan = planToolChoice([weather, time], "required")
  assert.ok(plan.mode === "forced")
  assert.equal(await validateStructuredValue(plan.schema, { name: "get_weather", arguments: { city: "Oslo" } }), true)
  assert.equal(await validateStructuredValue(plan.schema, { name: "get_time", arguments: { zone: "PST" } }), true)
  assert.equal(await validateStructuredValue(plan.schema, { name: "get_time", arguments: { city: "Oslo" } }), false)
  assert.equal(await validateStructuredValue(plan.schema, { name: "other", arguments: {} }), false)
  assert.deepEqual(plan.read({ name: "get_time", arguments: { zone: "UTC" } }), { name: "get_time", arguments: { zone: "UTC" } })
})

test("tools without parameters take an empty object; untyped parameters are objects", async () => {
  const plan = planToolChoice([tool("ping")], "required")
  assert.ok(plan.mode === "forced")
  assert.equal(await validateStructuredValue(plan.schema, {}), true)
  assert.equal(await validateStructuredValue(plan.schema, { x: 1 }), false)
  const untyped = planToolChoice([tool("f", { properties: { a: { type: "integer" } }, required: ["a"] })], "required")
  assert.ok(untyped.mode === "forced")
  assert.deepEqual(untyped.schema, { type: "object", properties: { a: { type: "integer" } }, required: ["a"], additionalProperties: false })
})

test("unknown names, no tools and non-object parameters fail", () => {
  assert.throws(() => planToolChoice([weather], { name: "nope" }), /nope/)
  assert.throws(() => planToolChoice([], "required"), /at least one tool/)
  assert.throws(() => planToolChoice([], { name: "nope" }), /nope/)
  for (const parameters of [{ type: "string" }, [1], "x"]) {
    let caught: unknown
    try { planToolChoice([tool("f", parameters)], "required") } catch (error) { caught = error }
    assert.equal(errorCode(caught), "invalid-schema")
  }
})

test("arguments that aren't an object are refused", () => {
  const plan = planToolChoice([weather], "required")
  assert.ok(plan.mode === "forced")
  assert.throws(() => plan.read([1]))
})

test("forced calls close objects that don't say otherwise, nested ones too", () => {
  const open = tool("lookup", {
    type: "object",
    properties: { city: { type: "string" }, filters: { type: "object", properties: { units: { type: "string" } } }, tags: { type: "array", items: { type: "object" } } },
    required: ["city"]
  })
  const plan = planToolChoice([open], "required")
  assert.equal(plan.mode, "forced")
  const schema = (plan as { schema: { additionalProperties?: unknown; properties: Record<string, { additionalProperties?: unknown; items?: { additionalProperties?: unknown } }> } }).schema
  assert.equal(schema.additionalProperties, false)
  assert.equal(schema.properties.filters.additionalProperties, false)
  assert.equal(schema.properties.tags.items?.additionalProperties, false)
  assert.equal(schema.properties.city.additionalProperties, undefined)
  // An explicit choice stays.
  const loose = planToolChoice([tool("loose", { type: "object", additionalProperties: true })], "required")
  assert.equal((loose as { schema: { additionalProperties?: unknown } }).schema.additionalProperties, true)
})

test("required with several keeps each tool's local $defs references", async () => {
  const setter = tool("set_value", {
    type: "object",
    properties: { value: { $ref: "#/$defs/value" } },
    required: ["value"],
    $defs: { value: { type: "integer" } }
  })
  const namer = tool("set_name", {
    type: "object",
    properties: { name: { $ref: "#/definitions/value" } },
    required: ["name"],
    definitions: { value: { type: "string" } }
  })
  // One tool alone compiles.
  const one = planToolChoice([setter], "required")
  assert.ok(one.mode === "forced")
  assert.equal(await validateStructuredValue(one.schema, { value: 3 }), true)
  // Several: each branch's references still resolve to its own definitions.
  const plan = planToolChoice([setter, namer], "required")
  assert.ok(plan.mode === "forced")
  assert.equal(await validateStructuredValue(plan.schema, { name: "set_value", arguments: { value: 3 } }), true)
  assert.equal(await validateStructuredValue(plan.schema, { name: "set_value", arguments: { value: "x" } }), false)
  assert.equal(await validateStructuredValue(plan.schema, { name: "set_name", arguments: { name: "x" } }), true)
  assert.equal(await validateStructuredValue(plan.schema, { name: "set_name", arguments: { name: 3 } }), false)
})
