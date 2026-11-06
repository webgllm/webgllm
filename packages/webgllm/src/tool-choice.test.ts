import assert from "node:assert/strict"
import { test } from "node:test"
import { closeObjects, planToolChoice } from "./tool-choice.js"
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

// Closing must only narrow what a schema accepts, and never reject a value
// that keeps to the properties the schema declares.
async function sameVerdicts(schema: unknown, values: Array<unknown>) {
  const closed = closeObjects(schema)
  for (const value of values)
    assert.equal(await validateStructuredValue(closed, value), await validateStructuredValue(schema, value), `${JSON.stringify(value)} against ${JSON.stringify(closed)}`)
}

test("closing leaves objects under not open", async () => {
  await sameVerdicts({ type: "object", properties: { a: {}, b: {} }, not: { type: "object", properties: { a: { type: "string" } } } }, [{}, { a: "x" }, { a: "x", b: 1 }, { b: 1 }])
  await sameVerdicts(
    { type: "object", properties: { v: { not: { $ref: "#/$defs/a" } } }, $defs: { a: { type: "object", properties: { a: { type: "string" } } } } },
    [{ v: { a: "x", b: 1 } }, { v: 1 }]
  )
})

test("closing leaves objects open whose properties come from allOf, a $ref, const or enum", async () => {
  await sameVerdicts(
    { type: "object", allOf: [{ type: "object", properties: { a: { type: "string" } } }, { type: "object", properties: { b: { type: "string" } } }] },
    [{ a: "x", b: "y" }, { a: 1 }]
  )
  await sameVerdicts(
    { allOf: [{ $ref: "#/$defs/a" }, { $ref: "#/$defs/b" }], $defs: { a: { type: "object", properties: { a: { type: "string" } } }, b: { type: "object", properties: { b: { type: "string" } } } } },
    [{ a: "x", b: "y" }, { a: 1 }]
  )
  await sameVerdicts({ type: "object", $ref: "#/$defs/x", $defs: { x: { type: "object", properties: { a: { type: "string" } } } } }, [{ a: "x" }, { a: 1 }])
  await sameVerdicts({ type: "object", const: { a: 1 } }, [{ a: 1 }, {}])
  await sameVerdicts({ type: "object", enum: [{ a: 1 }, { b: 2 }] }, [{ a: 1 }, { b: 2 }, {}])
})

test("closing keeps oneOf, if/then/else, dependentSchemas and contains as they were", async () => {
  await sameVerdicts({ oneOf: [{ type: "object", properties: { a: { type: "string" } } }, { type: "object", properties: { b: { type: "string" } } }] }, [{ a: "x" }, { b: "y" }, { a: "x", b: "y" }])
  await sameVerdicts(
    { type: "object", properties: { k: { type: "string" } }, if: { type: "object", properties: { k: { const: "a" } } }, then: { type: "object", properties: { x: { type: "string" } }, required: ["x"] } },
    [{ k: "a", x: "1" }, { k: "b" }, { k: "a" }]
  )
  await sameVerdicts({ type: "object", properties: { k: { type: "string" } }, dependentSchemas: { k: { type: "object", properties: { x: { type: "string" } }, required: ["x"] } } }, [{ k: "a", x: "1" }, {}])
  await sameVerdicts({ type: "object", properties: { k: { type: "string" } }, anyOf: [{ type: "object", properties: { x: { type: "string" } }, required: ["x"] }] }, [{ k: "a", x: "1" }])
  await sameVerdicts({ type: "array", contains: { type: "object", properties: { a: { type: "string" } } }, maxContains: 1 }, [[{ a: "x" }, { a: "y", b: 1 }], [{ a: "x" }]])
})

test("closing still closes plain objects, pure unions and referenced definitions", () => {
  const closed = closeObjects({
    type: "object",
    properties: { author: { $ref: "#/$defs/person" }, pet: { anyOf: [{ type: "object", properties: { cat: { type: "string" } } }, { type: "object", properties: { dog: { type: "string" } } }, { type: "null" }] } },
    $defs: { person: { type: "object", properties: { name: { type: "string" }, address: { type: "object", properties: { city: { type: "string" } } } } } }
  }) as any
  assert.equal(closed.additionalProperties, false)
  assert.equal(closed.$defs.person.additionalProperties, false)
  assert.equal(closed.$defs.person.properties.address.additionalProperties, false)
  assert.equal(closed.properties.pet.anyOf[0].additionalProperties, false)
  assert.equal(closed.properties.pet.anyOf[1].additionalProperties, false)
  // A root that only points at its model closes the model.
  const root = closeObjects({ $ref: "#/$defs/m", $defs: { m: { type: "object", properties: { a: { type: "string" } } } } }) as any
  assert.equal(root.$defs.m.additionalProperties, false)
  // Properties of an allOf branch are new values: they close.
  const all = closeObjects({ allOf: [{ type: "object", properties: { inner: { type: "object", properties: { a: { type: "string" } } } } }] }) as any
  assert.equal(all.allOf[0].additionalProperties, undefined)
  assert.equal(all.allOf[0].properties.inner.additionalProperties, false)
})

test("required with several scopes references under properties named like data keywords", async () => {
  const params = { type: "object", properties: { default: { $ref: "#/$defs/v" }, const: { $ref: "#/$defs/v" } }, required: ["default"], $defs: { v: { type: "integer" } } }
  const a = tool("a", params)
  const b = tool("b", { type: "object", properties: { x: { type: "string" } }, required: ["x"] })
  const one = planToolChoice([a], "required")
  assert.ok(one.mode === "forced")
  assert.equal(await validateStructuredValue(one.schema, { default: 3 }), true)
  const plan = planToolChoice([a, b], "required")
  assert.ok(plan.mode === "forced")
  assert.equal(await validateStructuredValue(plan.schema, { name: "a", arguments: { default: 3, const: 4 } }), true)
  assert.equal(await validateStructuredValue(plan.schema, { name: "a", arguments: { default: "x" } }), false)
  assert.equal(await validateStructuredValue(plan.schema, { name: "b", arguments: { x: "y" } }), true)
})
