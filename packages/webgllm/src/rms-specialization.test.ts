import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import * as K from "./gpu/kernels.js"

for (const parts of [16, 64]) {
  test(`normalization compiles the selected ${parts}-part reduction as a fixed loop`, () => {
    let source = ""
    const engine = Object.create(Engine.prototype) as Engine
    Object.assign(engine, {
      config: { hiddenSize: 2048, normEps: 1e-5 },
      tuned: { rmsParts: parts },
      gpu: {
        program: (value: string) => {
          source = value
          return {}
        },
        run: () => undefined
      }
    })
    Reflect.get(Engine.prototype, "normToH2").call(engine, {}, {}, {}, 512)
    assert.ok(source.includes(`p < ${parts};`))
    assert.ok(!source.includes("if (p >= uParts)"))
    assert.equal(source, K.withRms(K.NORM_TO_H2, parts))
  })
}
