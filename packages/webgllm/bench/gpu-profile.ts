/** Fence-bounded diagnostic timings; skipped operations are profiling only. */
import { Engine, TUNING } from "../src/engine.js"
import type { QuantMatrix } from "../src/weights.js"
import { GL, type Pass, type Texture } from "../src/gpu/gl.js"

export type UploadSample = { group: number; bytes: Uint8Array }

export async function profileGpu(
  engine: Engine,
  report: (data: object) => void,
  nativeSample?: QuantMatrix,
  qualityOnly = false,
  uploadSamples?: UploadSample[]
) {
  const compare = (a: Float32Array, b: Float32Array) => {
    let max = 0,
      dd = 0,
      aa = 0,
      pa = 0,
      pb = 0,
      kl = 0,
      ia = 0,
      ib = 0
    let ma = -Infinity,
      mb = -Infinity
    for (const v of a) ma = Math.max(ma, v)
    for (const v of b) mb = Math.max(mb, v)
    for (let i = 0; i < a.length; i++) {
      if (!Number.isFinite(a[i]) || !Number.isFinite(b[i]))
        throw new Error("Non-finite logits")
      max = Math.max(max, Math.abs(a[i] - b[i]))
      dd += (a[i] - b[i]) ** 2
      aa += a[i] ** 2
      pa += Math.exp(a[i] - ma)
      pb += Math.exp(b[i] - mb)
      if (a[i] > a[ia]) ia = i
      if (b[i] > b[ib]) ib = i
    }
    for (let i = 0; i < a.length; i++) {
      const p = Math.exp(a[i] - ma) / pa,
        q = Math.exp(b[i] - mb) / pb
      if (p > 0) kl += p * Math.log(p / Math.max(q, 1e-300))
    }
    return {
      maxAbs: max,
      relativeRms: Math.sqrt(dd / aa),
      kl,
      top1: ia === ib,
      logits: a.length
    }
  }
  {
    const original = TUNING.fusePromptInputs,
      rmsParts = TUNING.rmsParts,
      tunedParts = engine.tuned.rmsParts
    const values: Float32Array[] = []
    const tokens = Array.from(
      { length: 512 },
      (_, i) => (i * 7919 + 13) % engine.config.vocabSize
    )
    try {
      for (const fused of [0, 1]) {
        TUNING.fusePromptInputs = fused
        TUNING.rmsParts = fused ? rmsParts : 64
        engine.tuned.rmsParts = fused ? (tunedParts ?? rmsParts) : 64
        engine.reset()
        const row = await engine.forwardHidden(tokens)
        values.push(await engine.logits(row))
      }
      report({
        stage: "quality",
        kind: "fused-prefill-inputs-vs-original-full-model",
        ...compare(values[0], values[1])
      })
    } finally {
      TUNING.fusePromptInputs = original
      TUNING.rmsParts = rmsParts
      engine.tuned.rmsParts = tunedParts
      engine.reset()
    }
  }
  {
    const tokens = Array.from(
      { length: 2085 },
      (_, i) => (i * 7919 + 13) % engine.config.vocabSize
    )
    const values: Float32Array[] = []
    for (const prepared of [false, true]) {
      engine.reset()
      if (prepared) await engine.preparePrompt(tokens.length)
      values.push(await engine.logits(await engine.forwardHidden(tokens)))
    }
    report({
      stage: "quality",
      kind: "prepared-vs-incremental-cache-full-model",
      ...compare(values[0], values[1])
    })
    engine.reset()
  }
  if (uploadSamples?.length) {
    const embedding = Reflect.get(engine, "embedding") as {
      q6Expanded?: boolean
      quants: Texture
      quantPanels?: Texture[]
      rowsPerColumn: number
      rowTexels: number
    }
    if (embedding.q6Expanded) {
      const gpu = engine.gpu,
        scratch = gpu.texture(GL.RGBA32F, embedding.rowTexels, 1)
      const copy = gpu.program(`#version 300 es
precision highp float;precision highp int;uniform sampler2D uX;uniform int uX0;uniform int uY0;out vec4 o;
void main(){o=texelFetch(uX,ivec2(gl_FragCoord.xy)+ivec2(uX0,uY0),0);}`)
      let checked = 0,
        mismatches = 0
      try {
        const panels = embedding.quantPanels ?? [embedding.quants],
          panelRows = panels[0].height
        for (const sample of uploadSamples) {
          const y = sample.group % embedding.rowsPerColumn,
            column = Math.floor(sample.group / embedding.rowsPerColumn),
            panel = Math.floor(y / panelRows)
          gpu.run({
            program: copy,
            inputs: { uX: panels[panel] },
            uniforms: {
              uX0: { int: column * embedding.rowTexels },
              uY0: { int: y % panelRows }
            },
            output: scratch
          })
          const got = await gpu.read(scratch)
          for (let i = 0; i < sample.bytes.length; i++) {
            if (Math.round(got[i] * 255) !== sample.bytes[i]) mismatches++
            checked++
          }
        }
        report({
          stage: "quality",
          kind: "upload-byte-exact",
          groups: uploadSamples.length,
          bytesChecked: checked,
          mismatches
        })
        if (mismatches)
          throw new Error(
            `${mismatches} repacked upload bytes differ from the CPU reference`
          )
      } finally {
        gpu.delete(scratch)
      }
    }
  }
  if (Reflect.get(engine, "embedding").q6Expanded) {
    const choice = engine.headChoice
    const values: Float32Array[] = []
    try {
      for (const split of [1, 2]) {
        engine.headChoice = { expanded: true, split }
        values.push(await engine.logits(0))
      }
      report({
        stage: "quality",
        kind: "expanded-unsplit-vs-split-full-vocabulary",
        ...compare(values[0], values[1])
      })
    } finally {
      engine.headChoice = choice
    }
  }
  if (nativeSample) {
    type Matrix = {
      quants: Texture
      scales: Texture
      zeros: Texture | null
      quantPanels?: Texture[]
    }
    const e = engine as unknown as {
      embedding: Matrix
      config: { vocabSize: number }
      allocateMatrix: (n: number, k: number, bits: 6, zeros: boolean) => Matrix
      writeMatrix: (gpu: Matrix, m: QuantMatrix, row: number) => void
    }
    const original = e.embedding,
      vocab = e.config.vocabSize,
      choice = engine.headChoice,
      setting = TUNING.q6Expanded
    const made: Matrix[] = [],
      logits: Float32Array[] = []
    try {
      e.config.vocabSize = nativeSample.N
      for (const expanded of [false, true]) {
        TUNING.q6Expanded = expanded ? 1 : 0
        const gm = e.allocateMatrix(nativeSample.N, nativeSample.K, 6, false)
        e.writeMatrix(gm, nativeSample, 0)
        gm.quantPanels ??= [gm.quants]
        made.push(gm)
        e.embedding = gm
        engine.headChoice = { expanded, split: 2 }
        TUNING.q6Expanded = -1
        logits.push(await engine.logits(0))
      }
      report({
        stage: "quality",
        kind: "native-vs-expanded-real-hidden-state",
        ...compare(logits[0], logits[1])
      })
    } finally {
      e.embedding = original
      e.config.vocabSize = vocab
      engine.headChoice = choice
      TUNING.q6Expanded = setting
      for (const gm of made)
        for (const t of [
          ...(gm.quantPanels ?? [gm.quants]),
          gm.scales,
          gm.zeros
        ])
          if (t) engine.gpu.delete(t)
    }
    const at = engine.position,
      stream = TUNING.streamMatvec
    engine.snapshot()
    const values: Float32Array[] = []
    try {
      for (const enabled of [0, 1]) {
        engine.rewind(at)
        TUNING.streamMatvec = enabled
        values.push(await engine.forward([1000]))
      }
      report({
        stage: "quality",
        kind: "streamed-vs-original-full-model",
        ...compare(values[0], values[1])
      })
    } finally {
      TUNING.streamMatvec = stream
      engine.rewind(at)
    }
  }
  if (qualityOnly) return
  const gpu = engine.gpu
  const run = gpu.run.bind(gpu)
  const cpu = new Map<string, { calls: number; ms: number }>()
  let measuring = false
  gpu.run = (pass: Pass) => {
    const t = performance.now()
    run(pass)
    if (measuring) {
      const key = pass.name ?? "other",
        c = cpu.get(key) ?? { calls: 0, ms: 0 }
      c.calls++
      c.ms += performance.now() - t
      cpu.set(key, c)
    }
  }
  const body: number[] = [],
    head: number[] = []
  try {
    for (let i = 0; i < 12; i++) {
      await gpu.finish()
      let t = performance.now()
      measuring = true
      const row = await engine.forwardHidden([1000 + i])
      await gpu.finish()
      body.push(performance.now() - t)
      t = performance.now()
      await engine.argmax(row, 1, 1.1, [], false)
      await gpu.finish()
      head.push(performance.now() - t)
      measuring = false
    }
    report({
      stage: "body-head",
      bodyMs: body,
      headPickMs: head,
      cpuIssue: [...cpu].map(([name, v]) => ({ name, ...v })),
      readBudget: gpu.readsPerSlice
    })
    const skipResults: Record<string, number[]> = {}
    const configs = [
      "none",
      "matmul",
      "head",
      "scores",
      "attend",
      "conv",
      "add",
      "swiglu"
    ]
    for (let round = 0; round < 2; round++) {
      for (const key of round ? configs.slice().reverse() : configs) {
        gpu.skip = new Set(key === "none" ? [] : [key])
        await gpu.finish()
        const t = performance.now()
        for (let i = 0; i < 6; i++) {
          const row = await engine.forwardHidden([1000 + i])
          await engine.argmax(row, 1, 1.1, [], false)
        }
        await gpu.finish()
        ;(skipResults[key] ??= []).push((performance.now() - t) / 6)
        report({
          stage: "skip",
          round,
          key,
          msPerToken: skipResults[key].at(-1)
        })
      }
    }
    gpu.skip.clear()
    report({ stage: "summary", skipResults, tuning: { ...TUNING } })
  } finally {
    gpu.run = run
    gpu.skip.clear()
  }
}
