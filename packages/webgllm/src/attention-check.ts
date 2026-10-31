/**
 * Attention on this GPU against the CPU's: a short prompt through the
 * engine, every attention layer's output recomputed from the GPU's own
 * queries, cache and gates. GPU compilers differ (one leaked later keys
 * into every query), so the worker runs this at load instead of trusting
 * that the shaders mean the same thing everywhere.
 */
import type { Engine } from "./engine.js"

type Tensor = { data: Float32Array; rows: number; dims: number; offset: number }

/**
 * The largest relative error of any attention row (Infinity if any value
 * isn't finite), over `tokens` read from an empty cache. Leaves the cache
 * empty.
 */
export async function attentionError(
  engine: Engine,
  tokens: Array<number>
): Promise<number> {
  const c = engine.config
  const hd = c.headDim
  const group = c.numHeads / c.numKvHeads
  const kvDims = c.numKvHeads * hd
  const gateAt = (c.numHeads + 2 * c.numKvHeads) * hd
  const seen = new Map<number, Partial<Record<string, Tensor>>>()
  const want = new Float64Array(hd)
  let worst = 0
  engine.reset()
  try {
    await engine.forward(tokens, async (layer, name, read) => {
      if (name !== "q" && name !== "qkvg" && name !== "attn_out" && name !== "kv")
        return
      const got = seen.get(layer) ?? {}
      got[name] = await read()
      seen.set(layer, got)
      if (name !== "kv") return
      seen.delete(layer)
      const { q, qkvg, attn_out: out, kv } = got as Record<string, Tensor>
      const sliding = c.layerTypes[layer] === "sliding_attention"
      for (let m = 0; m < q.rows; m++) {
        const pos = q.offset + m
        const j0 = sliding ? Math.max(0, pos - c.slidingWindow + 1) : 0
        let err = 0
        let ref = 0
        for (let h = 0; h < c.numHeads; h++) {
          const kh = Math.floor(h / group)
          const qo = m * q.dims + h * hd
          const scores = new Float64Array(pos - j0 + 1)
          let mx = -Infinity
          for (let j = j0; j <= pos; j++) {
            let s = 0
            const ko = j * kv.dims + kh * hd
            for (let d = 0; d < hd; d++) s += q.data[qo + d] * kv.data[ko + d]
            scores[j - j0] = s
            mx = Math.max(mx, s)
          }
          let l = 0
          want.fill(0)
          for (let j = j0; j <= pos; j++) {
            const e = Math.exp(scores[j - j0] - mx)
            l += e
            const vo = j * kv.dims + kvDims + kh * hd
            for (let d = 0; d < hd; d++) want[d] += e * kv.data[vo + d]
          }
          const gate = c.attentionGate
            ? 1 / (1 + Math.exp(-qkvg.data[m * qkvg.dims + gateAt + h]))
            : 1
          for (let d = 0; d < hd; d++) {
            const r = (want[d] / l) * gate
            const g = out.data[m * out.dims + h * hd + d]
            if (!Number.isFinite(g)) worst = Infinity
            err += (g - r) ** 2
            ref += r * r
          }
        }
        worst = Math.max(worst, Math.sqrt(err / Math.max(ref, 1e-30)))
      }
    })
  } finally {
    engine.reset()
  }
  return worst
}
