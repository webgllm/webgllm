import type { Engine } from "../src/engine.js"
import type { Texture } from "../src/gpu/gl.js"

type Any = Record<string, any>

/** What each of the engine's textures is, by walking its fields. */
export function names(engine: Engine) {
  const e = engine as unknown as Any
  const map = new Map<Texture, string>()
  const put = (t: unknown, name: string) => {
    if (t && typeof t === "object" && "tex" in t) map.set(t as Texture, name)
  }
  const matrix = (m: Any | undefined, kind: string) => {
    if (!m) return
    put(m.quants, `weights ${kind}`)
    put(m.scales, `weights ${kind}`)
    put(m.zeros, `weights ${kind}`)
  }
  matrix(e.embedding, "embedding/head")
  for (const l of e.layers as Array<Any>) {
    matrix(l.gateUp, "ffn")
    matrix(l.down, "ffn")
    for (const n of ["operatorNorm", "postOperatorNorm", "ffnNorm", "postFfnNorm"])
      put(l[n], "weights norms+vectors")
    if (l.conv) {
      matrix(l.conv.inProj, "conv")
      matrix(l.conv.outProj, "conv")
      put(l.conv.kernel, "weights norms+vectors")
      l.conv.state.forEach((t: Texture) => put(t, "state conv"))
      put(l.conv.prefixes, "state conv prefixes")
    }
    if (l.attention) {
      matrix(l.attention.qkvg, "attention")
      matrix(l.attention.o, "attention")
      put(l.attention.qNorm, "weights norms+vectors")
      put(l.attention.kNorm, "weights norms+vectors")
      put(l.attention.kv, l.attention.ring ? "KV ring (sliding)" : "KV full attention")
    }
    if (l.linear) {
      matrix(l.linear.inProj, "deltanet")
      matrix(l.linear.ba, "deltanet")
      matrix(l.linear.outProj, "deltanet")
      for (const n of ["kernel", "a", "dtBias", "norm"]) put(l.linear[n], "weights norms+vectors")
      l.linear.convState.forEach((t: Texture) => put(t, "state deltanet"))
      l.linear.state.forEach((t: Texture) => put(t, "state deltanet"))
    }
  }
  put(e.finalNorm, "weights norms+vectors")
  put(e.ropeFull, "rope tables")
  put(e.ropeSliding, "rope tables")
  const ids = new Map<number, string>()
  for (const [k, v] of Object.entries((e.buffers ?? {}) as Any)) {
    const each = (t: unknown, name: string) => {
      put(t, `scratch ${name}`)
      if (t && typeof t === "object" && "id" in t) ids.set((t as Texture).id, name)
    }
    if (Array.isArray(v)) v.forEach((t) => each(t, k))
    else if (v && typeof v === "object" && !("tex" in v))
      for (const [k2, t] of Object.entries(v as Any)) each(t, `${k}.${k2}`)
    else each(v, k)
  }
  for (const [id, t] of e.quads as Map<number, Texture>)
    put(t, `scratch quad partner (${ids.get(id) ?? "?"})`)
  if (e.f16) {
    put(e.f16.w, "scratch f16 weights")
    put(e.f16.wh2, "scratch f16 packed weights")
    e.f16.a?.forEach((t: Texture) => put(t, "scratch f16 activations"))
  }
  for (const t of Object.values((e.linChunks?.textures ?? {}) as Any)) put(t, "scratch deltanet chunks")
  for (const bufs of (e.chunkBufs as Map<number, Array<Texture>>).values())
    bufs.forEach((t) => put(t, "scratch attention chunks"))
  for (const s of (e.snapshots as Map<number, Any>).values()) {
    s.conv.forEach((t: Texture) => put(t, "snapshots"))
    s.windows.forEach((t: Texture) => put(t, "snapshots"))
    s.linear.forEach((t: Texture) => put(t, "snapshots"))
    put(s.hidden, "snapshots")
  }
  put(e.headScratch, "scratch head slices")
  put(e.greedyOut, "scratch greedy")
  put(e.debug, "debug")
  put(e.kvDebug, "debug")
  return map
}
