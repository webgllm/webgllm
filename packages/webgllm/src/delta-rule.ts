/**
 * The gated delta rule (Qwen3.5's Gated DeltaNet recurrence) on the CPU,
 * a token at a time and in chunks, step for step as the GPU passes run it
 * (see Engine.linearAttend and the lin* kernels): a reference for tests.
 *
 * Per head, with q̂ and k̂ L2-normalized (q̂ also over √dk), decay a_t =
 * exp(g_t) and beta_t:
 *   d_t = beta_t (v_t - a_t Sᵀk̂_t), S_t = a_t S_{t-1} + k̂_t d_tᵀ, o_t = S_tᵀq̂_t.
 *
 * In a chunk of C tokens, with G_t the sum of g from the chunk's start to t:
 *   A[t][s] = beta_t exp(G_t - G_s) k̂_t·k̂_s (s < t), T = (I + A)⁻¹,
 *   U = T diag(beta) V, W = T diag(beta exp(G)) K, so that D = U - W S0;
 *   P[t][s] = exp(G_t - G_s) q̂_t·k̂_s (s ≤ t),
 *   O = P U + (diag(exp(G)) Q - P W) S0 = Oi + Q' S0;
 *   S = exp(G_last) S0 + Σ_s exp(G_last - G_s) k̂_s d_sᵀ.
 * Only the last two lines read the state, so a chunk costs two dependent
 * steps however many tokens it has; the rest runs for every chunk at once.
 * Decay factors are always exp of a difference G_t - G_s ≤ 0 (never a raw
 * product), so nothing overflows.
 */

/** One head's inputs, rows of `tokens`; state dk × dv, row-major (S[k][v]). */
export type DeltaInputs = {
  dk: number
  dv: number
  tokens: number
  /** tokens × dk each. */
  q: Float64Array
  k: Float64Array
  /** tokens × dv. */
  v: Float64Array
  beta: Float64Array
  /** log of the decay (≤ 0). */
  g: Float64Array
}

/** Token by token, as linDelta and linState do it. Returns outputs (tokens × dv) and the new state. */
export function deltaSequential(x: DeltaInputs, state: Float64Array) {
  const { dk, dv, tokens } = x
  const S = Float64Array.from(state)
  const o = new Float64Array(tokens * dv)
  const d = new Float64Array(dv)
  for (let t = 0; t < tokens; t++) {
    const a = Math.exp(x.g[t])
    let qk = 0
    for (let i = 0; i < dk; i++) qk += x.q[t * dk + i] * x.k[t * dk + i]
    for (let j = 0; j < dv; j++) {
      let kv = 0
      let qs = 0
      for (let i = 0; i < dk; i++) {
        kv += S[i * dv + j] * x.k[t * dk + i]
        qs += S[i * dv + j] * x.q[t * dk + i]
      }
      d[j] = x.beta[t] * (x.v[t * dv + j] - a * kv)
      o[t * dv + j] = a * qs + qk * d[j]
    }
    for (let i = 0; i < dk; i++)
      for (let j = 0; j < dv; j++)
        S[i * dv + j] = a * S[i * dv + j] + x.k[t * dk + i] * d[j]
  }
  return { o, state: S }
}

/**
 * In chunks of `chunk` tokens (the last may be shorter), as the chunked
 * passes do it: the state-free steps over every chunk first, then two
 * steps a chunk that read and write the state.
 */
export function deltaChunked(x: DeltaInputs, state: Float64Array, chunk: number) {
  const { dk, dv, tokens: R } = x
  const C = chunk
  const start = (t: number) => Math.floor(t / C) * C
  // linCum: G, the chunk's running sum of g.
  const G = new Float64Array(R)
  for (let t = 0; t < R; t++) G[t] = (t === start(t) ? 0 : G[t - 1]) + x.g[t]
  // linGram: A (strictly lower) and P (lower with the diagonal), C wide.
  const A = new Float64Array(R * C)
  const P = new Float64Array(R * C)
  for (let t = 0; t < R; t++) {
    const c0 = start(t)
    for (let s = c0; s <= t; s++) {
      let kk = 0
      let qk = 0
      for (let i = 0; i < dk; i++) {
        kk += x.k[t * dk + i] * x.k[s * dk + i]
        qk += x.q[t * dk + i] * x.k[s * dk + i]
      }
      const decay = Math.exp(G[t] - G[s])
      if (s < t) A[t * C + s - c0] = x.beta[t] * decay * kk
      P[t * C + s - c0] = decay * qk
    }
  }
  // linSolve: T = (I + A)⁻¹ by forward substitution, row t from the rows above it.
  const T = new Float64Array(R * C)
  for (let t = 0; t < R; t++) {
    const c0 = start(t)
    const r = t - c0
    for (let j = 0; j <= r; j++) {
      let acc = j === r ? 1 : 0
      for (let s = j; s < r; s++) acc -= A[t * C + s] * T[(c0 + s) * C + j]
      T[t * C + j] = acc
    }
  }
  // linUW, then linQO.
  const U = new Float64Array(R * dv)
  const W = new Float64Array(R * dk)
  for (let t = 0; t < R; t++) {
    const c0 = start(t)
    for (let s = c0; s <= t; s++) {
      const tau = T[t * C + s - c0] * x.beta[s]
      const eg = Math.exp(G[s])
      for (let j = 0; j < dv; j++) U[t * dv + j] += tau * x.v[s * dv + j]
      for (let i = 0; i < dk; i++) W[t * dk + i] += tau * eg * x.k[s * dk + i]
    }
  }
  const Oi = new Float64Array(R * dv)
  const Qp = new Float64Array(R * dk)
  for (let t = 0; t < R; t++) {
    const c0 = start(t)
    const eg = Math.exp(G[t])
    for (let i = 0; i < dk; i++) Qp[t * dk + i] = eg * x.q[t * dk + i]
    for (let s = c0; s <= t; s++) {
      const p = P[t * C + s - c0]
      for (let j = 0; j < dv; j++) Oi[t * dv + j] += p * U[s * dv + j]
      for (let i = 0; i < dk; i++) Qp[t * dk + i] -= p * W[s * dk + i]
    }
  }
  // The chain: linChunkOut (D and O from the chunk's starting state), then linChunkState.
  let S = Float64Array.from(state)
  const D = new Float64Array(R * dv)
  const o = new Float64Array(R * dv)
  for (let c0 = 0; c0 < R; c0 += C) {
    const c1 = Math.min(c0 + C, R)
    for (let t = c0; t < c1; t++) {
      for (let j = 0; j < dv; j++) {
        let ws = 0
        let qs = 0
        for (let i = 0; i < dk; i++) {
          ws += W[t * dk + i] * S[i * dv + j]
          qs += Qp[t * dk + i] * S[i * dv + j]
        }
        D[t * dv + j] = U[t * dv + j] - ws
        o[t * dv + j] = Oi[t * dv + j] + qs
      }
    }
    const last = G[c1 - 1]
    const next = new Float64Array(dk * dv)
    for (let i = 0; i < dk; i++) {
      for (let j = 0; j < dv; j++) {
        let acc = Math.exp(last) * S[i * dv + j]
        for (let s = c0; s < c1; s++)
          acc += Math.exp(last - G[s]) * x.k[s * dk + i] * D[s * dv + j]
        next[i * dv + j] = acc
      }
    }
    S = next
  }
  return { o, state: S }
}
