/**
 * The model as the GPU takes it, whatever file it came from. The GGUF and
 * ONNX loaders both produce these; the engine uploads them as textures.
 *
 * Quantized matrices are laid out for the matmul kernel already:
 *
 * - Rows go in groups of 4 ("row groups"), and blocks are 32 weights along
 *   K. For 4-bit weights, a block row is 16 bytes (4 uint32 words) in
 *   llama.cpp's Q4_0 nibble order: byte j holds weight j in its low nibble
 *   and weight j + 16 in its high nibble. For 8-bit weights, a block row is
 *   32 signed bytes (8 words), weight j in byte j.
 * - `quants` index, in words: ((g * blocks + b) * 4 + r) * wordsPerBlock + w,
 *   for row group g (rows 4g..4g+3), block b, row r in the group, word w.
 *   So one row group is contiguous: a texture row of blocks * 4 texels
 *   (4-bit: one RGBA32UI texel per block row; 8-bit: two).
 * - `scales` (IEEE half bits) and `zeros` index: (g * blocks + b) * 4 + r.
 *   One RGBA16F texel per row group and block holds the scales of its 4
 *   rows.
 *
 * A weight is scale * (q - zero): 4-bit q in 0..15 with `zeros` (null means
 * 8 everywhere, as Q4_0), 8-bit q in -128..127 with zero 0.
 *
 * 6-bit matrices are ggml's Q6_K blocks as they are, for an embedding a
 * GGUF stores that way (it's also the output head): super-blocks of 256
 * weights along K. A row's super-block is 52 words, ql (32 words: low
 * nibbles), qh (16: high 2 bits) and the 16 int8 sub-block scales (4), as
 * in the file; its fp16 d goes in `scales`. `quants` index, in words:
 * ((g * superBlocks + s) * 4 + r) * 52 + w; `scales`: (g * superBlocks + s)
 * * 4 + r. A weight is d * scale[sub-block] * (q - 32), as
 * dequantize_row_q6_K.
 */

export type LayerType =
  | "conv"
  | "full_attention"
  | "sliding_attention"
  /** Qwen3.5's Gated DeltaNet (see LinearAttentionWeights). */
  | "linear_attention"

export type ModelConfig = {
  /** Where the weights came from. */
  format: "gguf" | "onnx"
  hiddenSize: number
  intermediateSize: number
  numLayers: number
  numHeads: number
  numKvHeads: number
  headDim: number
  vocabSize: number
  layerTypes: Array<LayerType>
  /** Tokens a sliding-attention layer sees, the current one included. */
  slidingWindow: number
  /**
   * Positions the model reads: config.json's max_position_embeddings, the
   * GGUF's context_length.
   */
  contextLength: number
  ropeThetaFull: number
  ropeThetaSliding: number
  /**
   * Dimensions RoPE rotates per head, in full-attention layers (LFM3: 64 of
   * 128, partial rotary factor 0.5). Sliding layers rotate all of them.
   * The loaders put each head's rotated dims first: dim j pairs with
   * j + ropeDims / 2, at frequency theta^(-2j / headDim).
   */
  ropeDimsFull: number
  ropeDimsSliding: number
  /**
   * The dims RoPE's frequencies spread over: theta^(-2j / ropeFreqDims).
   * headDim when left out (LFM3); Qwen3.5 spreads them over its rotated
   * dims (64 of 256).
   */
  ropeFreqDims?: number
  normEps: number
  /** Short convolution width (LFM3: 7, LFM2: 3). */
  convKernel: number
  /**
   * Whether attention has LFM3's per-head sigmoid output gate (its rows
   * follow v's in `qkvg`). LFM2 has none.
   */
  attentionGate: boolean
  /**
   * The gate has one value per query dim (Qwen3.5: numHeads · headDim
   * rows after v's), not one per head (LFM3).
   */
  gatePerDim?: boolean
  /** Gated DeltaNet layers' shapes (Qwen3.5), when the model has them. */
  linear?: LinearConfig
  /**
   * The softmax's scale where it isn't 1/√headDim: latent attention scores
   * over headDim latent dims, scaled by its full keys' width (Raon: 1/√192).
   */
  attentionScale?: number
  /** The residual as several streams with learned mixing (Raon's mHC), when the model has them. */
  streams?: StreamConfig
  /** Latent attention's shapes (Raon's gated MLA), when the model has them. */
  latent?: LatentConfig
  /** Per-layer token embeddings (Raon), when the model has them. */
  perLayer?: PerLayerConfig
  bosTokenId: number
  eosTokenId: number
  padTokenId: number
}

/** Gated DeltaNet's shapes (Qwen3.5 2B: 16 heads of 128 for keys and values). */
export type LinearConfig = {
  numKeyHeads: number
  numValueHeads: number
  keyHeadDim: number
  valueHeadDim: number
  /** The causal convolution's width (4). */
  convKernel: number
  /**
   * Raon Delta Attention: the state decays per key dim, by
   * exp(lowerBound · sigmoid(low-rank projection + bias)), and what is
   * written is scaled by a write gate apart from the erase gate.
   */
  keyDecay?: { rank: number; lowerBound: number }
}

/** Raon's mHC: the residual is `count` streams of hiddenSize each. */
export type StreamConfig = {
  count: number
  /** The selector's RMS epsilon (over all streams). */
  eps: number
}

/**
 * Raon's gated MLA, in absorbed form: the cache holds one latent row a
 * token (rank values, then `shared` key dims every head reads as they are);
 * a head's query is projected into that latent (keyDim - shared dims
 * through its own rank × (keyDim - shared) matrix) and its attended latent
 * out to valueDim values.
 */
export type LatentConfig = {
  heads: number
  queryRank: number
  rank: number
  shared: number
  /** A head's key width before absorption (the softmax's scale is 1/√keyDim). */
  keyDim: number
  valueDim: number
  eps: number
}

/** Raon's per-layer embeddings: `dim` values a layer for every token. */
export type PerLayerConfig = {
  dim: number
  /** Layers the per-layer table and its projection hold inputs for, where not the model's own count (predictSpeed's probe: a layer of each kind standing in for all of them). */
  layers?: number
}

export type QuantMatrix = {
  /** Output rows; a multiple of 4. */
  N: number
  /** Inputs per row; a multiple of 32 (256 for 6-bit). */
  K: number
  bits: 4 | 6 | 8
  /** See the module comment for the layout. */
  quants: Uint32Array
  /** IEEE half bits, one per row and block (super-block for 6-bit). */
  scales: Uint16Array
  /** 4-bit only: one zero point (0..15) per row and block; null means 8. */
  zeros: Uint8Array | null
  /**
   * 4-bit only, instead of `zeros` (then null): the zero points as IEEE half
   * bits, where they aren't whole numbers. A GGUF's Q4_K sub-block (32
   * weights, d · scale · q - dmin · min) is this module's block with scale
   * d · scale and zero point dmin · min / (d · scale).
   */
  zeroHalves?: Uint16Array
  /**
   * 4-bit only, instead of `quants` (then empty): the rows as onnxruntime's
   * MatMulNBits stores them ([N][K / 32][16], element 2i of a block in byte
   * i's low nibble), for the engine to rearrange on the GPU.
   */
  raw?: Uint8Array
}

export type ConvWeights = {
  /** 2048 → 3 × 2048, rows in the order B, C, x (as the model chunks them). */
  inProj: QuantMatrix
  outProj: QuantMatrix
  /**
   * Depthwise causal convolution, channel-major: kernel[c * width + k] is
   * channel c's tap k, applied to the input k - (width - 1) tokens back
   * (tap width - 1 is the current token).
   */
  kernel: Float32Array
}

export type AttentionWeights = {
  /**
   * q, k, v and the output gate in one matrix: rows numHeads * headDim of q,
   * then numKvHeads * headDim of k, the same of v, then numHeads of gate
   * (when the model has one: config.attentionGate; numHeads * headDim
   * with config.gatePerDim).
   */
  qkvg: QuantMatrix
  o: QuantMatrix
  qNorm: Float32Array
  kNorm: Float32Array
}

/**
 * A Gated DeltaNet layer (Qwen3.5). With K = keyHeads · keyHeadDim and
 * V = valueHeads · valueHeadDim: qkv = conv(in_proj rows 0 .. 2K + V),
 * then silu; q and k L2-normalized per head; beta = sigmoid(b), decay =
 * exp(A · softplus(a + dtBias)); each value head keeps a keyHeadDim ×
 * valueHeadDim state S: S = decay · S, S += k (beta (v - Sᵀk))ᵀ, o = Sᵀq
 * (q scaled by 1/√keyHeadDim); out = out_proj(rms(o) · norm · silu(z)).
 */
export type LinearAttentionWeights = {
  /** Rows q (K), k (K), v (V), then z (V). */
  inProj: QuantMatrix
  /** Rows b (valueHeads), then a (valueHeads). */
  ba: QuantMatrix
  outProj: QuantMatrix
  /** Depthwise causal convolution over q|k|v, channel-major, oldest tap first (as ConvWeights.kernel). */
  conv: Float32Array
  /** A = -exp(A_log), one per value head. */
  a: Float32Array
  dtBias: Float32Array
  /** The gated RMSNorm's weights (valueHeadDim, shared by the heads; not zero-centered). */
  norm: Float32Array
  /**
   * Raon Delta Attention (config.linear.keyDecay), where `ba` holds the
   * decay's low-rank projection (rank rows), then b and b + write (value
   * heads each), `dtBias` is per key dim and `a` is unused: the rank's
   * outputs to every key dim's decay logit (K rows of rank). Its output
   * gate is sigmoid(z), not silu.
   */
  decay?: QuantMatrix
}

/** One of a layer's two mHC sites (Raon): how the branch reads the streams and writes back. */
export type StreamMixWeights = {
  /**
   * The selector: rows pre (count), post (count), then the (count - 1)²
   * mixing coordinates, padded with zero rows to a multiple of 4; inputs
   * the streams one after another (count · hiddenSize), over their RMS.
   */
  fn: QuantMatrix
  /** Added to the selector's outputs (its unpadded rows). */
  base: Float32Array
  /**
   * As KRAFTON's converter folds it: K (count² rows of the selector's
   * width), then r (count²), then c0 (count × count). With e = K · mixes
   * and rho = 1 + max(e · r): stream dst takes Σ_src (e / rho + c0)[dst +
   * count · src] · stream src, plus sigmoid(post[dst]) · the branch.
   */
  chart: Float32Array
}

/** Raon's gated MLA layer (see LatentConfig). */
export type LatentAttentionWeights = {
  /** Rows q_a (queryRank), kv_a (rank + shared), then the output gate (heads · valueDim). */
  inProj: QuantMatrix
  qNorm: Float32Array
  kvNorm: Float32Array
  /** heads · keyDim rows over queryRank: a head's keyDim - shared dims, then its shared dims. */
  qB: QuantMatrix
  /** Per head, rank rows of (keyDim - shared): halves, head-major, row-major. */
  kB: Uint16Array
  /** Per head, valueDim rows of rank: halves, head-major, row-major. */
  vB: Uint16Array
  o: QuantMatrix
}

/** A layer's part of Raon's per-layer embedding: streams += RMSNorm(proj(gelu(gate · Σ streams) · input)) · norm. */
export type PerLayerWeights = {
  gate: QuantMatrix
  proj: QuantMatrix
  norm: Float32Array
}

export type LayerWeights = {
  type: LayerType
  operatorNorm: Float32Array
  /** LFM3's sandwich norms on the operator's and the MLP's outputs; LFM2 has none. */
  postOperatorNorm?: Float32Array
  ffnNorm: Float32Array
  postFfnNorm?: Float32Array
  conv?: ConvWeights
  attention?: AttentionWeights
  linear?: LinearAttentionWeights
  latent?: LatentAttentionWeights
  /** Raon's mHC sites: before the operator, and before the MLP. */
  mix?: { operator: StreamMixWeights; ffn: StreamMixWeights }
  perLayer?: PerLayerWeights
  /** w1 (gate) rows, then w3 (up) rows: 2 × intermediateSize. */
  gateUp: QuantMatrix
  /** w2 (down). */
  down: QuantMatrix
}

/**
 * The model's weights. The embedding is also the output head (tied).
 */
export type ModelWeights = {
  config: ModelConfig
  embedding: QuantMatrix
  finalNorm: Float32Array
  layers: Array<LayerWeights>
}

/**
 * One piece of the model, as a loader hands it over: the engine uploads it
 * and drops it, so the whole model never sits in JavaScript memory at once.
 * `layer` is set for a layer's weights.
 */
export type WeightPiece =
  | { kind: "config"; config: ModelConfig }
  /**
   * The embedding, whole or in chunks of rows: `value` holds rows [row0,
   * row0 + value.N) of a totalRows-row matrix (row0 a multiple of 4).
   */
  | { kind: "embedding"; value: QuantMatrix; row0?: number; totalRows?: number }
  | { kind: "finalNorm"; value: Float32Array }
  /**
   * Raon's per-layer embeddings, the model's part: the projection of the
   * token embedding (layers · dim rows) and its norm (dim), and where the
   * table's rows are in the file (a token's row: layers · dim values of
   * tensor type `type`, rowBytes long, read when the token is).
   */
  | {
      kind: "perLayer"
      proj: QuantMatrix
      norm: Float32Array
      table: { offset: number; rowBytes: number; type: number; rows: number }
    }
  | {
      kind: "layer"
      layer: number
      value: LayerWeights
      /**
       * The layer's q and k rows were reordered from the source's head
       * layout (HF's) to each head's rotated dims first: a reference taken
       * in the source's layout needs the same reordering to be compared.
       */
      headsReordered?: boolean
    }

/** Random access to a file's bytes: a URL (range requests), OPFS, or disk in tests. */
export type ByteSource = {
  size: number
  read: (offset: number, length: number) => Promise<Uint8Array>
  /**
   * Takes back a buffer a read returned, once its bytes are used, for a
   * later read of the same length (instead of a new allocation). The caller
   * keeps no view of it.
   */
  recycle?: (buffer: ArrayBuffer) => void
}
