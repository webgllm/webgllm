/**
 * Raon-1 nano's passes that the other models don't have (see kernels.ts for
 * the tensor layouts and helpers): the residual as several streams with
 * learned mixing (Raon-mHC), per-layer token embeddings, Raon Delta
 * Attention's per-key decay and write gate, and the gated latent attention
 * (MLA, absorbed). Inputs a matmul wrote are read through pf, declared
 * `uniform usampler2D`: the engine rewrites them for a split matmul's
 * partial sums or a 4-token matmul's two textures.
 */
import { HEADER } from "./kernels.js"

/** The same string for the same options (a pass looks its program up by source). */
function memo<A extends Array<unknown>>(build: (...args: A) => string): ((...args: A) => string) & { source: string } {
  const cache = new Map<string, string>()
  const made = (...args: A) => {
    const key = JSON.stringify(args)
    let source = cache.get(key)
    if (source === undefined) cache.set(key, (source = build(...args)))
    return source
  }
  return Object.assign(made, { source: build.toString() })
}

/** 1 / RMS of row m from its partial sums (as kernels.ts's; see withRms). */
const RMS_SCALE = `
uniform sampler2D uR;
uniform int uROff;
uniform int uTexels;
uniform int uParts;
uniform float uEps;
float rmsScale(int m) {
  vec4 s = vec4(0.0);
  for (int p = 0; p < 64/* RMS_REDUCTION */; p++)
    s += texelFetch(uR, ivec2(uROff + p, m), 0);
  return inversesqrt(dot(s, vec4(1.0)) / float(uTexels * 4) + uEps);
}
`

const SIGMOID = "vec4 sigmoid4(vec4 x) { return 1.0 / (1.0 + exp(-x)); }"

// ------------------------------------------------------------- streams

/** Every stream starts as the embedding: texel x of the streams is texel x mod uT of uX. */
export const STREAMS_REPEAT = `${HEADER}
uniform sampler2D uX;
uniform int uT;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  o = texelFetch(uX, ivec2(x - (x / uT) * uT, int(gl_FragCoord.y)), 0);
}`

/** f32 rows as h2, as they are. */
export const F32_TO_H2 = `${HEADER}
uniform sampler2D uX;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  o = packH2(texelFetch(uX, ivec2(2 * x, m), 0), texelFetch(uX, ivec2(2 * x + 1, m), 0));
}`

/**
 * A site's coefficients from its selector's outputs (uMix: 4 pre, 4 post,
 * 9 mixing coordinates, read with their biases uBase): texel 0 sigmoid(pre),
 * texel 1 sigmoid(post), texels 2..5 the mixing matrix, texel 2 + src
 * holding what each stream dst takes of stream src. uChart: rows 0..15 are
 * K's rows (e = K · mixes, 5 texels), row 16 r, row 17 c0; with rho = 1 +
 * max(e · r), the matrix is e / rho + c0. Four streams.
 *
 * `inline` (a few rows: see STREAM_H2): the selector read the streams as
 * they are, times uPrescale, so its outputs are scaled here by the streams'
 * 1 / RMS over uPrescale; and texels 6..8 hold the streams' dot products
 * (s0·s0 s0·s1 s0·s2 s0·s3 | s1·s1 s1·s2 s1·s3 s2·s2 | s2·s3 s3·s3), from
 * which STREAM_INPUT takes its weighted sum's RMS. Both from STREAM_H2's
 * partial sums (uFlat, from texel 2·uT).
 */
function streamCoefficientsSource(inline: boolean) {
  return `${HEADER}
uniform usampler2D uMix;
uniform sampler2D uBase;
uniform sampler2D uChart;
${inline ? "uniform usampler2D uFlat;\nuniform int uT;\nuniform float uEps;\nuniform float uPrescale;" : ""}
out vec4 o;
${SIGMOID}
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
${inline ? `  int g0 = 2 * uT;
  if (x >= 6) {
    vec4 g = vec4(0.0);
    for (int p = 0; p < ${STREAM_PARTS}; p++) g += uintBitsToFloat(texelFetch(uFlat, ivec2(g0 + 3 * p + x - 6, m), 0));
    o = g;
    return;
  }
  // The streams' sum of squares: s0·s0 + s1·s1 + s2·s2 + s3·s3.
  float ss = 0.0;
  for (int p = 0; p < ${STREAM_PARTS}; p++) {
    vec4 a = uintBitsToFloat(texelFetch(uFlat, ivec2(g0 + 3 * p, m), 0));
    vec4 b = uintBitsToFloat(texelFetch(uFlat, ivec2(g0 + 3 * p + 1, m), 0));
    vec4 c = uintBitsToFloat(texelFetch(uFlat, ivec2(g0 + 3 * p + 2, m), 0));
    ss += a.x + b.x + b.w + c.y;
  }
  float scale = inversesqrt(ss / float(16 * uT) + uEps) / uPrescale;` : "  float scale = 1.0;"}
  vec4 m0 = pf(uMix, 0, m) * scale + texelFetch(uBase, ivec2(0, 0), 0);
  vec4 m1 = pf(uMix, 1, m) * scale + texelFetch(uBase, ivec2(1, 0), 0);
  o = sigmoid4(x == 0 ? m0 : m1);
  if (x < 2) return;
  vec4 m2 = pf(uMix, 2, m) * scale + texelFetch(uBase, ivec2(2, 0), 0);
  vec4 m3 = pf(uMix, 3, m) * scale + texelFetch(uBase, ivec2(3, 0), 0);
  vec4 m4 = pf(uMix, 4, m) * scale + texelFetch(uBase, ivec2(4, 0), 0);
  float rho = 0.0;
  vec4 mine = vec4(0.0);
  for (int s = 0; s < 4; s++) {
    vec4 e = vec4(0.0);
    for (int d = 0; d < 4; d++) {
      int j = d + 4 * s;
      float acc = dot(texelFetch(uChart, ivec2(0, j), 0), m0) + dot(texelFetch(uChart, ivec2(1, j), 0), m1)
        + dot(texelFetch(uChart, ivec2(2, j), 0), m2) + dot(texelFetch(uChart, ivec2(3, j), 0), m3)
        + dot(texelFetch(uChart, ivec2(4, j), 0), m4);
      e += acc * vec4(equal(ivec4(d), ivec4(0, 1, 2, 3)));
    }
    vec4 er = e * texelFetch(uChart, ivec2(s, 16), 0);
    rho = max(rho, max(max(er.x, er.y), max(er.z, er.w)));
    if (s == x - 2) mine = e;
  }
  o = mine / (1.0 + rho) + texelFetch(uChart, ivec2(x - 2, 17), 0);
}`
}
export const streamCoefficients = memo(streamCoefficientsSource)

/**
 * A few rows' streams for their site, one pass: as h2, times uScale (what
 * the selector reads: texels [0, 2·uT)), and after them the streams' dot
 * products in STREAM_PARTS partial sums of three texels each (their float
 * bits; the entries as streamCoefficients says), part p over each stream's
 * texels [p·chunk, (p+1)·chunk): many short sums, where one fragment
 * walking a whole row took an M3 Ultra 0.4 ms.
 */
export const STREAM_PARTS = 16
function streamH2Source(chunk: number) {
  return `${HEADER}
uniform sampler2D uS;
uniform float uScale;
uniform int uT;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int h = 2 * uT;
  if (x < h) {
    o = packH2(texelFetch(uS, ivec2(2 * x, m), 0) * uScale, texelFetch(uS, ivec2(2 * x + 1, m), 0) * uScale);
    return;
  }
  int p = (x - h) / 3;
  int t = (x - h) - 3 * p;
  vec4 g0 = vec4(0.0);
  vec4 g1 = vec4(0.0);
  vec4 g2 = vec4(0.0);
  // A constant bound, so the loop unrolls and the reads go out together.
  for (int k = 0; k < ${chunk}; k++) {
    int i = p * ${chunk} + k;
    float w = i < uT ? 1.0 : 0.0;
    i = min(i, uT - 1);
    vec4 a = texelFetch(uS, ivec2(i, m), 0);
    vec4 b = texelFetch(uS, ivec2(uT + i, m), 0);
    vec4 c = texelFetch(uS, ivec2(2 * uT + i, m), 0);
    vec4 d = texelFetch(uS, ivec2(3 * uT + i, m), 0);
    g0 += w * vec4(dot(a, a), dot(a, b), dot(a, c), dot(a, d));
    g1 += w * vec4(dot(b, b), dot(b, c), dot(b, d), dot(c, c));
    g2 += w * vec4(dot(c, d), dot(d, d), 0.0, 0.0);
  }
  o = floatBitsToUint(t == 0 ? g0 : t == 1 ? g1 : g2);
}`
}
/** `chunk`: each stream's texels a partial sum (their count over STREAM_PARTS, rounded up). */
export const streamH2 = memo(streamH2Source)

/**
 * The branch's input, h2: Σ pre[i] · stream i over its RMS, times uNorm.
 * The sum's mean square is preᵀ G pre over its width, G the streams' dot
 * products (uCoef's texels 6..8: see streamCoefficients' inline).
 */
export const STREAM_INPUT = `${HEADER}
uniform sampler2D uS;
uniform sampler2D uCoef;
uniform sampler2D uNorm;
uniform int uT;
uniform float uEps;
out uvec4 o;
vec4 sum(vec4 pre, int x, int m) {
  return pre.x * texelFetch(uS, ivec2(x, m), 0) + pre.y * texelFetch(uS, ivec2(uT + x, m), 0)
    + pre.z * texelFetch(uS, ivec2(2 * uT + x, m), 0) + pre.w * texelFetch(uS, ivec2(3 * uT + x, m), 0);
}
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 pre = texelFetch(uCoef, ivec2(0, m), 0);
  vec4 g0 = texelFetch(uCoef, ivec2(6, m), 0);
  vec4 g1 = texelFetch(uCoef, ivec2(7, m), 0);
  vec4 g2 = texelFetch(uCoef, ivec2(8, m), 0);
  float q = pre.x * pre.x * g0.x + pre.y * pre.y * g1.x + pre.z * pre.z * g1.w + pre.w * pre.w * g2.y
    + 2.0 * (pre.x * pre.y * g0.y + pre.x * pre.z * g0.z + pre.x * pre.w * g0.w
      + pre.y * pre.z * g1.y + pre.y * pre.w * g1.z + pre.z * pre.w * g2.x);
  float inv = inversesqrt(max(q, 0.0) / float(4 * uT) + uEps);
  o = packH2(
    sum(pre, 2 * x, m) * inv * texelFetch(uNorm, ivec2(2 * x, 0), 0),
    sum(pre, 2 * x + 1, m) * inv * texelFetch(uNorm, ivec2(2 * x + 1, 0), 0));
}`

/** The branch's input before its norm: Σ pre[i] · stream i (uT texels a stream). */
export const STREAM_AGGREGATE = `${HEADER}
uniform sampler2D uS;
uniform sampler2D uCoef;
uniform int uT;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 pre = texelFetch(uCoef, ivec2(0, m), 0);
  o = pre.x * texelFetch(uS, ivec2(x, m), 0) + pre.y * texelFetch(uS, ivec2(uT + x, m), 0)
    + pre.z * texelFetch(uS, ivec2(2 * uT + x, m), 0) + pre.w * texelFetch(uS, ivec2(3 * uT + x, m), 0);
}`

/** The streams after a branch: each a mix of the streams, plus its post · the branch's output uY. */
export const STREAM_MIX = `${HEADER}
uniform sampler2D uS;
uniform usampler2D uY;
uniform sampler2D uCoef;
uniform int uT;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int dst = x / uT;
  int c = x - dst * uT;
  vec4 acc = comp(texelFetch(uCoef, ivec2(1, m), 0), dst) * pf(uY, c, m);
  acc += comp(texelFetch(uCoef, ivec2(2, m), 0), dst) * texelFetch(uS, ivec2(c, m), 0);
  acc += comp(texelFetch(uCoef, ivec2(3, m), 0), dst) * texelFetch(uS, ivec2(uT + c, m), 0);
  acc += comp(texelFetch(uCoef, ivec2(4, m), 0), dst) * texelFetch(uS, ivec2(2 * uT + c, m), 0);
  acc += comp(texelFetch(uCoef, ivec2(5, m), 0), dst) * texelFetch(uS, ivec2(3 * uT + c, m), 0);
  o = acc;
}`

/** The four streams' sum times uScale, f32 (the hidden state the head reads: their mean). */
export const STREAM_SUM = `${HEADER}
uniform sampler2D uS;
uniform int uT;
uniform float uScale;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  o = (texelFetch(uS, ivec2(x, m), 0) + texelFetch(uS, ivec2(uT + x, m), 0)
    + texelFetch(uS, ivec2(2 * uT + x, m), 0) + texelFetch(uS, ivec2(3 * uT + x, m), 0)) * uScale;
}`

/** STREAM_SUM's rows' partial sums of squares: pixel (uBase + p, m) sums its texels [p·uChunk, (p+1)·uChunk). */
export const STREAM_SUM_SQUARES = `${HEADER}
uniform sampler2D uS;
uniform int uT;
uniform float uScale;
uniform int uTexels;
uniform int uChunk;
uniform int uBase;
out vec4 o;
void main() {
  int p = int(gl_FragCoord.x) - uBase;
  int m = int(gl_FragCoord.y);
  int i0 = p * uChunk;
  int i1 = min(uTexels, i0 + uChunk);
  vec4 s = vec4(0.0);
  for (int i = i0; i < i1; i++) {
    vec4 v = (texelFetch(uS, ivec2(i, m), 0) + texelFetch(uS, ivec2(uT + i, m), 0)
      + texelFetch(uS, ivec2(2 * uT + i, m), 0) + texelFetch(uS, ivec2(3 * uT + i, m), 0)) * uScale;
    s += v * v;
  }
  o = s;
}`

/** The four streams' sum as h2 (the per-layer gate's input). */
export const STREAM_SUM_H2 = `${HEADER}
uniform sampler2D uS;
uniform int uT;
out uvec4 o;
vec4 sum(int x, int m) {
  return texelFetch(uS, ivec2(x, m), 0) + texelFetch(uS, ivec2(uT + x, m), 0)
    + texelFetch(uS, ivec2(2 * uT + x, m), 0) + texelFetch(uS, ivec2(3 * uT + x, m), 0);
}
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  o = packH2(sum(2 * x, m), sum(2 * x + 1, m));
}`

// ------------------------------------------------- per-layer embeddings

/**
 * The per-layer inputs: (RMSNorm(projection, a layer's `dim` values at a
 * time) · uNorm + √dim · the token's table row) / √2.
 */
function perLayerInputsSource(dim: number) {
  const T = dim / 4
  return `${HEADER}
uniform usampler2D uProj;
uniform sampler2D uTable;
uniform sampler2D uNorm;
uniform float uEps;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int base = (x / ${T}) * ${T};
  vec4 s = vec4(0.0);
  for (int i = 0; i < ${T}; i++) {
    vec4 v = pf(uProj, base + i, m);
    s += v * v;
  }
  float inv = inversesqrt(dot(s, vec4(1.0)) / ${dim}.0 + uEps);
  o = (pf(uProj, x, m) * inv * texelFetch(uNorm, ivec2(x - base, 0), 0)
    + ${Math.sqrt(dim).toFixed(9)} * texelFetch(uTable, ivec2(x, m), 0)) * ${Math.SQRT1_2.toFixed(9)};
}`
}
export const perLayerInputs = memo(perLayerInputsSource)

/** h2 out = gelu(gate's output) · the layer's inputs (uInputs from texel uOff). */
export const PER_LAYER_GATE = `${HEADER}
uniform usampler2D uG;
uniform sampler2D uInputs;
uniform int uOff;
out uvec4 o;
/* gelu's tanh form: x · sigmoid(2 · √(2/π) · (x + 0.044715 x³)). */
vec4 gelu(vec4 x) { return x / (1.0 + exp(-1.5957691216 * (x + 0.044715 * x * x * x))); }
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  o = packH2(
    gelu(pf(uG, 2 * x, m)) * texelFetch(uInputs, ivec2(uOff + 2 * x, m), 0),
    gelu(pf(uG, 2 * x + 1, m)) * texelFetch(uInputs, ivec2(uOff + 2 * x + 1, m), 0));
}`

/** Every stream plus RMSNorm(uY) · uNorm (uY's partial sums in uR; uT texels a stream). */
export const PER_LAYER_ADD = `${HEADER}${RMS_SCALE}
uniform sampler2D uS;
uniform usampler2D uY;
uniform sampler2D uNorm;
uniform int uT;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int c = x - (x / uT) * uT;
  o = texelFetch(uS, ivec2(x, m), 0) + pf(uY, c, m) * rmsScale(m) * texelFetch(uNorm, ivec2(c, 0), 0);
}`

// ------------------------------------------------ Raon Delta Attention
/*
 * As Qwen3.5's Gated DeltaNet (kernels.ts's lin* passes, whose in_proj,
 * convolution and q̂|k̂ passes it shares), with a decay per key dim and a
 * write gate. Its gates tensor, per row: beta of heads 4i..4i+3 at texel i,
 * the write/erase ratio at uH4 + i, then the keys' log-decays (uK4 texels
 * from 2·uH4). The state decays row by row (S[k][v] by key k's decay);
 * what's written is beta (ratio · v - Sᵀk̂).
 */

/** beta = sigmoid(b) and ratio = sigmoid(b + write) / sigmoid(b), from the projection's rows after its rank. */
function rdaGatesSource(heads4: number, rank4: number) {
  return `${HEADER}
uniform usampler2D uBA;
uniform int uBAx;
out vec4 o;
${SIGMOID}
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  if (x < ${heads4}) {
    o = sigmoid4(pf(uBA, uBAx + ${rank4} + x, m));
  } else {
    int i = x - ${heads4};
    o = sigmoid4(pf(uBA, uBAx + ${rank4 + heads4} + i, m)) / sigmoid4(pf(uBA, uBAx + ${rank4} + i, m));
  }
}`
}
export const rdaGates = memo(rdaGatesSource)

/** The projection's first rows (the decay's rank) as h2, for the matmul to every key's logit. */
export const RDA_RANK = `${HEADER}
uniform usampler2D uBA;
uniform int uBAx;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  o = packH2(pf(uBA, uBAx + 2 * x, m), pf(uBA, uBAx + 2 * x + 1, m));
}`

/**
 * The keys' log-decays, uLower · sigmoid(logit + bias), drawn from texel
 * uX0 of the output (the logits from texel uFBx of uFB); with `decay`,
 * the decays themselves (their exp), for writing's fused passes.
 */
function rdaDecaySource(decay: boolean) {
  return `${HEADER}
uniform usampler2D uFB;
uniform sampler2D uDt;
uniform int uX0;
uniform int uFBx;
uniform float uLower;
out vec4 o;
${SIGMOID}
void main() {
  int i = int(gl_FragCoord.x) - uX0;
  int m = int(gl_FragCoord.y);
  vec4 g = uLower * sigmoid4(pf(uFB, uFBx + i, m) + texelFetch(uDt, ivec2(i, 0), 0));
  o = ${decay ? "exp(g)" : "g"};
}`
}
export const rdaDecay = memo(rdaDecaySource)

/**
 * Writing's rdaGates, linPrep and rdaDelta in one pass (as kernels.ts's
 * linDeltaFused): each fragment takes its head's q and k from the conv
 * output (uQKV) over their L2 norms, beta and the write ratio from the
 * projection's split partial sums (uBA: b from texel uBAx, b + write uH4
 * texels on), and the keys' decays from uDecay (rdaDecay's). Writes delta,
 * the output, and k̂ (texel x: a head's key texels line up with its value
 * texels), for rdaStateFused.
 */
function rdaDeltaFusedSource(o: { keyHeadDim: number; valueHeadDim: number }) {
  const DK4 = o.keyHeadDim / 4
  const DV4 = o.valueHeadDim / 4
  if (DK4 !== DV4) throw new Error("The fused recurrence needs as many key dims as value dims.")
  const scale = (1 / Math.sqrt(o.keyHeadDim)).toFixed(9)
  return `${HEADER}
uniform sampler2D uS;
uniform sampler2D uQKV;
uniform sampler2D uBA;
uniform sampler2D uDecay;
uniform int uK4;
uniform int uH4;
uniform int uBAx;
layout(location = 0) out vec4 oDelta;
layout(location = 1) out vec4 oOut;
layout(location = 2) out vec4 oK;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int h = x / ${DV4};
  int kb = h * ${DK4};
  vec4 sq = vec4(0.0);
  vec4 sk = vec4(0.0);
  for (int i = 0; i < ${DK4}; i++) {
    vec4 v = texelFetch(uQKV, ivec2(kb + i, m), 0);
    sq += v * v;
  }
  for (int i = 0; i < ${DK4}; i++) {
    vec4 v = texelFetch(uQKV, ivec2(uK4 + kb + i, m), 0);
    sk += v * v;
  }
  // (x / max(|x|, 1e-6), as kernels.ts's linPrep with its floor.)
  float invq = 1.0 / max(sqrt(dot(sq, vec4(1.0))), 1e-6);
  float invk = 1.0 / max(sqrt(dot(sk, vec4(1.0))), 1e-6);
  vec4 kv = vec4(0.0);
  vec4 qs = vec4(0.0);
  float qk = 0.0;
  for (int i = 0; i < ${DK4}; i++) {
    vec4 q = texelFetch(uQKV, ivec2(kb + i, m), 0) * invq * ${scale};
    vec4 k = texelFetch(uQKV, ivec2(uK4 + kb + i, m), 0) * invk;
    vec4 g = texelFetch(uDecay, ivec2(kb + i, m), 0);
    qk += dot(q, k);
    k *= g;
    q *= g;
    int r = 4 * i;
    vec4 s0 = texelFetch(uS, ivec2(x, r), 0);
    vec4 s1 = texelFetch(uS, ivec2(x, r + 1), 0);
    vec4 s2 = texelFetch(uS, ivec2(x, r + 2), 0);
    vec4 s3 = texelFetch(uS, ivec2(x, r + 3), 0);
    kv += s0 * k.x + s1 * k.y + s2 * k.z + s3 * k.w;
    qs += s0 * q.x + s1 * q.y + s2 * q.z + s3 * q.w;
  }
  int g4 = h >> 2;
  float beta = comp(1.0 / (1.0 + exp(-pf(uBA, uBAx + g4, m))), h & 3);
  float write = comp(1.0 / (1.0 + exp(-pf(uBA, uBAx + uH4 + g4, m))), h & 3);
  vec4 v = texelFetch(uQKV, ivec2(2 * uK4 + x, m), 0) * (write / beta);
  vec4 d = beta * (v - kv);
  oDelta = d;
  oOut = qs + qk * d;
  oK = texelFetch(uQKV, ivec2(uK4 + x, m), 0) * invk;
}`
}
export const rdaDeltaFused = memo(rdaDeltaFusedSource)

/** rdaState from rdaDeltaFused's k̂ (uK) and rdaDecay's decays (uDecay), a key's at texel (key / 4, uRow). */
function rdaStateFusedSource(o: { keyHeadDim: number; valueHeadDim: number }) {
  const DV4 = o.valueHeadDim / 4
  return `${HEADER}
uniform sampler2D uS;
uniform sampler2D uK;
uniform sampler2D uDelta;
uniform sampler2D uDecay;
uniform int uRow;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int k = int(gl_FragCoord.y);
  int h = x / ${DV4};
  int kc = h * ${o.keyHeadDim} + k;
  float decay = comp(texelFetch(uDecay, ivec2(kc >> 2, uRow), 0), kc & 3);
  float kk = comp(texelFetch(uK, ivec2(kc >> 2, uRow), 0), kc & 3);
  o = decay * texelFetch(uS, ivec2(x, k), 0) + kk * texelFetch(uDelta, ivec2(x, uRow), 0);
}`
}
export const rdaStateFused = memo(rdaStateFusedSource)

/** linDelta with a decay per key and the write ratio on v. */
function rdaDeltaSource(o: { keyHeadDim: number; valueHeadDim: number; out: "both" | "delta" | "o" }) {
  const DK4 = o.keyHeadDim / 4
  const DV4 = o.valueHeadDim / 4
  const outs = o.out === "both" ? "layout(location = 0) out vec4 oDelta;\nlayout(location = 1) out vec4 oOut;" : "out vec4 oResult;"
  const write = o.out === "both" ? "oDelta = d;\n  oOut = qs + qk * d;" : o.out === "delta" ? "oResult = d;" : "oResult = qs + qk * d;"
  return `${HEADER}
uniform sampler2D uS;
uniform sampler2D uQK;
uniform sampler2D uQKV;
uniform sampler2D uG;
uniform int uK4;
uniform int uH4;
${outs}
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int h = x / ${DV4};
  int kb = h * ${DK4};
  vec4 kv = vec4(0.0);
  vec4 qs = vec4(0.0);
  float qk = 0.0;
  for (int i = 0; i < ${DK4}; i++) {
    vec4 q = texelFetch(uQK, ivec2(kb + i, m), 0);
    vec4 k = texelFetch(uQK, ivec2(uK4 + kb + i, m), 0);
    vec4 g = exp(texelFetch(uG, ivec2(2 * uH4 + kb + i, m), 0));
    qk += dot(q, k);
    k *= g;
    q *= g;
    int r = 4 * i;
    vec4 s0 = texelFetch(uS, ivec2(x, r), 0);
    vec4 s1 = texelFetch(uS, ivec2(x, r + 1), 0);
    vec4 s2 = texelFetch(uS, ivec2(x, r + 2), 0);
    vec4 s3 = texelFetch(uS, ivec2(x, r + 3), 0);
    kv += s0 * k.x + s1 * k.y + s2 * k.z + s3 * k.w;
    qs += s0 * q.x + s1 * q.y + s2 * q.z + s3 * q.w;
  }
  float beta = comp(texelFetch(uG, ivec2(h >> 2, m), 0), h & 3);
  float ratio = comp(texelFetch(uG, ivec2(uH4 + (h >> 2), m), 0), h & 3);
  vec4 v = texelFetch(uQKV, ivec2(2 * uK4 + x, m), 0) * ratio;
  vec4 d = beta * (v - kv);
  ${write}
}`
}
export const rdaDelta = memo(rdaDeltaSource)

/** linState with a decay per key: S[k][v] = decay_k · S[k][v] + k̂_k · delta_v, for token row uRow. */
function rdaStateSource(o: { keyHeadDim: number; valueHeadDim: number }) {
  const DV4 = o.valueHeadDim / 4
  return `${HEADER}
uniform sampler2D uS;
uniform sampler2D uQK;
uniform sampler2D uDelta;
uniform sampler2D uG;
uniform int uRow;
uniform int uK4;
uniform int uH4;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int k = int(gl_FragCoord.y);
  int h = x / ${DV4};
  int kc = h * ${o.keyHeadDim} + k;
  float decay = exp(comp(texelFetch(uG, ivec2(2 * uH4 + (kc >> 2), uRow), 0), kc & 3));
  float kk = comp(texelFetch(uQK, ivec2(uK4 + (kc >> 2), uRow), 0), kc & 3);
  o = decay * texelFetch(uS, ivec2(x, k), 0) + kk * texelFetch(uDelta, ivec2(x, uRow), 0);
}`
}
export const rdaState = memo(rdaStateSource)

/** The gated RMSNorm, into h2 for the out projection: linNorm with sigmoid(z). */
function rdaNormSource(valueHeadDim: number) {
  const T = valueHeadDim / 4
  return `${HEADER}
uniform sampler2D uO;
uniform usampler2D uIn;
uniform sampler2D uNorm;
uniform int uZ;
uniform float uEps;
out uvec4 o;
vec4 gated(int t, int m, int base, float inv) {
  vec4 z = pf(uIn, uZ + t, m);
  return texelFetch(uO, ivec2(t, m), 0) * inv * texelFetch(uNorm, ivec2(t - base, 0), 0) / (1.0 + exp(-z));
}
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int base = ((2 * x) / ${T}) * ${T};
  vec4 s = vec4(0.0);
  for (int i = 0; i < ${T}; i++) {
    vec4 v = texelFetch(uO, ivec2(base + i, m), 0);
    s += v * v;
  }
  float inv = inversesqrt(dot(s, vec4(1.0)) / ${valueHeadDim}.0 + uEps);
  o = packH2(gated(2 * x, m, base, inv), gated(2 * x + 1, m, base, inv));
}`
}
export const rdaNorm = memo(rdaNormSource)

/*
 * The chunked recurrence (see delta-rule.ts's deltaChunkedKeys, which these
 * follow; kernels.ts's linSolve and linChunkOut are shared): G, the
 * chunk's running sum of log-decays, is per key (cum: uK4 texels a row), so
 * A[t][s] = beta_t Σ_i k̂_t[i] k̂_s[i] exp(G_t[i] - G_s[i]) and P likewise
 * with q̂_t; U = T diag(beta · ratio) V, W = T diag(beta) (exp(G) ∘ K),
 * Q' = exp(G) ∘ Q - P W, and S = exp(G_last) ∘ S0 + Σ_s (exp(G_last - G_s)
 * ∘ k̂_s) D_sᵀ. Every exponent is a difference ≤ 0 or a sum from the
 * chunk's start (≤ 0).
 */
const chunkStart = (chunk: number) => `uniform int uFirst;
int chunkStart(int t) { return t < uFirst ? 0 : uFirst + ((t - uFirst) / ${chunk}) * ${chunk}; }`

/** G of row m's keys 4x..4x+3: their log-decays summed from the chunk's start. */
function rdaCumSource(chunk: number) {
  return `${HEADER}
uniform sampler2D uG;
uniform int uH4;
${chunkStart(chunk)}
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 s = vec4(0.0);
  for (int r = chunkStart(m); r <= m; r++) s += texelFetch(uG, ivec2(2 * uH4 + x, r), 0);
  o = s;
}`
}
export const rdaCum = memo(rdaCumSource)

function rdaGramSource(o: { keyHeadDim: number; chunk: number; out: "both" | "a" | "p" }) {
  const DK4 = o.keyHeadDim / 4
  const C4 = o.chunk / 4
  const outs = o.out === "both" ? "layout(location = 0) out vec4 oA;\nlayout(location = 1) out vec4 oP;" : "out vec4 oResult;"
  const write = o.out === "both" ? "oA = a;\n  oP = p;" : o.out === "a" ? "oResult = a;" : "oResult = p;"
  return `${HEADER}
uniform sampler2D uQK;
uniform sampler2D uG;
uniform sampler2D uCum;
uniform int uK4;
${chunkStart(o.chunk)}
${outs}
void main() {
  int x = int(gl_FragCoord.x);
  int t = int(gl_FragCoord.y);
  int h = x / ${C4};
  int s0 = chunkStart(t) + 4 * (x - h * ${C4});
  int kb = h * ${DK4};
  ivec4 s = ivec4(s0, s0 + 1, s0 + 2, s0 + 3);
  // (Columns past t are masked out below; their rows are read as t's.)
  ivec4 r = min(s, ivec4(t));
  vec4 kk = vec4(0.0);
  vec4 qk = vec4(0.0);
  if (s0 <= t) {
    for (int i = 0; i < ${DK4}; i++) {
      vec4 q = texelFetch(uQK, ivec2(kb + i, t), 0);
      vec4 k = texelFetch(uQK, ivec2(uK4 + kb + i, t), 0);
      vec4 gt = texelFetch(uCum, ivec2(kb + i, t), 0);
      vec4 k0 = texelFetch(uQK, ivec2(uK4 + kb + i, r.x), 0) * exp(min(gt - texelFetch(uCum, ivec2(kb + i, r.x), 0), 0.0));
      vec4 k1 = texelFetch(uQK, ivec2(uK4 + kb + i, r.y), 0) * exp(min(gt - texelFetch(uCum, ivec2(kb + i, r.y), 0), 0.0));
      vec4 k2 = texelFetch(uQK, ivec2(uK4 + kb + i, r.z), 0) * exp(min(gt - texelFetch(uCum, ivec2(kb + i, r.z), 0), 0.0));
      vec4 k3 = texelFetch(uQK, ivec2(uK4 + kb + i, r.w), 0) * exp(min(gt - texelFetch(uCum, ivec2(kb + i, r.w), 0), 0.0));
      kk += vec4(dot(k, k0), dot(k, k1), dot(k, k2), dot(k, k3));
      qk += vec4(dot(q, k0), dot(q, k1), dot(q, k2), dot(q, k3));
    }
  }
  float beta = comp(texelFetch(uG, ivec2(h >> 2, t), 0), h & 3);
  vec4 a = beta * kk * vec4(lessThan(s, ivec4(t)));
  vec4 p = qk * vec4(lessThanEqual(s, ivec4(t)));
  ${write}
}`
}
export const rdaGram = memo(rdaGramSource)

/** U and W over the chunk's rows up to t (key dims as many as value dims: one texel index for both). */
function rdaUWSource(o: { keyHeadDim: number; valueHeadDim: number; chunk: number; out: "both" | "u" | "w" }) {
  const C4 = o.chunk / 4
  const D4 = (o.out === "w" ? o.keyHeadDim : o.valueHeadDim) / 4
  const outs = o.out === "both" ? "layout(location = 0) out vec4 oU;\nlayout(location = 1) out vec4 oW;" : "out vec4 oResult;"
  const write = o.out === "both" ? "oU = u;\n  oW = w;" : o.out === "u" ? "oResult = u;" : "oResult = w;"
  const wantU = o.out !== "w"
  const wantW = o.out !== "u"
  return `${HEADER}
uniform sampler2D uT;
uniform sampler2D uQK;
uniform sampler2D uQKV;
uniform sampler2D uG;
uniform sampler2D uCum;
uniform int uK4;
uniform int uH4;
${chunkStart(o.chunk)}
${outs}
void main() {
  int x = int(gl_FragCoord.x);
  int t = int(gl_FragCoord.y);
  int h = x / ${D4};
  int c0 = chunkStart(t);
  vec4 u = vec4(0.0);
  vec4 w = vec4(0.0);
  for (int s = c0; s <= t; s++) {
    int j = s - c0;
    float tau = comp(texelFetch(uT, ivec2(h * ${C4} + (j >> 2), t), 0), j & 3)
      * comp(texelFetch(uG, ivec2(h >> 2, s), 0), h & 3);
    ${wantU ? "u += tau * comp(texelFetch(uG, ivec2(uH4 + (h >> 2), s), 0), h & 3) * texelFetch(uQKV, ivec2(2 * uK4 + x, s), 0);" : ""}
    ${wantW ? "w += tau * exp(texelFetch(uCum, ivec2(x, s), 0)) * texelFetch(uQK, ivec2(uK4 + x, s), 0);" : ""}
  }
  ${write}
}`
}
export const rdaUW = memo(rdaUWSource)

/** Oi = P U and Q' = exp(G) ∘ Q - P W over the chunk's rows up to t. */
function rdaQOSource(o: { keyHeadDim: number; valueHeadDim: number; chunk: number; out: "both" | "oi" | "q" }) {
  const C4 = o.chunk / 4
  const D4 = (o.out === "q" ? o.keyHeadDim : o.valueHeadDim) / 4
  const outs = o.out === "both" ? "layout(location = 0) out vec4 oI;\nlayout(location = 1) out vec4 oQ;" : "out vec4 oResult;"
  const write = o.out === "both" ? "oI = oi;\n  oQ = q;" : o.out === "oi" ? "oResult = oi;" : "oResult = q;"
  return `${HEADER}
uniform sampler2D uP;
uniform sampler2D uU;
uniform sampler2D uW;
uniform sampler2D uQK;
uniform sampler2D uCum;
${chunkStart(o.chunk)}
${outs}
void main() {
  int x = int(gl_FragCoord.x);
  int t = int(gl_FragCoord.y);
  int h = x / ${D4};
  int c0 = chunkStart(t);
  vec4 oi = vec4(0.0);
  vec4 pw = vec4(0.0);
  for (int s = c0; s <= t; s++) {
    int j = s - c0;
    float p = comp(texelFetch(uP, ivec2(h * ${C4} + (j >> 2), t), 0), j & 3);
    ${o.out !== "q" ? "oi += p * texelFetch(uU, ivec2(x, s), 0);" : ""}
    ${o.out !== "oi" ? "pw += p * texelFetch(uW, ivec2(x, s), 0);" : ""}
  }
  vec4 q = ${o.out !== "oi" ? "exp(texelFetch(uCum, ivec2(x, t), 0)) * texelFetch(uQK, ivec2(x, t), 0) - pw" : "vec4(0.0)"};
  ${write}
}`
}
export const rdaQO = memo(rdaQOSource)

/** A chunk's state update, its rows [uC0, uC1), each key by its own decay. */
function rdaChunkStateSource(o: { keyHeadDim: number; valueHeadDim: number }) {
  const DV4 = o.valueHeadDim / 4
  return `${HEADER}
uniform sampler2D uS;
uniform sampler2D uQK;
uniform sampler2D uDelta;
uniform sampler2D uCum;
uniform int uK4;
uniform int uC0;
uniform int uC1;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int k = int(gl_FragCoord.y);
  int h = x / ${DV4};
  int kc = h * ${o.keyHeadDim} + k;
  float last = comp(texelFetch(uCum, ivec2(kc >> 2, uC1 - 1), 0), kc & 3);
  vec4 acc = exp(last) * texelFetch(uS, ivec2(x, k), 0);
  for (int s = uC0; s < uC1; s++) {
    float kk = comp(texelFetch(uQK, ivec2(uK4 + (kc >> 2), s), 0), kc & 3);
    acc += exp(last - comp(texelFetch(uCum, ivec2(kc >> 2, s), 0), kc & 3)) * kk * texelFetch(uDelta, ivec2(x, s), 0);
  }
  o = acc;
}`
}
export const rdaChunkState = memo(rdaChunkStateSource)

// ------------------------------------------------------ latent attention
/*
 * The gated MLA, absorbed: the layer's first projection (uIn) holds q_a
 * (queryRank), kv_a (rank + shared) and the output gate's rows.
 */

/** h2 out = RMSNorm(the first `dim` of uIn) · uNorm: the query's latent, for q_b. */
function latentNormSource(dim: number) {
  const T = dim / 4
  return `${HEADER}
uniform usampler2D uIn;
uniform sampler2D uNorm;
uniform float uEps;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 s = vec4(0.0);
  for (int i = 0; i < ${T}; i++) {
    vec4 v = pf(uIn, i, m);
    s += v * v;
  }
  float inv = inversesqrt(dot(s, vec4(1.0)) / ${dim}.0 + uEps);
  o = packH2(
    pf(uIn, 2 * x, m) * inv * texelFetch(uNorm, ivec2(2 * x, 0), 0),
    pf(uIn, 2 * x + 1, m) * inv * texelFetch(uNorm, ivec2(2 * x + 1, 0), 0));
}`
}
export const latentNorm = memo(latentNormSource)

/**
 * A token's cache row, h2: its key (RMSNorm(the latent) · uNorm, then the
 * shared dims as they are) and, beside it, the same again as its value
 * (the attention sums the latent). The fragment at (x, y) reads token
 * y - uRowOffset; the row starts at texel uXOff; kv_a's rows start at
 * texel uBase of uIn.
 */
function latentKeysSource(o: { rank: number; shared: number }) {
  const T = o.rank / 4
  const W = (o.rank + o.shared) / 8
  return `${HEADER}
uniform usampler2D uIn;
uniform sampler2D uNorm;
uniform int uBase;
uniform int uRowOffset;
uniform int uXOff;
uniform float uEps;
out uvec4 o;
void main() {
  int j = int(gl_FragCoord.x) - uXOff;
  int m = int(gl_FragCoord.y) - uRowOffset;
  if (j >= ${W}) j -= ${W};
  vec4 a = pf(uIn, uBase + 2 * j, m);
  vec4 b = pf(uIn, uBase + 2 * j + 1, m);
  if (2 * j < ${T}) {
    vec4 s = vec4(0.0);
    for (int i = 0; i < ${T}; i++) {
      vec4 v = pf(uIn, uBase + i, m);
      s += v * v;
    }
    float inv = inversesqrt(dot(s, vec4(1.0)) / ${o.rank}.0 + uEps);
    a *= inv * texelFetch(uNorm, ivec2(2 * j, 0), 0);
    b *= inv * texelFetch(uNorm, ivec2(2 * j + 1, 0), 0);
  }
  o = packH2(a, b);
}`
}
export const latentKeys = memo(latentKeysSource)

/** h2 out = sigmoid(the output gate's rows, from texel uBase of uIn). */
export const LATENT_GATE = `${HEADER}
uniform usampler2D uIn;
uniform int uBase;
out uvec4 o;
${SIGMOID}
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  o = packH2(sigmoid4(pf(uIn, uBase + 2 * x, m)), sigmoid4(pf(uIn, uBase + 2 * x + 1, m)));
}`

/**
 * The queries as the latent attention reads them, f32, times uScale: per
 * head, its first `nope` dims of uQ (q_b's output, keyDim a head) through
 * the head's rank × nope matrix (uKB: texel (h · nope/4 + i, j) holds row
 * j's weights 4i..4i+3), then its shared dims as they are.
 */
function latentQuerySource(o: { rank: number; shared: number; keyDim: number }) {
  const nope4 = (o.keyDim - o.shared) / 4
  const W = (o.rank + o.shared) / 4
  const R4 = o.rank / 4
  return `${HEADER}
uniform usampler2D uQ;
uniform sampler2D uKB;
uniform float uScale;
uniform int uQx;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int h = x / ${W};
  int j = x - h * ${W};
  int qb = uQx + h * ${o.keyDim / 4};
  if (j >= ${R4}) {
    o = pf(uQ, qb + ${nope4} + j - ${R4}, m) * uScale;
    return;
  }
  vec4 acc = vec4(0.0);
  for (int i = 0; i < ${nope4}; i++) {
    vec4 q = pf(uQ, qb + i, m);
    int wx = h * ${nope4} + i;
    acc += vec4(
      dot(q, texelFetch(uKB, ivec2(wx, 4 * j), 0)), dot(q, texelFetch(uKB, ivec2(wx, 4 * j + 1), 0)),
      dot(q, texelFetch(uKB, ivec2(wx, 4 * j + 2), 0)), dot(q, texelFetch(uKB, ivec2(wx, 4 * j + 3), 0)));
  }
  o = acc * uScale;
}`
}
export const latentQuery = memo(latentQuerySource)

/**
 * The attention's output for the out projection, h2: per head, its attended
 * latent (uA: h2, the head's row at texel h · (rank + shared) / 8) through
 * the head's valueDim × rank matrix (uVB: texel (h · rank/4 + i, v) holds
 * row v's weights 4i..4i+3), times the gate (uGate, h2).
 */
function latentValueSource(o: { rank: number; shared: number; valueDim: number; gate: "h2" | "projection" }) {
  const W = (o.rank + o.shared) / 8
  const R8 = o.rank / 8
  const V8 = o.valueDim / 8
  const row = (v: string) => `dot(lo, texelFetch(uVB, ivec2(wx, ${v}), 0)) + dot(hi, texelFetch(uVB, ivec2(wx + 1, ${v}), 0))`
  return `${HEADER}
uniform usampler2D uA;
uniform sampler2D uVB;
${o.gate === "h2" ? "uniform usampler2D uGate;" : "uniform usampler2D uIn;\nuniform int uBase;"}
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int h = x / ${V8};
  int v0 = 8 * (x - h * ${V8});
  vec4 a = vec4(0.0);
  vec4 b = vec4(0.0);
  for (int i = 0; i < ${R8}; i++) {
    uvec4 u = texelFetch(uA, ivec2(h * ${W} + i, m), 0);
    vec4 lo = h2lo(u);
    vec4 hi = h2hi(u);
    int wx = h * ${o.rank / 4} + 2 * i;
    a += vec4(${row("v0")}, ${row("v0 + 1")}, ${row("v0 + 2")}, ${row("v0 + 3")});
    b += vec4(${row("v0 + 4")}, ${row("v0 + 5")}, ${row("v0 + 6")}, ${row("v0 + 7")});
  }
  ${o.gate === "h2"
    ? "uvec4 g = texelFetch(uGate, ivec2(x, m), 0);\n  o = packH2(a * h2lo(g), b * h2hi(g));"
    : "o = packH2(a / (1.0 + exp(-pf(uIn, uBase + 2 * x, m))), b / (1.0 + exp(-pf(uIn, uBase + 2 * x + 1, m))));"}
}`
}
export const latentValue = memo(latentValueSource)
