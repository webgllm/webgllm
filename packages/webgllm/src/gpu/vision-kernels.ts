/**
 * The vision encoder's shaders (see vision.ts). Activations are RGBA32F,
 * texel (x, m) holding features 4x..4x+3 of patch m. They run once per
 * image, so they stay plain: no tiling, splits or half-float packing.
 */
import { HEADER } from "./kernels.js"

/** The activation after a projection's bias (GELU by tanh, or exact). */
export type VisionAct = "none" | "tanh" | "erf"

const ACT = `
vec4 geluTanh(vec4 x) {
  vec4 inner = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0);
  return 0.5 * x * (1.0 + tanh(inner));
}
/* erf by Abramowitz and Stegun 7.1.26 (error below 1.5e-7). */
vec4 erf4(vec4 x) {
  vec4 a = abs(x);
  vec4 t = 1.0 / (1.0 + 0.3275911 * a);
  vec4 y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return sign(x) * y;
}
vec4 geluErf(vec4 x) { return 0.5 * x * (1.0 + erf4(x * 0.7071067811865476)); }`

/**
 * out = act(A · Wᵀ + bias) (+ uR): one texel (rows 4g..4g+3 of W) of one
 * patch per fragment. Weights are in blocks of 32 along K (uA holds zeros
 * past K, where the last block is short); uS holds the 4 rows' scales of
 * row group g, block b, at texel (b, g).
 *
 * 4-bit weights are as MatMulNBits stores them: texel (b, n) holds row n's
 * block b (byte j: element 2j low, 2j + 1 high). 8-bit weights are a byte
 * each: texels (2b, n) and (2b + 1, n) hold row n's block b, element j in
 * byte j. A weight is scale · (q − zero): `zeros` reads the zero points
 * from uZ, laid out as the scales; without it they are 8 or 128.
 *
 * `group`: logical row m is physical rows m·uGroup.. side by side (the
 * merger's 2 × 2 blocks of patches).
 */
export function visionMatmul(o: { bits: 4 | 8; zeros: boolean; act: VisionAct; residual: boolean; group: boolean }) {
  const fetch = (k: string) =>
    o.group ? `texelFetch(uA, ivec2((${k}) % uInTexels, m * uGroup + (${k}) / uInTexels), 0)` : `texelFetch(uA, ivec2(${k}, m), 0)`
  const four = `
float row(uvec4 q, vec4 e[4], vec4 d[4]) {
  float s = 0.0;
  for (int w = 0; w < 4; w++) {
    uint word = q[w];
    vec4 lo = vec4((uvec4(word) >> uvec4(0u, 8u, 16u, 24u)) & 15u);
    vec4 hi = vec4((uvec4(word) >> uvec4(4u, 12u, 20u, 28u)) & 15u);
    s += dot(lo, e[w]) + dot(hi, d[w]);
  }
  return s;
}
vec4 block(int b, int g, int m, out float sum) {
  // Elements 8w + 0, 2, 4, 6 (e) and 8w + 1, 3, 5, 7 (d) of the block.
  vec4 e[4];
  vec4 d[4];
  sum = 0.0;
  for (int w = 0; w < 4; w++) {
    int k0 = 8 * b + 2 * w;
    vec4 a0 = ${fetch("k0")};
    vec4 a1 = ${fetch("k0 + 1")};
    e[w] = vec4(a0.x, a0.z, a1.x, a1.z);
    d[w] = vec4(a0.y, a0.w, a1.y, a1.w);
    sum += dot(a0 + a1, vec4(1.0));
  }
  return vec4(
    row(texelFetch(uW, ivec2(b, 4 * g), 0), e, d),
    row(texelFetch(uW, ivec2(b, 4 * g + 1), 0), e, d),
    row(texelFetch(uW, ivec2(b, 4 * g + 2), 0), e, d),
    row(texelFetch(uW, ivec2(b, 4 * g + 3), 0), e, d)
  );
}`
  const eight = `
float row(uvec4 q0, uvec4 q1, vec4 a[8]) {
  float s = 0.0;
  for (int w = 0; w < 4; w++) {
    s += dot(vec4((uvec4(q0[w]) >> uvec4(0u, 8u, 16u, 24u)) & 255u), a[w]);
    s += dot(vec4((uvec4(q1[w]) >> uvec4(0u, 8u, 16u, 24u)) & 255u), a[4 + w]);
  }
  return s;
}
vec4 block(int b, int g, int m, out float sum) {
  vec4 a[8];
  sum = 0.0;
  for (int w = 0; w < 8; w++) {
    a[w] = ${fetch("8 * b + w")};
    sum += dot(a[w], vec4(1.0));
  }
  return vec4(
    row(texelFetch(uW, ivec2(2 * b, 4 * g), 0), texelFetch(uW, ivec2(2 * b + 1, 4 * g), 0), a),
    row(texelFetch(uW, ivec2(2 * b, 4 * g + 1), 0), texelFetch(uW, ivec2(2 * b + 1, 4 * g + 1), 0), a),
    row(texelFetch(uW, ivec2(2 * b, 4 * g + 2), 0), texelFetch(uW, ivec2(2 * b + 1, 4 * g + 2), 0), a),
    row(texelFetch(uW, ivec2(2 * b, 4 * g + 3), 0), texelFetch(uW, ivec2(2 * b + 1, 4 * g + 3), 0), a)
  );
}`
  return `${HEADER}
uniform sampler2D uA;
uniform usampler2D uW;
uniform sampler2D uS;
${o.zeros ? "uniform usampler2D uZ;" : ""}
uniform sampler2D uBias;
${o.residual ? "uniform sampler2D uR;" : ""}
uniform int uBlocks;
${o.group ? "uniform int uInTexels;\nuniform int uGroup;" : ""}
out vec4 o;
${ACT}
${o.bits === 4 ? four : eight}
void main() {
  int g = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 acc = vec4(0.0);
  for (int b = 0; b < uBlocks; b++) {
    float sum;
    vec4 q = block(b, g, m, sum);
    vec4 s = texelFetch(uS, ivec2(b, g), 0);
    ${o.zeros ? "vec4 z = vec4(texelFetch(uZ, ivec2(b, g), 0));" : `vec4 z = vec4(${o.bits === 4 ? "8.0" : "128.0"});`}
    acc += s * (q - z * sum);
  }
  vec4 x = acc + texelFetch(uBias, ivec2(g, 0), 0);
  ${o.act === "tanh" ? "x = geluTanh(x);" : o.act === "erf" ? "x = geluErf(x);" : ""}
  ${o.residual ? "x += texelFetch(uR, ivec2(g, m), 0);" : ""}
  o = x;
}`
}

/** LayerNorm over each row of uTexels texels, with weight uW and bias uB. */
export const VISION_LAYER_NORM = `${HEADER}
uniform sampler2D uX;
uniform sampler2D uW;
uniform sampler2D uB;
uniform int uTexels;
uniform float uEps;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 s = vec4(0.0);
  for (int i = 0; i < uTexels; i++) s += texelFetch(uX, ivec2(i, m), 0);
  float mean = dot(s, vec4(1.0)) / float(4 * uTexels);
  vec4 v = vec4(0.0);
  for (int i = 0; i < uTexels; i++) {
    vec4 c = texelFetch(uX, ivec2(i, m), 0) - mean;
    v += c * c;
  }
  float inv = inversesqrt(dot(v, vec4(1.0)) / float(4 * uTexels) + uEps);
  o = (texelFetch(uX, ivec2(x, m), 0) - mean) * inv * texelFetch(uW, ivec2(x, 0), 0) + texelFetch(uB, ivec2(x, 0), 0);
}`

/** RMS norm over each row of uTexels texels, with weight uW: x · rsqrt(mean(x²) + eps) · w. */
export const VISION_RMS_NORM = `${HEADER}
uniform sampler2D uX;
uniform sampler2D uW;
uniform int uTexels;
uniform float uEps;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 v = vec4(0.0);
  for (int i = 0; i < uTexels; i++) {
    vec4 c = texelFetch(uX, ivec2(i, m), 0);
    v += c * c;
  }
  float inv = inversesqrt(dot(v, vec4(1.0)) / float(4 * uTexels) + uEps);
  o = texelFetch(uX, ivec2(x, m), 0) * inv * texelFetch(uW, ivec2(x, 0), 0);
}`

/**
 * q and k of the qkv projection (q | k | v; a head is `head` texels), made
 * ready for the scores: texels 0..uQ - 1 of the output are q (times
 * uScale), the next uK texels k.
 *
 * `norm`: each head RMS-normalized first, with weights uQW for q's heads
 * and uKW for k's. `rope`: each head rotated: dim d of its first half
 * pairs with d + half at frequency d (rotate half), cos and sin per patch
 * from uCS (texels 0..head/2 - 1 cos, the next head/2 sin, of
 * frequencies 0..2·head - 1).
 */
export function visionQk(o: { head: number; norm: boolean; rope: boolean }) {
  const half = o.head / 2
  if (o.rope && o.head % 2) throw new Error("RoPE rotates heads of a whole number of texel pairs.")
  return `${HEADER}
uniform sampler2D uQKV;
${o.rope ? "uniform sampler2D uCS;" : ""}
${o.norm ? "uniform sampler2D uQW;\nuniform sampler2D uKW;\nuniform float uEps;" : ""}
uniform int uQ;
uniform float uScale;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  bool isQ = x < uQ;
  int t = (isQ ? x : x - uQ) % ${o.head};
  int base = x - t;
  vec4 v = texelFetch(uQKV, ivec2(x, m), 0);
  ${o.rope ? `bool first = t < ${half};
  int tp = first ? t + ${half} : t - ${half};
  int f = first ? t : tp;
  vec4 p = texelFetch(uQKV, ivec2(base + tp, m), 0);` : ""}
  ${
    o.norm
      ? `vec4 ss = vec4(0.0);
  for (int i = 0; i < ${o.head}; i++) {
    vec4 c = texelFetch(uQKV, ivec2(base + i, m), 0);
    ss += c * c;
  }
  float inv = inversesqrt(dot(ss, vec4(1.0)) / float(${4 * o.head}) + uEps);
  v *= inv * (isQ ? texelFetch(uQW, ivec2(t, 0), 0) : texelFetch(uKW, ivec2(t, 0), 0));
  ${o.rope ? "p *= inv * (isQ ? texelFetch(uQW, ivec2(tp, 0), 0) : texelFetch(uKW, ivec2(tp, 0), 0));" : ""}`
      : ""
  }
  ${
    o.rope
      ? `vec4 c = texelFetch(uCS, ivec2(f, m), 0);
  vec4 s = texelFetch(uCS, ivec2(${half} + f, m), 0);
  v = first ? v * c - p * s : v * c + p * s;`
      : ""
  }
  o = isQ ? v * uScale : v;
}`
}

/**
 * Scores of query rows uQ0.. against every key: texel (x, h · uRows + i)
 * holds keys 4x..4x+3 for head h, query uQ0 + i (keys past uKeys: -1e30).
 * uQK is visionQk's output: q in texels 0..uTexels - 1, then k, whose
 * head h / uRep serves query head h (grouped-query attention).
 */
export function visionScores(head: number) {
  return `${HEADER}
uniform sampler2D uQK;
uniform int uTexels;
uniform int uRows;
uniform int uQ0;
uniform int uKeys;
uniform int uRep;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int h = y / uRows;
  int q = uQ0 + y - h * uRows;
  int kb = uTexels + (h / uRep) * ${head};
  vec4 qv[${head}];
  for (int i = 0; i < ${head}; i++) qv[i] = texelFetch(uQK, ivec2(h * ${head} + i, q), 0);
  vec4 r;
  for (int c = 0; c < 4; c++) {
    int key = 4 * x + c;
    float s = -1e30;
    if (key < uKeys) {
      s = 0.0;
      for (int i = 0; i < ${head}; i++) s += dot(qv[i], texelFetch(uQK, ivec2(kb + i, key), 0));
    }
    r[c] = s;
  }
  o = r;
}`
}

/** Each score row's max and 1 / Σ exp(score − max). */
export const VISION_SOFTMAX_STATS = `${HEADER}
uniform sampler2D uS;
uniform int uWidth;
out vec4 o;
void main() {
  int y = int(gl_FragCoord.y);
  vec4 mx = vec4(-1e30);
  for (int x = 0; x < uWidth; x++) mx = max(mx, texelFetch(uS, ivec2(x, y), 0));
  float top = max(max(mx.x, mx.y), max(mx.z, mx.w));
  vec4 sum = vec4(0.0);
  for (int x = 0; x < uWidth; x++) sum += exp(texelFetch(uS, ivec2(x, y), 0) - top);
  o = vec4(top, 1.0 / dot(sum, vec4(1.0)), 0.0, 0.0);
}`

/**
 * Attention's output for query rows uQ0.. (drawn at those rows): texel x
 * (head x / head) is Σ softmax(score) · v, v's head h / uRep at texel
 * uVBase + … of the qkv projection.
 */
export function visionValues(head: number) {
  return `${HEADER}
uniform sampler2D uS;
uniform sampler2D uStats;
uniform sampler2D uQKV;
uniform int uWidth;
uniform int uRows;
uniform int uQ0;
uniform int uKeys;
uniform int uVBase;
uniform int uRep;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int q = int(gl_FragCoord.y);
  int h = x / ${head};
  int y = h * uRows + q - uQ0;
  int vx = uVBase + (h / uRep) * ${head} + x % ${head};
  vec4 st = texelFetch(uStats, ivec2(0, y), 0);
  vec4 acc = vec4(0.0);
  for (int kx = 0; kx < uWidth; kx++) {
    vec4 p = exp(texelFetch(uS, ivec2(kx, y), 0) - st.x);
    int key = 4 * kx;
    for (int c = 0; c < 4; c++)
      if (key + c < uKeys) acc += p[c] * texelFetch(uQKV, ivec2(vx, key + c), 0);
  }
  o = acc * st.y;
}`
}

/**
 * LFM3-VL's gated depthwise 3 × 3 convolution over the patch grid. uX is
 * the in-projection (B | C | x, uTexels texels each); the output is
 * C · conv(B · x), zeros past the grid's edges. Patches are in merge order
 * (uMerge × uMerge blocks consecutive, blocks row-major); uK holds the
 * kernel, tap ky · 3 + kx of channels 4x..4x+3 at texel (x, ky · 3 + kx).
 */
export const VISION_CONV = `${HEADER}
uniform sampler2D uX;
uniform sampler2D uK;
uniform int uTexels;
uniform int uGridH;
uniform int uGridW;
uniform int uMerge;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int per = uMerge * uMerge;
  int blocksW = uGridW / uMerge;
  int blk = m / per;
  int inner = m - blk * per;
  int row = (blk / blocksW) * uMerge + inner / uMerge;
  int col = (blk % blocksW) * uMerge + inner % uMerge;
  vec4 acc = vec4(0.0);
  for (int ky = 0; ky < 3; ky++) {
    int r = row + ky - 1;
    if (r < 0 || r >= uGridH) continue;
    for (int kx = 0; kx < 3; kx++) {
      int c = col + kx - 1;
      if (c < 0 || c >= uGridW) continue;
      int p = ((r / uMerge) * blocksW + c / uMerge) * per + (r % uMerge) * uMerge + c % uMerge;
      acc += texelFetch(uK, ivec2(x, ky * 3 + kx), 0) * texelFetch(uX, ivec2(x, p), 0) * texelFetch(uX, ivec2(2 * uTexels + x, p), 0);
    }
  }
  o = acc * texelFetch(uX, ivec2(uTexels + x, m), 0);
}`

/** SwiGLU's product: silu(gate) · up, of a gate | up projection (uTexels texels each). */
export const VISION_SWIGLU = `${HEADER}
uniform sampler2D uX;
uniform int uTexels;
out vec4 o;
void main() {
  ivec2 at = ivec2(gl_FragCoord.xy);
  vec4 g = texelFetch(uX, at, 0);
  o = g / (1.0 + exp(-g)) * texelFetch(uX, ivec2(at.x + uTexels, at.y), 0);
}`
