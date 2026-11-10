/**
 * The vision encoder's shaders (see vision.ts). Activations are RGBA32F,
 * texel (x, m) holding features 4x..4x+3 of patch m. They run once per
 * image, so they stay plain: no tiling, splits or half-float packing.
 */
import { HEADER } from "./kernels.js"

/** The activation after a projection's bias. */
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
 * patch per fragment. W is 4-bit, as MatMulNBits stores it: texel (b, n)
 * holds row n's block b (byte j: element 2j low, 2j + 1 high); uS and uZ
 * hold the 4 rows' scales and zero points of row group g, block b, at
 * texel (b, g). `group`: logical row m is physical rows m·uGroup.. side by
 * side (the merger's 2 × 2 blocks of patches).
 */
export function visionMatmul(o: { act: VisionAct; residual: boolean; group: boolean }) {
  return `${HEADER}
uniform sampler2D uA;
uniform usampler2D uW;
uniform sampler2D uS;
uniform usampler2D uZ;
uniform sampler2D uBias;
${o.residual ? "uniform sampler2D uR;" : ""}
uniform int uBlocks;
${o.group ? "uniform int uInTexels;\nuniform int uGroup;" : ""}
out vec4 o;
${ACT}
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
void main() {
  int g = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 acc = vec4(0.0);
  for (int b = 0; b < uBlocks; b++) {
    // Elements 8w + 0, 2, 4, 6 (e) and 8w + 1, 3, 5, 7 (d) of the block.
    vec4 e[4];
    vec4 d[4];
    float sum = 0.0;
    for (int w = 0; w < 4; w++) {
      int k0 = 8 * b + 2 * w;
      ${
        o.group
          ? `vec4 a0 = texelFetch(uA, ivec2(k0 % uInTexels, m * uGroup + k0 / uInTexels), 0);
      vec4 a1 = texelFetch(uA, ivec2((k0 + 1) % uInTexels, m * uGroup + (k0 + 1) / uInTexels), 0);`
          : `vec4 a0 = texelFetch(uA, ivec2(k0, m), 0);
      vec4 a1 = texelFetch(uA, ivec2(k0 + 1, m), 0);`
      }
      e[w] = vec4(a0.x, a0.z, a1.x, a1.z);
      d[w] = vec4(a0.y, a0.w, a1.y, a1.w);
      sum += dot(a0 + a1, vec4(1.0));
    }
    vec4 s = texelFetch(uS, ivec2(b, g), 0);
    vec4 z = vec4(texelFetch(uZ, ivec2(b, g), 0));
    vec4 q = vec4(
      row(texelFetch(uW, ivec2(b, 4 * g), 0), e, d),
      row(texelFetch(uW, ivec2(b, 4 * g + 1), 0), e, d),
      row(texelFetch(uW, ivec2(b, 4 * g + 2), 0), e, d),
      row(texelFetch(uW, ivec2(b, 4 * g + 3), 0), e, d)
    );
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

/**
 * q and k of the qkv projection (q | k | v, each heads × 16 texels),
 * rotated: dim d < 32 of a head pairs with d + 32 at frequency d (rotate
 * half), cos and sin per patch from uCS (texels 0..7 cos, 8..15 sin of
 * frequencies 0..31). Texels 0..W/4 - 1 of the output are q (times uScale),
 * the next W/4 k.
 */
export const VISION_ROPE = `${HEADER}
uniform sampler2D uQKV;
uniform sampler2D uCS;
uniform int uTexels;
uniform float uScale;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int part = x / uTexels;
  int base = x - (x % 16);
  int t = x % 16;
  bool first = t < 8;
  int tp = first ? t + 8 : t - 8;
  int f = t % 8;
  vec4 v = texelFetch(uQKV, ivec2(x, m), 0);
  vec4 p = texelFetch(uQKV, ivec2(base + tp, m), 0);
  vec4 c = texelFetch(uCS, ivec2(f, m), 0);
  vec4 s = texelFetch(uCS, ivec2(8 + f, m), 0);
  vec4 r = first ? v * c - p * s : v * c + p * s;
  o = part == 0 ? r * uScale : r;
}`

/**
 * Scores of query rows uQ0.. against every key: texel (x, h · uRows + i)
 * holds keys 4x..4x+3 for head h, query uQ0 + i (keys past uKeys: -1e30).
 */
export const VISION_SCORES = `${HEADER}
uniform sampler2D uQK;
uniform int uTexels;
uniform int uRows;
uniform int uQ0;
uniform int uKeys;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int h = y / uRows;
  int q = uQ0 + y - h * uRows;
  vec4 qv[16];
  for (int i = 0; i < 16; i++) qv[i] = texelFetch(uQK, ivec2(h * 16 + i, q), 0);
  vec4 r;
  for (int c = 0; c < 4; c++) {
    int key = 4 * x + c;
    float s = -1e30;
    if (key < uKeys) {
      s = 0.0;
      for (int i = 0; i < 16; i++) s += dot(qv[i], texelFetch(uQK, ivec2(uTexels + h * 16 + i, key), 0));
    }
    r[c] = s;
  }
  o = r;
}`

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
 * (head x / 16) is Σ softmax(score) · v, v at texel uVBase + x of the qkv
 * projection.
 */
export const VISION_VALUES = `${HEADER}
uniform sampler2D uS;
uniform sampler2D uStats;
uniform sampler2D uQKV;
uniform int uWidth;
uniform int uRows;
uniform int uQ0;
uniform int uKeys;
uniform int uVBase;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int q = int(gl_FragCoord.y);
  int h = x / 16;
  int y = h * uRows + q - uQ0;
  vec4 st = texelFetch(uStats, ivec2(0, y), 0);
  vec4 acc = vec4(0.0);
  for (int kx = 0; kx < uWidth; kx++) {
    vec4 p = exp(texelFetch(uS, ivec2(kx, y), 0) - st.x);
    int key = 4 * kx;
    for (int c = 0; c < 4; c++)
      if (key + c < uKeys) acc += p[c] * texelFetch(uQKV, ivec2(uVBase + x, key + c), 0);
  }
  o = acc * st.y;
}`
