/**
 * The model's operations as fragment shaders. Each fragment writes one
 * texel of the output tensor.
 *
 * Tensor layouts (rows are tokens, or cache positions):
 * - f32: RGBA32F, 4 features per texel: element (row, c) at texel
 *   (c / 4, row), component c % 4.
 * - h2: RGBA32UI, 8 halves per texel: element (row, c) at texel
 *   (c / 8, row), word (c % 8) / 2, low half for even c. Matmul inputs
 *   and the KV cache use it: half the fetches of f32.
 * - Quantized weights: see weights.ts. Their textures may be split into
 *   columns when a matrix has more row groups than a texture is tall:
 *   row group g is in column g / H at row g % H.
 */

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
precision highp isampler2D;
`

/** Halves of an h2 texel: elements 0..3 and 4..7. */
const H2 = `
vec4 h2lo(uvec4 u) { return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); }
vec4 h2hi(uvec4 u) { return vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
uvec4 packH2(vec4 a, vec4 b) {
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}
`

export type MatmulOptions = {
  bits: 4 | 8
  zeros: boolean
  /**
   * One input row, its output spread over a 2D texture (the output head:
   * 32,000 texels is wider than some GPUs allow). Otherwise output texel
   * (g, m) is row group g of input row m.
   */
  wrap: boolean
  /** Precision of the per-block sums (accumulators stay highp). */
  precision: "highp" | "mediump"
  /**
   * Weights in byte textures (RGBA8UI, or RGBA8I for 8-bit): the texture
   * unit splits the bytes and nibbles split with float math. Otherwise
   * RGBA32UI texels, split with integer shifts.
   */
  bytes: boolean
  /**
   * Split K across fragments (for a few rows, where N / 4 fragments would
   * leave the GPU idle): output row m * uSplit + s holds slice s's partial
   * sums, over blocks [s * uKbSlice, (s + 1) * uKbSlice); REDUCE_SLICES adds
   * them up.
   */
  split?: boolean
}

/**
 * out = A · Wᵀ for h2 activations A (rows × K) and quantized W (N × K).
 * Uniforms: uKb (blocks per row), uH (row groups per weight column),
 * uRowTexels (quant texels per row group per column), uG (row groups);
 * with `wrap`, uOutW (output texels per row) and uRow (the input row).
 */
export function matmul(o: MatmulOptions) {
  const P = o.precision === "mediump" ? "mediump " : ""
  const rows = [0, 1, 2, 3]
  const body: Array<string> = []
  const line = (s: string) => body.push(`    ${s}`)
  line(`vec4 d = texelFetch(uS, ivec2(sx + kb, gr), 0);`)
  if (o.zeros) line(`vec4 z = vec4(texelFetch(uZ, ivec2(sx + kb, gr), 0));`)
  for (const j of [0, 1, 2, 3]) {
    line(`uvec4 a${j} = texelFetch(uA, ivec2(kb * 4 + ${j}, m), 0);`)
    line(`${P}vec4 A${2 * j} = h2lo(a${j});`)
    line(`${P}vec4 A${2 * j + 1} = h2hi(a${j});`)
  }
  if (o.bits === 4) {
    line(
      `${P}float sa = dot(A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7, vec4(1.0));`
    )
    for (const r of rows) {
      line(`{`)
      if (!o.bytes)
        line(`  uvec4 w = texelFetch(uW, ivec2(wx + kb * 4 + ${r}, gr), 0);`)
      line(`  ${P}float b = 0.0;`)
      for (const w of [0, 1, 2, 3]) {
        const c = "xyzw"[w]
        if (o.bytes) {
          // Bytes 4w..4w+3 of the block row: low nibbles are weights
          // 4w..4w+3, high nibbles 16+4w.., split exactly in floats.
          line(
            `  { ${P}vec4 B = vec4(texelFetch(uW, ivec2(wx + kb * 16 + ${r * 4 + w}, gr), 0));`
          )
          line(`    ${P}vec4 hi = floor(B * 0.0625);`)
          line(`    ${P}vec4 lo = B - 16.0 * hi;`)
        } else {
          line(
            `  { ${P}vec4 lo = vec4((uvec4(w.${c}) >> uvec4(0u, 8u, 16u, 24u)) & 15u);`
          )
          line(
            `    ${P}vec4 hi = vec4((uvec4(w.${c}) >> uvec4(4u, 12u, 20u, 28u)) & 15u);`
          )
        }
        line(`    b += dot(lo, A${w}) + dot(hi, A${w + 4}); }`)
      }
      const zero = o.zeros ? `z.${"xyzw"[r]}` : "8.0"
      line(
        `  acc.${"xyzw"[r]} += d.${"xyzw"[r]} * (float(b) - ${zero} * float(sa));`
      )
      line(`}`)
    }
  } else {
    // 8-bit: word w of a block row holds weights 4w..4w+3 as signed bytes.
    for (const r of rows) {
      line(`{`)
      if (!o.bytes) {
        line(
          `  uvec4 w0 = texelFetch(uW, ivec2(wx + kb * 8 + ${r * 2}, gr), 0);`
        )
        line(
          `  uvec4 w1 = texelFetch(uW, ivec2(wx + kb * 8 + ${r * 2 + 1}, gr), 0);`
        )
      }
      line(`  ${P}float b = 0.0;`)
      for (const w of [0, 1, 2, 3, 4, 5, 6, 7]) {
        if (o.bytes) {
          line(
            `  { ${P}vec4 q = vec4(texelFetch(uW, ivec2(wx + kb * 32 + ${r * 8 + w}, gr), 0)); b += dot(q, A${w}); }`
          )
        } else {
          const word = `w${w >> 2}.${"xyzw"[w & 3]}`
          line(
            `  { ${P}vec4 q = vec4(ivec4(uvec4(${word}) << uvec4(24u, 16u, 8u, 0u)) >> 24); b += dot(q, A${w}); }`
          )
        }
      }
      line(`  acc.${"xyzw"[r]} += d.${"xyzw"[r]} * float(b);`)
      line(`}`)
    }
  }
  const index = o.wrap
    ? `int g = y * uOutW + x;
  int m = uRow;
  if (g >= uG) { o = vec4(0.0); return; }
  int kb0 = 0;
  int kb1 = uKb;`
    : o.split
      ? `int g = x;
  int m = y / uSplit;
  int slice = y - m * uSplit;
  int kb0 = slice * uKbSlice;
  int kb1 = min(uKb, kb0 + uKbSlice);`
      : `int g = x;
  int m = y;
  int kb0 = 0;
  int kb1 = uKb;`
  const weights = o.bytes && o.bits === 8 ? "isampler2D" : "usampler2D"
  return `${HEADER}${H2}
uniform usampler2D uA;
uniform ${weights} uW;
uniform sampler2D uS;
${o.zeros ? "uniform usampler2D uZ;" : ""}
uniform int uKb;
uniform int uH;
uniform int uRowTexels;
uniform int uG;
uniform int uOutW;
uniform int uRow;
uniform int uSplit;
uniform int uKbSlice;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  ${index}
  int col = g / uH;
  int gr = g - col * uH;
  int wx = col * uRowTexels;
  int sx = col * uKb;
  vec4 acc = vec4(0.0);
  for (int kb = kb0; kb < kb1; kb++) {
${body.join("\n")}
  }
  o = acc;
}`
}

/**
 * Embedding rows for token ids (uIds: R32UI, one id per row) from the
 * quantized embedding: f32 output, 4 features per texel.
 */
export function embed(o: { bits: 4 | 8; zeros: boolean; bytes: boolean }) {
  const zero = o.zeros
    ? "float(texelFetch(uZ, ivec2(col * uKb + kb, gr), 0)[r])"
    : "8.0"
  let fetch: string
  if (o.bits === 4 && o.bytes) {
    fetch = `int word = j < 16 ? j >> 2 : (j - 16) >> 2;
  vec4 B = vec4(texelFetch(uW, ivec2(col * uRowTexels + kb * 16 + r * 4 + word, gr), 0));
  vec4 hi = floor(B * 0.0625);
  vec4 q = j < 16 ? B - 16.0 * hi : hi;
  o = d * (q - ${zero});`
  } else if (o.bits === 4) {
    fetch = `uvec4 t = texelFetch(uW, ivec2(col * uRowTexels + kb * 4 + r, gr), 0);
  uint word = j < 16 ? t[j >> 2] : t[(j - 16) >> 2];
  uvec4 shifts = j < 16 ? uvec4(0u, 8u, 16u, 24u) : uvec4(4u, 12u, 20u, 28u);
  vec4 q = vec4((uvec4(word) >> shifts) & 15u);
  o = d * (q - ${zero});`
  } else if (o.bytes) {
    fetch = `o = d * vec4(texelFetch(uW, ivec2(col * uRowTexels + kb * 32 + r * 8 + (j >> 2), gr), 0));`
  } else {
    fetch = `uvec4 t = texelFetch(uW, ivec2(col * uRowTexels + kb * 8 + r * 2 + (j >> 4), gr), 0);
  uint word = t[(j & 15) >> 2];
  vec4 q = vec4(ivec4(uvec4(word) << uvec4(24u, 16u, 8u, 0u)) >> 24);
  o = d * q;`
  }
  return `${HEADER}
uniform usampler2D uIds;
uniform ${o.bytes && o.bits === 8 ? "isampler2D" : "usampler2D"} uW;
uniform sampler2D uS;
${o.zeros ? "uniform usampler2D uZ;" : ""}
uniform int uKb;
uniform int uH;
uniform int uRowTexels;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int id = int(texelFetch(uIds, ivec2(0, y), 0).x);
  int g = id >> 2;
  int r = id & 3;
  int col = g / uH;
  int gr = g - col * uH;
  int c = x * 4;
  int kb = c >> 5;
  int j = c & 31;
  float d = texelFetch(uS, ivec2(col * uKb + kb, gr), 0)[r];
  ${fetch}
}`
}

/** Per row: x = inverse RMS of an f32 tensor's row (uTexels texels), y = mean square. */
export const RMS = `${HEADER}
uniform sampler2D uX;
uniform int uTexels;
uniform float uEps;
out vec4 o;
void main() {
  int m = int(gl_FragCoord.y);
  vec4 s = vec4(0.0);
  for (int i = 0; i < uTexels; i++) {
    vec4 v = texelFetch(uX, ivec2(i, m), 0);
    s += v * v;
  }
  float ms = dot(s, vec4(1.0)) / float(uTexels * 4);
  o = vec4(inversesqrt(ms + uEps), ms, 0.0, 0.0);
}`

/** h2 out = x · rms(x) · w, from f32 x and its RMS (RMS pass) and f32 weights. */
export const NORM_TO_H2 = `${HEADER}${H2}
uniform sampler2D uX;
uniform sampler2D uR;
uniform sampler2D uNorm;
uniform int uRowIn;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int m = uRowIn < 0 ? y : uRowIn;
  float r = texelFetch(uR, ivec2(0, m), 0).x;
  vec4 a = texelFetch(uX, ivec2(2 * x, m), 0) * r * texelFetch(uNorm, ivec2(2 * x, 0), 0);
  vec4 b = texelFetch(uX, ivec2(2 * x + 1, m), 0) * r * texelFetch(uNorm, ivec2(2 * x + 1, 0), 0);
  o = packH2(a, b);
}`

/** f32 out = h + y · rms(y) · w: a sandwich norm's output added to the residual. */
export const ADD_NORMED = `${HEADER}
uniform sampler2D uH;
uniform sampler2D uY;
uniform sampler2D uR;
uniform sampler2D uNorm;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float r = texelFetch(uR, ivec2(0, p.y), 0).x;
  o = texelFetch(uH, p, 0) + texelFetch(uY, p, 0) * r * texelFetch(uNorm, ivec2(p.x, 0), 0);
}`

/**
 * The short convolution: from in_proj's output (f32 B | C | x, uD/4 texels
 * each), z = C · Σ_k w[k] · (B·x)[t - (width-1) + k], with earlier tokens'
 * B·x from the state (uState: width-1 rows, oldest first). h2 output.
 * Weights uConv: texel (c / 4, k) holds tap k of channels c..c+3.
 */
export function shortConv(width: number) {
  const taps = Array.from({ length: width }, (_, k) => k)
    .map(
      (k) => `  {
    int t = m - ${width - 1 - k};
    vec4 y0;
    vec4 y1;
    if (t >= 0) {
      y0 = texelFetch(uBCx, ivec2(c0, t), 0) * texelFetch(uBCx, ivec2(2 * dt + c0, t), 0);
      y1 = texelFetch(uBCx, ivec2(c0 + 1, t), 0) * texelFetch(uBCx, ivec2(2 * dt + c0 + 1, t), 0);
    } else {
      y0 = texelFetch(uState, ivec2(c0, ${width - 1} + t), 0);
      y1 = texelFetch(uState, ivec2(c0 + 1, ${width - 1} + t), 0);
    }
    a0 += texelFetch(uConv, ivec2(c0, ${k}), 0) * y0;
    a1 += texelFetch(uConv, ivec2(c0 + 1, ${k}), 0) * y1;
  }`
    )
    .join("\n")
  return `${HEADER}${H2}
uniform sampler2D uBCx;
uniform sampler2D uState;
uniform sampler2D uConv;
uniform int uDt;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int dt = uDt;
  int c0 = 2 * x;
  vec4 a0 = vec4(0.0);
  vec4 a1 = vec4(0.0);
${taps}
  vec4 C0 = texelFetch(uBCx, ivec2(dt + c0, m), 0);
  vec4 C1 = texelFetch(uBCx, ivec2(dt + c0 + 1, m), 0);
  o = packH2(C0 * a0, C1 * a1);
}`
}

/**
 * The convolution's state after a batch of uM tokens: the last width-1
 * values of B·x, oldest first, from this batch or the previous state.
 */
export function convState(width: number) {
  return `${HEADER}
uniform sampler2D uBCx;
uniform sampler2D uState;
uniform int uDt;
uniform int uM;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  int t = uM - ${width - 1} + i;
  if (t >= 0) {
    o = texelFetch(uBCx, ivec2(x, t), 0) * texelFetch(uBCx, ivec2(2 * uDt + x, t), 0);
  } else {
    o = texelFetch(uState, ivec2(x, uM + i), 0);
  }
}`
}

/**
 * RMS-normalized and rotated head vectors. Reads one head (32 f32 texels
 * from texel uBase + head * 32) of the q|k|v|gate projection, normalizes it
 * with uNorm (32 texels), and applies RoPE from the table uRope (row =
 * position; texels 0..15 cos, 16..31 sin of frequencies 0..63). The
 * rotated dims come first (the loaders order them so): dim j pairs with
 * j + P for j < P, P = rotated dims / 2 = 4 · uPairTexels, with frequency
 * j; the dims after 2P aren't rotated. Writes f32 (`out: "f32"`, 4 dims per texel, for
 * queries) or h2 (8 dims per texel, for the key cache).
 */
export function headPrep(o: {
  out: "f32" | "h2"
  rope: boolean
  norm: boolean
}) {
  const texelsPerOut = o.out === "f32" ? 1 : 2
  const prep = `
vec4 rotated(int base, int t, int pos, float inv) {
  vec4 v = texelFetch(uQKVG, ivec2(base + t, m), 0)${o.norm ? " * inv * texelFetch(uNorm, ivec2(t, 0), 0)" : ""};
  ${
    o.rope
      ? `if (t >= 2 * uPairTexels) return v;
  bool first = t < uPairTexels;
  int tp = first ? t + uPairTexels : t - uPairTexels;
  int f = first ? t : tp;
  vec4 p = texelFetch(uQKVG, ivec2(base + tp, m), 0)${o.norm ? " * inv * texelFetch(uNorm, ivec2(tp, 0), 0)" : ""};
  vec4 cs = texelFetch(uRope, ivec2(f, pos), 0);
  vec4 sn = texelFetch(uRope, ivec2(16 + f, pos), 0);
  return first ? v * cs - p * sn : v * cs + p * sn;`
      : "return v;"
  }
}`
  return `${HEADER}${H2}
uniform sampler2D uQKVG;
uniform sampler2D uNorm;
uniform sampler2D uRope;
uniform int uBase;
uniform int uPairTexels;
uniform int uPos0;
uniform int uRowOffset;
uniform float uEps;
uniform float uScale;
int m;
${prep}
out ${o.out === "f32" ? "vec4" : "uvec4"} o;
void main() {
  int x = int(gl_FragCoord.x);
  m = int(gl_FragCoord.y) - uRowOffset;
  int head = x / ${32 / texelsPerOut};
  int t0 = (x % ${32 / texelsPerOut}) * ${texelsPerOut};
  int base = uBase + head * 32;
  float inv = 1.0;
  ${
    o.norm
      ? `vec4 s = vec4(0.0);
  for (int i = 0; i < 32; i++) {
    vec4 v = texelFetch(uQKVG, ivec2(base + i, m), 0);
    s += v * v;
  }
  inv = inversesqrt(dot(s, vec4(1.0)) / 128.0 + uEps);`
      : ""
  }
  int pos = uPos0 + m;
  ${
    o.out === "f32"
      ? "o = rotated(base, t0, pos, inv) * uScale;"
      : "o = packH2(rotated(base, t0, pos, inv), rotated(base, t0 + 1, pos, inv));"
  }
}`
}

/**
 * Attention scores, 4 keys per texel, for the uRows queries from uM0: row
 * h * uRows + (m - uM0) holds query m of head h against keys uJ0 + 4x ..
 * + 3. With uRing > 0 the cache is a ring of that many rows (position j at
 * row j % uRing). Queries are pre-scaled f32 (uQ), keys h2
 * in the cache (uK: row = position, 16 texels per KV head). Masked keys
 * (later than the query, outside the sliding window, past uKeys) get -1e30.
 */
export const SCORES = `${HEADER}${H2}
uniform sampler2D uQ;
uniform usampler2D uK;
uniform int uRows;
uniform int uM0;
uniform int uRing;
uniform int uJ0;
uniform int uKeys;
uniform int uPos0;
uniform int uWindow;
uniform int uGroup;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int h = y / uRows;
  int m = uM0 + (y - h * uRows);
  int kh = h / uGroup;
  int qpos = uPos0 + m;
  vec4 s = vec4(-1e30);
  for (int c = 0; c < 4; c++) {
    int j = uJ0 + 4 * x + c;
    if (j - uJ0 >= uKeys || j > qpos || (uWindow > 0 && qpos - j >= uWindow)) continue;
    float acc = 0.0;
    for (int u = 0; u < 16; u++) {
      uvec4 k = texelFetch(uK, ivec2(kh * 16 + u, uRing > 0 ? j % uRing : j), 0);
      acc += dot(texelFetch(uQ, ivec2(h * 32 + 2 * u, m), 0), h2lo(k));
      acc += dot(texelFetch(uQ, ivec2(h * 32 + 2 * u + 1, m), 0), h2hi(k));
    }
    s[c] = acc;
  }
  o = s;
}`

/**
 * Softmax over a query's scores, the weighted sum of values (uV: h2 cache),
 * times the head's sigmoid gate (from the q|k|v|gate projection at texel
 * uGateTexel). h2 output: 8 dims of one head per texel.
 */
export const ATTEND = `${HEADER}${H2}
uniform sampler2D uS;
uniform usampler2D uV;
uniform sampler2D uQKVG;
uniform int uRows;
uniform int uM0;
uniform int uRing;
uniform int uJ0;
uniform int uKeys;
uniform int uGroup;
uniform int uGateTexel;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int h = x / 16;
  int u = x - h * 16;
  int kh = h / uGroup;
  int row = h * uRows + (m - uM0);
  int texels = (uKeys + 3) / 4;
  float mx = -1e30;
  for (int i = 0; i < texels; i++) {
    vec4 s = texelFetch(uS, ivec2(i, row), 0);
    mx = max(mx, max(max(s.x, s.y), max(s.z, s.w)));
  }
  float l = 0.0;
  vec4 lo = vec4(0.0);
  vec4 hi = vec4(0.0);
  for (int i = 0; i < texels; i++) {
    vec4 e = exp(texelFetch(uS, ivec2(i, row), 0) - mx);
    l += dot(e, vec4(1.0));
    for (int c = 0; c < 4; c++) {
      int j = 4 * i + c;
      if (j >= uKeys) break;
      int vj = uJ0 + j;
      uvec4 v = texelFetch(uV, ivec2(kh * 16 + u, uRing > 0 ? vj % uRing : vj), 0);
      lo += e[c] * h2lo(v);
      hi += e[c] * h2hi(v);
    }
  }
  float gate = texelFetch(uQKVG, ivec2(uGateTexel + h / 4, m), 0)[h & 3];
  float scale = 1.0 / (l * (1.0 + exp(-gate)));
  o = packH2(lo * scale, hi * scale);
}`

/** Values into the h2 cache: 8 dims per texel, rows offset to the cache position. */
export const VALUES = `${HEADER}${H2}
uniform sampler2D uQKVG;
uniform int uBase;
uniform int uRowOffset;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y) - uRowOffset;
  int head = x / 16;
  int t = uBase + head * 32 + (x % 16) * 2;
  o = packH2(texelFetch(uQKVG, ivec2(t, m), 0), texelFetch(uQKVG, ivec2(t + 1, m), 0));
}`

/** SwiGLU: silu(gate) · up, from w1|w3's f32 output (uHalf texels each). h2 output. */
export const SWIGLU = `${HEADER}${H2}
uniform sampler2D uGU;
uniform int uHalf;
out uvec4 o;
vec4 silu(vec4 g) { return g / (1.0 + exp(-g)); }
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 a = silu(texelFetch(uGU, ivec2(2 * x, m), 0)) * texelFetch(uGU, ivec2(uHalf + 2 * x, m), 0);
  vec4 b = silu(texelFetch(uGU, ivec2(2 * x + 1, m), 0)) * texelFetch(uGU, ivec2(uHalf + 2 * x + 1, m), 0);
  o = packH2(a, b);
}`

/** Copies an f32 tensor (for debugging reads of h2 tensors: h2 → f32). */
export const H2_TO_F32 = `${HEADER}${H2}
uniform usampler2D uX;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  uvec4 u = texelFetch(uX, ivec2(p.x >> 1, p.y), 0);
  o = (p.x & 1) == 0 ? h2lo(u) : h2hi(u);
}`

/** Copies an f32 tensor. */
export const COPY = `${HEADER}
uniform sampler2D uX;
out vec4 o;
void main() {
  o = texelFetch(uX, ivec2(gl_FragCoord.xy), 0);
}`

/** NORM_TO_H2 with the row's RMS computed in place (for a few rows: decoding). */
export const NORM_TO_H2_FUSED = `${HEADER}${H2}
uniform sampler2D uX;
uniform sampler2D uNorm;
uniform int uRowIn;
uniform int uTexels;
uniform float uEps;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int m = uRowIn < 0 ? y : uRowIn;
  vec4 s = vec4(0.0);
  for (int i = 0; i < uTexels; i++) {
    vec4 v = texelFetch(uX, ivec2(i, m), 0);
    s += v * v;
  }
  float r = inversesqrt(dot(s, vec4(1.0)) / float(uTexels * 4) + uEps);
  vec4 a = texelFetch(uX, ivec2(2 * x, m), 0) * r * texelFetch(uNorm, ivec2(2 * x, 0), 0);
  vec4 b = texelFetch(uX, ivec2(2 * x + 1, m), 0) * r * texelFetch(uNorm, ivec2(2 * x + 1, 0), 0);
  o = packH2(a, b);
}`

/** ADD_NORMED with y's RMS computed in place (for a few rows: decoding). */
export const ADD_NORMED_FUSED = `${HEADER}
uniform sampler2D uH;
uniform sampler2D uY;
uniform sampler2D uNorm;
uniform int uTexels;
uniform float uEps;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 s = vec4(0.0);
  for (int i = 0; i < uTexels; i++) {
    vec4 v = texelFetch(uY, ivec2(i, p.y), 0);
    s += v * v;
  }
  float r = inversesqrt(dot(s, vec4(1.0)) / float(uTexels * 4) + uEps);
  o = texelFetch(uH, p, 0) + texelFetch(uY, p, 0) * r * texelFetch(uNorm, ivec2(p.x, 0), 0);
}`

/** Sums a split matmul's slices: out(x, m) = Σ_s partial(x, m * uSplit + s). */
export const REDUCE_SLICES = `${HEADER}
uniform sampler2D uP;
uniform int uSplit;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 acc = vec4(0.0);
  for (int s = 0; s < uSplit; s++) acc += texelFetch(uP, ivec2(x, m * uSplit + s), 0);
  o = acc;
}`

/*
 * Picking candidates on the GPU, so writing a token reads back ~50 KB
 * instead of every logit. Logits are the output head's wrapped layout
 * (uOutW RGBA texels, 4·uOutW logits, per row); the mask (R8UI, 4·uOutW
 * wide) marks tokens seen (1: repetition penalty) or never to draw (2).
 */
const PENALIZED = `
float penalized(float v, uint m) {
  if (m == 2u) return -1e30;
  if (m == 1u) return v < 0.0 ? v * uPenalty : v / uPenalty;
  return v;
}
vec4 chunkTexel(int c, int t) {
  int texel = c * 64 + t;
  ivec2 p = ivec2(texel % uOutW, texel / uOutW);
  vec4 v = texelFetch(uLogits, p, 0);
  int base = 4 * p.x;
  return vec4(
    penalized(v.x, texelFetch(uMask, ivec2(base, p.y), 0).x),
    penalized(v.y, texelFetch(uMask, ivec2(base + 1, p.y), 0).x),
    penalized(v.z, texelFetch(uMask, ivec2(base + 2, p.y), 0).x),
    penalized(v.w, texelFetch(uMask, ivec2(base + 3, p.y), 0).x));
}`

/** Each 256-logit chunk's largest (penalized) logit: x = chunk index. */
export const CHUNK_MAX = `${HEADER}
uniform sampler2D uLogits;
uniform usampler2D uMask;
uniform int uOutW;
uniform float uPenalty;
${PENALIZED}
out vec4 o;
void main() {
  int c = int(gl_FragCoord.x);
  float mx = -1e30;
  for (int t = 0; t < 64; t++) {
    vec4 v = chunkTexel(c, t);
    mx = max(mx, max(max(v.x, v.y), max(v.z, v.w)));
  }
  o = vec4(mx, 0.0, 0.0, 0.0);
}`

/** Each chunk's rank by its largest logit (0 = largest; ties by index). */
export const CHUNK_RANK = `${HEADER}
uniform sampler2D uMax;
uniform int uChunks;
out vec4 o;
void main() {
  int c = int(gl_FragCoord.x);
  float mine = texelFetch(uMax, ivec2(c, 0), 0).x;
  int rank = 0;
  for (int i = 0; i < uChunks; i++) {
    float v = texelFetch(uMax, ivec2(i, 0), 0).x;
    if (v > mine || (v == mine && i < c)) rank++;
  }
  o = vec4(float(rank), 0.0, 0.0, 0.0);
}`

/**
 * Row r: the logits of the chunk ranked r, 64 texels, then its index in
 * texel 64.
 */
export const CHUNK_GATHER = `${HEADER}
uniform sampler2D uLogits;
uniform usampler2D uMask;
uniform sampler2D uRank;
uniform int uOutW;
uniform int uChunks;
uniform float uPenalty;
${PENALIZED}
out vec4 o;
void main() {
  int t = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  int c = 0;
  for (int i = 0; i < uChunks; i++) {
    if (int(texelFetch(uRank, ivec2(i, 0), 0).x) == r) c = i;
  }
  o = t < 64 ? chunkTexel(c, t) : vec4(float(c), 0.0, 0.0, 0.0);
}`

/**
 * Copies cache rows (h2) between caches of different lengths: target row y
 * holds position uPos0 + (y - uRow0), read from row position % uMod of uX.
 */
export const COPY_POSITIONS = `${HEADER}
uniform usampler2D uX;
uniform int uPos0;
uniform int uRow0;
uniform int uMod;
out uvec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int pos = uPos0 + (p.y - uRow0);
  o = texelFetch(uX, ivec2(p.x, pos % uMod), 0);
}`

/*
 * ggml Q6_K, read natively (see weights.ts for the layout), from a byte
 * texture: a row's super-block of 256 weights is 52 RGBA8UI texels, ql
 * (texels 0..31), qh (32..47) and the int8 scales (48..51), with its fp16 d
 * in the scales texture. In half n of a super-block, for l = 0..31, as
 * dequantize_row_q6_K:
 *   y[128n + l]      = d · sc[8n + l/16 + 0] · ((ql[64n + l] & 15      | (qh[32n + l] >> 0 & 3) << 4) - 32)
 *   y[128n + l + 32] = d · sc[8n + l/16 + 2] · ((ql[64n + l + 32] & 15 | (qh[32n + l] >> 2 & 3) << 4) - 32)
 *   y[128n + l + 64] = d · sc[8n + l/16 + 4] · ((ql[64n + l] >> 4      | (qh[32n + l] >> 4 & 3) << 4) - 32)
 *   y[128n + l + 96] = d · sc[8n + l/16 + 6] · ((ql[64n + l + 32] >> 4 | (qh[32n + l] >> 6 & 3) << 4) - 32)
 * Texel w of a half covers l = 4w..4w+3: ql texels 16n + w and 16n + 8 + w,
 * qh texel 32 + 8n + w. The bits come apart with float math (exact on
 * integers up to 255), which mobile GPUs run faster than integer shifts.
 */
const Q6_K_UNPACK = `
// Low nibbles (lo) and high nibbles (hi) of 4 bytes.
void nibbles(vec4 B, out vec4 lo, out vec4 hi) { hi = floor(B * 0.0625); lo = B - 16.0 * hi; }
// The four 2-bit fields of 4 bytes, lowest first.
void pairs(vec4 B, out vec4 h0, out vec4 h1, out vec4 h2, out vec4 h3) {
  vec4 t1 = floor(B * 0.25);
  h0 = B - 4.0 * t1;
  vec4 t2 = floor(t1 * 0.25);
  h1 = t1 - 4.0 * t2;
  h3 = floor(t2 * 0.25);
  h2 = t2 - 4.0 * h3;
}
// An int8 scale from its byte.
float sc8(float b) { return b < 128.0 ? b : b - 256.0; }`

/**
 * The output head from a Q6_K embedding: `matmul` with `wrap`, for 6-bit
 * weights. For each word position the four groups' activations are decoded
 * once and shared by the fragment's 4 rows; each row reads its 3 weight
 * texels and the 2 scale texels for that half.
 */
export const MATMUL_Q6_K_HEAD = (() => {
  const C = "xyzw"
  const body: Array<string> = []
  const line = (s: string) => body.push(`    ${s}`)
  line(`vec4 s = vec4(0.0);`)
  for (const n of [0, 1]) {
    for (let w = 0; w < 8; w++) {
      const half = w & 1 ? "h2hi" : "h2lo"
      line(`{`)
      for (let g = 0; g < 4; g++) {
        line(
          `  vec4 A${g} = ${half}(texelFetch(uA, ivec2(sb * 32 + ${16 * n + 4 * g + (w >> 1)}, m), 0));`
        )
      }
      for (const r of [0, 1, 2, 3]) {
        line(`  {`)
        line(`    int t0 = wx + (sb * 4 + ${r}) * 52;`)
        line(`    vec4 a0, a1, b0, b1, h0, h1, h2, h3;`)
        line(
          `    nibbles(vec4(texelFetch(uW, ivec2(t0 + ${16 * n + w}, gr), 0)), a0, a1);`
        )
        line(
          `    nibbles(vec4(texelFetch(uW, ivec2(t0 + ${16 * n + 8 + w}, gr), 0)), b0, b1);`
        )
        line(
          `    pairs(vec4(texelFetch(uW, ivec2(t0 + ${32 + 8 * n + w}, gr), 0)), h0, h1, h2, h3);`
        )
        // Scales 8n + (w >> 2) + 2g: groups 0 and 1 in texel 48 + 2n,
        // groups 2 and 3 in the next.
        line(
          `    vec4 SA = vec4(texelFetch(uW, ivec2(t0 + ${48 + 2 * n}, gr), 0));`
        )
        line(
          `    vec4 SB = vec4(texelFetch(uW, ivec2(t0 + ${49 + 2 * n}, gr), 0));`
        )
        const c0 = C[w >> 2]
        const c2 = C[(w >> 2) + 2]
        line(`    s.${C[r]} += sc8(SA.${c0}) * dot(a0 + 16.0 * h0 - 32.0, A0)`)
        line(`      + sc8(SA.${c2}) * dot(b0 + 16.0 * h1 - 32.0, A1)`)
        line(`      + sc8(SB.${c0}) * dot(a1 + 16.0 * h2 - 32.0, A2)`)
        line(`      + sc8(SB.${c2}) * dot(b1 + 16.0 * h3 - 32.0, A3);`)
        line(`  }`)
      }
      line(`}`)
    }
  }
  line(`acc += d * s;`)
  return `${HEADER}${H2}${Q6_K_UNPACK}
uniform usampler2D uA;
uniform usampler2D uW;
uniform sampler2D uS;
uniform int uKb;
uniform int uH;
uniform int uRowTexels;
uniform int uG;
uniform int uOutW;
uniform int uRow;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int g = y * uOutW + x;
  int m = uRow;
  if (g >= uG) { o = vec4(0.0); return; }
  int col = g / uH;
  int gr = g - col * uH;
  int wx = col * uRowTexels;
  int sx = col * uKb;
  vec4 acc = vec4(0.0);
  for (int sb = 0; sb < uKb; sb++) {
    vec4 d = texelFetch(uS, ivec2(sx + sb, gr), 0);
${body.join("\n")}
  }
  o = acc;
}`
})()

/** Embedding rows for token ids from a Q6_K embedding: `embed`, for 6-bit weights. */
export const EMBED_Q6_K = `${HEADER}${Q6_K_UNPACK}
uniform usampler2D uIds;
uniform usampler2D uW;
uniform sampler2D uS;
uniform int uKb;
uniform int uH;
uniform int uRowTexels;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int id = int(texelFetch(uIds, ivec2(0, y), 0).x);
  int g = id >> 2;
  int r = id & 3;
  int col = g / uH;
  int gr = g - col * uH;
  int c = 4 * x;
  int sb = c >> 8;
  int n = (c >> 7) & 1;
  int grp = (c >> 5) & 3;
  int w = (c & 31) >> 2;
  int t0 = col * uRowTexels + (sb * 4 + r) * 52;
  vec4 lo, hi, h0, h1, h2, h3;
  nibbles(vec4(texelFetch(uW, ivec2(t0 + 16 * n + w + (grp & 1) * 8, gr), 0)), lo, hi);
  pairs(vec4(texelFetch(uW, ivec2(t0 + 32 + 8 * n + w, gr), 0)), h0, h1, h2, h3);
  vec4 low = grp < 2 ? lo : hi;
  vec4 high = grp == 0 ? h0 : grp == 1 ? h1 : grp == 2 ? h2 : h3;
  int k = 8 * n + (w >> 2) + 2 * grp;
  float sc = sc8(float(texelFetch(uW, ivec2(t0 + 48 + (k >> 2), gr), 0)[k & 3]));
  float d = texelFetch(uS, ivec2(col * uKb + sb, gr), 0)[r];
  o = (d * sc) * (low + 16.0 * high - 32.0);
}`
