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
/* Halves of an h2 texel: elements 0..3 and 4..7. */
vec4 h2lo(uvec4 u) { return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); }
vec4 h2hi(uvec4 u) { return vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
uvec4 packH2(vec4 a, vec4 b) {
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}
/* Texel x (features 4x..4x+3) of token m in a pair tensor. */
uniform int uPairSplit;
uniform int uSliceStride;
vec4 pf(highp usampler2D s, int x, int m) {
  // A K-split prompt matmul stacks its uPairSplit slices, uSliceStride
  // rows apart.
  vec4 acc = vec4(0.0);
  for (int k = 0; k < max(uPairSplit, 1); k++) {
    uvec4 u = texelFetch(s, ivec2(x, (m >> 1) + k * uSliceStride), 0);
    acc += (m & 1) == 0 ? h2lo(u) : h2hi(u);
  }
  return acc;
}
/* The same from a split matmul's uSplitIn partial sums (row m · uSplitIn + s). */
uniform int uSplitIn;
vec4 pf(highp sampler2D s, int x, int m) {
  // A constant bound, so the loop unrolls and the reads go out together.
  vec4 acc = vec4(0.0);
  for (int k = 0; k < 16; k++) {
    if (k >= uSplitIn) break;
    acc += texelFetch(s, ivec2(x, m * uSplitIn + k), 0);
  }
  return acc;
}

`

/** Kept for kernels that name it; the helpers are in HEADER. */
const H2 = ""

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
   * sums, over blocks [s * uKbSlice, (s + 1) * uKbSlice); REDUCE_PAIR adds
   * them up.
   */
  split?: boolean
  /**
   * Write a pair tensor: fragment (g, y) computes tokens 2y and 2y + 1
   * (for layer weights not in RGBA4: 8-bit).
   */
  pair?: boolean
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
out ${o.pair ? "uvec4" : "vec4"} o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  ${o.pair ? "vec4 r0 = vec4(0.0);\n  for (int tt = 0; tt < 2; tt++) {\n  int g = x;\n  int m = 2 * y + tt;\n  int kb0 = 0;\n  int kb1 = uKb;" : index}
  int col = g / uH;
  int gr = g - col * uH;
  int wx = col * uRowTexels;
  int sx = col * uKb;
  vec4 acc = vec4(0.0);
  for (int kb = kb0; kb < kb1; kb++) {
${body.join("\n")}
  }
  ${o.pair ? "if (tt == 0) r0 = acc;\n  else o = packH2(r0, acc);\n  }" : "o = acc;"}
}`
}

/**
 * The prompt matmul for 4-bit weights in RGBA4 textures (texel
 * (column · uRowTexels + kb · 32 + r · 8 + j, row group) holds row r's
 * weights 32kb + 4j .. + 3 as q / 15: the texture unit splits the nibbles
 * and converts them, exactly). Each fragment computes one row group (4 rows)
 * for two tokens, 2y and 2y + 1, into one pair texel: half the weight
 * fetches of one token per fragment, in the 16 bytes a fragment can write
 * without slowing down. Sums are f32: per block d · (Σ q·a − zero · Σ a).
 */
export function matmulPair(o: { zeros: boolean }) {
  const L: Array<string> = []
  const line = (s: string) => L.push(`    ${s}`)
  line(`vec4 d = texelFetch(uS, ivec2(sx + kb, gr), 0);`)
  line(
    o.zeros
      ? `vec4 z = vec4(texelFetch(uZ, ivec2(sx + kb, gr), 0));`
      : `const vec4 z = vec4(8.0);`
  )
  line(`vec4 b0 = vec4(0.0); vec4 b1 = vec4(0.0); float s0 = 0.0; float s1 = 0.0;`)
  for (let j = 0; j < 8; j++) {
    line(`{`)
    for (const r of [0, 1, 2, 3])
      line(`  vec4 w${r} = texelFetch(uW, ivec2(wx + kb * 32 + ${r * 8 + j}, gr), 0);`)
    const half = j & 1 ? "h2hi" : "h2lo"
    for (const t of [0, 1]) {
      line(`  { vec4 a = ${half}(texelFetch(uA, ivec2(kb * 4 + ${j >> 1}, m0 + ${t}), 0));`)
      line(`    s${t} += dot(a, vec4(1.0));`)
      line(`    b${t} += vec4(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a)); }`)
    }
    line(`}`)
  }
  line(`acc0 += d * (15.0 * b0 - z * s0);`)
  line(`acc1 += d * (15.0 * b1 - z * s1);`)
  return `${HEADER}
uniform usampler2D uA;
uniform sampler2D uW;
uniform sampler2D uS;
${o.zeros ? "uniform usampler2D uZ;" : ""}
uniform int uKb;
uniform int uH;
uniform int uRowTexels;
uniform int uSliceRows;
uniform int uKbSlice;
out uvec4 o;
void main() {
  // K split into slices stacked in the output: slice s's partial sums for
  // pair row py are at row s · uSliceRows + py (pf adds them up).
  int g = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int slice = y / uSliceRows;
  int m0 = 2 * (y - slice * uSliceRows);
  int kb0 = slice * uKbSlice;
  int kb1 = min(uKb, kb0 + uKbSlice);
  int col = g / uH;
  int gr = g - col * uH;
  int wx = col * uRowTexels;
  int sx = col * uKb;
  vec4 acc0 = vec4(0.0);
  vec4 acc1 = vec4(0.0);
  for (int kb = kb0; kb < kb1; kb++) {
${L.join("\n")}
  }
  o = packH2(acc0, acc1);
}`
}

/**
 * The few-token matmul for RGBA4 weights (see `matmulPair`), split along K:
 * output row m · uSplit + s holds row group x's partial sums for token m
 * over blocks [s · uKbSlice, (s + 1) · uKbSlice); pf adds them up. One
 * token per fragment: a fragment writing several tokens' sums (several
 * outputs) runs slower than their own fragments reading the same weights.
 */
export function matmulSplitRgba4(o: { zeros: boolean; normed?: boolean }) {
  const L: Array<string> = []
  const line = (s: string) => L.push(`    ${s}`)
  line(`vec4 d = texelFetch(uS, ivec2(sx + kb, gr), 0);`)
  line(
    o.zeros
      ? `vec4 z = vec4(texelFetch(uZ, ivec2(sx + kb, gr), 0));`
      : `const vec4 z = vec4(8.0);`
  )
  if (o.normed) {
    // x · norm weights from the f32 residual; the row's 1 / RMS is applied
    // to the sums at the end (it's the same for the whole row).
    for (let j = 0; j < 8; j++)
      line(
        `vec4 A${j} = texelFetch(uA, ivec2(kb * 8 + ${j}, m), 0) * texelFetch(uNorm, ivec2(kb * 8 + ${j}, 0), 0);`
      )
  } else
  for (const j of [0, 1, 2, 3]) {
    line(`uvec4 a${j} = texelFetch(uA, ivec2(kb * 4 + ${j}, m), 0);`)
    line(`vec4 A${2 * j} = h2lo(a${j}); vec4 A${2 * j + 1} = h2hi(a${j});`)
  }
  line(`float sa = dot(A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7, vec4(1.0));`)
  line(`vec4 b = vec4(0.0);`)
  for (const r of [0, 1, 2, 3]) {
    const terms = Array.from(
      { length: 8 },
      (_, j) => `dot(texelFetch(uW, ivec2(wx + kb * 32 + ${r * 8 + j}, gr), 0), A${j})`
    )
    line(`b.${"xyzw"[r]} = ${terms.join(" + ")};`)
  }
  line(`acc += d * (15.0 * b - z * sa);`)
  return `${HEADER}${o.normed ? `${RMS_SCALE}\nuniform sampler2D uA;\nuniform sampler2D uNorm;` : "uniform usampler2D uA;"}
uniform sampler2D uW;
uniform sampler2D uS;
${o.zeros ? "uniform usampler2D uZ;" : ""}
uniform int uKb;
uniform int uH;
uniform int uRowTexels;
uniform int uSplit;
uniform int uKbSlice;
out vec4 o;
void main() {
  int g = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int m = y / uSplit;
  int slice = y - m * uSplit;
  int kb0 = slice * uKbSlice;
  int kb1 = min(uKb, kb0 + uKbSlice);
  int col = g / uH;
  int gr = g - col * uH;
  int wx = col * uRowTexels;
  int sx = col * uKb;
  vec4 acc = vec4(0.0);
  for (int kb = kb0; kb < kb1; kb++) {
${L.join("\n")}
  }
  o = ${o.normed ? "acc * rmsScale(m)" : "acc"};
}`
}

/**
 * The output head for RGBA4 weights (see `matmulPair`): one token (row 0
 * of uA), output spread over uOutW-wide rows (the vocabulary is wider than
 * a texture). Texel (x, s · uSliceRows + y) holds row group y · uOutW + x's
 * sums over blocks [s · uKbSlice, (s + 1) · uKbSlice); HEAD_SUM adds the
 * slices when there's more than one.
 */
export function headRgba4(o: { zeros: boolean }) {
  const L: Array<string> = []
  const line = (s: string) => L.push(`    ${s}`)
  line(`vec4 d = texelFetch(uS, ivec2(sx + kb, gr), 0);`)
  line(
    o.zeros
      ? `vec4 z = vec4(texelFetch(uZ, ivec2(sx + kb, gr), 0));`
      : `const vec4 z = vec4(8.0);`
  )
  for (const j of [0, 1, 2, 3]) {
    line(`uvec4 a${j} = texelFetch(uA, ivec2(kb * 4 + ${j}, 0), 0);`)
    line(`vec4 A${2 * j} = h2lo(a${j}); vec4 A${2 * j + 1} = h2hi(a${j});`)
  }
  line(`float sa = dot(A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7, vec4(1.0));`)
  line(`vec4 b = vec4(0.0);`)
  for (const r of [0, 1, 2, 3]) {
    const terms = Array.from(
      { length: 8 },
      (_, j) => `dot(texelFetch(uW, ivec2(wx + kb * 32 + ${r * 8 + j}, gr), 0), A${j})`
    )
    line(`b.${"xyzw"[r]} = ${terms.join(" + ")};`)
  }
  line(`acc += d * (15.0 * b - z * sa);`)
  return `${HEADER}
uniform usampler2D uA;
uniform sampler2D uW;
uniform sampler2D uS;
${o.zeros ? "uniform usampler2D uZ;" : ""}
uniform int uKb;
uniform int uH;
uniform int uRowTexels;
uniform int uG;
uniform int uOutW;
uniform int uSliceRows;
uniform int uKbSlice;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int slice = y / uSliceRows;
  int g = (y - slice * uSliceRows) * uOutW + x;
  vec4 acc = vec4(0.0);
  if (g < uG) {
    int kb0 = slice * uKbSlice;
    int kb1 = min(uKb, kb0 + uKbSlice);
    int col = g / uH;
    int gr = g - col * uH;
    int wx = col * uRowTexels;
    int sx = col * uKb;
    for (int kb = kb0; kb < kb1; kb++) {
${L.join("\n")}
    }
  }
  o = acc;
}`
}

/** The output head's slices (see `headRgba4`) added up: texel (x, y). */
export const HEAD_SUM = `${HEADER}
uniform sampler2D uP;
uniform int uSplit;
uniform int uSliceRows;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 acc = vec4(0.0);
  for (int s = 0; s < uSplit; s++) acc += texelFetch(uP, ivec2(p.x, p.y + s * uSliceRows), 0);
  o = acc;
}`

/**
 * Sums a split matmul's slices into a pair tensor: texel (x, y) holds
 * tokens 2y and 2y + 1 (zero past uRows), each Σ_s partial(x, m · uSplit + s).
 */
export const REDUCE_PAIR = `${HEADER}
uniform sampler2D uP;
uniform int uSplit;
uniform int uRows;
out uvec4 o;
vec4 total(int x, int m) {
  vec4 acc = vec4(0.0);
  if (m >= uRows) return acc;
  for (int s = 0; s < uSplit; s++) acc += texelFetch(uP, ivec2(x, m * uSplit + s), 0);
  return acc;
}
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  o = packH2(total(x, 2 * y), total(x, 2 * y + 1));
}`

/**
 * A split gate|up matmul's slices summed and through SwiGLU, straight into
 * the h2 activations: texel (x, m) holds silu(gate) · up for features
 * 8x..8x+7 of token m (gate texel t, up texel uHalf + t).
 */
export const REDUCE_SWIGLU = `${HEADER}
uniform sampler2D uP;
uniform int uHalf;
out uvec4 o;
vec4 silu(vec4 g) { return g / (1.0 + exp(-g)); }
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 a = silu(pf(uP, 2 * x, m)) * pf(uP, uHalf + 2 * x, m);
  vec4 b = silu(pf(uP, 2 * x + 1, m)) * pf(uP, uHalf + 2 * x + 1, m);
  o = packH2(a, b);
}`

/** A pair tensor as f32 (for probes): texel (x, m). */
export const PAIR_TO_F32 = `${HEADER}
uniform usampler2D uX;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = pf(uX, p.x, p.y);
}`

/**
 * Embedding rows for token ids (uIds: R32UI, one id per row) from the
 * quantized embedding: f32 output, 4 features per texel.
 */
export function embed(o: {
  bits: 4 | 8
  zeros: boolean
  bytes: boolean
  rgba4?: boolean
}) {
  const zero = o.zeros
    ? "float(texelFetch(uZ, ivec2(col * uKb + kb, gr), 0)[r])"
    : "8.0"
  let fetch: string
  if (o.rgba4) {
    // Texel kb · 32 + r · 8 + j / 4: row r's weights 32kb + j .. + 3, as q / 15.
    fetch = `vec4 q = floor(15.0 * texelFetch(uW, ivec2(col * uRowTexels + kb * 32 + r * 8 + (j >> 2), gr), 0) + 0.5);
  o = d * (q - ${zero});`
  } else if (o.bits === 4 && o.bytes) {
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
uniform ${o.rgba4 ? "sampler2D" : o.bytes && o.bits === 8 ? "isampler2D" : "usampler2D"} uW;
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
/** Pixels per row that each sum a slice of the row's squares (RMS_PARTIAL). */
export const RMS_PARTS = 64

/**
 * Partial sums of squares: pixel (p, m) sums texels [p·uChunk, (p+1)·uChunk)
 * of row m. The norms add up the RMS_PARTS partials, so no pixel walks a
 * whole row.
 */
/**
 * `kind`: x is f32, a pair tensor, or a split matmul's partial sums
 * (read through pf).
 */
export function rmsPartial(kind: "f32" | "pair" | "split") {
  const pair = kind !== "f32"
  return `${HEADER}
uniform ${kind === "pair" ? "usampler2D" : "sampler2D"} uX;
uniform int uTexels;
uniform int uChunk;
out vec4 o;
void main() {
  int p = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int i0 = p * uChunk;
  int i1 = min(uTexels, i0 + uChunk);
  vec4 s = vec4(0.0);
  // Unrolled 4 at a time, so the reads go out together.
  for (int i = i0; i < i1; i += 4) {
    for (int k = 0; k < 4; k++) {
      if (i + k >= i1) break;
      vec4 v = ${pair ? "pf(uX, i + k, m)" : "texelFetch(uX, ivec2(i + k, m), 0)"};
      s += v * v;
    }
  }
  o = s;
}`
}

/** 1 / RMS of row m from its RMS_PARTIAL sums (uTexels: the row's texels). */
const RMS_SCALE = `
uniform sampler2D uR;
uniform int uROff;
uniform int uTexels;
uniform int uParts;
uniform float uEps;
float rmsScale(int m) {
  // A constant bound (RMS_PARTS), so the loop unrolls and the reads go out
  // together.
  vec4 s = vec4(0.0);
  for (int p = 0; p < 64; p++) {
    if (p >= uParts) break;
    s += texelFetch(uR, ivec2(uROff + p, m), 0);
  }
  return inversesqrt(dot(s, vec4(1.0)) / float(uTexels * 4) + uEps);
}
`

/** h2 out = x · rms(x) · w, from f32 x, its RMS_PARTIAL sums and f32 weights. */
export const NORM_TO_H2 = `${HEADER}${H2}${RMS_SCALE}
uniform sampler2D uX;
uniform sampler2D uNorm;
uniform int uRowIn;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int m = uRowIn < 0 ? y : uRowIn;
  float r = rmsScale(m);
  vec4 a = texelFetch(uX, ivec2(2 * x, m), 0) * r * texelFetch(uNorm, ivec2(2 * x, 0), 0);
  vec4 b = texelFetch(uX, ivec2(2 * x + 1, m), 0) * r * texelFetch(uNorm, ivec2(2 * x + 1, 0), 0);
  o = packH2(a, b);
}`

/**
 * y as added to the residual: through a sandwich norm (LFM3: y · rms(y) · w,
 * with y's RMS_PARTIAL sums in uR) or as it is (LFM2).
 */
function addedY(normed: boolean) {
  return normed
    ? `${RMS_SCALE}
uniform sampler2D uNorm;
float rowScale(int m) { return rmsScale(m); }
vec4 added(int x, int m, float rs) {
  return pf(uY, x, m) * rs * texelFetch(uNorm, ivec2(x, 0), 0);
}`
    : `
uniform int uTexels;
float rowScale(int m) { return 1.0; }
vec4 added(int x, int m, float rs) { return pf(uY, x, m); }`
}

/** f32 out = h + y, y through the sandwich norm when `normed`. */
export function addNormed(normed: boolean) {
  return `${HEADER}
uniform sampler2D uH;
uniform usampler2D uY;
${addedY(normed)}
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = texelFetch(uH, p, 0) + added(p.x, p.y, rowScale(p.y));
}`
}

/**
 * The partial sums of squares of ADD_NORMED's output x = h + y · rms(y) · w,
 * computed from its inputs, so they can be drawn in the same pass as the
 * add itself (into the residual's last RMS_PARTS columns, from uBase).
 */
export function rmsOfAdd(normed: boolean) {
  return `${HEADER}
uniform sampler2D uH;
uniform usampler2D uY;
${addedY(normed)}
uniform int uChunk;
uniform int uBase;
out vec4 o;
void main() {
  int p = int(gl_FragCoord.x) - uBase;
  int m = int(gl_FragCoord.y);
  int i0 = p * uChunk;
  int i1 = min(uTexels, i0 + uChunk);
  float rs = rowScale(m);
  vec4 s = vec4(0.0);
  for (int i = i0; i < i1; i++) {
    vec4 v = texelFetch(uH, ivec2(i, m), 0) + added(i, m, rs);
    s += v * v;
  }
  o = s;
}`
}

/**
 * A kernel that reads matmul outputs (`names`) as pair tensors, made to
 * read a split matmul's partial sums instead (pf's other overload).
 */
export function fromPartials(source: string, names: Array<string>) {
  let out = source
  for (const name of names)
    out = out.replace(`uniform usampler2D ${name};`, `uniform sampler2D ${name};`)
  return out
}

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
      y0 = pf(uBCx, c0, t) * pf(uBCx, 2 * dt + c0, t);
      y1 = pf(uBCx, c0 + 1, t) * pf(uBCx, 2 * dt + c0 + 1, t);
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
uniform usampler2D uBCx;
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
  vec4 C0 = pf(uBCx, dt + c0, m);
  vec4 C1 = pf(uBCx, dt + c0 + 1, m);
  o = packH2(C0 * a0, C1 * a1);
}`
}

/**
 * The convolution's state after a batch of uM tokens: the last width-1
 * values of B·x, oldest first, from this batch or the previous state.
 */
export function convState(width: number) {
  return `${HEADER}
uniform usampler2D uBCx;
uniform sampler2D uState;
uniform int uDt;
uniform int uM;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  int t = uM - ${width - 1} + i;
  if (t >= 0) {
    o = pf(uBCx, x, t) * pf(uBCx, 2 * uDt + x, t);
  } else {
    o = texelFetch(uState, ivec2(x, uM + i), 0);
  }
}`
}

/**
 * The convolution's state after each prefix of a few-token batch (for
 * speculative decoding, which may keep only some of a batch's tokens):
 * block p - 1 (width - 1 rows) holds convState's output for uM = p.
 */
export function convPrefixes(width: number) {
  const w = width - 1
  return `${HEADER}
uniform usampler2D uBCx;
uniform sampler2D uState;
uniform int uDt;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int p = y / ${w} + 1;
  int i = y - (p - 1) * ${w};
  int t = p - ${w} + i;
  if (t >= 0) {
    o = pf(uBCx, x, t) * pf(uBCx, 2 * uDt + x, t);
  } else {
    o = texelFetch(uState, ivec2(x, p + i), 0);
  }
}`
}

/** Copies rows [uOff, uOff + height) of an f32 tensor to rows [0, height). */
export const COPY_ROWS = `${HEADER}
uniform sampler2D uX;
uniform int uOff;
out vec4 o;
void main() {
  o = texelFetch(uX, ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y) + uOff), 0);
}`

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
  /** Dims per head (LFM3: 128, LFM2: 64). */
  headDim: number
}) {
  const texelsPerOut = o.out === "f32" ? 1 : 2
  // f32 texels per head; the RoPE table's sines start half a head in.
  const T = o.headDim / 4
  const prep = `
vec4 rotated(int base, int t, int pos, float inv) {
  vec4 v = pf(uQKVG, base + t, m)${o.norm ? " * inv * texelFetch(uNorm, ivec2(t, 0), 0)" : ""};
  ${
    o.rope
      ? `if (t >= 2 * uPairTexels) return v;
  bool first = t < uPairTexels;
  int tp = first ? t + uPairTexels : t - uPairTexels;
  int f = first ? t : tp;
  vec4 p = pf(uQKVG, base + tp, m)${o.norm ? " * inv * texelFetch(uNorm, ivec2(tp, 0), 0)" : ""};
  vec4 cs = texelFetch(uRope, ivec2(f, pos), 0);
  vec4 sn = texelFetch(uRope, ivec2(${T / 2} + f, pos), 0);
  return first ? v * cs - p * sn : v * cs + p * sn;`
      : "return v;"
  }
}`
  return `${HEADER}${H2}
uniform usampler2D uQKVG;
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
  int head = x / ${T / texelsPerOut};
  int t0 = (x % ${T / texelsPerOut}) * ${texelsPerOut};
  int base = uBase + head * ${T};
  float inv = 1.0;
  ${
    o.norm
      ? `vec4 s = vec4(0.0);
  for (int i = 0; i < ${T}; i++) {
    vec4 v = pf(uQKVG, base + i, m);
    s += v * v;
  }
  inv = inversesqrt(dot(s, vec4(1.0)) / ${o.headDim}.0 + uEps);`
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
export function scores(headDim: number) {
  const U = headDim / 8
  return `${HEADER}
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
uniform int uKt;
uniform int uPanelRows;
uniform int uTotalRows;
out vec4 o;
void main() {
  // Panels of uKt texels side by side, each holding uPanelRows rows: row r
  // (query m of head h, r = h · uRows + m − uM0) at (panel · uKt + x, y).
  int panel = int(gl_FragCoord.x) / uKt;
  int x = int(gl_FragCoord.x) - panel * uKt;
  int y = panel * uPanelRows + int(gl_FragCoord.y);
  int h = y / uRows;
  if (y >= uTotalRows) { o = vec4(-1e30); return; }
  int m = uM0 + (y - h * uRows);
  int kh = h / uGroup;
  int qpos = uPos0 + m;
  // The 4 keys' rows (masked ones read row 0 and are dropped below), and
  // each query texel read once for all 4.
  ivec4 j = uJ0 + 4 * x + ivec4(0, 1, 2, 3);
  bvec4 live = bvec4(true);
  ivec4 row = j;
  for (int c = 0; c < 4; c++) {
    live[c] = !(j[c] - uJ0 >= uKeys || j[c] > qpos || (uWindow > 0 && qpos - j[c] >= uWindow));
    row[c] = live[c] ? (uRing > 0 ? j[c] % uRing : j[c]) : 0;
  }
  if (!any(live)) { o = vec4(-1e30); return; }
  vec4 acc = vec4(0.0);
  for (int u = 0; u < ${U}; u++) {
    vec4 qa = texelFetch(uQ, ivec2(h * ${2 * U} + 2 * u, m), 0);
    vec4 qb = texelFetch(uQ, ivec2(h * ${2 * U} + 2 * u + 1, m), 0);
    uvec4 k0 = texelFetch(uK, ivec2(kh * ${U} + u, row.x), 0);
    uvec4 k1 = texelFetch(uK, ivec2(kh * ${U} + u, row.y), 0);
    uvec4 k2 = texelFetch(uK, ivec2(kh * ${U} + u, row.z), 0);
    uvec4 k3 = texelFetch(uK, ivec2(kh * ${U} + u, row.w), 0);
    acc += vec4(
      dot(qa, h2lo(k0)) + dot(qb, h2hi(k0)),
      dot(qa, h2lo(k1)) + dot(qb, h2hi(k1)),
      dot(qa, h2lo(k2)) + dot(qb, h2hi(k2)),
      dot(qa, h2lo(k3)) + dot(qb, h2hi(k3)));
  }
  o = mix(vec4(-1e30), acc, vec4(live));
}`
}

/**
 * Softmax over a query's scores, the weighted sum of values (uV: h2 cache),
 * times the head's sigmoid gate (from the q|k|v|gate projection at texel
 * uGateTexel). h2 output: 8 dims of one head per texel.
 */
export function attend(headDim: number, gate: boolean) {
  const U = headDim / 8
  return `${HEADER}
uniform sampler2D uS;
uniform usampler2D uV;
uniform usampler2D uQKVG;
uniform int uRows;
uniform int uM0;
uniform int uRing;
uniform int uJ0;
uniform int uKeys;
uniform int uGroup;
uniform int uGateTexel;
uniform int uKt;
uniform int uPanelRows;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  int h = x / ${U};
  int u = x - h * ${U};
  int kh = h / uGroup;
  int r = h * uRows + (m - uM0);
  int panel = r / uPanelRows;
  int row = r - panel * uPanelRows;
  int sx = panel * uKt;
  int texels = (uKeys + 3) / 4;
  float mx = -1e30;
  for (int i = 0; i < texels; i++) {
    vec4 s = texelFetch(uS, ivec2(sx + i, row), 0);
    mx = max(mx, max(max(s.x, s.y), max(s.z, s.w)));
  }
  float l = 0.0;
  vec4 lo = vec4(0.0);
  vec4 hi = vec4(0.0);
  for (int i = 0; i < texels; i++) {
    vec4 e = exp(texelFetch(uS, ivec2(sx + i, row), 0) - mx);
    l += dot(e, vec4(1.0));
    for (int c = 0; c < 4; c++) {
      int j = 4 * i + c;
      if (j >= uKeys) break;
      int vj = uJ0 + j;
      uvec4 v = texelFetch(uV, ivec2(kh * ${U} + u, uRing > 0 ? vj % uRing : vj), 0);
      lo += e[c] * h2lo(v);
      hi += e[c] * h2hi(v);
    }
  }
  ${
    gate
      ? `float gate = pf(uQKVG, uGateTexel + h / 4, m)[h & 3];
  float scale = 1.0 / (l * (1.0 + exp(-gate)));`
      : "float scale = 1.0 / l;"
  }
  o = packH2(lo * scale, hi * scale);
}`
}

/** Values into the h2 cache: 8 dims per texel, rows offset to the cache position. */
export function values(headDim: number) {
  const U = headDim / 8
  return `${HEADER}
uniform usampler2D uQKVG;
uniform int uBase;
uniform int uRowOffset;
out uvec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y) - uRowOffset;
  int head = x / ${U};
  int t = uBase + head * ${2 * U} + (x % ${U}) * 2;
  o = packH2(pf(uQKVG, t, m), pf(uQKVG, t + 1, m));
}`
}

/** SwiGLU: silu(gate) · up, from w1|w3's f32 output (uHalf texels each). h2 output. */
export const SWIGLU = `${HEADER}${H2}
uniform usampler2D uGU;
uniform int uHalf;
out uvec4 o;
vec4 silu(vec4 g) { return g / (1.0 + exp(-g)); }
void main() {
  int x = int(gl_FragCoord.x);
  int m = int(gl_FragCoord.y);
  vec4 a = silu(pf(uGU, 2 * x, m)) * pf(uGU, uHalf + 2 * x, m);
  vec4 b = silu(pf(uGU, 2 * x + 1, m)) * pf(uGU, uHalf + 2 * x + 1, m);
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

/** Pixel r: the index of the chunk ranked r (CHUNK_RANK inverted). */
export const CHUNK_BY_RANK = `${HEADER}
uniform sampler2D uRank;
uniform int uChunks;
out vec4 o;
void main() {
  int r = int(gl_FragCoord.x);
  int c = 0;
  for (int i = 0; i < uChunks; i++) {
    if (int(texelFetch(uRank, ivec2(i, 0), 0).x) == r) c = i;
  }
  o = vec4(float(c), 0.0, 0.0, 0.0);
}`

/**
 * Row r: the logits of the chunk ranked r (from CHUNK_BY_RANK), 64
 * texels, then its index in texel 64.
 */
export const CHUNK_GATHER = `${HEADER}
uniform sampler2D uLogits;
uniform usampler2D uMask;
uniform sampler2D uByRank;
uniform int uOutW;
uniform float uPenalty;
${PENALIZED}
out vec4 o;
void main() {
  int t = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  int c = int(texelFetch(uByRank, ivec2(r, 0), 0).x);
  o = t < 64 ? chunkTexel(c, t) : vec4(float(c), 0.0, 0.0, 0.0);
}`

/**
 * The likeliest token (penalized), as a token id into the ids texture:
 * the first largest logit of the top-ranked chunk (CHUNK_GATHER's row 0),
 * as the CPU's pick from the same candidates would be.
 */
export const CANDIDATE_ARGMAX = `${HEADER}
uniform sampler2D uCand;
out uvec4 o;
void main() {
  float best = -3e38;
  int at = 0;
  for (int t = 0; t < 64; t++) {
    vec4 v = texelFetch(uCand, ivec2(t, 0), 0);
    for (int c = 0; c < 4; c++) {
      if (v[c] > best) { best = v[c]; at = 4 * t + c; }
    }
  }
  int chunk = int(texelFetch(uCand, ivec2(64, 0), 0).x + 0.5);
  o = uvec4(uint(chunk * 256 + at), 0u, 0u, 0u);
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
