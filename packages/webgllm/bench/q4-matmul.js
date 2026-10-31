/**
 * A Q4_0 matrix multiply for WebGL 2 fragment shaders: out = A · Wᵀ, with
 * W (N rows × K) in llama.cpp's Q4_0 blocks and A (M tokens × K) in floats.
 *
 * Layouts
 * - Weights: an RGBA32UI texture. One texel is the 16 quant bytes of one
 *   Q4_0 block (32 weights), as they are in the GGUF file. Rows are grouped
 *   in fours, interleaved: texel (kb * 4 + r, n4) is block kb of row
 *   4 * n4 + r. Width K / 32 * 4, height N / 4.
 * - Scales: an RGBA16F texture, texel (kb, n4) holding block kb's scale for
 *   rows 4 * n4 .. 4 * n4 + 3. Width K / 32, height N / 4.
 * - Activations: an RGBA32F (or RGBA16F) 2D array texture. A texel holds 4
 *   consecutive features of one token. With T tokens per fragment and S
 *   feature texels per fragment, token m's feature texel i is at
 *   (i / S, m / T, (m % T) * S + i % S): width K / 4 / S, height M / T,
 *   T * S layers. Outputs use the same layout, so one multiply's output
 *   feeds the next.
 *
 * A fragment computes 4 * S output features for T tokens, written to
 * T * S color attachments (one per layer).
 *
 * Q4_0: weight j of a block is d * ((qs[j] & 15) - 8) and weight j + 16 is
 * d * ((qs[j] >> 4) - 8). So word w of the texel holds weights 4w..4w+3 in
 * its bytes' low nibbles and 16+4w..16+4w+3 in their high nibbles, which
 * line up with activation texels w and w + 4 of the block. The -8 is
 * applied once per block: d * (Σ q·a - 8 Σ a).
 */

/**
 * @typedef {{
 *   S: 1 | 2,
 *   T: 1 | 2 | 4 | 8,
 *   unpack: "shift" | "u16",
 *   dep?: boolean,
 * }} MatmulVariant
 */

export const VERTEX_SHADER = `#version 300 es
void main() {
  // A triangle that covers the viewport.
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`

const range = (n) => Array.from({ length: n }, (_, i) => i)

/**
 * The fragment shader for a variant.
 *
 * - `act`: activations as RGBA32F ("f32"), RGBA16F ("f16"), or RGBA32UI
 *   texels of 8 packed halves ("h2", S = 1 only: a block's 32 activations
 *   in 4 texels instead of 8).
 * - `unpack`: nibbles with integer shifts from an RGBA32UI texel ("shift"),
 *   or with float ops from 4 RGBA8 texels of the same bytes ("float").
 * - `prec`: precision of the per-block math; accumulators stay highp.
 * @param {MatmulVariant & { act?: "f32" | "f16" | "h2", prec?: "highp" | "mediump" }} v
 */
export function matmulShader(v) {
  const { S, T } = v
  const R = 4 * S
  const act = v.act ?? "f32"
  const P = v.prec === "mediump" ? "mediump " : ""
  if (act === "h2" && S !== 1) throw new Error("h2 activations need S = 1")
  const outs = range(T * S)
    .map((o) => `layout(location = ${o}) out vec4 o${o};`)
    .join("\n")

  const lines = []
  const decl = (s) => lines.push(`  ${s}`)
  for (const r of range(R))
    for (const t of range(T)) decl(`float acc${r}_${t} = 0.0;`)

  lines.push(`  for (int kb = 0; kb < uKb; kb++) {`)
  for (const s of range(S)) {
    decl(`  vec4 d${s} = texelFetch(uS, ivec2(kb, n4 + ${s}), 0);`)
  }
  // Weights: per row, 4 words (one RGBA32UI texel, or 4 RGBA8 texels).
  for (const s of range(S)) {
    for (const r of range(4)) {
      const row = s * 4 + r
      if (v.unpack === "float") {
        for (const w of range(4)) {
          decl(
            `  ${P}vec4 B${row}_${w} = texelFetch(uW8, ivec2(kb * 16 + ${r * 4 + w}, n4 + ${s}), 0) * 255.0;`
          )
        }
      } else {
        decl(
          `  uvec4 w${row} = texelFetch(uW, ivec2(kb * 4 + ${r}, n4 + ${s}), 0);`
        )
      }
    }
  }
  for (const r of range(R))
    for (const t of range(T)) decl(`  ${P}float b${r}_${t} = 0.0;`)
  for (const t of range(T)) decl(`  ${P}float sa${t} = 0.0;`)
  // Activation texels of the block, per token.
  if (act === "h2") {
    for (const t of range(T)) {
      for (const j of range(4)) {
        decl(
          `  uvec4 u${t}_${j} = texelFetch(uAh, ivec3(kb * 4 + ${j}, ty, ${t}), 0);`
        )
      }
    }
  }
  for (const w of range(4)) {
    for (const t of range(T)) {
      for (const [name, j] of [
        ["a", w],
        ["c", w + 4]
      ]) {
        if (act === "h2") {
          // Features 4j..4j+3: texel j >> 1, halves (x, y) or (z, w).
          const u = `u${t}_${j >> 1}`
          const [c0, c1] = j & 1 ? ["z", "w"] : ["x", "y"]
          decl(
            `  ${P}vec4 ${name}${t}_${w} = vec4(unpackHalf2x16(${u}.${c0}), unpackHalf2x16(${u}.${c1}));`
          )
        } else {
          const i = `kb * 8 + ${j}`
          decl(
            `  ${P}vec4 ${name}${t}_${w} = texelFetch(uA, ivec3((${i}) / ${S}, ty, ${t} * ${S} + (${i}) % ${S}), 0);`
          )
        }
      }
      decl(`  sa${t} += dot(a${t}_${w} + c${t}_${w}, vec4(1.0));`)
    }
    for (const r of range(R)) {
      if (v.unpack === "float") {
        decl(
          `  { ${P}vec4 h = floor(B${r}_${w} * 0.0625 + 0.015625); ${P}vec4 l = B${r}_${w} - 16.0 * h;`
        )
      } else {
        const comp = "xyzw"[w]
        decl(
          `  { ${P}vec4 l = vec4((uvec4(w${r}.${comp}) >> uvec4(0u, 8u, 16u, 24u)) & 15u);`
        )
        decl(
          `    ${P}vec4 h = vec4((uvec4(w${r}.${comp}) >> uvec4(4u, 12u, 20u, 28u)) & 15u);`
        )
      }
      for (const t of range(T)) {
        decl(`    b${r}_${t} += dot(l, a${t}_${w}) + dot(h, c${t}_${w});`)
      }
      decl(`  }`)
    }
  }
  for (const r of range(R)) {
    const s = Math.floor(r / 4)
    const comp = "xyzw"[r % 4]
    for (const t of range(T)) {
      decl(
        `  acc${r}_${t} += d${s}.${comp} * (float(b${r}_${t}) - 8.0 * float(sa${t}));`
      )
    }
  }
  lines.push(`  }`)
  for (const t of range(T)) {
    for (const s of range(S)) {
      const rs = range(4)
        .map((c) => `acc${s * 4 + c}_${t}`)
        .join(", ")
      lines.push(`  o${t * S + s} = vec4(${rs});`)
    }
  }

  // For benchmarks: a read of the previous draw's output, times a uniform
  // zero, so the driver can't merge or skip draws (each depends on the last,
  // as a model's layers do).
  const dep = v.dep ? `  o0.x += texelFetch(uPrev, ivec3(0), 0).x * uZero;` : ""
  return `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp usampler2DArray;
precision highp sampler2D;
precision highp sampler2DArray;
uniform usampler2D uW;
uniform sampler2D uW8;
uniform sampler2D uS;
uniform sampler2DArray uA;
uniform usampler2DArray uAh;
uniform int uKb;
${v.dep ? "uniform sampler2DArray uPrev;\nuniform float uZero;" : ""}
${outs}
void main() {
  int x = int(gl_FragCoord.x);
  int ty = int(gl_FragCoord.y);
  int n4 = x * ${S};
${lines.join("\n")}
${dep}
}`
}

/** Float to IEEE half, as a uint16 (round to nearest). */
export function toHalf(value) {
  const f = new Float32Array(1)
  const u = new Uint32Array(f.buffer)
  f[0] = value
  const x = u[0]
  const sign = (x >>> 16) & 0x8000
  const exp = ((x >>> 23) & 0xff) - 127 + 15
  let mant = x & 0x7fffff
  if (exp <= 0) {
    if (exp < -10) return sign
    mant = (mant | 0x800000) >>> (1 - exp)
    return sign | ((mant + 0x1000) >>> 13)
  }
  if (exp >= 31) return sign | 0x7c00
  const h = sign | (exp << 10) | ((mant + 0x1000) >>> 13)
  return h
}

/** IEEE half (uint16) to float. */
export function fromHalf(h) {
  const sign = h & 0x8000 ? -1 : 1
  const exp = (h >>> 10) & 0x1f
  const mant = h & 0x3ff
  if (exp === 0) return sign * mant * 2 ** -24
  if (exp === 31) return mant ? NaN : sign * Infinity
  return sign * (1 + mant / 1024) * 2 ** (exp - 15)
}

/**
 * Repacks Q4_0 blocks (18 bytes each: fp16 scale, 16 quant bytes) of an
 * N × K matrix into the weight and scale textures' data.
 * @param {Uint8Array} blocks
 * @param {number} N
 * @param {number} K
 */
export function packQ4_0(blocks, N, K) {
  const kb = K / 32
  const quants = new Uint32Array(N * kb * 4)
  const scales = new Uint16Array(N * kb)
  const view = new DataView(blocks.buffer, blocks.byteOffset, blocks.byteLength)
  for (let n = 0; n < N; n++) {
    const n4 = n >> 2
    const r = n & 3
    for (let b = 0; b < kb; b++) {
      const src = (n * kb + b) * 18
      // Scale texel (b, n4), component r.
      scales[(n4 * kb + b) * 4 + r] = view.getUint16(src, true)
      // Weight texel (b * 4 + r, n4): 4 words.
      const dst = (n4 * kb * 4 + b * 4 + r) * 4
      for (let w = 0; w < 4; w++) {
        quants[dst + w] = view.getUint32(src + 2 + w * 4, true)
      }
    }
  }
  return { quants, scales }
}
