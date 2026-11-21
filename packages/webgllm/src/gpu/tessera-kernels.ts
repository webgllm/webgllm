/**
 * Tessera's shaders (see tessera.ts): the denoiser and CLIP's text tower,
 * written once in GLSL for both backends. Activations are RGBA32F, texel
 * (x, r) holding features 4x..4x+3 of row r. Rows are token rows, batch
 * row b's `seq` tokens at rows b·seq.., [registers | text | image].
 *
 * Weights stay int8: a projection's texture (RGBA32UI) holds 16 weights a
 * texel, texel (t, n) row n's inputs 16t..16t+15 (byte j of word w is
 * input 16t + 4w + j), each signed. Its scales are per output row
 * (applied to the sum) or per input (applied to the input as it's read),
 * whichever the file quantized.
 */
import { HEADER } from "./kernels.js"

/** int8 bytes of a word, as signed floats. */
const SIGNED = `
vec4 signed8(uint word) {
  vec4 b = vec4((uvec4(word) >> uvec4(0u, 8u, 16u, 24u)) & 255u);
  return b - 256.0 * step(128.0, b);
}`

/** Activations: SiLU, and CLIP's quick GELU (x · σ(1.702 x)). */
export type TesseraAct = "none" | "silu" | "quickGelu"

const ACTS = `
vec4 silu(vec4 x) { return x / (1.0 + exp(-x)); }
vec4 quickGelu(vec4 x) { return x / (1.0 + exp(-1.702 * x)); }`

/**
 * out = act(f(A) · Wᵀ · s + bias), then (+ uR, or uR + gate · …): one
 * texel (outputs 4g..4g+3) of one row per fragment, drawn at column
 * uOut0 + g.
 *
 * - `axis` "out": uS (N / 4 texels) scales the sums; "in": uS (K / 4
 *   texels) scales the inputs.
 * - `siluIn`: f = SiLU (adaLN's SiLU before its projection), else identity.
 * - `gate`: the residual is uR + gate · value, gate being chunk uGate of
 *   the modulation (uMod row b = row / uSeq, plus uModBias's chunk).
 */
export function tesseraMatmul(o: { axis: "out" | "in"; bias: boolean; act: TesseraAct; residual: boolean; gate: boolean; siluIn: boolean }) {
  return `${HEADER}
uniform sampler2D uA;
uniform usampler2D uW;
uniform sampler2D uS;
${o.bias ? "uniform sampler2D uBias;" : ""}
${o.residual || o.gate ? "uniform sampler2D uR;" : ""}
${o.gate ? "uniform sampler2D uMod;\nuniform sampler2D uModBias;\nuniform int uSeq;\nuniform int uGate;\nuniform int uDimTexels;" : ""}
uniform int uSteps;
uniform int uOut0;
out vec4 o;
${SIGNED}
${ACTS}
void main() {
  int g = int(gl_FragCoord.x) - uOut0;
  int m = int(gl_FragCoord.y);
  vec4 acc = vec4(0.0);
  for (int t = 0; t < uSteps; t++) {
    vec4 a[4];
    for (int i = 0; i < 4; i++) {
      vec4 v = texelFetch(uA, ivec2(4 * t + i, m), 0);
      ${o.siluIn ? "v = silu(v);" : ""}
      ${o.axis === "in" ? "v *= texelFetch(uS, ivec2(4 * t + i, 0), 0);" : ""}
      a[i] = v;
    }
    for (int r = 0; r < 4; r++) {
      uvec4 w = texelFetch(uW, ivec2(t, 4 * g + r), 0);
      acc[r] += dot(signed8(w.x), a[0]) + dot(signed8(w.y), a[1]) + dot(signed8(w.z), a[2]) + dot(signed8(w.w), a[3]);
    }
  }
  ${o.axis === "out" ? "acc *= texelFetch(uS, ivec2(g, 0), 0);" : ""}
  ${o.bias ? "acc += texelFetch(uBias, ivec2(g, 0), 0);" : ""}
  ${o.act === "silu" ? "acc = silu(acc);" : o.act === "quickGelu" ? "acc = quickGelu(acc);" : ""}
  ${
    o.gate
      ? `int b = m / uSeq;
  vec4 gate = texelFetch(uMod, ivec2(uGate * uDimTexels + g, b), 0) + texelFetch(uModBias, ivec2(uGate * uDimTexels + g, 0), 0);
  acc = texelFetch(uR, ivec2(g, m), 0) + gate * acc;`
      : o.residual
        ? "acc += texelFetch(uR, ivec2(g, m), 0);"
        : ""
  }
  o = acc;
}`
}

/**
 * RMS norm of each row over uTexels texels, then × uW (with `weight`), or
 * adaLN's modulation (with `modulate`): × (1 + scale) + shift, chunks
 * uScale and uShift of the modulation (uMod row b = row / uSeq, uModBias
 * added when `modBias`).
 */
export function tesseraRmsNorm(o: { weight: boolean; modulate: boolean; modBias: boolean }) {
  return `${HEADER}
uniform sampler2D uX;
${o.weight ? "uniform sampler2D uW;" : ""}
${o.modulate ? "uniform sampler2D uMod;\nuniform int uSeq;\nuniform int uShift;\nuniform int uScale;" : ""}
${o.modBias ? "uniform sampler2D uModBias;" : ""}
uniform int uTexels;
uniform float uEps;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  float s = 0.0;
  for (int t = 0; t < uTexels; t++) {
    vec4 v = texelFetch(uX, ivec2(t, r), 0);
    s += dot(v, v);
  }
  vec4 y = texelFetch(uX, ivec2(x, r), 0) * inversesqrt(s / float(4 * uTexels) + uEps);
  ${o.weight ? "y *= texelFetch(uW, ivec2(x, 0), 0);" : ""}
  ${
    o.modulate
      ? `int b = r / uSeq;
  vec4 scale = texelFetch(uMod, ivec2(uScale * uTexels + x, b), 0);
  vec4 shift = texelFetch(uMod, ivec2(uShift * uTexels + x, b), 0);
  ${o.modBias ? "scale += texelFetch(uModBias, ivec2(uScale * uTexels + x, 0), 0);\n  shift += texelFetch(uModBias, ivec2(uShift * uTexels + x, 0), 0);" : ""}
  y = y * (1.0 + scale) + shift;`
      : ""
  }
  o = y;
}`
}

/** LayerNorm of each row over uTexels texels: (x − mean) / σ · uW + uB. */
export const TESSERA_LAYER_NORM = `${HEADER}
uniform sampler2D uX;
uniform sampler2D uW;
uniform sampler2D uB;
uniform int uTexels;
uniform float uEps;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  vec4 s = vec4(0.0);
  for (int t = 0; t < uTexels; t++) s += texelFetch(uX, ivec2(t, r), 0);
  float mean = dot(s, vec4(1.0)) / float(4 * uTexels);
  float v = 0.0;
  for (int t = 0; t < uTexels; t++) {
    vec4 d = texelFetch(uX, ivec2(t, r), 0) - mean;
    v += dot(d, d);
  }
  vec4 y = (texelFetch(uX, ivec2(x, r), 0) - mean) * inversesqrt(v / float(4 * uTexels) + uEps);
  o = y * texelFetch(uW, ivec2(x, 0), 0) + texelFetch(uB, ivec2(x, 0), 0);
}`

/**
 * q | k of a qkv projection (uQKV, rows × 3 · dim), each head RMS-normed
 * (uQN, uKN: headDim weights) and rotated by 2D axial RoPE (uRope row
 * pos = row % uSeq: 8 texels of cos, then 8 of sin, for the head's 32
 * pairs (d, d + 32)). Drawn as rows × 2 · dim: q's texels, then k's.
 */
export const TESSERA_QK_NORM_ROPE = `${HEADER}
uniform sampler2D uQKV;
uniform sampler2D uQN;
uniform sampler2D uKN;
uniform sampler2D uRope;
uniform int uSeq;
uniform int uDimTexels;
uniform float uEps;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  bool isK = x >= uDimTexels;
  int c = isK ? x - uDimTexels : x;
  int head = c / 16;
  int t = c - head * 16;
  int base = (isK ? uDimTexels : 0) + head * 16;
  float s = 0.0;
  for (int i = 0; i < 16; i++) {
    vec4 v = texelFetch(uQKV, ivec2(base + i, r), 0);
    s += dot(v, v);
  }
  float inv = inversesqrt(s / 64.0 + uEps);
  int pair = t < 8 ? t : t - 8;
  vec4 self = texelFetch(uQKV, ivec2(base + t, r), 0) * inv * (isK ? texelFetch(uKN, ivec2(t, 0), 0) : texelFetch(uQN, ivec2(t, 0), 0));
  int other = t < 8 ? t + 8 : t - 8;
  vec4 mate = texelFetch(uQKV, ivec2(base + other, r), 0) * inv * (isK ? texelFetch(uKN, ivec2(other, 0), 0) : texelFetch(uQN, ivec2(other, 0), 0));
  int pos = r - (r / uSeq) * uSeq;
  vec4 cs = texelFetch(uRope, ivec2(pair, pos), 0);
  vec4 sn = texelFetch(uRope, ivec2(8 + pair, pos), 0);
  // First half: x1·cos − x2·sin; second half: x1·sin + x2·cos (x1 = mate there).
  o = t < 8 ? self * cs - mate * sn : mate * sn + self * cs;
}`

/**
 * Attention scores for query rows uQ0 + y % uRows, head y / uRows: texel
 * x holds keys 4x..4x+3 of the query's batch row (uSeq tokens), scaled by
 * 1 / √64. Keys past the row, and with `causal` past the query, are −∞.
 * q at texel uQBase + head · 16 of uQ, k at uKBase + head · 16 of uK.
 */
export function tesseraScores(o: { causal: boolean }) {
  return `${HEADER}
uniform sampler2D uQ;
uniform sampler2D uK;
uniform int uRows;
uniform int uQ0;
uniform int uSeq;
uniform int uQBase;
uniform int uKBase;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int head = y / uRows;
  int q = uQ0 + y - head * uRows;
  int b = q / uSeq;
  int qpos = q - b * uSeq;
  vec4 qv[16];
  for (int i = 0; i < 16; i++) qv[i] = texelFetch(uQ, ivec2(uQBase + head * 16 + i, q), 0);
  vec4 r = vec4(-1e30);
  for (int c = 0; c < 4; c++) {
    int key = 4 * x + c;
    if (key < uSeq${o.causal ? " && key <= qpos" : ""}) {
      float s = 0.0;
      for (int i = 0; i < 16; i++) s += dot(qv[i], texelFetch(uK, ivec2(uKBase + head * 16 + i, b * uSeq + key), 0));
      r[c] = 0.125 * s;
    }
  }
  o = r;
}`
}

/** Each score row's max and 1 / Σ exp(score − max), over uWidth texels. */
export const TESSERA_SOFTMAX_STATS = `${HEADER}
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
 * (head x / 16) is Σ softmax(score) · v over the batch row's keys, v at
 * texel uVBase + x of uV.
 */
export const TESSERA_VALUES = `${HEADER}
uniform sampler2D uS;
uniform sampler2D uStats;
uniform sampler2D uV;
uniform int uWidth;
uniform int uRows;
uniform int uQ0;
uniform int uSeq;
uniform int uVBase;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int q = int(gl_FragCoord.y);
  int head = x / 16;
  int y = head * uRows + q - uQ0;
  int b = q / uSeq;
  vec4 st = texelFetch(uStats, ivec2(0, y), 0);
  vec4 acc = vec4(0.0);
  for (int t = 0; t < uWidth; t++) {
    vec4 p = exp(texelFetch(uS, ivec2(t, y), 0) - st.x);
    for (int c = 0; c < 4; c++) {
      int key = 4 * t + c;
      if (key < uSeq) acc += p[c] * texelFetch(uV, ivec2(uVBase + x, b * uSeq + key), 0);
    }
  }
  o = acc * st.y;
}`

/**
 * Conv-SwiGLU's middle: a · SiLU(gate) of the w12 projection (uX: a's
 * uHidden texels, then the gate's), where an image row's gate first gets
 * its depthwise 3 × 3 convolution over the uGrid × uGrid image (uK: tap
 * j's texels at row j; uB its bias) added.
 */
export const TESSERA_CONV_SWIGLU = `${HEADER}
uniform sampler2D uX;
uniform sampler2D uK;
uniform sampler2D uB;
uniform int uSeq;
uniform int uPrefix;
uniform int uGrid;
uniform int uHidden;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  int pos = r - (r / uSeq) * uSeq;
  vec4 gate = texelFetch(uX, ivec2(uHidden + x, r), 0);
  if (pos >= uPrefix) {
    int p = pos - uPrefix;
    int i = p / uGrid;
    int j = p - i * uGrid;
    vec4 conv = texelFetch(uB, ivec2(x, 0), 0);
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int yy = i + dy;
        int xx = j + dx;
        if (yy >= 0 && yy < uGrid && xx >= 0 && xx < uGrid)
          conv += texelFetch(uK, ivec2(x, (dy + 1) * 3 + dx + 1), 0) * texelFetch(uX, ivec2(uHidden + x, r + dy * uGrid + dx), 0);
      }
    }
    gate += conv;
  }
  o = texelFetch(uX, ivec2(x, r), 0) * (gate / (1.0 + exp(-gate)));
}`

/**
 * A step's input rows, uSeq a batch row: registers (uReg), then the text
 * rows (uText for batch rows below uCount, uNullText after), then the
 * image: token tok's embedding (uEmbed row tok) plus its position's
 * (uPos), plus the self-conditioning projection (uSc) where it's MASK.
 * Tokens: uTok texel (p / 4, b), component p % 4.
 */
export const TESSERA_ASSEMBLE = `${HEADER}
uniform sampler2D uReg;
uniform sampler2D uText;
uniform sampler2D uNullText;
uniform sampler2D uEmbed;
uniform sampler2D uPos;
uniform sampler2D uSc;
uniform sampler2D uTok;
uniform int uSeq;
uniform int uRegisters;
uniform int uTextLen;
uniform int uCount;
uniform int uMask;
uniform int uPixels;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  int b = r / uSeq;
  int pos = r - b * uSeq;
  if (pos < uRegisters) {
    o = texelFetch(uReg, ivec2(x, pos), 0);
  } else if (pos < uRegisters + uTextLen) {
    int t = pos - uRegisters;
    o = b < uCount ? texelFetch(uText, ivec2(x, t), 0) : texelFetch(uNullText, ivec2(x, t), 0);
  } else {
    int p = pos - uRegisters - uTextLen;
    vec4 toks = texelFetch(uTok, ivec2(p / 4, b), 0);
    int tok = int(toks[p - (p / 4) * 4] + 0.5);
    vec4 v = texelFetch(uEmbed, ivec2(x, tok), 0) + texelFetch(uPos, ivec2(x, p), 0);
    if (tok == uMask) v += texelFetch(uSc, ivec2(x, b * uPixels + p), 0);
    o = v;
  }
}`

/** Rows n of every uSeq-row batch row from row uFrom on, packed: output row r is input row (r / n)·uSeq + uFrom + r % n. */
export const TESSERA_TAKE_ROWS = `${HEADER}
uniform sampler2D uX;
uniform int uSeq;
uniform int uFrom;
uniform int uN;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  int b = r / uN;
  o = texelFetch(uX, ivec2(x, b * uSeq + uFrom + r - b * uN), 0);
}`

/**
 * The hourglass's 2 × 2 merge: output row (b, token (gi, gj)) holds the
 * 4 pixels (2gi + s / 2, 2gj + s % 2) side by side, s's texels at s ·
 * uDimTexels..; pixels from uX's image rows (batch row b of uSeq, image
 * from uFrom, uGrid wide).
 */
export const TESSERA_MERGE = `${HEADER}
uniform sampler2D uX;
uniform int uSeq;
uniform int uFrom;
uniform int uGrid;
uniform int uDimTexels;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  int hw = uGrid / 2;
  int b = r / (hw * hw);
  int tok = r - b * hw * hw;
  int gi = tok / hw;
  int gj = tok - gi * hw;
  int s = x / uDimTexels;
  int c = x - s * uDimTexels;
  int i = 2 * gi + s / 2;
  int j = 2 * gj + (s - (s / 2) * 2);
  o = texelFetch(uX, ivec2(c, b * uSeq + uFrom + i * uGrid + j), 0);
}`

/**
 * The core's rows: batch row b's uPrefix rows from uPre (rows of uPreSeq a
 * batch row), then its uN rows of uImg (rows of uN a batch row).
 */
export const TESSERA_JOIN = `${HEADER}
uniform sampler2D uPre;
uniform sampler2D uImg;
uniform int uSeq;
uniform int uPreSeq;
uniform int uPrefix;
uniform int uN;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  int b = r / uSeq;
  int pos = r - b * uSeq;
  o = pos < uPrefix ? texelFetch(uPre, ivec2(x, b * uPreSeq + pos), 0) : texelFetch(uImg, ivec2(x, b * uN + pos - uPrefix), 0);
}`

/**
 * The hourglass's split back to pixels, plus the skip: a prefix row is the
 * core's (uCore, uCoreSeq rows a batch row); image pixel (i, j) is the
 * skip's (uSkip, the same rows as the output) plus slot (i % 2) · 2 +
 * j % 2 of its token's up projection (uUp, (uGrid / 2)² rows a batch row).
 */
export const TESSERA_UNMERGE = `${HEADER}
uniform sampler2D uCore;
uniform sampler2D uSkip;
uniform sampler2D uUp;
uniform int uSeq;
uniform int uCoreSeq;
uniform int uPrefix;
uniform int uGrid;
uniform int uDimTexels;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  int b = r / uSeq;
  int pos = r - b * uSeq;
  if (pos < uPrefix) {
    o = texelFetch(uCore, ivec2(x, b * uCoreSeq + pos), 0);
  } else {
    int p = pos - uPrefix;
    int i = p / uGrid;
    int j = p - i * uGrid;
    int hw = uGrid / 2;
    int tok = (i / 2) * hw + j / 2;
    int s = (i - (i / 2) * 2) * 2 + (j - (j / 2) * 2);
    o = texelFetch(uSkip, ivec2(x, r), 0) + texelFetch(uUp, ivec2(s * uDimTexels + x, b * hw * hw + tok), 0);
  }
}`

/** uA + uB, texel by texel; uB's row r % uRowsB (a row broadcast over a batch). */
export const TESSERA_ADD = `${HEADER}
uniform sampler2D uA;
uniform sampler2D uB;
uniform int uRowsB;
out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  o = texelFetch(uA, ivec2(x, r), 0) + texelFetch(uB, ivec2(x, r - (r / uRowsB) * uRowsB), 0);
}`
