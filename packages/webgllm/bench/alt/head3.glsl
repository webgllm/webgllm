#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
precision highp isampler2D;
/* Halves of an h2 texel: elements 0..3 and 4..7. */
vec4 h2lo(uvec4 u) { return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); }
vec4 h2hi(uvec4 u) { return vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
/* Clamped to half's range: overflow saturates alike on every GPU (the spec
   leaves packHalf2x16 of a too-large value to the implementation). */
uvec4 packH2(vec4 a, vec4 b) {
  a = clamp(a, -65504.0, 65504.0);
  b = clamp(b, -65504.0, 65504.0);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}
/* Component i of v, as a select: some compilers get v[i] wrong when i
   varies by fragment. */
float comp(vec4 v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
uint comp(uvec4 v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
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
/*
 * The same from a 4-token matmul's two outputs (see matmulQuad): tokens
 * 4q, 4q + 1 in a, 4q + 2, 4q + 3 in b, row q (plus its K-slices).
 */
vec4 pfq(highp usampler2D a, highp usampler2D b, int x, int m) {
  vec4 acc = vec4(0.0);
  int row = m >> 2;
  bool second = ((m >> 1) & 1) == 1;
  for (int k = 0; k < max(uPairSplit, 1); k++) {
    ivec2 p = ivec2(x, row + k * uSliceStride);
    uvec4 u = second ? texelFetch(b, p, 0) : texelFetch(a, p, 0);
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


const uvec4 BYTE_SHIFT = uvec4(0u, 8u, 16u, 24u);
// 4 6-bit values in the bytes of q, minus 32: each byte as the mantissa
// of a half whose exponent makes it 1024 + byte, two to an unpackHalf2x16.
vec4 q6v(uint q) {
  vec2 e = unpackHalf2x16((q & 0x00FF00FFu) | 0x64006400u);
  vec2 f = unpackHalf2x16(((q >> 8u) & 0x00FF00FFu) | 0x64006400u);
  return vec4(e.x, f.x, e.y, f.y) - 1056.0;
}
// The 6-bit values of groups 0..3 (see Q6_K_UNPACK) for 4 word positions,
// one a byte, from ql word a, ql word b (32 bytes on) and qh word h.
uint q6g0(uint a, uint h) { return (a & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4u); }
uint q6g1(uint b, uint h) { return (b & 0x0F0F0F0Fu) | ((h & 0x0C0C0C0Cu) << 2u); }
uint q6g2(uint a, uint h) { return ((a >> 4u) & 0x0F0F0F0Fu) | (h & 0x30303030u); }
uint q6g3(uint b, uint h) { return ((b >> 4u) & 0x0F0F0F0Fu) | ((h >> 2u) & 0x30303030u); }
// An int8 scale: byte b of u.
float sc8w(uint u, uint b) { return float(int(u << (24u - 8u * b)) >> 24); }
uniform usampler2D uA;
uniform usampler2D uW;
uniform sampler2D uS;
uniform int uKb;
uniform int uH;
uniform int uRowTexels;
uniform int uG;
uniform int uOutW;
uniform int uLogitRows;
uniform int uKbSlice;
uniform int uRows;

out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  int m = y / uLogitRows;
  y -= m * uLogitRows;
  int slice = m / uRows;
  m -= slice * uRows;
  int sb0 = slice * uKbSlice;
  int sb1 = min(uKb, sb0 + uKbSlice);
  int g = y * uOutW + x;
  if (g >= uG) { o = vec4(0.0); return; }
  int col = g / uH;
  int gr = g - col * uH;
  int wx = col * uRowTexels;
  int sx = col * uKb;
  vec4 acc = vec4(0.0);
  for (int sb = sb0; sb < sb1; sb++) {
    vec4 d = texelFetch(uS, ivec2(sx + sb, gr), 0);
    uvec4 S0 = texelFetch(uW, ivec2(wx + (sb * 4 + 0) * 13 + 12, gr), 0);
    uvec4 S1 = texelFetch(uW, ivec2(wx + (sb * 4 + 1) * 13 + 12, gr), 0);
    uvec4 S2 = texelFetch(uW, ivec2(wx + (sb * 4 + 2) * 13 + 12, gr), 0);
    uvec4 S3 = texelFetch(uW, ivec2(wx + (sb * 4 + 3) * 13 + 12, gr), 0);
    vec4 P = vec4(0.0);
  for (int q4 = 0; q4 < 4; q4++) {
    int n = (q4 >> 1) & 1;
    int hw = q4 & 1;
    int a0 = sb * 32 + 16 * n + 2 * hw;
    uvec4 u00 = texelFetch(uA, ivec2(a0 + 0, m), 0);
    vec4 A00 = h2lo(u00); vec4 A10 = h2hi(u00);
    uvec4 u01 = texelFetch(uA, ivec2(a0 + 4, m), 0);
    vec4 A01 = h2lo(u01); vec4 A11 = h2hi(u01);
    uvec4 u02 = texelFetch(uA, ivec2(a0 + 8, m), 0);
    vec4 A02 = h2lo(u02); vec4 A12 = h2hi(u02);
    uvec4 u03 = texelFetch(uA, ivec2(a0 + 12, m), 0);
    vec4 A03 = h2lo(u03); vec4 A13 = h2hi(u03);
    uvec4 u10 = texelFetch(uA, ivec2(a0 + 1, m), 0);
    vec4 A20 = h2lo(u10); vec4 A30 = h2hi(u10);
    uvec4 u11 = texelFetch(uA, ivec2(a0 + 5, m), 0);
    vec4 A21 = h2lo(u11); vec4 A31 = h2hi(u11);
    uvec4 u12 = texelFetch(uA, ivec2(a0 + 9, m), 0);
    vec4 A22 = h2lo(u12); vec4 A32 = h2hi(u12);
    uvec4 u13 = texelFetch(uA, ivec2(a0 + 13, m), 0);
    vec4 A23 = h2lo(u13); vec4 A33 = h2hi(u13);
    int qa = 4 * n + hw;
    int qh = 8 + 2 * n + hw;
    uint b0 = uint(hw), b2 = uint(hw + 2);
    {
      int t = wx + (sb * 4 + 0) * 13;
      uvec4 QA = texelFetch(uW, ivec2(t + qa, gr), 0);
      uvec4 QB = texelFetch(uW, ivec2(t + qa + 2, gr), 0);
      uvec4 QH = texelFetch(uW, ivec2(t + qh, gr), 0);
      uvec4 S = S0;
      uvec2 SS = n == 0 ? S.xy : S.zw;
      vec4 p = vec4(0.0);
      p += vec4(dot(q6v(q6g0(QA.x, QH.x)), A00), dot(q6v(q6g1(QB.x, QH.x)), A01), dot(q6v(q6g2(QA.x, QH.x)), A02), dot(q6v(q6g3(QB.x, QH.x)), A03));
      p += vec4(dot(q6v(q6g0(QA.y, QH.y)), A10), dot(q6v(q6g1(QB.y, QH.y)), A11), dot(q6v(q6g2(QA.y, QH.y)), A12), dot(q6v(q6g3(QB.y, QH.y)), A13));
      p += vec4(dot(q6v(q6g0(QA.z, QH.z)), A20), dot(q6v(q6g1(QB.z, QH.z)), A21), dot(q6v(q6g2(QA.z, QH.z)), A22), dot(q6v(q6g3(QB.z, QH.z)), A23));
      p += vec4(dot(q6v(q6g0(QA.w, QH.w)), A30), dot(q6v(q6g1(QB.w, QH.w)), A31), dot(q6v(q6g2(QA.w, QH.w)), A32), dot(q6v(q6g3(QB.w, QH.w)), A33));
      P.x += dot(p, vec4(sc8w(SS.x, b0), sc8w(SS.x, b2), sc8w(SS.y, b0), sc8w(SS.y, b2)));
    }
    {
      int t = wx + (sb * 4 + 1) * 13;
      uvec4 QA = texelFetch(uW, ivec2(t + qa, gr), 0);
      uvec4 QB = texelFetch(uW, ivec2(t + qa + 2, gr), 0);
      uvec4 QH = texelFetch(uW, ivec2(t + qh, gr), 0);
      uvec4 S = S1;
      uvec2 SS = n == 0 ? S.xy : S.zw;
      vec4 p = vec4(0.0);
      p += vec4(dot(q6v(q6g0(QA.x, QH.x)), A00), dot(q6v(q6g1(QB.x, QH.x)), A01), dot(q6v(q6g2(QA.x, QH.x)), A02), dot(q6v(q6g3(QB.x, QH.x)), A03));
      p += vec4(dot(q6v(q6g0(QA.y, QH.y)), A10), dot(q6v(q6g1(QB.y, QH.y)), A11), dot(q6v(q6g2(QA.y, QH.y)), A12), dot(q6v(q6g3(QB.y, QH.y)), A13));
      p += vec4(dot(q6v(q6g0(QA.z, QH.z)), A20), dot(q6v(q6g1(QB.z, QH.z)), A21), dot(q6v(q6g2(QA.z, QH.z)), A22), dot(q6v(q6g3(QB.z, QH.z)), A23));
      p += vec4(dot(q6v(q6g0(QA.w, QH.w)), A30), dot(q6v(q6g1(QB.w, QH.w)), A31), dot(q6v(q6g2(QA.w, QH.w)), A32), dot(q6v(q6g3(QB.w, QH.w)), A33));
      P.y += dot(p, vec4(sc8w(SS.x, b0), sc8w(SS.x, b2), sc8w(SS.y, b0), sc8w(SS.y, b2)));
    }
    {
      int t = wx + (sb * 4 + 2) * 13;
      uvec4 QA = texelFetch(uW, ivec2(t + qa, gr), 0);
      uvec4 QB = texelFetch(uW, ivec2(t + qa + 2, gr), 0);
      uvec4 QH = texelFetch(uW, ivec2(t + qh, gr), 0);
      uvec4 S = S2;
      uvec2 SS = n == 0 ? S.xy : S.zw;
      vec4 p = vec4(0.0);
      p += vec4(dot(q6v(q6g0(QA.x, QH.x)), A00), dot(q6v(q6g1(QB.x, QH.x)), A01), dot(q6v(q6g2(QA.x, QH.x)), A02), dot(q6v(q6g3(QB.x, QH.x)), A03));
      p += vec4(dot(q6v(q6g0(QA.y, QH.y)), A10), dot(q6v(q6g1(QB.y, QH.y)), A11), dot(q6v(q6g2(QA.y, QH.y)), A12), dot(q6v(q6g3(QB.y, QH.y)), A13));
      p += vec4(dot(q6v(q6g0(QA.z, QH.z)), A20), dot(q6v(q6g1(QB.z, QH.z)), A21), dot(q6v(q6g2(QA.z, QH.z)), A22), dot(q6v(q6g3(QB.z, QH.z)), A23));
      p += vec4(dot(q6v(q6g0(QA.w, QH.w)), A30), dot(q6v(q6g1(QB.w, QH.w)), A31), dot(q6v(q6g2(QA.w, QH.w)), A32), dot(q6v(q6g3(QB.w, QH.w)), A33));
      P.z += dot(p, vec4(sc8w(SS.x, b0), sc8w(SS.x, b2), sc8w(SS.y, b0), sc8w(SS.y, b2)));
    }
    {
      int t = wx + (sb * 4 + 3) * 13;
      uvec4 QA = texelFetch(uW, ivec2(t + qa, gr), 0);
      uvec4 QB = texelFetch(uW, ivec2(t + qa + 2, gr), 0);
      uvec4 QH = texelFetch(uW, ivec2(t + qh, gr), 0);
      uvec4 S = S3;
      uvec2 SS = n == 0 ? S.xy : S.zw;
      vec4 p = vec4(0.0);
      p += vec4(dot(q6v(q6g0(QA.x, QH.x)), A00), dot(q6v(q6g1(QB.x, QH.x)), A01), dot(q6v(q6g2(QA.x, QH.x)), A02), dot(q6v(q6g3(QB.x, QH.x)), A03));
      p += vec4(dot(q6v(q6g0(QA.y, QH.y)), A10), dot(q6v(q6g1(QB.y, QH.y)), A11), dot(q6v(q6g2(QA.y, QH.y)), A12), dot(q6v(q6g3(QB.y, QH.y)), A13));
      p += vec4(dot(q6v(q6g0(QA.z, QH.z)), A20), dot(q6v(q6g1(QB.z, QH.z)), A21), dot(q6v(q6g2(QA.z, QH.z)), A22), dot(q6v(q6g3(QB.z, QH.z)), A23));
      p += vec4(dot(q6v(q6g0(QA.w, QH.w)), A30), dot(q6v(q6g1(QB.w, QH.w)), A31), dot(q6v(q6g2(QA.w, QH.w)), A32), dot(q6v(q6g3(QB.w, QH.w)), A33));
      P.w += dot(p, vec4(sc8w(SS.x, b0), sc8w(SS.x, b2), sc8w(SS.y, b0), sc8w(SS.y, b2)));
    }
  }
    acc += d * P;
  }
  o = acc;
}