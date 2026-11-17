/**
 * The engine's fragment shaders as WebGPU compute shaders: a translator for
 * the GLSL ES 3.00 the kernels are written in (see kernels.ts), so one set
 * of kernels runs on both backends.
 *
 * A fragment shader computes the texel at gl_FragCoord of its outputs; the
 * compute shader it becomes runs one invocation per texel of the pass's
 * rectangle, sets gl_FragCoord the same way, runs the shader's main and
 * stores its outputs into storage textures. Uniforms go in one uniform
 * buffer (the rectangle first), samplers become textures read with
 * textureLoad.
 *
 * The GLSL it reads is the subset the kernels use: no structs, no
 * multi-component swizzle writes, no out parameters. Anything else throws.
 */

type Base = "f" | "i" | "u" | "b"
/** A GLSL type's name: float, ivec2, usampler2D, void, or an array: ivec4[16]. */
type Type = string

const SCALARS: Record<string, Base> = { float: "f", int: "i", uint: "u", bool: "b" }
const PREFIX: Record<string, Base> = { "": "f", i: "i", u: "u", b: "b" }
const SAMPLERS: Record<string, Base> = { sampler2D: "f", usampler2D: "u", isampler2D: "i" }

function shape(t: Type): { base: Base; n: number } | null {
  if (t in SCALARS) return { base: SCALARS[t], n: 1 }
  const m = /^([iub]?)vec([234])$/.exec(t)
  return m ? { base: PREFIX[m[1]], n: Number(m[2]) } : null
}
function make(base: Base, n: number): Type {
  if (n === 1) return { f: "float", i: "int", u: "uint", b: "bool" }[base]
  return `${base === "f" ? "" : base}vec${n}`
}
const isType = (t: string) =>
  t in SCALARS || t in SAMPLERS || t === "void" || /^[iub]?vec[234]$/.test(t)

const WGSL_BASE: Record<Base, string> = { f: "f32", i: "i32", u: "u32", b: "bool" }
function wgslType(t: Type): string {
  const arr = /^(\w+)\[(\d+)\]$/.exec(t)
  if (arr) return `array<${wgslType(arr[1])}, ${arr[2]}>`
  const s = shape(t)
  if (!s) throw new Error(`No WGSL type for ${t}.`)
  return s.n === 1 ? WGSL_BASE[s.base] : `vec${s.n}<${WGSL_BASE[s.base]}>`
}

// ---------------------------------------------------------------- lexing

type Token = { k: "id" | "num" | "op" | "eof"; v: string }

const OPS = [
  "<<=", ">>=", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "^^",
  "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
  "+", "-", "*", "/", "%", "<", ">", "=", "!", "~", "&", "|", "^", "?", ":",
  "(", ")", "[", "]", "{", "}", ",", ";", "."
]

function lex(source: string): Array<Token> {
  const tokens: Array<Token> = []
  let i = 0
  const s = source
  while (i < s.length) {
    const c = s[i]
    if (c === "\n" || c === " " || c === "\t" || c === "\r") { i++; continue }
    if (c === "/" && s[i + 1] === "/") { while (i < s.length && s[i] !== "\n") i++; continue }
    if (c === "/" && s[i + 1] === "*") { const e = s.indexOf("*/", i + 2); i = e < 0 ? s.length : e + 2; continue }
    if (c === "#") { while (i < s.length && s[i] !== "\n") i++; continue }
    if (/[A-Za-z_]/.test(c)) {
      let j = i + 1
      while (j < s.length && /\w/.test(s[j])) j++
      tokens.push({ k: "id", v: s.slice(i, j) })
      i = j
      continue
    }
    if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(s[i + 1]))) {
      const m = /^(0[xX][0-9a-fA-F]+[uU]?|(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?[fFuU]?)/.exec(s.slice(i))
      if (!m) throw new Error(`Bad number at ${s.slice(i, i + 20)}`)
      tokens.push({ k: "num", v: m[0] })
      i += m[0].length
      continue
    }
    const op = OPS.find((o) => s.startsWith(o, i))
    if (!op) throw new Error(`Unexpected ${JSON.stringify(c)} in GLSL.`)
    tokens.push({ k: "op", v: op })
    i += op.length
  }
  tokens.push({ k: "eof", v: "" })
  return tokens
}

// ---------------------------------------------------------------- parsing

type Expr =
  | { k: "num"; v: string }
  | { k: "id"; v: string }
  | { k: "bin"; op: string; a: Expr; b: Expr }
  | { k: "un"; op: string; a: Expr }
  | { k: "post"; op: string; a: Expr }
  | { k: "call"; f: string; args: Array<Expr> }
  | { k: "field"; a: Expr; f: string }
  | { k: "index"; a: Expr; i: Expr }
  | { k: "tern"; c: Expr; a: Expr; b: Expr }
  | { k: "assign"; op: string; a: Expr; b: Expr }

type Decl = { type: Type; name: string; init?: Expr }
type Stmt =
  | { k: "block"; body: Array<Stmt> }
  | { k: "decl"; konst: boolean; vars: Array<Decl> }
  | { k: "expr"; e: Expr }
  | { k: "if"; c: Expr; a: Stmt; b?: Stmt }
  | { k: "for"; init?: Stmt; c?: Expr; step?: Expr; body: Stmt }
  | { k: "while"; c: Expr; body: Stmt }
  | { k: "break" }
  | { k: "continue" }
  | { k: "return"; e?: Expr }
  | { k: "empty" }

type Fn = { ret: Type; name: string; params: Array<{ type: Type; name: string }>; body: Array<Stmt> }

type Shader = {
  uniforms: Array<{ type: Type; name: string }>
  samplers: Array<{ type: Type; name: string }>
  outputs: Array<{ type: Type; name: string; location: number }>
  globals: Array<{ konst: boolean; decl: Decl }>
  fns: Array<Fn>
}

const QUALIFIERS = new Set(["highp", "mediump", "lowp", "flat", "in"])

class Parser {
  private p = 0
  constructor(private t: Array<Token>) {}
  private peek(o = 0) { return this.t[this.p + o] }
  private next() { return this.t[this.p++] }
  private is(v: string, o = 0) { const t = this.peek(o); return t.k !== "num" && t.v === v }
  private eat(v: string) {
    const t = this.next()
    if (t.v !== v) throw new Error(`Expected ${v}, got ${t.v || "end"} (GLSL token ${this.p}).`)
  }
  private ident() {
    const t = this.next()
    if (t.k !== "id") throw new Error(`Expected a name, got ${t.v}.`)
    return t.v
  }
  private skipQualifiers() { while (this.peek().k === "id" && QUALIFIERS.has(this.peek().v)) this.p++ }
  /** Whether a declaration starts here (after qualifiers). */
  private atType() {
    let o = 0
    while (this.peek(o).k === "id" && (QUALIFIERS.has(this.peek(o).v) || this.peek(o).v === "const")) o++
    const t = this.peek(o)
    return t.k === "id" && isType(t.v) && this.peek(o + 1).k === "id"
  }

  shader(): Shader {
    const out: Shader = { uniforms: [], samplers: [], outputs: [], globals: [], fns: [] }
    while (this.peek().k !== "eof") {
      if (this.is(";")) { this.p++; continue }
      if (this.is("precision")) { while (!this.is(";")) this.p++; this.p++; continue }
      let location = -1
      if (this.is("layout")) {
        this.p++
        this.eat("(")
        if (this.ident() !== "location") throw new Error("Only layout(location) is supported.")
        this.eat("=")
        location = Number(this.next().v)
        this.eat(")")
      }
      if (this.is("uniform")) {
        this.p++
        this.skipQualifiers()
        const type = this.ident()
        for (;;) {
          const name = this.ident()
          if (type in SAMPLERS) out.samplers.push({ type, name })
          else out.uniforms.push({ type: this.arraySuffix(type), name })
          if (!this.is(",")) break
          this.p++
        }
        this.eat(";")
        continue
      }
      if (this.is("out")) {
        this.p++
        this.skipQualifiers()
        const type = this.ident()
        const name = this.ident()
        this.eat(";")
        out.outputs.push({ type, name, location: location < 0 ? 0 : location })
        continue
      }
      const konst = this.is("const")
      if (konst) this.p++
      this.skipQualifiers()
      const type = this.ident()
      const name = this.ident()
      if (this.is("(")) {
        this.p++
        const params: Fn["params"] = []
        while (!this.is(")")) {
          this.skipQualifiers()
          if (this.is("void") && this.is(")", 1)) { this.p++; break }
          if (this.is("out") || this.is("inout")) throw new Error("out parameters aren't supported.")
          this.skipQualifiers()
          const pt = this.ident()
          params.push({ type: pt, name: this.ident() })
          if (this.is(",")) this.p++
        }
        this.eat(")")
        if (this.is(";")) { this.p++; continue }
        out.fns.push({ ret: type, name, params, body: this.blockBody() })
        continue
      }
      for (const decl of this.declarators(type, name)) out.globals.push({ konst, decl })
      this.eat(";")
    }
    return out
  }

  private arraySuffix(type: Type) {
    if (!this.is("[")) return type
    this.p++
    const n = this.next().v
    this.eat("]")
    return `${type}[${n}]`
  }

  private declarators(type: Type, first: string): Array<Decl> {
    const vars: Array<Decl> = []
    let name = first
    for (;;) {
      const t = this.arraySuffix(type)
      let init: Expr | undefined
      if (this.is("=")) { this.p++; init = this.assignment() }
      vars.push({ type: t, name, ...(init && { init }) })
      if (!this.is(",")) break
      this.p++
      name = this.ident()
    }
    return vars
  }

  private blockBody(): Array<Stmt> {
    this.eat("{")
    const body: Array<Stmt> = []
    while (!this.is("}")) body.push(this.statement())
    this.eat("}")
    return body
  }

  private statement(): Stmt {
    if (this.is("{")) return { k: "block", body: this.blockBody() }
    if (this.is(";")) { this.p++; return { k: "empty" } }
    if (this.is("if")) {
      this.p++
      this.eat("(")
      const c = this.expression()
      this.eat(")")
      const a = this.statement()
      if (this.is("else")) { this.p++; return { k: "if", c, a, b: this.statement() } }
      return { k: "if", c, a }
    }
    if (this.is("for")) {
      this.p++
      this.eat("(")
      let init: Stmt | undefined
      if (this.is(";")) this.p++
      else init = this.simple(true)
      const c = this.is(";") ? undefined : this.expression()
      this.eat(";")
      const step = this.is(")") ? undefined : this.expression()
      this.eat(")")
      return { k: "for", ...(init && { init }), ...(c && { c }), ...(step && { step }), body: this.statement() }
    }
    if (this.is("while")) {
      this.p++
      this.eat("(")
      const c = this.expression()
      this.eat(")")
      return { k: "while", c, body: this.statement() }
    }
    if (this.is("break")) { this.p++; this.eat(";"); return { k: "break" } }
    if (this.is("continue")) { this.p++; this.eat(";"); return { k: "continue" } }
    if (this.is("return")) {
      this.p++
      if (this.is(";")) { this.p++; return { k: "return" } }
      const e = this.expression()
      this.eat(";")
      return { k: "return", e }
    }
    return this.simple(true)
  }

  /** A declaration or expression statement, through its semicolon. */
  private simple(semicolon: boolean): Stmt {
    if (this.atType()) {
      const konst = this.is("const")
      if (konst) this.p++
      this.skipQualifiers()
      const type = this.ident()
      const vars = this.declarators(type, this.ident())
      if (semicolon) this.eat(";")
      return { k: "decl", konst, vars }
    }
    const e = this.expression()
    if (semicolon) this.eat(";")
    return { k: "expr", e }
  }

  expression(): Expr {
    const e = this.assignment()
    if (this.is(",")) throw new Error("The comma operator isn't supported.")
    return e
  }

  private assignment(): Expr {
    const a = this.ternary()
    const t = this.peek()
    if (t.k === "op" && /^([-+*/%&|^]|<<|>>)?=$/.test(t.v) && t.v !== "==") {
      this.p++
      return { k: "assign", op: t.v, a, b: this.assignment() }
    }
    return a
  }

  private ternary(): Expr {
    const c = this.binary(0)
    if (!this.is("?")) return c
    this.p++
    const a = this.assignment()
    this.eat(":")
    return { k: "tern", c, a, b: this.assignment() }
  }

  private static LEVELS = [
    ["||"], ["^^"], ["&&"], ["|"], ["^"], ["&"], ["==", "!="], ["<", ">", "<=", ">="],
    ["<<", ">>"], ["+", "-"], ["*", "/", "%"]
  ]

  private binary(level: number): Expr {
    if (level === Parser.LEVELS.length) return this.unary()
    let a = this.binary(level + 1)
    for (;;) {
      const t = this.peek()
      if (t.k !== "op" || !Parser.LEVELS[level].includes(t.v)) return a
      this.p++
      a = { k: "bin", op: t.v, a, b: this.binary(level + 1) }
    }
  }

  private unary(): Expr {
    const t = this.peek()
    if (t.k === "op" && ["-", "+", "!", "~", "++", "--"].includes(t.v)) {
      this.p++
      return { k: "un", op: t.v, a: this.unary() }
    }
    return this.postfix()
  }

  private postfix(): Expr {
    let e = this.primary()
    for (;;) {
      if (this.is(".")) { this.p++; e = { k: "field", a: e, f: this.ident() }; continue }
      if (this.is("[")) { this.p++; const i = this.expression(); this.eat("]"); e = { k: "index", a: e, i }; continue }
      if (this.is("++") || this.is("--")) { e = { k: "post", op: this.next().v, a: e }; continue }
      return e
    }
  }

  private primary(): Expr {
    const t = this.next()
    if (t.k === "num") return { k: "num", v: t.v }
    if (t.k === "op" && t.v === "(") { const e = this.expression(); this.eat(")"); return e }
    if (t.k !== "id") throw new Error(`Unexpected ${t.v || "end"} in an expression.`)
    if (this.is("(")) {
      this.p++
      const args: Array<Expr> = []
      while (!this.is(")")) {
        if (this.is("void") && this.is(")", 1)) { this.p++; break }
        args.push(this.assignment())
        if (this.is(",")) this.p++
      }
      this.eat(")")
      return { k: "call", f: t.v, args }
    }
    return { k: "id", v: t.v }
  }
}

// ---------------------------------------------------------------- emitting

/** How a pass binds the shader: its outputs' formats, and inputs to read as RGBA4. */
export type Variant = {
  /** Storage texture format of each output, by location. */
  outputs: Array<GPUTextureFormat>
  /** Samplers bound to RGBA4 textures (held as r16uint: see webgpu.ts). */
  rgba4?: ReadonlyArray<string>
  workgroup: [number, number]
  /**
   * Threads per output texel (a power of two; see coopLoop): with more than
   * one, the workgroup is (lanes, workgroup[0] · workgroup[1] / lanes) and
   * covers that many texels of a row, its threads sharing main's K loop.
   */
  lanes?: number
}

export type Translation = {
  code: string
  /** The samplers main reads, in binding order (bindings 1, 2, …). */
  samplers: Array<{ name: string; kind: Base }>
  /** Each uniform's byte offset in the uniform buffer (after the 16-byte rectangle). */
  uniforms: Map<string, { type: Type; offset: number }>
  uniformBytes: number
  outputs: number
  /** Threads per output texel: 1 unless the variant asked for more and main's loop allows it. */
  lanes: number
  workgroup: [number, number]
}

const BUILTIN_SAME = new Set([
  "abs", "sign", "floor", "ceil", "fract", "exp", "exp2", "log", "log2", "sqrt", "sin", "cos",
  "tan", "tanh", "round", "trunc", "min", "max", "clamp", "mix", "pow", "smoothstep", "roundEven"
])
const RENAME: Record<string, string> = {
  inversesqrt: "inverseSqrt",
  unpackHalf2x16: "unpack2x16float",
  packHalf2x16: "pack2x16float",
  roundEven: "round"
}
const COMPARE: Record<string, string> = {
  lessThan: "<", lessThanEqual: "<=", greaterThan: ">", greaterThanEqual: ">=", equal: "==", notEqual: "!="
}
const BITCAST: Record<string, Base> = {
  floatBitsToUint: "u", floatBitsToInt: "i", uintBitsToFloat: "f", intBitsToFloat: "f"
}

type Scope = Map<string, { type: Type; out: string }>
type Ctx = { pre: Array<string>; hoist: boolean; indent: string }

/** Every identifier the shader declares gets this prefix: none collides with WGSL's words. */
const P = "v_"

class Emitter {
  private scopes: Array<Scope> = []
  private globals: Scope = new Map()
  private overloads = new Map<string, Array<Fn>>()
  private fnOut = new Map<Fn, string>()
  private temps = 0
  /** Per function: the samplers and functions it uses. */
  private uses = new Map<Fn, { samplers: Set<string>; fns: Set<Fn> }>()
  private current: Fn | null = null
  private rgba4: Set<string>
  /** The K loop the workgroup's lanes share, and its accumulator (see coopLoop). */
  private coop: { loop: Stmt; acc: string } | null = null
  private lanes = 1

  constructor(private s: Shader, private variant: Variant) {
    this.rgba4 = new Set(variant.rgba4 ?? [])
    for (const f of s.fns) {
      const list = this.overloads.get(f.name) ?? []
      list.push(f)
      this.overloads.set(f.name, list)
    }
    for (const [name, list] of this.overloads)
      list.forEach((f, i) => this.fnOut.set(f, list.length === 1 ? P + name : `${P}${name}_${i}`))
    for (const u of s.uniforms) this.globals.set(u.name, { type: u.type, out: `U.${P}${u.name}` })
    for (const t of s.samplers) this.globals.set(t.name, { type: t.type, out: P + t.name })
    for (const o of s.outputs) this.globals.set(o.name, { type: o.type, out: P + o.name })
    for (const g of s.globals) this.globals.set(g.decl.name, { type: g.decl.type, out: P + g.decl.name })
    this.globals.set("gl_FragCoord", { type: "vec4", out: "fc" })
    if ((variant.lanes ?? 1) > 1) {
      this.coop = coopLoop(s)
      if (this.coop) this.lanes = variant.lanes!
    }
  }

  private lookup(name: string) {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const v = this.scopes[i].get(name)
      if (v) return v
    }
    const g = this.globals.get(name)
    if (!g) throw new Error(`Unknown name ${name}.`)
    if (g.type in SAMPLERS && this.current) this.uses.get(this.current)?.samplers.add(name)
    return g
  }

  /** The type of an expression, without emitting it. */
  typeOf(e: Expr): Type {
    switch (e.k) {
      case "num":
        if (/^0[xX]/.test(e.v)) return /[uU]$/.test(e.v) ? "uint" : "int"
        if (/[uU]$/.test(e.v)) return "uint"
        return /[.eE]|[fF]$/.test(e.v) ? "float" : "int"
      case "id":
        if (e.v === "true" || e.v === "false") return "bool"
        return this.lookup(e.v).type
      case "assign": return this.typeOf(e.a)
      case "tern": return this.typeOf(e.a)
      case "post": return this.typeOf(e.a)
      case "un": return e.op === "!" ? "bool" : this.typeOf(e.a)
      case "field": {
        const s = shape(this.typeOf(e.a))
        if (!s) throw new Error(`.${e.f} of a non-vector.`)
        return make(s.base, e.f.length)
      }
      case "index": {
        const t = this.typeOf(e.a)
        const arr = /^(\w+)\[\d+\]$/.exec(t)
        if (arr) return arr[1]
        const s = shape(t)
        if (!s) throw new Error("Indexing a non-vector.")
        return make(s.base, 1)
      }
      case "bin": {
        if (["<", ">", "<=", ">=", "==", "!=", "&&", "||", "^^"].includes(e.op)) return "bool"
        const a = this.typeOf(e.a)
        if (e.op === "<<" || e.op === ">>") return a
        const b = this.typeOf(e.b)
        return (shape(a)?.n ?? 1) >= (shape(b)?.n ?? 1) ? a : b
      }
      case "call": return this.callType(e)
    }
  }

  private callType(e: Extract<Expr, { k: "call" }>): Type {
    const f = e.f
    if (isType(f)) return f
    if (f === "texelFetch") {
      const s = this.typeOf(e.args[0])
      return make(SAMPLERS[s] ?? "f", 4)
    }
    if (BUILTIN_SAME.has(f)) {
      // (step(edge, x) and mix(a, b, t) take x's and a's shape; min/max/clamp the widest.)
      if (f === "mix") return this.typeOf(e.args[0])
      const types = e.args.map((a) => this.typeOf(a))
      return types.reduce((w, t) => ((shape(t)?.n ?? 1) > (shape(w)?.n ?? 1) ? t : w))
    }
    if (f === "step") return this.typeOf(e.args[1])
    if (f === "inversesqrt" || f === "normalize") return this.typeOf(e.args[0])
    if (f === "dot" || f === "length" || f === "distance") return "float"
    if (f === "unpackHalf2x16") return "vec2"
    if (f === "packHalf2x16") return "uint"
    if (f in BITCAST) {
      const s = shape(this.typeOf(e.args[0]))
      return make(BITCAST[f], s?.n ?? 1)
    }
    if (f in COMPARE) return make("b", shape(this.typeOf(e.args[0]))?.n ?? 1)
    if (f === "any" || f === "all") return "bool"
    if (f === "not") return this.typeOf(e.args[0])
    return this.resolve(e).ret
  }

  private resolve(e: Extract<Expr, { k: "call" }>): Fn {
    const list = this.overloads.get(e.f)
    if (!list) throw new Error(`Unknown function ${e.f}.`)
    if (list.length === 1) return list[0]
    const types = e.args.map((a) => this.typeOf(a))
    const f = list.find((c) => c.params.length === types.length && c.params.every((p, i) => p.type === types[i]))
    if (!f) throw new Error(`No overload of ${e.f} takes (${types.join(", ")}).`)
    return f
  }

  /** Whether evaluating e reads a texture or calls a function of the shader's (a ternary of it is branched, not selected). */
  private heavy(e: Expr): boolean {
    switch (e.k) {
      case "num": case "id": return false
      case "call":
        return e.f === "texelFetch" || this.overloads.has(e.f) || e.args.some((a) => this.heavy(a))
      case "bin": return this.heavy(e.a) || this.heavy(e.b)
      case "un": case "post": case "field": return this.heavy(e.a)
      case "index": return this.heavy(e.a) || this.heavy(e.i)
      case "tern": return this.heavy(e.c) || this.heavy(e.a) || this.heavy(e.b)
      case "assign": return true
    }
  }

  /** e converted to type t (constructor arguments, mixed-shape operands). */
  private convert(code: string, from: Type, to: Type): string {
    if (from === to) return code
    const a = shape(from), b = shape(to)
    if (!a || !b) return code
    if (a.n === b.n) return `${wgslType(to)}(${code})`
    if (a.n === 1) return `${wgslType(to)}(${a.base === b.base ? code : `${WGSL_BASE[b.base]}(${code})`})`
    throw new Error(`Can't convert ${from} to ${to}.`)
  }

  expr(e: Expr, ctx: Ctx): string {
    switch (e.k) {
      case "num": {
        let v = e.v
        if (/^0[xX]/.test(v)) return v.replace(/U$/, "u")
        v = v.replace(/[fF]$/, "")
        if (/[uU]$/.test(v)) return v.replace(/U$/, "u")
        if (v.startsWith(".")) v = "0" + v
        v = v.replace(/\.(?=[eE]|$)/, ".0")
        return v
      }
      case "id":
        if (e.v === "true" || e.v === "false") return e.v
        return this.lookup(e.v).out
      case "field": return `${this.expr(e.a, ctx)}.${e.f.replace(/[rgba]/g, (c) => "xyzw"["rgba".indexOf(c)])}`
      case "index": return `${this.expr(e.a, ctx)}[${this.expr(e.i, ctx)}]`
      case "post":
      case "un":
        if (e.op === "++" || e.op === "--") {
          if (!ctx.hoist) throw new Error("++/-- inside an expression isn't supported.")
          ctx.pre.push(`${this.expr(e.a, ctx)}${e.op};`)
          if (e.k === "un") return this.expr(e.a, ctx)
          throw new Error("Postfix ++/-- inside an expression isn't supported.")
        }
        return `(${e.op}${this.expr(e.a, ctx)})`
      case "assign": throw new Error("Assignments inside expressions aren't supported.")
      case "tern": {
        const t = this.typeOf(e.a)
        if (ctx.hoist && (this.heavy(e.a) || this.heavy(e.b))) {
          const tmp = `t${this.temps++}`
          const c = this.expr(e.c, ctx)
          const ca: Ctx = { pre: [], hoist: true, indent: ctx.indent + "  " }
          const a = this.coerce(e.a, t, ca)
          const cb: Ctx = { pre: [], hoist: true, indent: ctx.indent + "  " }
          const b = this.coerce(e.b, t, cb)
          const i = ctx.indent
          ctx.pre.push(
            `var ${tmp}: ${wgslType(t)};`,
            `if (${c}) {`, ...ca.pre.map((l) => `  ${l}`), `  ${tmp} = ${a};`,
            `} else {`, ...cb.pre.map((l) => `  ${l}`), `  ${tmp} = ${b};`, `}`
          )
          void i
          return tmp
        }
        const quiet = { ...ctx, hoist: false }
        return `select(${this.coerce(e.b, t, quiet)}, ${this.coerce(e.a, t, quiet)}, ${this.expr(e.c, quiet)})`
      }
      case "bin": return this.binary(e, ctx)
      case "call": return this.call(e, ctx)
    }
  }

  /** e emitted as type t: a scalar splat to a vector where GLSL allows mixing them. */
  private coerce(e: Expr, t: Type, ctx: Ctx) {
    const from = this.typeOf(e)
    const code = this.expr(e, ctx)
    if (from === t) return code
    const a = shape(from), b = shape(t)
    if (a && b && a.n === 1 && b.n > 1 && a.base === b.base) return `${wgslType(t)}(${code})`
    return code
  }

  private binary(e: Extract<Expr, { k: "bin" }>, ctx: Ctx): string {
    const ta = this.typeOf(e.a), tb = this.typeOf(e.b)
    const sa = shape(ta), sb = shape(tb)
    if (e.op === "&&" || e.op === "||") {
      const rhs = this.expr(e.b, { ...ctx, hoist: false })
      return `(${this.expr(e.a, ctx)} ${e.op} ${rhs})`
    }
    if (e.op === "^^") return `(${this.expr(e.a, ctx)} != ${this.expr(e.b, ctx)})`
    let a = this.expr(e.a, ctx)
    let b = this.expr(e.b, ctx)
    if (e.op === "<<" || e.op === ">>") {
      const n = sa?.n ?? 1
      const ut = make("u", n)
      if (e.b.k === "num" && !/^0[xX]/.test(e.b.v) && !/[.eE]/.test(e.b.v)) {
        b = `${e.b.v.replace(/[uU]$/, "")}u`
        if (n > 1) b = `${wgslType(ut)}(${b})`
      } else if (sb && (sb.base !== "u" || sb.n !== n)) b = this.convert(b, tb, ut)
      return `(${a} ${e.op} ${b})`
    }
    if ((e.op === "==" || e.op === "!=") && sa && sa.n > 1)
      return e.op === "==" ? `all(${a} == ${b})` : `any(${a} != ${b})`
    // Bitwise operators and comparisons mix a vector with a scalar only by a splat.
    if (sa && sb && sa.n !== sb.n && ["&", "|", "^", "<", ">", "<=", ">=", "==", "!="].includes(e.op)) {
      if (sa.n === 1) a = `${wgslType(tb)}(${a})`
      else b = `${wgslType(ta)}(${b})`
    }
    return `(${a} ${e.op} ${b})`
  }

  private call(e: Extract<Expr, { k: "call" }>, ctx: Ctx): string {
    const f = e.f
    const args = () => e.args.map((a) => this.expr(a, ctx))
    if (isType(f)) {
      const target = shape(f)
      if (!target) throw new Error(`Can't construct ${f}.`)
      const types = e.args.map((a) => this.typeOf(a))
      const codes = args()
      if (target.n === 1) {
        const s = shape(types[0])
        const c = s && s.n > 1 ? `${codes[0]}.x` : codes[0]
        return `${WGSL_BASE[target.base]}(${c})`
      }
      if (codes.length === 1) {
        const s = shape(types[0])
        if (s && s.n > target.n) return this.convert(`${codes[0]}.${"xyzw".slice(0, target.n)}`, make(s.base, target.n), f)
        return this.convert(codes[0], types[0], f)
      }
      const parts = codes.map((c, i) => {
        const s = shape(types[i])
        if (!s || s.base === target.base) return c
        return s.n === 1 ? `${WGSL_BASE[target.base]}(${c})` : `${wgslType(make(target.base, s.n))}(${c})`
      })
      return `${wgslType(f)}(${parts.join(", ")})`
    }
    if (f === "texelFetch") {
      const s = e.args[0]
      const tex = this.expr(s, ctx)
      const load = `textureLoad(${tex}, ${this.expr(e.args[1], ctx)}, ${this.expr(e.args[2], ctx)})`
      if (s.k === "id" && this.rgba4.has(s.v)) return `rgba4(${load}.x)`
      return load
    }
    if (f in COMPARE) {
      const [a, b] = args()
      return `(${a} ${COMPARE[f]} ${b})`
    }
    if (f in BITCAST) {
      const t = this.callType(e)
      return `bitcast<${wgslType(t)}>(${args()[0]})`
    }
    if (f === "not") return `(!${args()[0]})`
    if (f === "mix" && shape(this.typeOf(e.args[2]))?.base === "b") {
      const [a, b, c] = args()
      return `select(${a}, ${b}, ${c})`
    }
    if (BUILTIN_SAME.has(f) || f === "step") {
      // WGSL wants one shape: splat scalar arguments to the result's.
      const t = this.callType(e)
      const st = shape(t)
      const codes = e.args.map((a, i) => {
        const at = this.typeOf(a)
        const s = shape(at)
        const c = this.expr(a, ctx)
        if (f === "mix" && i === 2) return c
        if (st && s && st.n > 1 && s.n === 1) return `${wgslType(t)}(${c})`
        return c
      })
      return `${RENAME[f] ?? f}(${codes.join(", ")})`
    }
    if (f in RENAME || ["dot", "length", "distance", "normalize", "any", "all"].includes(f))
      return `${RENAME[f] ?? f}(${args().join(", ")})`
    const fn = this.resolve(e)
    if (this.current) this.uses.get(this.current)?.fns.add(fn)
    e.args.forEach((a) => {
      if (a.k === "id" && this.rgba4.has(a.v) && !this.scopes.some((s) => s.has(a.v)))
        throw new Error(`An RGBA4 texture (${a.v}) passed to ${e.f}: not supported.`)
    })
    const codes = e.args.map((a, i) => this.coerce(a, fn.params[i].type, ctx))
    return `${this.fnOut.get(fn)}(${codes.join(", ")})`
  }

  private stmt(s: Stmt, out: Array<string>, indent: string) {
    const ctx = (): Ctx => ({ pre: [], hoist: true, indent })
    const flush = (c: Ctx, line: string) => {
      for (const l of c.pre) out.push(indent + l)
      out.push(indent + line)
    }
    switch (s.k) {
      case "empty": return
      case "block": {
        out.push(`${indent}{`)
        this.block(s.body, out, indent + "  ")
        out.push(`${indent}}`)
        return
      }
      case "decl": {
        for (const v of s.vars) {
          const c = ctx()
          const init = v.init ? this.coerce(v.init, v.type, c) : undefined
          const scope = this.scopes[this.scopes.length - 1]
          const name = `${P}${v.name}${scope.has(v.name) || this.shadows(v.name) ? `_${this.temps++}` : ""}`
          scope.set(v.name, { type: v.type, out: name })
          flush(c, init === undefined
            ? `var ${name}: ${wgslType(v.type)};`
            : `${s.konst ? "let" : "var"} ${name}: ${wgslType(v.type)} = ${init};`)
        }
        return
      }
      case "expr": {
        const e = s.e
        const c = ctx()
        if (e.k === "assign") {
          const lt = this.typeOf(e.a)
          const lhs = this.expr(e.a, c)
          if (e.a.k === "field" && e.a.f.length > 1) throw new Error("Swizzle writes aren't supported.")
          let rhs: string
          const op = e.op.slice(0, -1)
          if (op === "<<" || op === ">>") {
            rhs = this.binary({ k: "bin", op, a: e.a, b: e.b }, c)
            flush(c, `${lhs} = ${rhs};`)
            return
          }
          const rt = this.typeOf(e.b)
          rhs = this.expr(e.b, c)
          const sl = shape(lt), sr = shape(rt)
          if (sl && sr && sl.n > 1 && sr.n === 1 && (op === "" || ["&", "|", "^"].includes(op)))
            rhs = `${wgslType(lt)}(${rhs})`
          flush(c, `${lhs} ${e.op} ${rhs};`)
          return
        }
        if (e.k === "post" || (e.k === "un" && (e.op === "++" || e.op === "--"))) {
          flush(c, `${this.expr(e.a, c)}${e.op};`)
          return
        }
        if (e.k === "call") {
          const call = this.expr(e, c)
          flush(c, this.typeOf(e) === "void" ? `${call};` : `_ = ${call};`)
          return
        }
        throw new Error("An expression statement that does nothing.")
      }
      case "if": {
        const c = ctx()
        flush(c, `if (${this.expr(s.c, c)}) {`)
        this.body(s.a, out, indent + "  ")
        if (s.b) {
          out.push(`${indent}} else {`)
          this.body(s.b, out, indent + "  ")
        }
        out.push(`${indent}}`)
        return
      }
      case "for": {
        if (this.coop && s === this.coop.loop) {
          this.coopFor(s, out, indent)
          return
        }
        out.push(`${indent}{`)
        const inner = indent + "  "
        this.scopes.push(new Map())
        const init: Array<string> = []
        if (s.init) this.stmt(s.init, init, "")
        // A one-line init (a declaration) goes in the for; anything longer before it.
        let head = ""
        if (init.length === 1) head = init[0].replace(/;$/, "")
        else for (const l of init) out.push(inner + l)
        const cc: Ctx = { pre: [], hoist: false, indent: inner }
        const cond = s.c ? this.expr(s.c, cc) : ""
        let step = ""
        if (s.step) {
          const st: Array<string> = []
          this.stmt({ k: "expr", e: s.step }, st, "")
          if (st.length !== 1) throw new Error("A for step must be one statement.")
          step = st[0].replace(/;$/, "")
        }
        out.push(`${inner}for (${head}; ${cond}; ${step}) {`)
        this.body(s.body, out, inner + "  ")
        out.push(`${inner}}`)
        this.scopes.pop()
        out.push(`${indent}}`)
        return
      }
      case "while": {
        const cc: Ctx = { pre: [], hoist: false, indent }
        out.push(`${indent}while (${this.expr(s.c, cc)}) {`)
        this.body(s.body, out, indent + "  ")
        out.push(`${indent}}`)
        return
      }
      case "break": out.push(`${indent}break;`); return
      case "continue": out.push(`${indent}continue;`); return
      case "return": {
        if (!s.e) { out.push(`${indent}return;`); return }
        const c = ctx()
        flush(c, `return ${this.coerce(s.e, this.current?.ret ?? "void", c)};`)
        return
      }
    }
  }

  /**
   * main's K loop shared by the workgroup's lanes: lane l takes iterations
   * l, l + lanes, …, then the lanes sum their accumulators in workgroup
   * memory. The trip count every lane loops to is the workgroup's largest
   * (made uniform with workgroupUniformLoad, as the barriers need).
   */
  private coopFor(s: Extract<Stmt, { k: "for" }>, out: Array<string>, indent: string) {
    const init = s.init as Extract<Stmt, { k: "decl" }>
    const v = init.vars[0]
    const cond = s.c as Extract<Expr, { k: "bin" }>
    const L = this.lanes
    const i = indent + "  "
    out.push(`${indent}{`)
    const c: Ctx = { pre: [], hoist: false, indent: i }
    const lo = this.coerce(v.init!, "int", c)
    const hi = this.coerce(cond.b, "int", c)
    const acc = this.lookup(this.coop!.acc)
    out.push(
      `${i}let lo_: i32 = ${lo};`,
      `${i}let hi_: i32 = ${hi};`,
      `${i}coopTrips[li] = max(0, (hi_ - lo_ - i32(lane) + ${L - 1}) / ${L});`,
      `${i}workgroupBarrier();`,
      `${i}if (li == 0u) { var t_ = 0; for (var j_ = 0u; j_ < ${this.coopThreads()}u; j_++) { t_ = max(t_, coopTrips[j_]); } coopMax = t_; }`,
      `${i}let n_ = workgroupUniformLoad(&coopMax);`,
      `${i}for (var it_ = 0; it_ < n_; it_++) {`
    )
    this.scopes.push(new Map([[v.name, { type: "int", out: `${P}${v.name}` }]]))
    out.push(`${i}  let ${P}${v.name}: i32 = lo_ + i32(lane) + it_ * ${L};`, `${i}  if (${P}${v.name} < hi_) {`)
    this.body(s.body, out, i + "    ")
    this.scopes.pop()
    out.push(
      `${i}  }`,
      `${i}}`,
      `${i}coopAcc[li] = ${acc.out};`,
      `${i}workgroupBarrier();`,
      `${i}for (var s_ = ${L >> 1}u; s_ > 0u; s_ >>= 1u) {`,
      `${i}  if (lane < s_) { coopAcc[li] += coopAcc[li + s_]; }`,
      `${i}  workgroupBarrier();`,
      `${i}}`,
      `${i}${acc.out} = coopAcc[li - lane];`,
      `${indent}}`
    )
  }

  private coopThreads() {
    return this.variant.workgroup[0] * this.variant.workgroup[1]
  }

  /** Whether a name is a parameter of the current function (WGSL can't redeclare one at its top level). */
  private shadows(name: string) {
    return this.scopes.length === 2 && !!this.scopes[0].get(name)
  }

  private body(s: Stmt, out: Array<string>, indent: string) {
    if (s.k === "block") this.block(s.body, out, indent)
    else {
      this.scopes.push(new Map())
      this.stmt(s, out, indent)
      this.scopes.pop()
    }
  }

  private block(body: Array<Stmt>, out: Array<string>, indent: string) {
    this.scopes.push(new Map())
    for (const s of body) this.stmt(s, out, indent)
    this.scopes.pop()
  }

  private fn(f: Fn): Array<string> {
    this.current = f
    this.uses.set(f, { samplers: new Set(), fns: new Set() })
    const params = new Map<string, { type: Type; out: string }>()
    for (const p of f.params) params.set(p.name, { type: p.type, out: P + p.name })
    this.scopes = [params]
    const out: Array<string> = []
    // (WGSL parameters are constants: those a GLSL function assigns to are copied into variables.)
    const assigned = f.params.filter((p) => !(p.type in SAMPLERS) && assigns(f.body, p.name))
    const sig = f.params
      .map((p) => `${assigned.includes(p) ? "a_" : P}${p.name}: ${p.type in SAMPLERS ? `texture_2d<${WGSL_BASE[SAMPLERS[p.type]]}>` : wgslType(p.type)}`)
      .join(", ")
    const name = f.name === "main" ? `${P}main` : (this.fnOut.get(f) ?? P + f.name)
    out.push(`fn ${name}(${sig})${f.ret === "void" ? "" : ` -> ${wgslType(f.ret)}`} {`)
    for (const p of assigned) out.push(`  var ${P}${p.name} = a_${p.name};`)
    this.block(f.body, out, "  ")
    out.push("}")
    this.scopes = []
    this.current = null
    return out
  }

  translate(): Translation {
    const s = this.s
    const main = s.fns.find((f) => f.name === "main")
    if (!main) throw new Error("The shader has no main.")
    const bodies = new Map<Fn, Array<string>>()
    for (const f of s.fns) bodies.set(f, this.fn(f))
    // What main reaches: only those functions are emitted, and only the
    // samplers they read declared (GL drops unread samplers too).
    const reached = new Set<Fn>()
    const visit = (f: Fn) => {
      if (reached.has(f)) return
      reached.add(f)
      for (const g of this.uses.get(f)?.fns ?? []) visit(g)
    }
    visit(main)
    const read = new Set<string>()
    for (const f of reached) for (const n of this.uses.get(f)?.samplers ?? []) read.add(n)
    const samplers = s.samplers.filter((t) => read.has(t.name))

    // Uniform layout: WGSL's uniform rules (16-byte vec3/vec4 and array strides).
    const uniforms = new Map<string, { type: Type; offset: number }>()
    let offset = 16
    const fields = ["  rect: vec4<u32>,"]
    for (const u of s.uniforms) {
      const arr = /^(\w+)\[(\d+)\]$/.exec(u.type)
      const n = arr ? 4 : (shape(u.type)?.n ?? 1)
      if (shape(arr ? arr[1] : u.type)?.base === "b") throw new Error("bool uniforms aren't supported.")
      const align = arr ? 16 : n === 1 ? 4 : n === 2 ? 8 : 16
      offset = Math.ceil(offset / align) * align
      uniforms.set(u.name, { type: u.type, offset })
      offset += arr ? 16 * Number(arr[2]) : 4 * n
      fields.push(`  ${P}${u.name}: ${wgslType(u.type)},`)
    }
    const uniformBytes = Math.ceil(offset / 16) * 16

    const lines: Array<string> = []
    lines.push(`struct Uniforms {\n${fields.join("\n")}\n}`)
    lines.push(`@group(0) @binding(0) var<uniform> U: Uniforms;`)
    samplers.forEach((t, i) => {
      const kind = this.rgba4.has(t.name) ? "u32" : WGSL_BASE[SAMPLERS[t.type]]
      lines.push(`@group(0) @binding(${i + 1}) var ${P}${t.name}: texture_2d<${kind}>;`)
    })
    const outputs = [...s.outputs].sort((a, b) => a.location - b.location)
    if (outputs.length !== this.variant.outputs.length)
      throw new Error(`The shader writes ${outputs.length} outputs; the pass gives ${this.variant.outputs.length}.`)
    outputs.forEach((o, i) => {
      lines.push(`@group(0) @binding(${samplers.length + 1 + i}) var out${i}: texture_storage_2d<${this.variant.outputs[i]}, write>;`)
      lines.push(`var<private> ${P}${o.name}: ${wgslType(o.type)};`)
    })
    lines.push(`var<private> fc: vec4<f32>;`)
    const L = this.lanes
    if (L > 1) {
      const n = this.coopThreads()
      const accType = wgslType(this.globalsOfMain(main).get(this.coop!.acc)!)
      lines.push(
        `var<private> lane: u32;`,
        `var<private> li: u32;`,
        `var<workgroup> coopAcc: array<${accType}, ${n}>;`,
        `var<workgroup> coopTrips: array<i32, ${n}>;`,
        `var<workgroup> coopMax: i32;`
      )
    }
    if (this.rgba4.size)
      lines.push(`fn rgba4(u: u32) -> vec4<f32> { return vec4<f32>(vec4<u32>(u >> 12u, u >> 8u, u >> 4u, u) & vec4<u32>(15u)) / 15.0; }`)
    for (const g of s.globals) {
      const c: Ctx = { pre: [], hoist: false, indent: "" }
      if (g.konst && g.decl.init)
        lines.push(`const ${P}${g.decl.name}: ${wgslType(g.decl.type)} = ${this.coerce(g.decl.init, g.decl.type, c)};`)
      else if (g.decl.init) throw new Error("Initialized global variables aren't supported.")
      else lines.push(`var<private> ${P}${g.decl.name}: ${wgslType(g.decl.type)};`)
    }
    for (const f of s.fns) if (reached.has(f)) lines.push(bodies.get(f)!.join("\n"))
    const threads = this.coopThreads()
    const [wx, wy]: [number, number] = L > 1 ? [L, threads / L] : this.variant.workgroup
    const stores = outputs.map((o, i) => {
      const t = shape(o.type)!
      const want = /uint$/.test(this.variant.outputs[i]) ? "u" : /sint$/.test(this.variant.outputs[i]) ? "i" : "f"
      let v = `${P}${o.name}`
      if (t.n < 4) v = `${wgslType(make(t.base, 4))}(${v}${t.n === 1 ? ", 0, 0, 1" : t.n === 2 ? ", 0, 1" : ", 1"})`
      if (t.base !== want) v = `${wgslType(make(want, 4))}(${v})`
      return `  textureStore(out${i}, p, ${v});`
    })
    if (L > 1)
      // Texel x of the row is lane-group l.y of workgroup w.x; every lane
      // runs main (for the barriers), lane 0 stores.
      lines.push(
        `@compute @workgroup_size(${wx}, ${wy})`,
        `fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {`,
        `  lane = l.x;`,
        `  li = l.y * ${L}u + l.x;`,
        `  let tx = w.x * ${wy}u + l.y;`,
        `  let inside = tx < U.rect.z;`,
        `  let p = vec2<u32>(min(tx, U.rect.z - 1u), w.y) + U.rect.xy;`,
        `  fc = vec4<f32>(vec2<f32>(p) + 0.5, 0.0, 1.0);`,
        `  ${P}main();`,
        `  if (inside && lane == 0u) {`,
        ...stores.map((l) => `  ${l}`),
        `  }`,
        `}`
      )
    else
      lines.push(
        `@compute @workgroup_size(${wx}, ${wy})`,
        `fn main(@builtin(global_invocation_id) id: vec3<u32>) {`,
        `  if (id.x >= U.rect.z || id.y >= U.rect.w) { return; }`,
        `  let p = id.xy + U.rect.xy;`,
        `  fc = vec4<f32>(vec2<f32>(p) + 0.5, 0.0, 1.0);`,
        `  ${P}main();`,
        ...stores,
        `}`
      )
    return {
      code: lines.join("\n"),
      samplers: samplers.map((t) => ({ name: t.name, kind: this.rgba4.has(t.name) ? "u" : SAMPLERS[t.type] })),
      uniforms,
      uniformBytes,
      outputs: outputs.length,
      lanes: L,
      workgroup: [wx, wy]
    }
  }

  /** The types of the variables main declares at its top level. */
  private globalsOfMain(main: Fn) {
    const types = new Map<string, Type>()
    for (const st of main.body) if (st.k === "decl") for (const v of st.vars) types.set(v.name, v.type)
    return types
  }
}

/**
 * main's loop the lanes of a workgroup can share, if it has one: a top-level
 * `for (int k = a; k < b; k++)` whose body only adds into one variable
 * declared before it (`acc += …`, or a component of it), with no break,
 * continue or return in it and no return anywhere in main (the lanes meet
 * at barriers after it). The kernels' K loops (matmuls, the head) are.
 */
export function coopLoop(s: Shader): { loop: Stmt; acc: string } | null {
  const main = s.fns.find((f) => f.name === "main")
  if (!main) return null
  let returns = false
  const walk = (st: Stmt | undefined, f: (st: Stmt) => void) => {
    if (!st) return
    f(st)
    if (st.k === "block") st.body.forEach((x) => walk(x, f))
    else if (st.k === "if") { walk(st.a, f); walk(st.b, f) }
    else if (st.k === "for") { walk(st.init, f); walk(st.body, f) }
    else if (st.k === "while") walk(st.body, f)
  }
  main.body.forEach((st) => walk(st, (x) => { if (x.k === "return") returns = true }))
  if (returns) return null
  const declared = new Set<string>()
  for (const st of main.body) {
    if (st.k === "decl") st.vars.forEach((v) => declared.add(v.name))
    if (st.k !== "for" || !st.init || st.init.k !== "decl" || st.init.vars.length !== 1) continue
    const v = st.init.vars[0]
    if (v.type !== "int" || !v.init) continue
    const c = st.c
    if (!c || c.k !== "bin" || c.op !== "<" || c.a.k !== "id" || c.a.v !== v.name) continue
    const step = st.step
    if (!step || !((step.k === "post" || step.k === "un") && step.op === "++" && step.a.k === "id" && step.a.v === v.name)) continue
    // The body: no jumps, only `+=` into one outer variable, nothing else outer written.
    let ok = true
    const inner = new Set<string>()
    const writes = new Map<string, Array<string>>()
    const target = (e: Expr): string | null => (e.k === "id" ? e.v : e.k === "field" || e.k === "index" ? target(e.a) : null)
    const exprWrites = (e: Expr | undefined): void => {
      if (!e) return
      switch (e.k) {
        case "assign": { const t = target(e.a); if (t) writes.set(t, [...(writes.get(t) ?? []), e.op]); exprWrites(e.b); return }
        case "post": case "un": {
          if (e.op === "++" || e.op === "--") { const t = target(e.a); if (t) writes.set(t, [...(writes.get(t) ?? []), e.op]) }
          exprWrites(e.a); return
        }
        case "bin": exprWrites(e.a); exprWrites(e.b); return
        case "tern": exprWrites(e.c); exprWrites(e.a); exprWrites(e.b); return
        case "call": e.args.forEach(exprWrites); return
        case "field": exprWrites(e.a); return
        case "index": exprWrites(e.a); exprWrites(e.i); return
      }
    }
    walk(st.body, (x) => {
      if (x.k === "break" || x.k === "continue" || x.k === "return") ok = false
      else if (x.k === "decl") x.vars.forEach((d) => { inner.add(d.name); exprWrites(d.init) })
      else if (x.k === "expr") exprWrites(x.e)
      else if (x.k === "if") exprWrites(x.c)
      else if (x.k === "for") { exprWrites(x.c); exprWrites(x.step) }
      else if (x.k === "while") exprWrites(x.c)
    })
    if (!ok) return null
    const outer = [...writes].filter(([name]) => !inner.has(name))
    if (outer.length !== 1) return null
    const [acc, ops] = outer[0]
    if (acc === v.name || !declared.has(acc) || ops.some((op) => op !== "+=")) return null
    // The bound mustn't change inside the loop.
    const reads = new Set<string>()
    const collect = (e: Expr): void => {
      if (e.k === "id") reads.add(e.v)
      else if (e.k === "bin") { collect(e.a); collect(e.b) }
      else if (e.k === "un" || e.k === "post" || e.k === "field") collect(e.a)
      else if (e.k === "call") e.args.forEach(collect)
      else if (e.k === "index") { collect(e.a); collect(e.i) }
      else if (e.k === "tern") { collect(e.c); collect(e.a); collect(e.b) }
    }
    collect(c.b)
    if (reads.has(acc) || reads.has(v.name)) return null
    return { loop: st, acc }
  }
  return null
}

/** Whether a statement list assigns to (or increments) `name`. */
function assigns(body: Array<Stmt>, name: string): boolean {
  const target = (e: Expr): boolean =>
    e.k === "id" ? e.v === name : e.k === "field" || e.k === "index" ? target(e.a) : false
  const inExpr = (e: Expr | undefined): boolean => {
    if (!e) return false
    switch (e.k) {
      case "assign": return target(e.a) || inExpr(e.b)
      case "post": return target(e.a)
      case "un": return ((e.op === "++" || e.op === "--") && target(e.a)) || inExpr(e.a)
      case "bin": return inExpr(e.a) || inExpr(e.b)
      case "tern": return inExpr(e.c) || inExpr(e.a) || inExpr(e.b)
      case "call": return e.args.some(inExpr)
      case "field": return inExpr(e.a)
      case "index": return inExpr(e.a) || inExpr(e.i)
      default: return false
    }
  }
  const inStmt = (s: Stmt | undefined): boolean => {
    if (!s) return false
    switch (s.k) {
      case "block": return s.body.some(inStmt)
      case "decl": return s.vars.some((v) => inExpr(v.init))
      case "expr": return inExpr(s.e)
      case "if": return inExpr(s.c) || inStmt(s.a) || inStmt(s.b)
      case "for": return inStmt(s.init) || inExpr(s.c) || inExpr(s.step) || inStmt(s.body)
      case "while": return inExpr(s.c) || inStmt(s.body)
      case "return": return inExpr(s.e)
      default: return false
    }
  }
  return body.some(inStmt)
}

/** A GLSL fragment shader's parse, reused for each variant. */
export function parseGlsl(source: string): Shader {
  return new Parser(lex(source)).shader()
}

/** The compute shader for a parsed fragment shader, bound as `variant` says. */
export function toWgsl(shader: Shader, variant: Variant): Translation {
  return new Emitter(shader, variant).translate()
}

export type { Shader }
