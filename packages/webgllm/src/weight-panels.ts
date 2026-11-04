/** Route a logical integer weight texture's row coordinates to bounded panels. */
export function weightPanelSource(
  source: string,
  rows: number,
  count: number,
  normalized = false
) {
  if (count <= 1) return source
  if (normalized)
    source = source
      .replace(
        "vec4 q6bytes(",
        "vec4 weightTexel(ivec2 p,int level);\nvec4 q6bytes("
      )
      .replace("texelFetch(s,p,0)", "weightTexel(p,0)")
  const samplers = Array.from(
    { length: count - 1 },
    (_, i) =>
      `uniform highp ${normalized ? "sampler2D" : "usampler2D"} uWP${i + 1};`
  ).join("\n")
  const branches = Array.from({ length: count - 1 }, (_, i) => {
    const panel = i + 1
    return `if (p.y < ${panel * rows}) return texelFetch(${i ? `uWP${i}` : "uW"}, p - ivec2(0, ${i * rows}), level);`
  }).join("\n")
  const read = `\n${samplers}\n${normalized ? "vec4" : "uvec4"} weightTexel(ivec2 p, int level) {\n${branches}\nreturn texelFetch(uWP${count - 1}, p - ivec2(0, ${(count - 1) * rows}), level);\n}\n`
  return source
    .replaceAll("texelFetch(uW,", "weightTexel(")
    .replace(
      normalized ? "uniform sampler2D uW;" : "uniform usampler2D uW;",
      `${normalized ? "uniform sampler2D uW;" : "uniform usampler2D uW;"}${read}`
    )
}

/** Vocabulary rectangles that read only one physical weight panel. */
export function* weightHeadRegions(o: {
  groups: number
  rowsPerColumn: number
  panelRows: number
  heights: number[]
  planes: number
  logitRows: number
  width: number
}) {
  const columns = Math.ceil(o.groups / o.rowsPerColumn)
  for (let plane = 0; plane < o.planes; plane++) {
    for (let panel = 0; panel < o.heights.length; panel++) {
      for (let col = 0; col < columns; col++) {
        const start = col * o.rowsPerColumn + panel * o.panelRows
        const end = Math.min(
          o.groups,
          (col + 1) * o.rowsPerColumn,
          start + o.heights[panel]
        )
        for (let g = start; g < end;) {
          const x = g % o.width
          const width = Math.min(o.width - x, end - g)
          const height =
            x === 0 && width === o.width ? Math.floor((end - g) / o.width) : 1
          yield {
            panel,
            row0: panel * o.panelRows,
            rect: {
              x,
              y: plane * o.logitRows + Math.floor(g / o.width),
              width,
              height
            }
          }
          g += width * height
        }
      }
    }
  }
}
