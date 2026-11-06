import assert from "node:assert/strict"
import test from "node:test"
import { weightHeadRegions, weightPanelSource } from "./weight-panels.js"
import { EMBED_Q6_K_WORDS } from "./gpu/kernels.js"

test("embedding panel selection preserves logical row boundaries", () => {
  const got = weightPanelSource(EMBED_Q6_K_WORDS, 2048, 4)
  for (let p = 1; p < 4; p++) {
    assert.ok(got.includes(`uniform highp usampler2D uWP${p};`))
    assert.ok(got.includes(`p - ivec2(0, ${p * 2048})`))
  }
  assert.equal(got.split("texelFetch(uW,").length - 1, 1)
})
test("single-panel embedding shaders remain identical", () => {
  assert.equal(weightPanelSource(EMBED_Q6_K_WORDS, 2048, 1), EMBED_Q6_K_WORDS)
})

test("head rectangles cover every logit once and address the right column and panel", () => {
  for (const groups of [17, 2049, 31999, 32064]) {
    for (const rowsPerColumn of [1023, 2048, 8016]) {
      const panelRows = 512,
        width = 1024
      const heights = Array.from(
        { length: Math.ceil(rowsPerColumn / panelRows) },
        (_, i) => Math.min(panelRows, rowsPerColumn - i * panelRows)
      )
      const logitRows = Math.ceil(groups / width),
        planes = 3
      const seen = new Uint8Array(planes * groups)
      for (const { panel, row0, rect } of weightHeadRegions({
        groups,
        rowsPerColumn,
        panelRows,
        heights,
        planes,
        logitRows,
        width
      })) {
        const plane = Math.floor(rect.y / logitRows)
        for (let y = rect.y; y < rect.y + rect.height; y++)
          for (let x = rect.x; x < rect.x + rect.width; x++) {
            const g = (y % logitRows) * width + x
            assert.ok(g < groups)
            const localRow = (g % rowsPerColumn) - row0
            assert.ok(localRow >= 0 && localRow < heights[panel])
            seen[plane * groups + g]++
          }
      }
      assert.ok(seen.every((n) => n === 1))
    }
  }
})
