import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { isOldVersion } from "./storage.js"

const repo = "onnx-community/LFM3-1.7B-ONNX"
const commit = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
const config = { repo, path: "config.json" }

describe("isOldVersion", () => {
  test("a file and its markers saved at a pinned commit", () => {
    const old = `onnx-community_LFM3-1.7B-ONNX@${commit}_config.json`
    assert.ok(isOldVersion(old, config))
    assert.ok(isOldVersion(`${old}.done`, config))
    assert.ok(isOldVersion(`${old}.etag`, config))
  })

  test("not the file at main", () => {
    assert.ok(!isOldVersion("onnx-community_LFM3-1.7B-ONNX@main_config.json", config))
  })

  test("not another file whose name ends the same", () => {
    // config.json's cleanup deleted a fresh tokenizer_config.json.
    for (const name of [
      "onnx-community_LFM3-1.7B-ONNX@main_tokenizer_config.json",
      "onnx-community_LFM3-1.7B-ONNX@main_tokenizer_config.json.done",
      `onnx-community_LFM3-1.7B-ONNX@${commit}_tokenizer_config.json`
    ]) {
      assert.ok(!isOldVersion(name, config), name)
    }
  })

  test("not another repo's file", () => {
    const other = `onnx-community_LFM3-1.7B-ONNX-Web@${commit}_config.json`
    assert.ok(!isOldVersion(other, config))
  })

  test("a file in a folder", () => {
    const data = { repo, path: "onnx/model_q4f16.onnx_data" }
    const graph = { repo, path: "onnx/model_q4f16.onnx" }
    const old = `onnx-community_LFM3-1.7B-ONNX@${commit}_onnx_model_q4f16.onnx_data`
    assert.ok(isOldVersion(old, data))
    assert.ok(!isOldVersion(old, graph))
  })
})
