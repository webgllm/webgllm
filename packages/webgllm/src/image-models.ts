/**
 * The image models webgllm runs, and their files: Tessera-small (Jagrat
 * Patel, TheJagStudio; CC BY-NC 4.0), as cfahlgren1/tessera-small-onnx
 * released it at one commit: its int8 denoiser, CLIP ViT-B/32's text
 * tower (OpenAI, MIT), CLIP's tokenizer, its palette and its self-test.
 */
import type { ModelFile } from "./model-meta.js"

export const TESSERA = "cfahlgren1/tessera-small-onnx"
const TESSERA_REVISION = "bef21be58afea4d16be21f5fbe85eaa0cae69290"

export const WEBGL_IMAGE_MODELS = [TESSERA] as const
export type WebGlImageModel = (typeof WEBGL_IMAGE_MODELS)[number]

export type ImageModelMeta = {
  id: WebGlImageModel
  /** Each file and its bytes (for download progress before its size is known). */
  files: Record<"denoiser" | "textEncoder" | "tokenizer" | "palette" | "selftest", ModelFile & { bytes: number }>
  /** The license, as its card states it. */
  license: string
}

// (A folder URL as the repository: files come from that commit, not main.)
const tesseraFile = (path: string, bytes: number) => ({ repo: `https://huggingface.co/${TESSERA}/resolve/${TESSERA_REVISION}`, path, bytes })

export const IMAGE_MODEL_META: Record<WebGlImageModel, ImageModelMeta> = {
  [TESSERA]: {
    id: TESSERA,
    files: {
      denoiser: tesseraFile("tessera_w8.onnx", 50834182),
      textEncoder: tesseraFile("clip_text_w8.onnx", 64060515),
      tokenizer: tesseraFile("tokenizer.json", 2224041),
      palette: tesseraFile("palette.json", 8394),
      selftest: tesseraFile("selftest.json", 2508)
    },
    license: "cc-by-nc-4.0"
  }
}

export const isImageModel = (model: unknown): model is WebGlImageModel =>
  typeof model === "string" && (WEBGL_IMAGE_MODELS as ReadonlyArray<string>).includes(model)
