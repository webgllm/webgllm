/**
 * How the worker reads a prompt into the engine: in forward batches from
 * where the cache stops sharing it, with a snapshot after each batch so a
 * later prompt can resume there (and after the last, the last token's
 * hidden row, for a retry: see Engine.resume). The bench pages read
 * prompts through this too, so they measure what the worker does.
 */
import type { Engine } from "./engine.js"

/**
 * Reads prompt[start..] into `engine` (whose cache holds prompt[..start]),
 * appending each batch to `cached` once it's read. Stops after a batch if
 * `stop` says so. Returns the last batch's row of the last token read
 * (see Engine.forwardHidden).
 */
export async function readPrompt(
  engine: Engine,
  prompt: Array<number>,
  start: number,
  cached: Array<number>,
  stop: () => boolean = () => false
): Promise<number> {
  let row = 0
  let i = start
  for (const size of engine.batchSizes(prompt.length - start)) {
    const batch = prompt.slice(i, i + size)
    i += size
    row = await engine.forwardHidden(batch)
    cached.push(...batch)
    engine.snapshot(i === prompt.length ? row : undefined)
    if (stop()) break
  }
  return row
}
