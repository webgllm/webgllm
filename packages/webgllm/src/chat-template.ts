/**
 * Which chat template a model renders, chosen one way for the resolver
 * (which reads it for capabilities) and the tokenizer (which renders it):
 * chat_template.jinja when there is one, else tokenizer_config.json's
 * `chat_template`, which is a string or, as Transformers allows, a list of
 * named templates: the one named "default", else the only one.
 */
export function chatTemplateOf(config: Record<string, unknown> | undefined, file?: string): string | undefined {
  if (file !== undefined) return file
  const template = config?.chat_template
  if (typeof template === "string") return template
  if (Array.isArray(template)) {
    const named = template.filter(
      (t): t is { name?: unknown; template: string } => !!t && typeof t === "object" && typeof (t as { template?: unknown }).template === "string"
    )
    return (named.find((t) => t.name === "default") ?? (named.length === 1 ? named[0] : undefined))?.template
  }
  return undefined
}
