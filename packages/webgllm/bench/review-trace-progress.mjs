// Additional diagnostic packets in the isolated review worker only.
export function instrumentReviewProgress(source){
 if(source.includes('const reviewProgress ='))throw Error('Review progress already instrumented.');
 const replace=(before,after)=>{if(!source.includes(before))throw Error('Review progress anchor changed: '+before);source=source.replace(before,after)};
 replace('  const { id } = request','  const { id } = request\n  const reviewProgress = (stage: string, detail: Record<string, unknown> = {}) => post({ type: "review-progress", id, stage, at: performance.now(), detail })\n  reviewProgress("load-before")');
 replace('  const model = await startLoad(request.model, request.contextLength)','  const model = await startLoad(request.model, request.contextLength)\n  reviewProgress("load-after")');
 replace('  const tokenizeAt = performance.now()','  reviewProgress("tokenize-before")\n  const tokenizeAt = performance.now()');
 replace('  const tokenizeMs = performance.now() - tokenizeAt','  const tokenizeMs = performance.now() - tokenizeAt\n  reviewProgress("tokenize-after", { promptTokens: prompt.length })');
 replace('  const schemaAt = performance.now()','  reviewProgress("schema-before", { structured: request.outputSchema !== undefined })\n  const schemaAt = performance.now()\n  const reviewConstraintTokenizer = request.outputSchema === undefined ? undefined : (() => {\n    reviewProgress("schema-tokenizer-before")\n    const value = tokenizer.constraintTokenizer\n    reviewProgress("schema-tokenizer-after")\n    return value\n  })()');
 replace('await createConstraint({ schema: request.outputSchema, tokenizer: tokenizer.constraintTokenizer })','await createConstraint({ schema: request.outputSchema, tokenizer: reviewConstraintTokenizer! })');
 replace('  const schemaMs = performance.now() - schemaAt','  const schemaMs = performance.now() - schemaAt\n  reviewProgress("schema-after", { schemaMs })');
 replace('  const promptStart = performance.now()','  reviewProgress("prefill-before", { start, promptTokens: prompt.length })\n  const promptStart = performance.now()');
 replace('  const stats: RunTimings = {','  reviewProgress("prefill-after", { cached: model.cached.length })\n  const stats: RunTimings = {');
 replace('  const updateAllowed = () => {','  const updateAllowed = () => {\n    reviewProgress("mask-before", { structured: !!structured })');
 replace('    appliedMask = mask\n    return mask ?? undefined','    appliedMask = mask\n    reviewProgress("mask-after", { structured: !!structured })\n    return mask ?? undefined');
 replace('  let token = openThinking ?','  reviewProgress("generating-before")\n  let token = openThinking ?');
 replace('    stats.outputTokens++\n    structured?.observe(t)','    stats.outputTokens++\n    reviewProgress("token-selected", { tokenId: t, acceptedTokenCount: reviewAcceptedTokenIds.length })\n    structured?.observe(t)');
 replace('    if (t === eos) return false','    if (t === eos) {\n      reviewProgress("token", { tokenId: t, acceptedTokenCount: reviewAcceptedTokenIds.length, eos: true, piece: "" })\n      return false\n    }');
 replace('    reviewPushes.push({ id: t, piece: text })','    reviewPushes.push({ id: t, piece: text })\n    reviewProgress("token", { tokenId: t, acceptedTokenCount: reviewAcceptedTokenIds.length, piece: text })');
 return source;
}
