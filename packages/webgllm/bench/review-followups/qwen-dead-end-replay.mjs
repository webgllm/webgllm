/** CPU-only matcher replay: no inference, sampling, prompt cache, or environment reads.
 * Usage: node qwen-dead-end-replay.mjs /absolute/path/to/Qwen/tokenizer.json
 * The exact accepted IDs come from an INVALID native harness; the isolated replay
 * independently verifies this mechanism, not attribution of a historical failure.
 */
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire(new URL('../../../structured-output/package.json', import.meta.url));
const { loadJsonGuidance } = await import(pathToFileURL(require.resolve('llguidance/json-schema')));
const file = process.argv[2];
if (!file) throw Error('Pass the exact public Qwen tokenizer.json path.');
const bytes = await readFile(file), tokenizerJSON = JSON.parse(bytes), runtime = await loadJsonGuidance();
const eos = tokenizerJSON.added_tokens.find(token => token.content === '<|im_end|>').id;
const tokenizer = { tokenizerJSON, eos_token_id: eos, special_token_ids: tokenizerJSON.added_tokens.filter(token => token.special).map(token => token.id) };
const schema = { type: 'object', properties: { city: { type: 'string', enum: ['Paris', 'Tokyo', 'London', 'Berlin'] } }, required: ['city'], additionalProperties: false, 'x-guidance': { whitespace_flexible: false, key_separator: ':', item_separator: ',' } };
const create = () => runtime.createInterpreter({ tokenizer, response_format: { type: 'json_schema', json_schema: schema }, limits: { maxTokens: Number.MAX_SAFE_INTEGER, maxOutputBytes: Number.MAX_SAFE_INTEGER } });
// Diagnostic-only introspection retrieves exact converted bytes without reimplementing its bridge.
const probe = create(), exact = Reflect.get(probe, 'tokenizer'), decoder = new TextDecoder();
const asciiMask = new Uint32Array(Math.ceil(exact.vocabSize() / 32));
for (let id = 0; id < exact.vocabSize(); id++) if (id === eos || (!exact.isSpecial(id) && exact.token(id).every(byte => byte < 128))) asciiMask[id >>> 5] |= 1 << (id & 31);
const allowed = (mask, token) => 'mask' in mask && !!(mask.mask[token >>> 5] & (1 << (token & 31)));
const byteToken = byte => { for (let id = 0; id < 256; id++) if (exact.token(id).length === 1 && exact.token(id)[0] === byte) return id; throw Error('Missing byte token ' + byte); };
const report = { scope: 'Independent CPU mechanism replay; historical native London attribution remains unproven.', tokenizerSha256: createHash('sha256').update(bytes).digest('hex'), vocabSize: exact.vocabSize(), schema, unfiltered: [], asciiIntersectionProof: [], escapedPrefixLimit: [] };
for (const [name, tokens] of [['captured-Paris-prefix', [90, 1, 8656, 763, 65399, 157, 116]], ['London-prefix', [...new TextEncoder().encode('{"city":"L')].map(byteToken).concat([157, 116])]]) {
  const interpreter = create(), steps = [];
  for (const token of tokens) { const mask = interpreter.computeMask(); steps.push({ token, bytes: [...exact.token(token)], piece: decoder.decode(exact.token(token)), allowed: allowed(mask, token) }); if (!allowed(mask, token)) throw Error('Replay token was disallowed'); interpreter.commitToken(token); }
  const result = interpreter.computeMask();
  report.unfiltered.push({ name, steps, result: 'mask' in result ? { mask: true } : result, utf8State: Reflect.get(interpreter, 'engine').state.machine.utf8, maskInstrumentation: interpreter.maskInstrumentation() });
  interpreter.dispose();
}
for (const city of ['Paris', 'Tokyo', 'London', 'Berlin']) {
  const interpreter = create(); let filteredMasks = 0;
  for (const byte of new TextEncoder().encode(JSON.stringify({ city }))) { const raw = interpreter.computeMask(); if (!('mask' in raw)) throw Error('Unexpected stop'); const mask = { mask: raw.mask.map((word, index) => word & asciiMask[index]) }; if (raw.mask.some((word, index) => word !== mask.mask[index])) filteredMasks++; const token = byteToken(byte); if (!allowed(mask, token)) throw Error('Removed a valid continuation'); interpreter.commitToken(token); }
  report.asciiIntersectionProof.push({ city, complete: allowed(interpreter.computeMask(), eos), filteredMasks }); interpreter.dispose();
}
for (const prefix of ['{"city":"P' + String.fromCharCode(92) + 'u00e9', '{"city":"P' + String.fromCharCode(92) + 'u0041']) {
  const interpreter = create(), steps = [];
  for (const token of [...new TextEncoder().encode(prefix)].map(byteToken)) { const mask = interpreter.computeMask(); if (!allowed(mask, token)) throw Error('Escaped replay token disallowed'); interpreter.commitToken(token); steps.push(token); }
  const machine = Reflect.get(interpreter, 'engine').state.machine, value = machine.lexeme.value, mask = interpreter.computeMask();
  const asciiAllowed = 'mask' in mask ? mask.mask.map((word, index) => word & asciiMask[index]) : new Uint32Array();
  report.escapedPrefixLimit.push({ prefix, tokens: steps, decodedValue: value, possibleCityContinuations: schema.properties.city.enum.filter(city => city.startsWith(value)), matcherStillHasMask: 'mask' in mask, asciiFilterStillHasMask: asciiAllowed.some(word => word !== 0), limitation: 'ASCII intersection does not repair escaped-character prefix viability.' });
  interpreter.dispose();
}
probe.dispose();
console.log(JSON.stringify(report, null, 2));
