import type { AttachmentAdapter, ChatModelAdapter, PendingAttachment, ThreadMessage, ToolCallMessagePart } from '@assistant-ui/react';
import { useSyncExternalStore } from 'react';
import type {
  ChatCompletionContentPartImage,
  ChatCompletionContentPartText,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  CompletionTimings,
  CompletionUsage,
  ReasoningEffort,
} from 'webgllm';
import { WEBGL_MODEL_META, WebGLLM, type WebGlModelMeta } from 'webgllm';
import type { ModelOption, ModelSelectorEffortOption } from './model-selection';

/**
 * The models the demo offers: regular builds, each with tools. Those with a
 * vision encoder also read images: Qwen3.5 2B's ONNX build, and Liquid AI's
 * vision-language models (LFM3-VL and LFM2.5-VL, in their GGUF builds).
 */
const MODEL_NAMES = [
  ['LiquidAI/LFM3-1.7B-GGUF', 'LFM3 1.7B'],
  ['unsloth/Qwen3.5-2B-GGUF', 'Qwen3.5 2B'],
  ['onnx-community/Qwen3.5-2B-ONNX', 'Qwen3.5 2B ONNX'],
  ['LiquidAI/LFM2.5-1.2B-Instruct-GGUF', 'LFM2.5 1.2B Instruct'],
  ['LiquidAI/LFM2.5-1.2B-Thinking-GGUF', 'LFM2.5 1.2B Thinking'],
  ['LiquidAI/LFM2.5-350M-GGUF', 'LFM2.5 350M'],
  ['LiquidAI/LFM3-VL-2B-GGUF', 'LFM3-VL 2B'],
  ['LiquidAI/LFM2.5-VL-3B-GGUF', 'LFM2.5-VL 3B'],
  ['LiquidAI/LFM2.5-VL-1.6B-GGUF', 'LFM2.5-VL 1.6B'],
  ['LiquidAI/LFM2.5-VL-450M-GGUF', 'LFM2.5-VL 450M'],
] as const;

/**
 * webgllm's thinking is on or off ("none" turns it off, any other level on),
 * so a model that can switch it off offers those two levels.
 */
const THINKING_LEVELS: ModelSelectorEffortOption[] = [
  { id: 'none', name: 'Off' },
  { id: 'high', name: 'High' },
];

/** The models, as assistant-ui's model selector takes them. */
export const MODELS: ModelOption[] = MODEL_NAMES.map(([id, name]) => {
  const reasoning = WEBGL_MODEL_META[id].reasoning;
  const switchable = reasoning !== undefined && 'off' in reasoning.levels;
  return { id, name, description: id, ...(switchable && { efforts: THINKING_LEVELS }) };
});

export type ModelId = (typeof MODEL_NAMES)[number][0];

/** What a model reads besides text, from webgllm's meta (its files decide it). */
export type MediaInput = Exclude<WebGlModelMeta['supports']['input'][number], 'text'>;
export const mediaInputs = (model: string): MediaInput[] =>
  ((WEBGL_MODEL_META[model as ModelId]?.supports.input ?? []) as WebGlModelMeta['supports']['input']).filter(
    (input): input is MediaInput => input !== 'text',
  );

/** The model picked in the composer (set by its model selector). */
let selected: string = MODELS[0]!.id;
const selectedListeners = new Set<() => void>();
export const selectedModel = {
  get: () => selected,
  set: (model: string) => {
    selected = model;
    for (const listener of selectedListeners) listener();
  },
  subscribe: (listener: () => void) => {
    selectedListeners.add(listener);
    return () => selectedListeners.delete(listener);
  },
};

/** What the selected model reads besides text: the composer offers attachments for these. */
export function useMediaInputs(): MediaInput[] {
  const model = useSyncExternalStore(selectedModel.subscribe, selectedModel.get, () => MODELS[0]!.id);
  return mediaInputs(model);
}

/** The longest side images are sent at: webgllm's largest pixel budget ("high", 1024²) reads no more. */
const MAX_IMAGE_SIDE = 1024;

/** An image file as a data URL, scaled down to MAX_IMAGE_SIDE (saved chats keep it, in localStorage). */
async function imageDataUrl(file: File): Promise<string> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#fff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return canvas.toDataURL('image/jpeg', 0.9);
}

/**
 * assistant-ui's attachments for what the selected model reads: images for
 * a model with a vision encoder; nothing for the others (their composer
 * shows no attach button, and a pasted or dropped file is refused).
 */
export class MediaAttachmentAdapter implements AttachmentAdapter {
  get accept() {
    return mediaInputs(selectedModel.get()).includes('image') ? 'image/*' : '';
  }

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    if (!file.type.startsWith('image/') || !mediaInputs(selectedModel.get()).includes('image'))
      throw new Error(`${selectedModel.get()} doesn't read this kind of file.`);
    return {
      id: crypto.randomUUID(),
      type: 'image',
      name: file.name,
      contentType: file.type,
      file,
      status: { type: 'requires-action', reason: 'composer-send' },
    };
  }

  async send(attachment: PendingAttachment) {
    return {
      ...attachment,
      status: { type: 'complete' as const },
      content: [{ type: 'image' as const, image: await imageDataUrl(attachment.file) }],
    };
  }

  async remove() {}
}

/** Where the selected model is: loading it shows here, as the chat waits for it. */
export type ModelStatus =
  | { state: 'idle' }
  | {
      state: 'downloading' | 'loading';
      model: string;
      /** webgllm's progress across both phases: downloading takes 0–0.9, loading the rest. */
      progress: number;
      /** The files were already stored, so loading runs across 0–1. */
      cached: boolean;
    }
  | { state: 'ready' }
  | { state: 'error'; message: string };

let status: ModelStatus = { state: 'idle' };
const listeners = new Set<() => void>();
const setStatus = (next: ModelStatus) => {
  status = next;
  for (const listener of listeners) listener();
};

export const modelStatus = {
  get: () => status,
  subscribe: (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

let client: WebGLLM | undefined;

/** One WebGLLM for the page, made on first use (it starts workers). */
export function webgllm(): WebGLLM {
  client ??= new WebGLLM({
    contextLength: 8192,
    onProgress: ({ model, phase, progress, cached }) =>
      setStatus({ state: phase === 'download' ? 'downloading' : 'loading', model, progress, cached }),
  });
  return client;
}

type ReadonlyJSONObject = ToolCallMessagePart['args'];

const text = (message: ThreadMessage) =>
  message.content
    .filter((part) => part.type === 'text')
    .map((part) => (part as { text: string }).text)
    .join('\n\n');

/** A message's images (its own and its attachments'), as data URLs. */
const images = (message: ThreadMessage): string[] =>
  [...message.content, ...(message.attachments ?? []).flatMap((a) => a.content ?? [])].flatMap((part) =>
    part.type === 'image' ? [(part as { image: string }).image] : [],
  );

/**
 * The thread as OpenAI chat messages, tool calls and their results
 * included; user messages' images as image_url parts where the model reads
 * them (a model that doesn't gets their text only).
 */
function toOpenAI(messages: readonly ThreadMessage[], readsImages: boolean): ChatCompletionMessageParam[] {
  const out: ChatCompletionMessageParam[] = [];
  for (const message of messages) {
    if (message.role === 'user' && readsImages && images(message).length > 0) {
      const parts: Array<ChatCompletionContentPartImage | ChatCompletionContentPartText> = images(message).map((url) => ({
        type: 'image_url',
        image_url: { url },
      }));
      if (text(message)) parts.push({ type: 'text', text: text(message) });
      out.push({ role: 'user', content: parts });
      continue;
    }
    if (message.role !== 'assistant') {
      out.push({ role: message.role, content: text(message) });
      continue;
    }
    const calls = message.content.filter((part): part is ToolCallMessagePart => part.type === 'tool-call');
    out.push({
      role: 'assistant',
      content: text(message),
      ...(calls.length > 0 && {
        tool_calls: calls.map((call) => ({
          id: call.toolCallId,
          type: 'function' as const,
          function: { name: call.toolName, arguments: call.argsText || JSON.stringify(call.args) },
        })),
      }),
    });
    for (const call of calls) {
      if (call.result === undefined) continue;
      out.push({
        role: 'tool',
        tool_call_id: call.toolCallId,
        content: typeof call.result === 'string' ? call.result : JSON.stringify(call.result),
      });
    }
  }
  return out;
}

/** The pages a web tool's result lists, from its "Title:" and "URL:" lines. */
function resultSources(text: string): Array<{ url: string; title?: string }> {
  const out: Array<{ url: string; title?: string }> = [];
  for (const match of text.matchAll(/^(?:Title: (.*)\n)?URL: (https?:\/\/\S+)/gm)) {
    const title = match[1]?.trim();
    out.push({ url: match[2]!, ...(title && title !== 'N/A' && { title }) });
  }
  return out;
}

/** Source parts for the pages the reply's tool calls returned, once each. */
function replySources(message: ThreadMessage) {
  const seen = new Set<string>();
  return message.content.flatMap((part) =>
    part.type !== 'tool-call' || part.isError || typeof part.result !== 'string'
      ? []
      : resultSources(part.result)
          .filter(({ url }) => !seen.has(url) && Boolean(seen.add(url)))
          .map(({ url, title }) => ({ type: 'source' as const, sourceType: 'url' as const, id: url, url, ...(title && { title }) })),
  );
}

/** A reply's prompt, cache and token counts, from webgllm's usage and timings. */
export type ReplyStats = {
  promptTokensPerSecond?: number;
  promptTokens?: number;
  cachedTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
  totalTokens?: number;
};

const parse = (json: string): ReadonlyJSONObject => {
  try {
    const value: unknown = JSON.parse(json);
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as ReadonlyJSONObject) : {};
  } catch {
    return {};
  }
};

/**
 * assistant-ui's chat model: calls webgllm's OpenAI-style API in the page,
 * with the thread's tools, and streams the reply's reasoning, text and tool
 * calls. Each yield is the whole reply so far.
 */
export function chatModel(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, context, unstable_getMessage }) {
      // `messages` ends at the user's message. After a tool call, the reply
      // so far (its calls and their results) is only in the message being
      // written, so it goes to the model too.
      const current = unstable_getMessage();
      const thread = current.content.length > 0 ? [...messages, current] : messages;
      const tools: ChatCompletionTool[] = Object.entries(context.tools ?? {})
        .filter(([, tool]) => !tool.disabled)
        .map(([name, tool]) => ({
          type: 'function',
          function: {
            name,
            ...(tool.description && { description: tool.description }),
            parameters: (tool.parameters ?? { type: 'object', properties: {} }) as Record<string, unknown>,
          },
        }));
      let reasoning = '';
      let answer = '';
      const calls: Array<{ id: string; name: string; args: string }> = [];
      const content = () => [
        ...(reasoning ? [{ type: 'reasoning' as const, text: reasoning }] : []),
        ...(answer ? [{ type: 'text' as const, text: answer }] : []),
        ...calls.map((call) => ({
          type: 'tool-call' as const,
          toolCallId: call.id,
          toolName: call.name,
          argsText: call.args,
          args: parse(call.args),
        })),
      ];
      try {
        // Set by assistant-ui's model selector through the model context.
        const model = context.config?.modelName ?? MODELS[0]!.id;
        const stream = await webgllm().chat.completions.create(
          {
            model,
            ...(context.config?.reasoningEffort && {
              reasoning_effort: context.config.reasoningEffort as ReasoningEffort,
            }),
            messages: [
              ...(context.system ? [{ role: 'system' as const, content: context.system }] : []),
              ...toOpenAI(thread, mediaInputs(model).includes('image')),
            ],
            ...(tools.length > 0 && { tools }),
            stream: true,
            stream_options: { include_usage: true },
          },
          { signal: abortSignal },
        );
        setStatus({ state: 'ready' });
        // Timed from here, once the model is loaded, so a download doesn't count.
        const streamStartTime = Date.now();
        let totalChunks = 0;
        let usage: CompletionUsage | undefined;
        let timings: CompletionTimings | undefined;
        for await (const chunk of stream) {
          totalChunks++;
          usage = chunk.usage ?? usage;
          timings = chunk.timings ?? timings;
          const delta = chunk.choices[0]?.delta;
          if (!delta) continue;
          if (delta.reasoning_content) reasoning += delta.reasoning_content;
          if (delta.content) answer += delta.content;
          for (const call of delta.tool_calls ?? []) {
            const entry = (calls[call.index] ??= { id: '', name: '', args: '' });
            if (call.id) entry.id = call.id;
            if (call.function.name) entry.name += call.function.name;
            if (call.function.arguments) entry.args += call.function.arguments;
          }
          yield { content: content() };
        }
        // assistant-ui's message timing, from what webgllm measured.
        const timing = {
          streamStartTime,
          totalStreamTime: Date.now() - streamStartTime,
          ...(timings?.first_token_ms != null && { firstTokenTime: timings.first_token_ms }),
          ...(usage && { tokenCount: usage.completion_tokens }),
          ...(timings && { tokensPerSecond: timings.decode_tokens_per_second }),
          totalChunks,
          toolCallCount: calls.length,
        };
        // What assistant-ui's timing has no fields for, for the stats badge.
        const stats: ReplyStats = {
          ...(timings && { promptTokensPerSecond: timings.prompt_tokens_per_second }),
          ...(usage && {
            promptTokens: usage.prompt_tokens,
            cachedTokens: usage.prompt_tokens_details.cached_tokens,
            outputTokens: usage.completion_tokens,
            reasoningTokens: usage.completion_tokens_details.reasoning_tokens,
            totalTokens: usage.total_tokens,
          }),
        };
        if (calls.length === 0) {
          // The answer: the pages its web searches found go under it as
          // source parts, which assistant-ui renders as chips.
          yield { content: [...content(), ...replySources(current)], metadata: { timing, custom: { stats } } };
          return;
        }
        // assistant-ui's local runtime doesn't run tools itself: run each
        // call's tool here, then hand back. With every call answered, the
        // runtime calls the model again with the results (up to maxSteps).
        const results = await Promise.all(
          calls.map(async (call) => {
            const tool = context.tools?.[call.name];
            if (!tool?.execute) return { result: `Unknown tool: ${call.name}`, isError: true };
            try {
              const result: unknown = await tool.execute(parse(call.args), {
                toolCallId: call.id,
                abortSignal,
                human: () => Promise.reject(new Error('This tool cannot ask for input')),
              });
              return { result, isError: false };
            } catch (error) {
              if (abortSignal.aborted) throw error;
              return { result: error instanceof Error ? error.message : String(error), isError: true };
            }
          }),
        );
        yield {
          content: content().map((part) => {
            if (part.type !== 'tool-call') return part;
            const index = calls.findIndex((call) => call.id === part.toolCallId);
            return { ...part, ...results[index] };
          }),
          status: { type: 'requires-action', reason: 'tool-calls' },
          metadata: { steps: [{}], timing, custom: { stats } },
        };
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        // assistant-ui keeps an error with a string code and message as is,
        // and saves it with the chat. An Error's message doesn't survive
        // JSON, so a reload would show "[object Object]": hand it a plain one.
        if (error instanceof Error) {
          throw { code: 'code' in error && typeof error.code === 'string' ? error.code : 'unknown', message: error.message };
        }
        throw error;
      }
    },
  };
}
