import type { ChatModelAdapter, ThreadMessage, ToolCallMessagePart } from '@assistant-ui/react';
import type { ChatCompletionMessageParam, ChatCompletionTool } from 'webgllm';
import { WebGLLM } from 'webgllm';

/** The models the demo offers: regular GGUF builds, each with tools. */
export const MODELS = [
  { id: 'LiquidAI/LFM3-1.7B-GGUF', label: 'LFM3 1.7B' },
  { id: 'unsloth/Qwen3.5-2B-GGUF', label: 'Qwen3.5 2B' },
  { id: 'LiquidAI/LFM2.5-1.2B-Instruct-GGUF', label: 'LFM2.5 1.2B' },
  { id: 'LiquidAI/LFM2.5-350M-GGUF', label: 'LFM2.5 350M' },
] as const;

export type ModelId = (typeof MODELS)[number]['id'];

/** Where the selected model is: loading it shows here, as the chat waits for it. */
export type ModelStatus =
  | { state: 'idle' }
  | { state: 'downloading' | 'loading'; progress: number }
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

let selected: ModelId = MODELS[0].id;
const selectionListeners = new Set<() => void>();

/** The model new replies use, picked in the header. */
export const selectedModel = {
  get: () => selected,
  set: (model: ModelId) => {
    selected = model;
    for (const listener of selectionListeners) listener();
  },
  subscribe: (listener: () => void) => {
    selectionListeners.add(listener);
    return () => selectionListeners.delete(listener);
  },
};

let client: WebGLLM | undefined;

/** One WebGLLM for the page, made on first use (it starts workers). */
export function webgllm(): WebGLLM {
  client ??= new WebGLLM({
    contextLength: 8192,
    onProgress: ({ phase, progress }) =>
      setStatus({ state: phase === 'download' ? 'downloading' : 'loading', progress }),
  });
  return client;
}

/** Loads `model` ahead of the first message, so its download shows at once. */
export function preload(model: ModelId) {
  setStatus({ state: 'loading', progress: 0 });
  webgllm()
    .loadModel(model)
    .then(
      () => setStatus({ state: 'ready' }),
      (error: unknown) => setStatus({ state: 'error', message: error instanceof Error ? error.message : String(error) }),
    );
}

type ReadonlyJSONObject = ToolCallMessagePart['args'];

const text = (message: ThreadMessage) =>
  message.content
    .filter((part) => part.type === 'text')
    .map((part) => (part as { text: string }).text)
    .join('\n\n');

/** The thread as OpenAI chat messages, tool calls and their results included. */
function toOpenAI(messages: readonly ThreadMessage[]): ChatCompletionMessageParam[] {
  const out: ChatCompletionMessageParam[] = [];
  for (const message of messages) {
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
export function chatModel(getModel: () => ModelId): ChatModelAdapter {
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
        const stream = await webgllm().chat.completions.create(
          {
            model: getModel(),
            messages: [
              ...(context.system ? [{ role: 'system' as const, content: context.system }] : []),
              ...toOpenAI(thread),
            ],
            ...(tools.length > 0 && { tools }),
            stream: true,
          },
          { signal: abortSignal },
        );
        setStatus({ state: 'ready' });
        for await (const chunk of stream) {
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
        if (calls.length === 0) return;
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
          metadata: { steps: [{}] },
        };
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        throw error;
      }
    },
  };
}
