import type { Toolkit } from '@assistant-ui/react';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { useEffect, useState } from 'react';
import { ToolFallback } from './tool-fallback';

/** Exa's hosted MCP server: it allows browser calls and needs no key. */
const EXA_URL = 'https://mcp.exa.ai/mcp';

/** The Exa tools the demo offers; small models pick better from few. */
const TOOLS = new Set(['web_search_exa', 'web_fetch_exa']);

/** A tool result stays small enough for an 8K context. */
const MAX_RESULT = 6000;

type ToolResult = { content?: Array<{ type: string; text?: string }>; isError?: boolean };

const resultText = (result: ToolResult) => {
  const text = (result.content ?? [])
    .map((part) => (part.type === 'text' ? (part.text ?? '') : `[${part.type}]`))
    .join('\n\n');
  return text.length > MAX_RESULT ? `${text.slice(0, MAX_RESULT)}\n\n[truncated]` : text;
};

let connection: Promise<Toolkit> | undefined;

/** Connects to Exa once and turns its tools into frontend tools. */
function connect(): Promise<Toolkit> {
  connection ??= (async () => {
    const client = new Client({ name: 'webgllm-demo', version: '1.0.0' });
    await client.connect(new StreamableHTTPClientTransport(new URL(EXA_URL)));
    const { tools } = await client.listTools();
    const toolkit: Toolkit = {};
    for (const tool of tools) {
      if (!TOOLS.has(tool.name)) continue;
      toolkit[tool.name] = {
        type: 'frontend',
        description: tool.description ?? tool.name,
        parameters: tool.inputSchema,
        execute: async (args: Record<string, unknown>, { abortSignal }: { abortSignal: AbortSignal }) => {
          const result = (await client.callTool({ name: tool.name, arguments: args }, undefined, {
            signal: abortSignal,
          })) as ToolResult;
          const text = resultText(result);
          if (result.isError) throw new Error(text || `${tool.name} failed`);
          return text;
        },
        render: ToolFallback,
      };
    }
    return toolkit;
  })();
  connection.catch(() => {
    connection = undefined;
  });
  return connection;
}

/** Exa's tools once connected; empty until then, or if Exa can't be reached. */
export function useExaToolkit(): { toolkit: Toolkit; error?: string } {
  const [state, setState] = useState<{ toolkit: Toolkit; error?: string }>({ toolkit: {} });
  useEffect(() => {
    let live = true;
    connect().then(
      (toolkit) => live && setState({ toolkit }),
      (error: unknown) => live && setState({ toolkit: {}, error: error instanceof Error ? error.message : String(error) }),
    );
    return () => {
      live = false;
    };
  }, []);
  return state;
}
