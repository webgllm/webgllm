import {
  AssistantRuntimeProvider,
  AuiConfig,
  useAuiState,
  type AssistantState,
  createSimpleTitleAdapter,
  Suggestions,
  Tools,
  useLocalRuntime,
  useAuiEvent,
  useRemoteThreadListRuntime,
  WebSpeechDictationAdapter,
  WebSpeechSynthesisAdapter,
} from '@assistant-ui/react';
import { createLocalStorageAdapter, type AsyncStorageLike } from '@assistant-ui/core/react';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { useSidebar } from 'fumadocs-ui/layouts/docs/slots/sidebar';
import { useMemo, type FC } from 'react';
import { baseOptions } from '@/lib/layout.shared';
import { useExaToolkit } from './exa';
import { chatModel, MODELS } from './model';
import { ModelProgress } from './model-progress';
import { ModelSelector } from './model-selector';
import { Thread } from './thread';
import { ThreadList } from './thread-list';

const chat = chatModel();

const suggestions = [
  { title: 'Search the web', label: 'for the latest WebGL news', prompt: 'Search the web for the latest WebGL news and summarize it.' },
  { title: 'Explain', label: 'how attention works in transformers', prompt: 'Explain how attention works in transformers, briefly.' },
  { title: 'Write code', label: 'a debounce function in TypeScript', prompt: 'Write a debounce function in TypeScript.' },
  { title: 'Plan', label: 'a weekend in Lisbon', prompt: 'Plan a weekend in Lisbon for someone who likes food and walking.' },
];

/** The browser's localStorage, as assistant-ui's async storage. */
const browserStorage: AsyncStorageLike = {
  getItem: async (key) => localStorage.getItem(key),
  setItem: async (key, value) => localStorage.setItem(key, value),
  removeItem: async (key) => localStorage.removeItem(key),
};

/** One local runtime per chat; chats and their messages are kept in localStorage. */
function useDemoRuntime(chatId: string | undefined, onChatChange: (chatId: string | undefined) => void) {
  const adapter = useMemo(
    () =>
      createLocalStorageAdapter({
        storage: browserStorage,
        prefix: 'webgllm-demo:',
        titleGenerator: createSimpleTitleAdapter(),
      }),
    [],
  );
  const adapters = useMemo(
    () => ({
      speech: new WebSpeechSynthesisAdapter(),
      // Checked in the browser only: the page also renders on the server.
      ...(typeof window !== 'undefined' &&
        WebSpeechDictationAdapter.isSupported() && { dictation: new WebSpeechDictationAdapter() }),
    }),
    [],
  );
  return useRemoteThreadListRuntime({
    runtimeHook: () => useLocalRuntime(chat, { maxSteps: 5, adapters }),
    adapter,
    threadId: chatId,
    onThreadIdChange: onChatChange,
  });
}

/**
 * The chat in the URL isn't on screen yet: the server can't read the saved
 * chats, and the browser loads them after it starts. The thread shows its
 * loading skeleton meanwhile, not the welcome. A chat that no longer exists
 * shows the welcome once the list has loaded.
 */
const isRestoring = (s: AssistantState, chatId: string | undefined) => {
  if (chatId === undefined) return false;
  if (s.threads.isLoading) return true;
  const items = s.threads.threadItems;
  if (!items.some((item) => item.remoteId === chatId)) return false;
  const main = items.find((item) => item.id === s.threads.mainThreadId);
  return main?.remoteId !== chatId || s.thread.isLoading;
};

/** assistant-ui's model selector, in the composer's slot for it. */
// Thinking starts High, so the selector shows a level from the start.
const ComposerModelSelector: FC = () => (
  <ModelSelector models={MODELS} defaultEffort="high" variant="ghost" size="sm" />
);

const threadComponents = { ComposerExtra: ComposerModelSelector, Indicator: ModelProgress };

/**
 * assistant-ui's starter layout (templates/default/app/assistant.tsx): the
 * thread list in a sidebar beside a header and the thread. The sidebar is
 * Fumadocs' docs sidebar, where the template has shadcn's.
 */
export function DemoApp({
  chatId,
  onChatChange,
}: {
  /** The chat to show; a new one when left out. */
  chatId?: string;
  onChatChange: (chatId: string | undefined) => void;
}) {
  const runtime = useDemoRuntime(chatId, onChatChange);
  const { toolkit } = useExaToolkit();
  const config = useMemo(() => AuiConfig({ tools: Tools({ toolkit }), suggestions: Suggestions(suggestions) }), [toolkit]);

  return (
    <AssistantRuntimeProvider runtime={runtime} config={config}>
      <DocsLayout
        {...baseOptions()}
        links={[{ text: 'Docs', url: '/docs', active: 'nested-url' }]}
        tree={{ name: 'Chats', children: [] }}
        sidebar={{ banner: <ThreadList /> }}
      >
        <CloseSidebarOnSwitch />
        {/* Fumadocs' main area, under its mobile header (its sidebar button opens the chat list there). */}
        <div className="flex h-[calc(var(--fd-docs-height)-var(--fd-header-height))] w-full min-w-0 flex-col [grid-area:main]">
          {/* On phones, Fumadocs' own header stands in for this one. */}
          <header className="hidden h-16 shrink-0 items-center gap-2 border-b px-4 md:flex">
            <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1.5 text-sm text-fd-muted-foreground">
              <a href="/" className="hidden hover:text-fd-foreground md:block">
                WebGLLM
              </a>
              <span className="hidden md:block">/</span>
              <span className="font-medium text-fd-foreground">Demo</span>
            </nav>
          </header>
          <div className="flex-1 overflow-hidden">
            <DemoThread chatId={chatId} />
          </div>
        </div>
      </DocsLayout>
    </AssistantRuntimeProvider>
  );
}

/** Their thread, with its loading skeleton while the chat in the URL comes back from storage. */
const DemoThread: FC<{ chatId: string | undefined }> = ({ chatId }) => {
  const restoring = useAuiState((s) => isRestoring(s, chatId));
  return <Thread components={threadComponents} autoFocus={false} loading={restoring} />;
};

/** Closes Fumadocs' mobile drawer once a chat is picked or started from it. */
const CloseSidebarOnSwitch: FC = () => {
  const { setOpen } = useSidebar();
  useAuiEvent('threads.selectionChanged', () => setOpen(false));
  return null;
};
