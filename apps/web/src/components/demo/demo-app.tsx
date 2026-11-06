import {
  AssistantRuntimeProvider,
  AuiConfig,
  createSimpleTitleAdapter,
  Suggestions,
  Tools,
  useLocalRuntime,
  useRemoteThreadListRuntime,
  WebSpeechDictationAdapter,
  WebSpeechSynthesisAdapter,
} from '@assistant-ui/react';
import { createLocalStorageAdapter, type AsyncStorageLike } from '@assistant-ui/core/react';
import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import { Popover, PopoverClose, PopoverContent, PopoverTrigger } from 'fumadocs-ui/components/ui/popover';
import { Check, ChevronDown, PanelLeft, X } from 'lucide-react';
import { useMemo, useState, useSyncExternalStore, type FC } from 'react';
import { cn } from '@/lib/cn';
import { useExaToolkit } from './exa';
import { IconButton } from './icon-button';
import { chatModel, MODELS, modelStatus, selectedModel } from './model';
import { Thread } from './thread';
import { ThreadList } from './thread-list';

const chat = chatModel(selectedModel.get);

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
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <AssistantRuntimeProvider runtime={runtime} config={config}>
      <div className="relative flex h-[calc(100svh-3.5rem)] min-h-0 w-full overflow-hidden">
        <aside
          className={cn(
            'absolute inset-y-0 start-0 z-30 flex w-64 shrink-0 flex-col border-e bg-fd-card transition-transform md:static md:translate-x-0',
            sidebarOpen ? 'translate-x-0' : '-translate-x-full',
          )}
        >
          <div className="flex h-12 items-center justify-between border-b px-3">
            <span className="text-sm font-medium">Chats</span>
            <IconButton label="Close sidebar" className="md:hidden" onClick={() => setSidebarOpen(false)}>
              <X />
            </IconButton>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            <ThreadList />
          </div>
          <p className="border-t px-3 py-2 text-xs text-fd-muted-foreground">
            Chats are saved in this browser.
          </p>
        </aside>
        {sidebarOpen && (
          <button
            type="button"
            aria-label="Close sidebar"
            className="absolute inset-0 z-20 bg-fd-background/60 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
        <main className="flex min-w-0 flex-1 flex-col">
          <header className="flex h-12 shrink-0 items-center gap-2 border-b px-3">
            <IconButton label="Open sidebar" size="icon-sm" className="md:hidden" onClick={() => setSidebarOpen(true)}>
              <PanelLeft />
            </IconButton>
            <ModelPicker />
            <ModelStatusLine />
          </header>
          <div className="min-h-0 flex-1">
            <Thread chatId={chatId} />
          </div>
        </main>
      </div>
    </AssistantRuntimeProvider>
  );
}

const ModelPicker: FC = () => {
  const model = useSyncExternalStore(selectedModel.subscribe, selectedModel.get, selectedModel.get);
  const label = MODELS.find((entry) => entry.id === model)?.label ?? model;
  return (
    <Popover>
      <PopoverTrigger className={cn(buttonVariants({ variant: 'ghost', size: 'sm' }), 'gap-1.5 text-sm')}>
        {label}
        <ChevronDown className="size-3.5 text-fd-muted-foreground" />
      </PopoverTrigger>
      <PopoverContent align="start" className="flex w-72 flex-col p-1">
        {MODELS.map((entry) => (
          <PopoverClose
            key={entry.id}
            onClick={() => selectedModel.set(entry.id)}
            className="flex items-start gap-2 rounded-lg px-2.5 py-2 text-start text-sm hover:bg-fd-accent hover:text-fd-accent-foreground"
          >
            <Check className={cn('mt-0.5 size-4 shrink-0', entry.id !== model && 'invisible')} />
            <span className="flex min-w-0 flex-col">
              <span className="font-medium">{entry.label}</span>
              <code className="truncate text-xs text-fd-muted-foreground">{entry.id}</code>
            </span>
          </PopoverClose>
        ))}
      </PopoverContent>
    </Popover>
  );
};

const ModelStatusLine: FC = () => {
  const status = useSyncExternalStore(modelStatus.subscribe, modelStatus.get, modelStatus.get);
  if (status.state === 'idle') {
    return <span className="truncate text-xs text-fd-muted-foreground">Loads with your first message</span>;
  }
  if (status.state === 'error') {
    return <span className="truncate text-xs text-red-500">{status.message}</span>;
  }
  if (status.state === 'ready') {
    return <span className="truncate text-xs text-fd-muted-foreground">Running on your device</span>;
  }
  const percent = Math.round(status.progress * 100);
  return (
    <span className="flex min-w-0 items-center gap-2 text-xs text-fd-muted-foreground">
      <span className="h-1.5 w-24 shrink-0 overflow-hidden rounded-full bg-fd-muted">
        <span className="block h-full rounded-full bg-fd-primary transition-[width]" style={{ width: `${percent}%` }} />
      </span>
      <span className="truncate">
        {status.state === 'downloading' ? 'Downloading' : 'Loading'} {percent}%
      </span>
    </span>
  );
};
