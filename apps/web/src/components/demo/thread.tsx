import {
  ActionBarMorePrimitive,
  ActionBarPrimitive,
  AuiIf,
  type AssistantState,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  groupPartByType,
  MessagePrimitive,
  SuggestionPrimitive,
  ThreadPrimitive,
  useAuiEvent,
  useAuiState,
} from '@assistant-ui/react';
import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import {
  ArrowDown,
  ArrowUp,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Mic,
  MoreHorizontal,
  Pencil,
  RefreshCw,
  Square,
  Volume2,
} from 'lucide-react';
import type { CSSProperties, FC } from 'react';
import { cn } from '@/lib/cn';
import { IconButton } from './icon-button';
import { MarkdownText } from './markdown-text';
import { Source } from './sources';
import { ReasoningGroup, ToolFallback, ToolGroup } from './tool-ui';

const groupBy = groupPartByType({
  reasoning: ['group-chainOfThought', 'group-reasoning'],
  'tool-call': ['group-chainOfThought', 'group-tool'],
  'standalone-tool-call': [],
  source: ['group-sources'],
});

// The startup placeholder thread counts as a new chat, so the composer mounts centered.
const isNewChatView = (s: AssistantState) =>
  s.thread.messages.length === 0 && (!s.thread.isLoading || s.threads.isLoading);

const menuClass = 'z-50 min-w-40 overflow-hidden rounded-xl border bg-fd-popover p-1 text-fd-popover-foreground shadow-lg';
const menuItemClass =
  'flex cursor-default items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm outline-none select-none data-[highlighted]:bg-fd-accent data-[highlighted]:text-fd-accent-foreground hover:bg-fd-accent hover:text-fd-accent-foreground';

/**
 * The chat in the URL isn't on screen yet: the server can't read the saved
 * chats, and the browser loads them after it starts. Until then the welcome
 * stays hidden, so a reload doesn't flash it. A chat that no longer exists
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

export const Thread: FC<{ chatId?: string }> = ({ chatId }) => {
  const restoring = useAuiState((s) => isRestoring(s, chatId));
  const showWelcome = useAuiState((s) => isNewChatView(s) && !isRestoring(s, chatId));
  const isEmpty = useAuiState(isNewChatView);
  return (
    <ThreadPrimitive.Root
      className="flex h-full min-h-0 flex-col bg-fd-background"
      style={{ '--thread-max-width': '44rem' } as CSSProperties}
    >
      <ThreadPrimitive.Viewport className="relative flex flex-1 flex-col overflow-y-scroll overscroll-contain scroll-smooth">
        <div
          className={cn(
            'mx-auto flex w-full max-w-(--thread-max-width) flex-1 flex-col px-4 pt-4',
            isEmpty && !restoring && 'justify-center',
          )}
        >
          {showWelcome && <ThreadWelcome />}

          <div className="mb-14 flex flex-col gap-y-6 empty:hidden">
            <ThreadPrimitive.Messages>{() => <ThreadMessage />}</ThreadPrimitive.Messages>
          </div>

          <ThreadPrimitive.ViewportFooter
            className={cn(
              'flex flex-col gap-4 overflow-visible bg-fd-background pb-4 md:pb-6',
              (!isEmpty || restoring) && 'sticky bottom-0 mt-auto',
            )}
          >
            <ThreadScrollToBottom />
            <Composer />
            <AuiIf condition={(s) => isNewChatView(s) && s.composer.isEmpty && !isRestoring(s, chatId)}>
              <ThreadSuggestions />
            </AuiIf>
          </ThreadPrimitive.ViewportFooter>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadMessage: FC = () => {
  const role = useAuiState((s) => s.message.role);
  const isEditing = useAuiState((s) => s.message.composer.isEditing);
  if (isEditing) return <EditComposer />;
  if (role === 'user') return <UserMessage />;
  return <AssistantMessage />;
};

const ThreadScrollToBottom: FC = () => (
  <ThreadPrimitive.ScrollToBottom asChild>
    <IconButton
      label="Scroll to bottom"
      variant="outline"
      size="icon-sm"
      className="absolute -top-12 z-10 self-center rounded-full bg-fd-background disabled:invisible"
    >
      <ArrowDown />
    </IconButton>
  </ThreadPrimitive.ScrollToBottom>
);

const ThreadWelcome: FC = () => (
  <div className="mb-6 flex flex-col gap-2 px-2">
    <p className="text-2xl font-semibold tracking-tight">How can I help you today?</p>
    <p className="text-sm text-fd-muted-foreground">
      Runs on your device with WebGL 2. The first message downloads the model; web search goes through Exa.
    </p>
  </div>
);

const ThreadSuggestions: FC = () => (
  <div className="grid w-full gap-2 sm:grid-cols-2">
    <ThreadPrimitive.Suggestions>{() => <ThreadSuggestionItem />}</ThreadPrimitive.Suggestions>
  </div>
);

const ThreadSuggestionItem: FC = () => (
  <SuggestionPrimitive.Trigger send asChild>
    <button
      type="button"
      className="flex w-full flex-col items-start gap-0.5 rounded-xl border bg-fd-card px-4 py-3 text-start text-sm transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground"
    >
      <SuggestionPrimitive.Title className="font-medium" />
      <SuggestionPrimitive.Description className="text-fd-muted-foreground empty:hidden" />
    </button>
  </SuggestionPrimitive.Trigger>
);

const Composer: FC = () => {
  // Sending drops focus from the box, so a phone's keyboard closes.
  useAuiEvent('thread.runStart', () => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
  });
  return (
  <ComposerPrimitive.Root className="relative flex w-full flex-col">
    <div className="flex w-full cursor-text flex-col gap-2 rounded-2xl border bg-fd-card p-2 shadow-sm transition-colors focus-within:border-fd-ring">
      <ComposerPrimitive.Input
        placeholder="Send a message..."
        className="max-h-48 min-h-10 w-full resize-none bg-transparent px-2.5 py-1 text-base leading-6 outline-none placeholder:text-fd-muted-foreground"
        rows={1}
        enterKeyHint="send"
        aria-label="Message input"
      />
      <ComposerAction />
    </div>
  </ComposerPrimitive.Root>
  );
};

const ComposerAction: FC = () => (
  <div className="flex items-center justify-end gap-1.5">
    <AuiIf condition={(s) => s.thread.capabilities.dictation}>
      <AuiIf condition={(s) => s.composer.dictation == null}>
        <ComposerPrimitive.Dictate asChild>
          <IconButton label="Voice input" size="icon-sm" className="rounded-full">
            <Mic />
          </IconButton>
        </ComposerPrimitive.Dictate>
      </AuiIf>
      <AuiIf condition={(s) => s.composer.dictation != null}>
        <ComposerPrimitive.StopDictation asChild>
          <IconButton label="Stop voice input" size="icon-sm" className="rounded-full text-red-500">
            <Square className="animate-pulse fill-current" />
          </IconButton>
        </ComposerPrimitive.StopDictation>
      </AuiIf>
    </AuiIf>
    <AuiIf condition={(s) => !s.composer.canCancel}>
      <ComposerPrimitive.Send asChild>
        <IconButton label="Send message" variant="primary" size="icon-sm" className="rounded-full text-fd-primary-foreground">
          <ArrowUp />
        </IconButton>
      </ComposerPrimitive.Send>
    </AuiIf>
    <AuiIf condition={(s) => s.composer.canCancel}>
      <ComposerPrimitive.Cancel asChild>
        <IconButton label="Stop generating" variant="primary" size="icon-sm" className="rounded-full text-fd-primary-foreground">
          <Square className="fill-current" />
        </IconButton>
      </ComposerPrimitive.Cancel>
    </AuiIf>
  </div>
);

const MessageError: FC = () => (
  <MessagePrimitive.Error>
    <ErrorPrimitive.Root className="mt-2 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-600 dark:text-red-300">
      <ErrorPrimitive.Message className="line-clamp-3" />
    </ErrorPrimitive.Root>
  </MessagePrimitive.Error>
);

const AssistantMessage: FC = () => (
  <MessagePrimitive.Root data-role="assistant" className="relative">
    <div className="px-2 leading-relaxed break-words">
      <MessagePrimitive.GroupedParts groupBy={groupBy}>
        {({ part, children }) => {
          switch (part.type) {
            case 'group-chainOfThought':
              return <div>{children}</div>;
            case 'group-tool':
              return <ToolGroup group={part}>{children}</ToolGroup>;
            case 'group-sources':
              return <div className="mt-3 mb-1 flex flex-wrap gap-1.5">{children}</div>;
            case 'source':
              return <Source {...part} />;
            case 'group-reasoning':
              return <ReasoningGroup group={part}>{children}</ReasoningGroup>;
            case 'text':
            case 'reasoning':
              return <MarkdownText />;
            case 'tool-call':
              return part.toolUI ?? <ToolFallback {...part} />;
            case 'indicator':
              return (
                <span className="animate-pulse text-fd-muted-foreground" aria-label="Assistant is working">
                  ●
                </span>
              );
            default:
              return null;
          }
        }}
      </MessagePrimitive.GroupedParts>
      <MessageError />
    </div>
    <div className="ms-2 flex min-h-8 items-center pt-1.5">
      <BranchPicker />
      <AssistantActionBar />
    </div>
  </MessagePrimitive.Root>
);

const AssistantActionBar: FC = () => (
  <ActionBarPrimitive.Root hideWhenRunning autohide="not-last" className="-ms-1 flex gap-1">
    <ActionBarPrimitive.Copy asChild>
      <IconButton label="Copy">
        <AuiIf condition={(s) => s.message.isCopied}>
          <Check />
        </AuiIf>
        <AuiIf condition={(s) => !s.message.isCopied}>
          <Copy />
        </AuiIf>
      </IconButton>
    </ActionBarPrimitive.Copy>
    <AuiIf condition={(s) => s.thread.capabilities.speech}>
      <AuiIf condition={(s) => s.message.speech == null}>
        <ActionBarPrimitive.Speak asChild>
          <IconButton label="Read aloud">
            <Volume2 />
          </IconButton>
        </ActionBarPrimitive.Speak>
      </AuiIf>
      <AuiIf condition={(s) => s.message.speech != null}>
        <ActionBarPrimitive.StopSpeaking asChild>
          <IconButton label="Stop reading">
            <Square className="fill-current" />
          </IconButton>
        </ActionBarPrimitive.StopSpeaking>
      </AuiIf>
    </AuiIf>
    <ActionBarPrimitive.Reload asChild>
      <IconButton label="Regenerate">
        <RefreshCw />
      </IconButton>
    </ActionBarPrimitive.Reload>
    <ActionBarMorePrimitive.Root>
      <ActionBarMorePrimitive.Trigger asChild>
        <IconButton label="More" className="data-[state=open]:bg-fd-accent">
          <MoreHorizontal />
        </IconButton>
      </ActionBarMorePrimitive.Trigger>
      <ActionBarMorePrimitive.Content side="bottom" align="start" sideOffset={6} className={menuClass}>
        <ActionBarPrimitive.ExportMarkdown asChild>
          <ActionBarMorePrimitive.Item className={menuItemClass}>
            <Download className="size-4" />
            Export as Markdown
          </ActionBarMorePrimitive.Item>
        </ActionBarPrimitive.ExportMarkdown>
      </ActionBarMorePrimitive.Content>
    </ActionBarMorePrimitive.Root>
  </ActionBarPrimitive.Root>
);

const UserMessage: FC = () => (
  <MessagePrimitive.Root
    data-role="user"
    className="grid auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] content-start gap-y-2 px-2 [&:where(>*)]:col-start-2"
  >
    <div className="relative col-start-2 min-w-0">
      <div className="peer rounded-2xl bg-fd-secondary px-4 py-2 break-words whitespace-pre-wrap text-fd-secondary-foreground empty:hidden">
        <MessagePrimitive.Parts />
      </div>
      <div className="absolute start-0 top-1/2 -translate-x-full -translate-y-1/2 pe-2 peer-empty:hidden">
        <ActionBarPrimitive.Root hideWhenRunning autohide="not-last" className="flex flex-col items-end">
          <ActionBarPrimitive.Edit asChild>
            <IconButton label="Edit">
              <Pencil />
            </IconButton>
          </ActionBarPrimitive.Edit>
        </ActionBarPrimitive.Root>
      </div>
    </div>
    <BranchPicker className="col-span-full col-start-1 -me-1 justify-end" />
  </MessagePrimitive.Root>
);

const EditComposer: FC = () => (
  <MessagePrimitive.Root className="flex flex-col px-2">
    <ComposerPrimitive.Root className="ms-auto flex w-full max-w-[85%] flex-col rounded-2xl border bg-fd-card transition-colors focus-within:border-fd-ring">
      <ComposerPrimitive.Input
        className="min-h-14 w-full resize-none bg-transparent px-4 pt-3 pb-1 text-base outline-none"
        autoFocus
      />
      <div className="mx-2.5 mb-2.5 flex items-center gap-1.5 self-end">
        <ComposerPrimitive.Cancel asChild>
          <button type="button" className={cn(buttonVariants({ variant: 'ghost', size: 'sm' }), 'h-8 px-3')}>
            Cancel
          </button>
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send asChild>
          <button type="button" className={cn(buttonVariants({ variant: 'primary', size: 'sm' }), 'h-8 px-3')}>
            Update
          </button>
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  </MessagePrimitive.Root>
);

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({ className, ...rest }) => (
  <BranchPickerPrimitive.Root
    hideWhenSingleBranch
    className={cn('-ms-2 me-2 inline-flex items-center text-xs text-fd-muted-foreground', className)}
    {...rest}
  >
    <BranchPickerPrimitive.Previous asChild>
      <IconButton label="Previous">
        <ChevronLeft />
      </IconButton>
    </BranchPickerPrimitive.Previous>
    <span className="font-medium">
      <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
    </span>
    <BranchPickerPrimitive.Next asChild>
      <IconButton label="Next">
        <ChevronRight />
      </IconButton>
    </BranchPickerPrimitive.Next>
  </BranchPickerPrimitive.Root>
);
