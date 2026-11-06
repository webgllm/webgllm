import {
  AuiIf,
  ThreadListItemMorePrimitive,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAui,
  useAuiState,
} from '@assistant-ui/react';
import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import { Archive, Loader2, MoreHorizontal, Pencil, Plus, Search, Trash2 } from 'lucide-react';
import { Fragment, useEffect, useMemo, useRef, useState, type FC } from 'react';
import { cn } from '@/lib/cn';

const menuClass = 'z-50 min-w-36 overflow-hidden rounded-xl border bg-fd-popover p-1 text-fd-popover-foreground shadow-lg';
const menuItemClass =
  'flex cursor-default items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm outline-none select-none data-[highlighted]:bg-fd-accent data-[highlighted]:text-fd-accent-foreground hover:bg-fd-accent hover:text-fd-accent-foreground';
const inputClass =
  'w-full rounded-md border bg-fd-background text-sm outline-none placeholder:text-fd-muted-foreground focus-visible:ring-2 focus-visible:ring-fd-ring';

export const ThreadList: FC = () => {
  const [search, setSearch] = useState('');
  const hasThreads = useAuiState((s) => s.threads.threadIds.length > 0);
  return (
    <ThreadListPrimitive.Root className="flex flex-col gap-0.5">
      <ThreadListPrimitive.New asChild>
        <button
          type="button"
          className={cn(
            buttonVariants({ variant: 'ghost' }),
            'h-8 justify-start gap-2 px-2.5 font-normal data-active:bg-fd-accent',
          )}
        >
          <Plus className="size-4 shrink-0" />
          New chat
        </button>
      </ThreadListPrimitive.New>
      {hasThreads && (
        <div className="relative px-0.5 py-1">
          <Search className="pointer-events-none absolute start-3 top-1/2 size-4 -translate-y-1/2 text-fd-muted-foreground" />
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            aria-label="Search chats"
            placeholder="Search chats"
            className={cn(inputClass, 'h-8 ps-8')}
          />
        </div>
      )}
      <AuiIf condition={(s) => !s.threads.isLoading}>
        <ThreadListGroups query={hasThreads ? search : ''} />
      </AuiIf>
    </ThreadListPrimitive.Root>
  );
};

const startOfDay = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();

/** The chats matching `query`, newest first, under Today, Yesterday and Earlier. */
const ThreadListGroups: FC<{ query: string }> = ({ query }) => {
  const threadIds = useAuiState((s) => s.threads.threadIds);
  const threadItems = useAuiState((s) => s.threads.threadItems);
  const needle = query.trim().toLowerCase();
  const groups = useMemo(() => {
    const items = new Map(threadItems.map((item) => [item.id, item]));
    const today = startOfDay(new Date());
    const yesterday = today - 86_400_000;
    const time = (index: number) => items.get(threadIds[index]!)?.lastMessageAt?.getTime() ?? Number.MAX_SAFE_INTEGER;
    const matches = threadIds
      .map((id, index) => ({ id, index }))
      .filter(({ id }) => !needle || (items.get(id)?.title || 'New chat').toLowerCase().includes(needle))
      .map(({ index }) => index)
      .sort((a, b) => time(b) - time(a));
    const result: Array<{ label: string; indices: number[] }> = [];
    for (const index of matches) {
      const at = time(index);
      const label = at >= today ? 'Today' : at >= yesterday ? 'Yesterday' : 'Earlier';
      const last = result.at(-1);
      if (last?.label === label) last.indices.push(index);
      else result.push({ label, indices: [index] });
    }
    return result;
  }, [threadIds, threadItems, needle]);

  if (needle && groups.length === 0) {
    return <p className="px-2.5 py-4 text-sm text-fd-muted-foreground">No chats found</p>;
  }
  return groups.map((group) => (
    <Fragment key={group.label}>
      <p className="px-2.5 pt-3 pb-1 text-xs font-medium text-fd-muted-foreground">{group.label}</p>
      {group.indices.map((index) => (
        <ThreadListPrimitive.ItemByIndex key={threadIds[index]} index={index} components={{ ThreadListItem }} />
      ))}
    </Fragment>
  ));
};

const ThreadListItem: FC = () => {
  const isRunning = useAuiState((s) => s.threadListItem.isRunning);
  const [renaming, setRenaming] = useState(false);
  return (
    <ThreadListItemPrimitive.Root className="group relative flex h-8 items-center rounded-md text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground data-active:bg-fd-primary/10 data-active:text-fd-primary has-data-[state=open]:bg-fd-accent">
      {renaming ? (
        <ThreadListItemRename onDone={() => setRenaming(false)} />
      ) : (
        <ThreadListItemPrimitive.Trigger className="flex h-full min-w-0 flex-1 items-center rounded-md px-2.5 text-start text-sm outline-none group-hover:pe-9 group-data-active:pe-9 focus-visible:ring-2 focus-visible:ring-fd-ring">
          {isRunning && <Loader2 aria-hidden className="me-1.5 size-3.5 shrink-0 animate-spin" />}
          <span className="min-w-0 flex-1 truncate">
            <ThreadListItemPrimitive.Title fallback="New chat" />
          </span>
          {isRunning && <span className="sr-only">Running</span>}
        </ThreadListItemPrimitive.Trigger>
      )}
      <ThreadListItemMorePrimitive.Root>
        <ThreadListItemMorePrimitive.Trigger asChild>
          <button
            type="button"
            title="More options"
            aria-label="More options"
            className={cn(
              buttonVariants({ variant: 'ghost', size: 'icon-xs' }),
              'absolute end-1.5 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 group-data-active:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100',
            )}
          >
            <MoreHorizontal />
          </button>
        </ThreadListItemMorePrimitive.Trigger>
        <ThreadListItemMorePrimitive.Content side="right" align="start" sideOffset={6} className={menuClass}>
          <ThreadListItemMorePrimitive.Item className={menuItemClass} onSelect={() => setRenaming(true)}>
            <Pencil className="size-4" />
            Rename
          </ThreadListItemMorePrimitive.Item>
          <ThreadListItemPrimitive.Archive asChild>
            <ThreadListItemMorePrimitive.Item className={menuItemClass}>
              <Archive className="size-4" />
              Archive
            </ThreadListItemMorePrimitive.Item>
          </ThreadListItemPrimitive.Archive>
          <ThreadListItemPrimitive.Delete asChild>
            <ThreadListItemMorePrimitive.Item className={cn(menuItemClass, 'text-red-500 data-[highlighted]:text-red-500 hover:text-red-500')}>
              <Trash2 className="size-4" />
              Delete
            </ThreadListItemMorePrimitive.Item>
          </ThreadListItemPrimitive.Delete>
        </ThreadListItemMorePrimitive.Content>
      </ThreadListItemMorePrimitive.Root>
    </ThreadListItemPrimitive.Root>
  );
};

const ThreadListItemRename: FC<{ onDone: () => void }> = ({ onDone }) => {
  const aui = useAui();
  const title = useAuiState((s) => s.threadListItem.title) ?? '';
  const [value, setValue] = useState(title);
  const ref = useRef<HTMLInputElement>(null);
  const settled = useRef(false);
  useEffect(() => ref.current?.select(), []);

  const commit = () => {
    if (settled.current) return;
    settled.current = true;
    const next = value.trim();
    if (!next || next === title) return onDone();
    Promise.resolve()
      .then(() => aui.threadListItem.rename(next))
      .then(onDone, () => {
        settled.current = false;
      });
  };

  return (
    <input
      ref={ref}
      autoFocus
      aria-label="Rename chat"
      value={value}
      className={cn(inputClass, 'h-7 min-w-0 flex-1 ps-2.5 pe-9 text-fd-foreground')}
      onChange={(event) => setValue(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return;
        if (event.key === 'Enter') {
          event.preventDefault();
          commit();
        } else if (event.key === 'Escape') {
          event.preventDefault();
          settled.current = true;
          onDone();
        }
      }}
    />
  );
};
