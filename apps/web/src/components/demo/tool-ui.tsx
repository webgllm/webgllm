import type { MessagePrimitive, ToolCallMessagePartComponent } from '@assistant-ui/react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from 'fumadocs-ui/components/ui/collapsible';
import { Brain, ChevronRight, Globe, Loader2, TriangleAlert, Wrench } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { cn } from '@/lib/cn';

type GroupPart = MessagePrimitive.GroupedParts.GroupPart;

const triggerClass =
  'group/trigger flex w-full items-center gap-2 py-1 text-sm text-fd-muted-foreground transition-colors hover:text-fd-foreground';

const Chevron = () => (
  <ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[panel-open]/trigger:rotate-90" />
);

const seconds = (timing: GroupPart['timing']) =>
  timing?.completedAt === undefined ? undefined : Math.max(1, Math.round((timing.completedAt - timing.startedAt) / 1000));

/** The model's thinking, folded away once it answers. */
export function ReasoningGroup({ group, children }: { group: GroupPart; children: ReactNode }) {
  const running = group.status.type === 'running';
  const duration = seconds(group.timing);
  return (
    <Collapsible className="mb-2">
      <CollapsibleTrigger className={triggerClass}>
        <Brain className={cn('size-4 shrink-0', running && 'animate-pulse')} />
        <span>{running ? 'Thinking…' : duration ? `Thought for ${duration}s` : 'Thoughts'}</span>
        <Chevron />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-1 border-l-2 ps-4 text-fd-muted-foreground [&_.prose]:text-fd-muted-foreground">
          {children}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

/** A run of tool calls, behind one line. */
export function ToolGroup({ group, children }: { group: GroupPart; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const running = group.status.type === 'running';
  const count = group.indices.length;
  return (
    <Collapsible open={open || group.counts.requiresAction > 0} onOpenChange={setOpen} className="mb-2">
      <CollapsibleTrigger className={triggerClass}>
        {running ? <Loader2 className="size-4 shrink-0 animate-spin" /> : <Wrench className="size-4 shrink-0" />}
        <span>{count === 1 ? '1 tool call' : `${count} tool calls`}</span>
        <Chevron />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-1 flex flex-col gap-2">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  );
}

const pretty = (text: string) => {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
};

/** A tool call: its name, arguments and result, in a Fumadocs card. */
export const ToolFallback: ToolCallMessagePartComponent = ({ toolName, argsText, result, status, isError }) => {
  const running = status.type === 'running' || status.type === 'requires-action';
  const failed = isError || status.type === 'incomplete';
  const query = (() => {
    try {
      const args = JSON.parse(argsText) as Record<string, unknown>;
      return typeof args.query === 'string' ? args.query : typeof args.url === 'string' ? args.url : undefined;
    } catch {
      return undefined;
    }
  })();
  return (
    <Collapsible className="rounded-lg border bg-fd-card text-sm text-fd-card-foreground">
      <CollapsibleTrigger className="group/trigger flex w-full items-center gap-2 px-3 py-2 text-start">
        {running ? (
          <Loader2 className="size-4 shrink-0 animate-spin text-fd-muted-foreground" />
        ) : failed ? (
          <TriangleAlert className="size-4 shrink-0 text-red-500" />
        ) : (
          <Globe className="size-4 shrink-0 text-fd-muted-foreground" />
        )}
        <code className="shrink-0 font-medium">{toolName}</code>
        {query && <span className="min-w-0 truncate text-fd-muted-foreground">{query}</span>}
        <span className="ms-auto text-fd-muted-foreground">
          <Chevron />
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="flex flex-col gap-2 border-t px-3 py-2">
          <p className="text-xs font-medium text-fd-muted-foreground">Arguments</p>
          <pre className="max-h-40 overflow-auto rounded-md bg-fd-muted p-2 text-xs whitespace-pre-wrap">
            {pretty(argsText)}
          </pre>
          {result !== undefined && (
            <>
              <p className="text-xs font-medium text-fd-muted-foreground">Result</p>
              <pre className="max-h-64 overflow-auto rounded-md bg-fd-muted p-2 text-xs whitespace-pre-wrap">
                {typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
              </pre>
            </>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};
