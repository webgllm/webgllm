"use client";

import { useAuiState, useMessageTiming } from "@assistant-ui/react";
import type { ReplyStats } from "./model";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui";
import { cn } from "@/lib/cn";
import { useState, type FC } from "react";

const formatTimingMs = (ms: number | undefined): string => {
  if (ms === undefined) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

/**
 * Shows streaming stats (TTFT, total time, tok/s, chunks) as a badge with a
 * hover/focus tooltip. Renders nothing until the stream completes.
 *
 * Place it inside `ActionBarPrimitive.Root` in your `thread.tsx` so it
 * inherits the action bar's autohide behaviour:
 *
 * ```tsx
 * import { MessageTiming } from "./message-timing";
 *
 * <ActionBarPrimitive.Root >
 *   <ActionBarPrimitive.Copy />
 *   <ActionBarPrimitive.Reload />
 *   <MessageTiming />  // <-- add this
 * </ActionBarPrimitive.Root>
 * ```
 *
 * @param side - Side of the tooltip relative to the badge trigger.
 * @default "right"
 */
export const MessageTiming: FC<{
  className?: string;
  side?: "top" | "right" | "bottom" | "left";
}> = ({ className, side = "right" }) => {
  const timing = useMessageTiming();
  // Phones have no hover: a tap on the badge opens and closes the stats too.
  const [open, setOpen] = useState(false);
  // webgllm's prompt, cache and token counts, which assistant-ui's timing has no fields for.
  const stats = useAuiState(
    (s) => (s.message.metadata.custom as { stats?: ReplyStats } | undefined)?.stats,
  );
  if (timing?.totalStreamTime === undefined) return null;

  return (
    <TooltipProvider>
      <Tooltip open={open} onOpenChange={setOpen}>
        <TooltipTrigger asChild>
          <button
            type="button"
            data-slot="message-timing-trigger"
            aria-label="Message timing"
            onClick={() => setOpen((value) => !value)}
            className={cn(
              "text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-accent-foreground flex items-center rounded-md p-1 font-mono text-xs tabular-nums transition-colors",
              className,
            )}
          >
            {formatTimingMs(timing.totalStreamTime)}
          </button>
        </TooltipTrigger>
        <TooltipContent
          side={side}
          sideOffset={8}
          data-slot="message-timing-popover"
          className="bg-fd-popover text-fd-popover-foreground border px-3 py-2 [&_[data-slot=tooltip-arrow]]:hidden"
        >
          <div className="grid min-w-35 gap-1.5 text-xs">
            {timing.firstTokenTime !== undefined && (
              <div className="flex items-center justify-between gap-4">
                <span className="text-fd-muted-foreground">First token</span>
                <span className="font-mono tabular-nums">
                  {formatTimingMs(timing.firstTokenTime)}
                </span>
              </div>
            )}
            <div className="flex items-center justify-between gap-4">
              <span className="text-fd-muted-foreground">Total time</span>
              <span className="font-mono tabular-nums">
                {formatTimingMs(timing.totalStreamTime)}
              </span>
            </div>
            {stats?.promptTokensPerSecond !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-fd-muted-foreground">Prompt speed</span>
              <span className="font-mono tabular-nums">
                {stats.promptTokensPerSecond.toFixed(1)} tok/s
              </span>
            </div>
            )}
            {timing.tokensPerSecond !== undefined && (
              <div className="flex items-center justify-between gap-4">
                <span className="text-fd-muted-foreground">Decode speed</span>
                <span className="font-mono tabular-nums">
                  {timing.tokensPerSecond.toFixed(1)} tok/s
                </span>
              </div>
            )}
            {stats?.promptTokens !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-fd-muted-foreground">Prompt tokens</span>
              <span className="font-mono tabular-nums">
                {stats.promptTokens}
              </span>
            </div>
            )}
            {stats?.cachedTokens !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-fd-muted-foreground">Cached tokens</span>
              <span className="font-mono tabular-nums">
                {stats.cachedTokens}
              </span>
            </div>
            )}
            {stats?.outputTokens !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-fd-muted-foreground">Completion tokens</span>
              <span className="font-mono tabular-nums">
                {stats.outputTokens}
              </span>
            </div>
            )}
            {!!stats?.reasoningTokens && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-fd-muted-foreground">Reasoning</span>
              <span className="font-mono tabular-nums">
                {stats.reasoningTokens}
              </span>
            </div>
            )}
            {stats?.totalTokens !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-fd-muted-foreground">Total tokens</span>
              <span className="font-mono tabular-nums">
                {stats.totalTokens}
              </span>
            </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};
