import type { SourceMessagePartComponent } from '@assistant-ui/react';
import { memo, useState } from 'react';

// assistant-ui's sources element (registry "sources"), on Fumadocs tokens.

const ALLOWED_PROTOCOLS = new Set(['http:', 'https:']);

/** `url` when following it runs no script; links come from web results. */
function safeHref(url: string): string | undefined {
  try {
    return ALLOWED_PROTOCOLS.has(new URL(url).protocol) ? url : undefined;
  } catch {
    return undefined;
  }
}

/** The host `url` points at, without a leading `www.`. */
function hostOf(url: string): string | undefined {
  const href = safeHref(url);
  return href ? new URL(href).hostname.replace(/^www\./, '') || undefined : undefined;
}

function SourceIcon({ domain }: { domain: string | undefined }) {
  const [failed, setFailed] = useState(false);
  if (!domain || failed) {
    return (
      <span className="flex size-3.5 shrink-0 items-center justify-center rounded-sm bg-fd-muted text-[10px] font-medium">
        {domain?.charAt(0).toUpperCase() || '?'}
      </span>
    );
  }
  return (
    <img
      src={`https://icons.duckduckgo.com/ip3/${domain}.ico`}
      alt=""
      className="size-3.5 shrink-0 rounded-sm"
      onError={() => setFailed(true)}
    />
  );
}

/** One web source: favicon, title and a link that opens in a new tab. */
export const Source: SourceMessagePartComponent = memo(function Source(part) {
  if (part.sourceType !== 'url') return null;
  const href = safeHref(part.url);
  const domain = hostOf(part.url);
  const content = (
    <>
      <SourceIcon domain={domain} />
      <span className="max-w-40 truncate">{part.title || domain || part.url}</span>
    </>
  );
  const className =
    'inline-flex items-center gap-1.5 rounded-md border bg-fd-card px-2 py-1 text-xs font-medium text-fd-muted-foreground transition-colors';
  return href ? (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      title={part.title ? `${part.title}\n${part.url}` : part.url}
      className={`${className} hover:bg-fd-accent hover:text-fd-accent-foreground`}
    >
      {content}
    </a>
  ) : (
    <span className={className}>{content}</span>
  );
});
