'use client';

import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock.core';
import type { HighlighterCore } from 'shiki/core';

// A Shiki highlighter on the JavaScript regex engine with only the languages the
// homepage shows. Fumadocs' default one also references Shiki's WASM engine,
// which the Cloudflare server build can't resolve.
let highlighter: Promise<HighlighterCore> | undefined;

function getHighlighter() {
  highlighter ??= Promise.all([
    import('shiki/core'),
    import('shiki/engine/javascript'),
  ]).then(([{ createHighlighterCore }, { createJavaScriptRegexEngine }]) =>
    createHighlighterCore({
      langs: [import('shiki/langs/typescript.mjs'), import('shiki/langs/bash.mjs')],
      themes: [import('shiki/themes/github-light.mjs'), import('shiki/themes/github-dark.mjs')],
      engine: createJavaScriptRegexEngine(),
    }),
  );
  return highlighter;
}

export function CodeBlock({ lang, code }: { lang: 'ts' | 'bash'; code: string }) {
  return (
    <DynamicCodeBlock
      highlighter={getHighlighter}
      lang={lang}
      code={code}
      options={{ themes: { light: 'github-light', dark: 'github-dark' } }}
    />
  );
}
