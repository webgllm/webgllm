import '@assistant-ui/react-markdown/styles/dot.css';
import { MarkdownTextPrimitive, memoizeMarkdownComponents } from '@assistant-ui/react-markdown';
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';
import { memo } from 'react';
import remarkGfm from 'remark-gfm';

// Fumadocs' prose styles the text; code blocks are Fumadocs' own, with its
// copy button, so assistant-ui's code header is left out.
const components = memoizeMarkdownComponents({
  pre: (props) => (
    <CodeBlock className="my-3">
      <Pre {...props} />
    </CodeBlock>
  ),
  CodeHeader: () => null,
});

export const MarkdownText = memo(function MarkdownText() {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      className="prose prose-sm max-w-none dark:prose-invert [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
      components={components}
      defer
    />
  );
});
