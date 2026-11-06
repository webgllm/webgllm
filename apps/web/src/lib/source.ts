import { llms, loader } from 'fumadocs-core/source';
import { defineDocs } from 'fumadocs-mdx/macro';
import { Rocket, type LucideIcon } from 'lucide-react';
import { createElement } from 'react';
import { docsRoute } from './shared';

// Icons the docs can name in frontmatter (`icon: Rocket`) or meta.json. An
// explicit map instead of fumadocs' lucideIconsPlugin, which imports every
// Lucide icon into a chunk the homepage and docs preload. To use another
// icon, import it from lucide-react and add it here.
const icons: Record<string, LucideIcon> = { Rocket };

function icon(name: string | undefined) {
  if (name === undefined) return;
  const Icon = icons[name];
  // Fails the prerender (failOnError), so an unmapped icon can't ship.
  if (!Icon) throw new Error(`Unknown docs icon "${name}": add it to the map in src/lib/source.ts`);
  return createElement(Icon);
}

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    async: true,
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
});

export const source = loader({
  source: docs.toFumadocsSource(),
  baseUrl: docsRoute,
  icon,
});

export const docsLlms = llms(source, {
  renderPage: async (page) => `# ${page.data.title} (${page.url})

${await page.data.getText('processed')}`,
});
