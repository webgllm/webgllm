import react from '@vitejs/plugin-react';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import { fumadocsMdx } from 'fumadocs-mdx/vite';
import { nitro } from 'nitro/vite';
import { glob } from 'node:fs/promises';
import { getSlugs } from 'fumadocs-core/source';
import { getPageImageUrl } from './src/lib/shared';

// OG images are binary, so Nitro prerenders them (TanStack Start writes responses as text).
const imageRoutes: string[] = [];
for await (const entry of glob('**/*.mdx', { cwd: 'content/docs' })) {
  imageRoutes.push(getPageImageUrl({ slugs: getSlugs(entry) }).url);
}

export default defineConfig({
  server: {
    port: 3000,
  },
  plugins: [
    fumadocsMdx(),
    tailwindcss(),
    tanstackStart({
      // Every page is rendered to HTML at build time; links are crawled from the home page.
      prerender: {
        enabled: true,
        crawlLinks: true,
        failOnError: true,
      },
      // Routes no page links to.
      pages: [{ path: '/api/search' }, { path: '/llms.txt' }, { path: '/llms-full.txt' }],
    }),
    react(),
    // Static output in .output/public, served by Cloudflare as static assets (wrangler.jsonc).
    nitro({
      // Prerender in Node so static server functions can write their JSON to disk.
      preset: 'node-server',
      prerender: { routes: imageRoutes },
    }),
  ],
  resolve: {
    tsconfigPaths: true,
    alias: {
      tslib: 'tslib/tslib.es6.js',
    },
  },
});
