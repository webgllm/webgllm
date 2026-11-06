import { cloudflare } from '@cloudflare/vite-plugin';
import react from '@vitejs/plugin-react';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import { fumadocsMdx } from 'fumadocs-mdx/vite';

export default defineConfig({
  server: {
    port: 3000,
    // Cloudflare quick tunnels, for trying the dev server on a phone.
    allowedHosts: ['.trycloudflare.com'],
  },
  plugins: [
    fumadocsMdx(),
    tailwindcss(),
    cloudflare({ viteEnvironment: { name: 'ssr' } }),
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
  ],
  resolve: {
    tsconfigPaths: true,
    alias: {
      tslib: 'tslib/tslib.es6.js',
    },
  },
});
