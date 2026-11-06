# WebGLLM

LLM inference in the browser on WebGL 2, for devices without WebGPU.

- [`packages/webgllm`](packages/webgllm): the engine, published to npm as [`webgllm`](https://www.npmjs.com/package/webgllm).
- [`apps/web`](apps/web): the docs site, [webgllm.dev](https://webgllm.dev). Fumadocs on TanStack Start, prerendered to static HTML.

## Development

```bash
pnpm install
pnpm dev          # docs site on http://localhost:3000
pnpm build        # builds every project
```

## License

[Apache 2.0](LICENSE)
