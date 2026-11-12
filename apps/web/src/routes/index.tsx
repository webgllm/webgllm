import { createFileRoute, Link } from '@tanstack/react-router';
import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import { Card, Cards } from 'fumadocs-ui/components/card';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import { CodeBlock } from '@/components/code-block';
import {
  ArrowRight,
  Boxes,
  Braces,
  Brain,
  Cpu,
  Gauge,
  HardDriveDownload,
  MessageSquare,
  Plug,
  ShieldCheck,
  Wrench,
} from 'lucide-react';
import { baseOptions } from '@/lib/layout.shared';
import { gitConfig } from '@/lib/shared';
import { cn } from '@/lib/cn';

export const Route = createFileRoute('/')({
  component: Home,
});

const githubUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;

function Home() {
  return (
    <HomeLayout
      {...baseOptions()}
      links={[
        { text: 'Docs', url: '/docs', active: 'nested-url' },
        { text: 'Demo', url: '/demo', active: 'url' },
      ]}
    >
      <main className="flex flex-col">
        <Hero />
        <Platforms />
        <Features />
        <Models />
        <CallToAction />
      </main>
      <Footer />
    </HomeLayout>
  );
}

function Hero() {
  return (
    <section className="relative overflow-hidden border-b">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,var(--color-fd-border)_1px,transparent_1px),linear-gradient(to_bottom,var(--color-fd-border)_1px,transparent_1px)] bg-[size:48px_48px] [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black,transparent)] opacity-60"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/2 h-[480px] w-[900px] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,var(--color-fd-primary),transparent)] opacity-[0.08]"
      />

      <div className="relative mx-auto grid w-full max-w-(--fd-layout-width) gap-12 px-4 pt-20 pb-16 md:px-6 lg:grid-cols-[1fr_1.1fr] lg:items-center lg:pt-28 lg:pb-24">
        <div className="flex flex-col items-start">
          <a
            href={githubUrl}
            className="mb-6 inline-flex items-center gap-2 rounded-full border bg-fd-card px-3 py-1 text-xs font-medium text-fd-muted-foreground transition-colors hover:text-fd-foreground"
          >
            <span className="size-1.5 rounded-full bg-fd-primary" />
            Early release, 0.x
            <ArrowRight className="size-3" />
          </a>
          <h1 className="text-4xl font-semibold tracking-tight text-balance sm:text-5xl lg:text-6xl">
            LLMs in the browser,
            <br />
            <span className="text-fd-muted-foreground">on WebGL 2.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg text-fd-muted-foreground text-pretty">
            WebGLLM runs LFM3, LFM2.5 and Qwen3.5 on its own WebGL 2 engine, with an
            OpenAI-compatible API. It works where WebGPU doesn't: Firefox on Android,
            older iPhones, and everywhere else.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              to="/docs/$"
              params={{ _splat: '' }}
              className={cn(buttonVariants({ variant: 'primary' }), 'h-10 gap-2 px-5')}
            >
              Get started
              <ArrowRight className="size-4" />
            </Link>
            <Link
              to="/demo"
              className={cn(buttonVariants({ variant: 'outline' }), 'h-10 gap-2 px-5')}
            >
              <MessageSquare className="size-4" />
              Try the demo
            </Link>
            <a
              href={githubUrl}
              className={cn(buttonVariants({ variant: 'outline' }), 'h-10 gap-2 px-5')}
            >
              <GitHubIcon className="size-4" />
              GitHub
            </a>
          </div>
          <div className="mt-8 w-full max-w-sm [&_figure]:my-0">
            <CodeBlock lang="bash" code="npm install webgllm" />
          </div>
        </div>

        <CodeTabs />
      </div>
    </section>
  );
}

const chatCode = `import { WebGLLM } from "webgllm"

const webgllm = new WebGLLM()

const completion = await webgllm.chat.completions.create({
  model: "LiquidAI/LFM3-1.7B-GGUF",
  messages: [{ role: "user", content: "What is the capital of France?" }]
})

console.log(completion.choices[0].message.content)`;

const streamCode = `const stream = await webgllm.chat.completions.create({
  model: "unsloth/Qwen3.5-2B-GGUF",
  messages,
  reasoning_effort: "high",
  stream: true
})

for await (const chunk of stream) {
  const { reasoning_content, content } = chunk.choices[0].delta
  output.textContent += content ?? ""
}`;

const jsonCode = `const reply = await webgllm.chat.completions.create({
  model: "LiquidAI/LFM3-1.7B-GGUF",
  messages: [{
    role: "user",
    content: "Extract as JSON: Ada Lovelace, born 1815 in London."
  }],
  response_format: {
    type: "json_schema",
    json_schema: { name: "person", schema }
  }
})

const person = JSON.parse(reply.choices[0].message.content!)`;

const toolsCode = `const reply = await webgllm.chat.completions.create({
  model: "LiquidAI/LFM3-1.7B-GGUF",
  messages,
  tools: [{
    type: "function",
    function: { name: "get_weather", parameters: weatherSchema }
  }],
  tool_choice: "auto"
})

for (const call of reply.choices[0].message.tool_calls ?? []) {
  const args = JSON.parse(call.function.arguments)
}`;

function CodeTabs() {
  return (
    <div className="relative min-w-0">
      <div
        aria-hidden
        className="absolute -inset-4 rounded-3xl bg-gradient-to-b from-fd-primary/10 to-transparent blur-2xl"
      />
      <Tabs
        items={['Chat', 'Streaming', 'Structured output', 'Tools']}
        className="relative my-0 shadow-xl [&_figure]:my-0"
      >
        <Tab value="Chat">
          <CodeBlock lang="ts" code={chatCode} />
        </Tab>
        <Tab value="Streaming">
          <CodeBlock lang="ts" code={streamCode} />
        </Tab>
        <Tab value="Structured output">
          <CodeBlock lang="ts" code={jsonCode} />
        </Tab>
        <Tab value="Tools">
          <CodeBlock lang="ts" code={toolsCode} />
        </Tab>
      </Tabs>
    </div>
  );
}

const platforms = ['Chrome', 'Firefox', 'Safari', 'Firefox on Android', 'Older iPhones'];

function Platforms() {
  return (
    <section className="border-b bg-fd-card/40">
      <div className="mx-auto flex w-full max-w-(--fd-layout-width) flex-col items-center gap-4 px-4 py-8 md:flex-row md:justify-between md:px-6">
        <p className="text-sm text-fd-muted-foreground">
          Needs only WebGL 2, no WebGPU and no cross-origin isolation.
        </p>
        <ul className="flex flex-wrap justify-center gap-2">
          {platforms.map((name) => (
            <li
              key={name}
              className="rounded-full border bg-fd-background px-3 py-1 text-xs font-medium"
            >
              {name}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

const features = [
  {
    icon: <Plug />,
    title: 'OpenAI-compatible',
    description:
      'chat.completions.create with streaming, images, logprobs, stop sequences, seeds, usage and timings. Unsupported fields throw instead of being ignored.',
  },
  {
    icon: <Gauge />,
    title: 'Keeps the page smooth',
    description:
      'Each model runs in its own Web Worker on an OffscreenCanvas, with big passes split into bands, so the page keeps painting while it reads and writes.',
  },
  {
    icon: <HardDriveDownload />,
    title: 'Downloads once',
    description:
      'Models are stored in the Origin Private File System. Dropped downloads resume, and later page loads reuse the files.',
  },
  {
    icon: <Brain />,
    title: 'Thinking',
    description:
      'reasoning.effort turns thinking on or off, reasoning.max_tokens caps it, and reasoning streams separately from the answer.',
  },
  {
    icon: <Wrench />,
    title: 'Tool calls that work',
    description:
      'tools and tool_choice as in OpenAI. Automatic calls are checked against their schema and generated again if they fail.',
  },
  {
    icon: <Braces />,
    title: 'Structured output',
    description:
      'json_schema responses are constrained token by token, so the answer is always valid against the schema.',
  },
  {
    icon: <Boxes />,
    title: 'Several models',
    description:
      'Load several models at once under a memory budget picked from the device. The least recently used one is unloaded first.',
  },
  {
    icon: <Cpu />,
    title: 'GGUF and ONNX',
    description:
      'Reads regular GGUF and ONNX builds from Hugging Face into the same 4-bit weights. No special web builds needed.',
  },
  {
    icon: <ShieldCheck />,
    title: 'Private by default',
    description:
      'Everything runs on the device. Prompts and answers never leave the browser, and there is no server to run.',
  },
];

function Features() {
  return (
    <section className="mx-auto w-full max-w-(--fd-layout-width) px-4 py-20 md:px-6 lg:py-28">
      <SectionHeading
        eyebrow="Features"
        title="Everything you need to ship a local model"
        description="A familiar API on top of an engine written for WebGL 2 from the ground up."
      />
      <Cards className="mt-12 lg:grid-cols-3">
        {features.map((feature) => (
          <Card
            key={feature.title}
            icon={feature.icon}
            title={feature.title}
            description={feature.description}
          />
        ))}
      </Cards>
    </section>
  );
}

const models = [
  {
    name: 'LFM3',
    vendor: 'Liquid AI',
    note: '1.7B',
    description: 'Thinks by default, with a 262K-token context.',
    ids: ['LiquidAI/LFM3-1.7B-GGUF', 'onnx-community/LFM3-1.7B-ONNX'],
  },
  {
    name: 'Qwen3.5',
    vendor: 'Qwen',
    note: '2B',
    description: 'Gated DeltaNet layers, with thinking on request. Its ONNX build reads images.',
    ids: ['unsloth/Qwen3.5-2B-GGUF', 'onnx-community/Qwen3.5-2B-ONNX'],
  },
  {
    name: 'LFM2.5',
    vendor: 'Liquid AI',
    note: '350M to 1.2B',
    description: 'Instruct and Thinking at 1.2B, and a 350M model, in GGUF and ONNX.',
    ids: [
      'LiquidAI/LFM2.5-1.2B-Instruct-GGUF',
      'LiquidAI/LFM2.5-1.2B-Thinking-GGUF',
      'LiquidAI/LFM2.5-350M-GGUF',
    ],
  },
];

function Models() {
  return (
    <section className="border-y bg-fd-card/40">
      <div className="mx-auto w-full max-w-(--fd-layout-width) px-4 py-20 md:px-6 lg:py-28">
        <SectionHeading
          eyebrow="Models"
          title="Small models, loaded straight from Hugging Face"
          description="Pass a repository name as the model. 4-bit weights, quantization-aware trained where the vendor ships them."
        />
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {models.map((model) => (
            <div
              key={model.name}
              className="flex flex-col gap-4 rounded-xl border bg-fd-card p-6 text-fd-card-foreground"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-xs text-fd-muted-foreground">{model.vendor}</p>
                  <p className="mt-1 text-xl font-semibold">{model.name}</p>
                </div>
                <span className="rounded-full bg-fd-secondary px-2 py-0.5 text-xs text-fd-secondary-foreground">
                  {model.note}
                </span>
              </div>
              <p className="text-sm text-fd-muted-foreground">{model.description}</p>
              <ul className="mt-auto flex flex-col gap-1 border-t pt-4">
                {model.ids.map((id) => (
                  <li key={id}>
                    <code className="text-xs break-all text-fd-muted-foreground">{id}</code>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CallToAction() {
  return (
    <section className="mx-auto w-full max-w-(--fd-layout-width) px-4 py-20 md:px-6 lg:py-28">
      <div className="relative overflow-hidden rounded-2xl border bg-fd-card px-6 py-14 text-center">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_80%_at_50%_120%,var(--color-fd-primary),transparent)] opacity-10"
        />
        <h2 className="relative text-3xl font-semibold tracking-tight sm:text-4xl">
          Run your first model in a minute
        </h2>
        <p className="relative mx-auto mt-4 max-w-lg text-fd-muted-foreground">
          Install the package, create a WebGLLM and send a chat. The first call downloads
          the model; every call after that is local.
        </p>
        <div className="relative mt-8 flex flex-wrap justify-center gap-3">
          <Link
            to="/docs/$"
            params={{ _splat: '' }}
            className={cn(buttonVariants({ variant: 'primary' }), 'h-10 gap-2 px-5')}
          >
            Read the docs
            <ArrowRight className="size-4" />
          </Link>
          <a
            href="https://www.npmjs.com/package/webgllm"
            className={cn(buttonVariants({ variant: 'outline' }), 'h-10 px-5')}
          >
            View on npm
          </a>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t">
      <div className="mx-auto flex w-full max-w-(--fd-layout-width) flex-col gap-4 px-4 py-8 text-sm text-fd-muted-foreground sm:flex-row sm:items-center sm:justify-between md:px-6">
        <p>WebGLLM. Apache 2.0.</p>
        <nav className="flex gap-6">
          <Link to="/docs/$" params={{ _splat: '' }} className="hover:text-fd-foreground">
            Docs
          </Link>
          <Link to="/demo" className="hover:text-fd-foreground">
            Demo
          </Link>
          <a href={githubUrl} className="hover:text-fd-foreground">
            GitHub
          </a>
          <a href="https://www.npmjs.com/package/webgllm" className="hover:text-fd-foreground">
            npm
          </a>
        </nav>
      </div>
    </footer>
  );
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mx-auto max-w-2xl text-center">
      <p className="text-sm font-medium text-fd-primary">{eyebrow}</p>
      <h2 className="mt-3 text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
        {title}
      </h2>
      <p className="mt-4 text-fd-muted-foreground text-pretty">{description}</p>
    </div>
  );
}

function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden className={className}>
      <path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 0-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-6 0-1.2.5-2.3 1.3-3.1-.2-.4-.6-1.6 0-3.2 0 0 1-.3 3.4 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.6.2 2.8.1 3.2.8.8 1.3 1.9 1.3 3.1 0 4.6-2.8 5.6-5.5 6 .5.3.8 1 .8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .3" />
    </svg>
  );
}
