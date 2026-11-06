import { createFileRoute, Link } from '@tanstack/react-router';
import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { baseOptions } from '@/lib/layout.shared';

export const Route = createFileRoute('/')({
  component: Home,
});

function Home() {
  return (
    <HomeLayout {...baseOptions()}>
      <div className="flex flex-col flex-1 items-center justify-center gap-4 px-4 py-8 text-center">
        <h1 className="text-3xl font-semibold">WebGLLM</h1>
        <p className="text-fd-muted-foreground max-w-md">
          LLM inference in the browser on WebGL 2, for devices without WebGPU.
        </p>
        <Link
          to="/docs/$"
          params={{ _splat: '' }}
          className="px-3 py-2 rounded-lg bg-fd-primary text-fd-primary-foreground font-medium text-sm"
        >
          Read the docs
        </Link>
      </div>
    </HomeLayout>
  );
}
