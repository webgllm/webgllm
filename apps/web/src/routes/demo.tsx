import { createFileRoute } from '@tanstack/react-router';
import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { DemoApp } from '@/components/demo/demo-app';
import { baseOptions } from '@/lib/layout.shared';

export const Route = createFileRoute('/demo')({
  // The open chat, so a refresh or a shared link reopens it.
  validateSearch: (search: Record<string, unknown>): { chat?: string } =>
    typeof search.chat === 'string' && search.chat ? { chat: search.chat } : {},
  head: () => ({ meta: [{ title: 'Demo | WebGLLM' }] }),
  component: Demo,
});

function Demo() {
  const { chat } = Route.useSearch();
  const navigate = Route.useNavigate();
  return (
    <HomeLayout
      {...baseOptions()}
      links={[
        { text: 'Docs', url: '/docs', active: 'nested-url' },
        { text: 'Demo', url: '/demo', active: 'url' },
      ]}
    >
      <DemoApp
        chatId={chat}
        onChatChange={(next) => navigate({ search: next ? { chat: next } : {}, replace: true })}
      />
    </HomeLayout>
  );
}
