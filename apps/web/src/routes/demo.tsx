import { createFileRoute } from '@tanstack/react-router';
import { DemoApp } from '@/components/demo/demo-app';

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
    <DemoApp
      chatId={chat}
      onChatChange={(next) => navigate({ search: next ? { chat: next } : {}, replace: true })}
    />
  );
}
