import { useSyncExternalStore, type ReactNode } from 'react';
import { JobProgress, type JobStage } from './job-progress';
import { MODELS, modelStatus } from './model';

/** webgllm's two load phases: downloading takes 0–0.9 of its progress, loading the rest. */
const STAGES: readonly JobStage[] = [
  { name: 'Download', weight: 0.9, description: 'Saved in this browser, so it downloads once.' },
  { name: 'Load', weight: 0.1, description: 'Reading the weights into the GPU.' },
];

/**
 * The selected model's download and load, as assistant-ui's job progress
 * card, in place of the thread's waiting dot (`children`) until it's ready.
 */
export function ModelProgress({ children }: { children?: ReactNode }) {
  const status = useSyncExternalStore(modelStatus.subscribe, modelStatus.get, modelStatus.get);
  if (status.state !== 'downloading' && status.state !== 'loading') return <>{children}</>;

  const name = MODELS.find((model) => model.id === status.model)?.name ?? status.model;
  const loading = status.state === 'loading';
  // A model already stored skips the download and loads across all of 0–1.
  const stageProgress = loading
    ? status.cached
      ? status.progress
      : (status.progress - 0.9) / 0.1
    : status.progress / 0.9;

  return (
    <JobProgress
      title={`Loading ${name}`}
      stages={STAGES}
      stageIndex={loading ? 1 : 0}
      stageProgress={stageProgress}
      eta={`${Math.round(status.progress * 100)}%`}
    />
  );
}
