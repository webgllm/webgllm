/**
 * The demo's image tool: the chat model calls it with a short description,
 * and Tessera-small draws it in the page (`webgllm.images.generate`), in the
 * chat model's place: the chat model unloads meanwhile and loads again
 * after, so the two never sit in memory together (a phone's tab holds one or
 * the other). The model reads a line of the result; the images are for the
 * chat only. The model says something with the image, the call's `message`,
 * shown under it as plain text; a model turn that only drew ends the reply
 * there (see model.ts), so the model never writes after the image, where it
 * tends to add a link or file name of its own.
 */
import type { ToolCallMessagePartComponent, Toolkit } from '@assistant-ui/react';
import { Loader2Icon } from 'lucide-react';
import { useSyncExternalStore } from 'react';
import type { LoadProgress } from 'webgllm';
import { type ShownToolResult, webgllm, whileUnloaded } from './model';

export const IMAGE_TOOL = 'generate_image';

const IMAGE_MODEL = 'cfahlgren1/tessera-small-onnx';
const IMAGE_MODEL_NAME = 'Tessera-small';

/** Images one call draws: one, so the image model's memory stays inside a phone tab's. */
const IMAGES_PER_CALL = 1;

/** What the tool returns: `text` for the model, the images (PNG data URLs) for its widget. */
type ImageResult = ShownToolResult & { images: string[] };

/** The image model's download the first time, 0 to 1 (undefined when it isn't downloading). */
let download: number | undefined;
const downloadListeners = new Set<() => void>();
const setDownload = (next: number | undefined) => {
  if (next === download) return;
  download = next;
  for (const listener of downloadListeners) listener();
};
const subscribeDownload = (listener: () => void) => {
  downloadListeners.add(listener);
  return () => downloadListeners.delete(listener);
};
// (webgllm's progress gives a download 0–0.9 of the whole load.)
const onProgress = ({ phase, progress, cached }: LoadProgress) =>
  setDownload(phase === 'download' && !cached ? Math.min(1, progress / 0.9) : undefined);

/** The result's line for the model, which it reads in later turns. */
const resultLine = (prompt: string) =>
  `The image of "${prompt}" is already shown to the person. Don't add an image or link to it.`;

async function draw(args: Record<string, unknown>, { abortSignal }: { abortSignal: AbortSignal }): Promise<ImageResult> {
  const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : '';
  if (!prompt) throw new Error('A prompt is needed.');
  // The whole swap (unload the chat models, draw, unload the image model, load this
  // reply's chat model again) is one turn alone at the models: see `whileUnloaded`.
  const response = await whileUnloaded(async () => {
    try {
      return await webgllm().images.generate(
        { model: IMAGE_MODEL, prompt, n: IMAGES_PER_CALL },
        { signal: abortSignal, onProgress },
      );
    } finally {
      setDownload(undefined);
      await webgllm().unloadModel(IMAGE_MODEL);
    }
  }, abortSignal);
  return {
    text: resultLine(prompt),
    endsReply: true,
    images: response.data.flatMap((image) => (image.b64_json ? [`data:image/png;base64,${image.b64_json}`] : [])),
  };
}

const imagesOf = (result: unknown): string[] => {
  const images = (result as { images?: unknown } | null | undefined)?.images;
  return Array.isArray(images) ? images.filter((image): image is string => typeof image === 'string') : [];
};

/**
 * The tool's widget, in place of the usual tool row: the drawn images,
 * scaled up with their pixels kept sharp, then the model's message. A line
 * while it draws (with the model's download, the first time), and a short
 * line if it fails or stops.
 */
const ImageTool: ToolCallMessagePartComponent = ({ args, result, status, isError }) => {
  const downloading = useSyncExternalStore(subscribeDownload, () => download, () => undefined);
  const input = (args ?? {}) as { prompt?: unknown; message?: unknown };
  const prompt = typeof input.prompt === 'string' ? input.prompt : undefined;
  const message = typeof input.message === 'string' ? input.message.trim() : '';
  const images = imagesOf(result);

  if (images.length > 0) {
    return (
      <div className="my-2 flex flex-col gap-3">
        <div role="group" aria-label={prompt ? `Images of ${prompt}` : 'Images'} className="flex flex-wrap gap-2">
          {images.map((src, i) => (
            <img
              key={i}
              src={src}
              alt={prompt ?? 'Generated image'}
              width={16}
              height={16}
              className="size-32 rounded-md bg-fd-muted [image-rendering:pixelated]"
            />
          ))}
        </div>
        {/* Plain text, not markdown: the message can't hold an image or link. */}
        {message && <p>{message}</p>}
      </div>
    );
  }
  if (isError) {
    return (
      <p className="my-2 text-sm text-red-600 dark:text-red-400">
        Couldn&apos;t draw{prompt ? ` ${prompt}` : ' the image'}. {typeof result === 'string' ? result : ''}
      </p>
    );
  }
  // No result and nothing running: stopped, or the page was ended mid-drawing (a phone
  // ends a tab past its memory limit, and the chat comes back with the call but no result).
  if (status.type !== 'running' && status.type !== 'requires-action') {
    return <p className="my-2 text-sm text-fd-muted-foreground">Drawing didn&apos;t finish.</p>;
  }
  return (
    <p className="my-2 flex items-center gap-2 text-sm text-fd-muted-foreground">
      <Loader2Icon className="size-3.5 shrink-0 animate-spin" aria-hidden />
      {downloading !== undefined
        ? `Downloading ${IMAGE_MODEL_NAME}… ${Math.round(downloading * 100)}%`
        : prompt
          ? `Drawing ${prompt}…`
          : 'Drawing…'}
    </p>
  );
};

/** The image tool, as a frontend tool: shown on its own, not folded in with the reply's other tool calls. */
export const imageToolkit: Toolkit = {
  [IMAGE_TOOL]: {
    type: 'frontend',
    display: 'standalone',
    description: 'Draws a pixel-art image from a short description and shows it to the person.',
    parameters: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'What to draw, in a few words.' },
        message: { type: 'string', description: 'A sentence to say to the person with the image.' },
      },
      required: ['prompt', 'message'],
    },
    execute: draw,
    render: ImageTool,
  },
};
