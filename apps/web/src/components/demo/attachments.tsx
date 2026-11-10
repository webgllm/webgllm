"use client";

import {
  AttachmentPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useAuiState,
} from "@assistant-ui/react";
import { PaperclipIcon, XIcon } from "lucide-react";
import { useEffect, useState, type FC } from "react";
import { useMediaInputs } from "./model";
import { TooltipIconButton } from "./tooltip-icon-button";

/** The attachment's image: its file while in the composer, its data URL once sent. */
function useAttachmentSrc(): string | undefined {
  const attachment = useAuiState((s) => s.attachment);
  const file = attachment && "file" in attachment ? attachment.file : undefined;
  const sent = attachment?.content?.find((part) => part.type === "image") as { image: string } | undefined;
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!file) return setUrl(undefined);
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);
  return sent?.image ?? url;
}

/** One attachment: a thumbnail, with a remove button in the composer. */
const AttachmentTile: FC<{ removable: boolean }> = ({ removable }) => {
  const src = useAttachmentSrc();
  return (
    <AttachmentPrimitive.Root className="aui-attachment-root border-fd-foreground/10 bg-fd-muted relative size-16 overflow-hidden rounded-lg border">
      {src ? (
        <img src={src} alt="" className="size-full object-cover" />
      ) : (
        <AttachmentPrimitive.Name />
      )}
      {removable && (
        <AttachmentPrimitive.Remove asChild>
          <TooltipIconButton
            tooltip="Remove"
            side="top"
            className="bg-fd-background/80 hover:bg-fd-background absolute end-1 top-1 size-5 rounded-full p-0.5"
          >
            <XIcon />
          </TooltipIconButton>
        </AttachmentPrimitive.Remove>
      )}
    </AttachmentPrimitive.Root>
  );
};

/** The composer's attachments, above its input (nothing when there are none). */
export const ComposerAttachments: FC = () => (
  <div className="aui-composer-attachments flex w-full flex-row flex-wrap gap-2 px-1 empty:hidden">
    <ComposerPrimitive.Attachments>{() => <AttachmentTile removable />}</ComposerPrimitive.Attachments>
  </div>
);

/** A sent message's attachments, above its text. */
export const UserMessageAttachments: FC = () => (
  <div className="aui-user-message-attachments col-start-2 flex flex-row flex-wrap justify-end gap-2 empty:hidden">
    <MessagePrimitive.Attachments>{() => <AttachmentTile removable={false} />}</MessagePrimitive.Attachments>
  </div>
);

/** The attach button, only for a model that reads images (what webgllm's meta says it takes). */
export const ComposerAddAttachment: FC = () => {
  const inputs = useMediaInputs();
  if (!inputs.includes("image")) return null;
  return (
    <ComposerPrimitive.AddAttachment asChild multiple>
      <TooltipIconButton tooltip="Attach an image" side="top" className="aui-composer-add-attachment size-8 p-1.5">
        <PaperclipIcon />
      </TooltipIconButton>
    </ComposerPrimitive.AddAttachment>
  );
};

/** Whether files dropped on the composer are taken: only by a model that reads them. */
export function useTakesDrops(): boolean {
  return useMediaInputs().length > 0;
}

/**
 * Whether the draft has attachments the selected model can't read (picked
 * under another model): sending waits until the user switches back or
 * removes them, rather than sending the text without them.
 */
export function useAttachmentsBlocked(): boolean {
  const inputs = useMediaInputs();
  const attached = useAuiState((s) => s.composer.attachments.length);
  return attached > 0 && !inputs.includes("image");
}

/** Says why the draft can't be sent, while it can't. */
export const AttachmentsBlockedNote: FC = () =>
  useAttachmentsBlocked() ? (
    <p className="aui-composer-attachments-note text-fd-muted-foreground px-2.5 text-xs">
      This model doesn&apos;t read images. Switch to one that does, or remove the image to send.
    </p>
  ) : null;
