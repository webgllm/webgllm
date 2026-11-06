/**
 * The shadcn/ui pieces assistant-ui's components import, as Fumadocs
 * components (or the Base UI parts Fumadocs is built on) with Fumadocs
 * styling, so the ported components keep their own markup.
 */
import { Popover as BasePopover } from '@base-ui/react/popover';
import { Tooltip as BaseTooltip } from '@base-ui/react/tooltip';
import { cva, type VariantProps } from 'class-variance-authority';
import { Command as CommandPrimitive } from 'cmdk';
import { buttonVariants as fdButtonVariants } from 'fumadocs-ui/components/ui/button';
import {
  Collapsible,
  CollapsibleContent as FdCollapsibleContent,
  CollapsibleTrigger as FdCollapsibleTrigger,
} from 'fumadocs-ui/components/ui/collapsible';
import { SearchIcon } from 'lucide-react';
import {
  forwardRef,
  isValidElement,
  useEffect,
  useLayoutEffect,
  type ComponentProps,
  type ComponentPropsWithoutRef,
  type ReactElement,
  type ReactNode,
} from 'react';
import { cn } from '@/lib/cn';

export const useIsomorphicLayoutEffect = typeof window === 'undefined' ? useEffect : useLayoutEffect;

// Button: Fumadocs' button, under shadcn's variant and size names.

const BUTTON_VARIANTS = {
  default: 'primary',
  outline: 'outline',
  ghost: 'ghost',
  secondary: 'secondary',
} as const;

type ButtonProps = ComponentPropsWithoutRef<'button'> & {
  variant?: 'default' | 'outline' | 'ghost' | 'secondary' | 'destructive' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'default', size = 'default', className, type = 'button', ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        // shadcn's button spacing and icon sizing, which the ported components are laid out for.
        'gap-2 whitespace-nowrap [&_svg]:pointer-events-none [&_svg]:shrink-0',
        variant === 'destructive'
          ? fdButtonVariants({ size: size === 'icon' ? 'icon-sm' : undefined }) + ' bg-red-500 text-white hover:bg-red-500/90'
          : variant === 'link'
            ? 'inline-flex items-center text-sm font-medium text-fd-primary underline-offset-4 hover:underline'
            : fdButtonVariants({
                variant: BUTTON_VARIANTS[variant],
                size: size === 'icon' ? 'icon-sm' : undefined,
              }),
        size === 'default' && 'h-9 px-4 py-2',
        size === 'sm' && 'h-8 gap-1.5 px-3',
        size === 'lg' && 'h-10 px-6',
        size === 'icon' && 'size-9',
        className,
      )}
      {...props}
    />
  );
});

// Form fields, in Fumadocs tokens.

const fieldClass =
  'w-full min-w-0 rounded-md border bg-fd-background px-3 py-1 text-sm shadow-xs outline-none transition-colors placeholder:text-fd-muted-foreground focus-visible:ring-2 focus-visible:ring-fd-ring disabled:pointer-events-none disabled:opacity-50';

export const Input = forwardRef<HTMLInputElement, ComponentProps<'input'>>(function Input({ className, ...props }, ref) {
  return <input ref={ref} data-slot="input" className={cn(fieldClass, 'h-9', className)} {...props} />;
});

export const Textarea = forwardRef<HTMLTextAreaElement, ComponentProps<'textarea'>>(function Textarea(
  { className, ...props },
  ref,
) {
  return <textarea ref={ref} data-slot="textarea" className={cn(fieldClass, 'min-h-16 py-2', className)} {...props} />;
});

export function Skeleton({ className, ...props }: ComponentProps<'div'>) {
  return <div data-slot="skeleton" className={cn('animate-pulse rounded-md bg-fd-accent', className)} {...props} />;
}

const badgeVariants = cva(
  'inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-md border px-2 py-0.5 text-xs font-medium whitespace-nowrap [&>svg]:pointer-events-none [&>svg]:size-3',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-fd-primary text-fd-primary-foreground',
        secondary: 'border-transparent bg-fd-secondary text-fd-secondary-foreground',
        destructive: 'border-transparent bg-red-500 text-white',
        outline: 'text-fd-foreground',
      },
    },
    defaultVariants: { variant: 'default' },
  },
);

export function Badge({ className, variant, ...props }: ComponentProps<'span'> & VariantProps<typeof badgeVariants>) {
  return <span data-slot="badge" className={cn(badgeVariants({ variant }), className)} {...props} />;
}

// Collapsible: Fumadocs' collapsible.

export const CollapsibleRoot = Collapsible;
export const CollapsibleTrigger = FdCollapsibleTrigger;
export const CollapsibleContent = FdCollapsibleContent;

// Tooltip: Base UI's tooltip (what Fumadocs builds on), in Fumadocs' popover style.

/** Radix-style `asChild`: Base UI renders the child element instead. */
const asRender = (asChild: boolean | undefined, children: ReactNode) =>
  asChild && isValidElement(children) ? { render: children as ReactElement<Record<string, unknown>> } : { children };

export function TooltipProvider({ delayDuration = 0, children }: { delayDuration?: number; children: ReactNode }) {
  return <BaseTooltip.Provider delay={delayDuration}>{children}</BaseTooltip.Provider>;
}

export const Tooltip = BaseTooltip.Root;

export function TooltipTrigger({ asChild, children, ...props }: { asChild?: boolean; children: ReactNode }) {
  return <BaseTooltip.Trigger {...props} {...asRender(asChild, children)} />;
}

export function TooltipContent({
  side = 'top',
  sideOffset = 4,
  className,
  children,
}: {
  side?: 'top' | 'bottom' | 'left' | 'right';
  sideOffset?: number;
  className?: string;
  children: ReactNode;
}) {
  return (
    <BaseTooltip.Portal>
      <BaseTooltip.Positioner side={side} sideOffset={sideOffset} className="z-50">
        <BaseTooltip.Popup
          className={cn(
            'rounded-md border bg-fd-popover px-2 py-1 text-xs text-fd-popover-foreground shadow-md',
            className,
          )}
        >
          {children}
        </BaseTooltip.Popup>
      </BaseTooltip.Positioner>
    </BaseTooltip.Portal>
  );
}

// Popover: Fumadocs' popover, with the side the ported components choose.

export function Popover({
  open,
  onOpenChange,
  children,
}: {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
}) {
  return (
    <BasePopover.Root open={open} onOpenChange={(next) => onOpenChange?.(next)}>
      {children}
    </BasePopover.Root>
  );
}

export const PopoverTrigger = BasePopover.Trigger;

export const PopoverContent = forwardRef<
  HTMLDivElement,
  Omit<ComponentPropsWithoutRef<'div'>, 'className'> & {
    className?: string;
    align?: 'start' | 'center' | 'end';
    side?: 'top' | 'bottom' | 'left' | 'right';
    sideOffset?: number;
  }
>(function PopoverContent({ className, align = 'center', side = 'bottom', sideOffset = 4, ...props }, ref) {
  return (
    <BasePopover.Portal>
      <BasePopover.Positioner align={align} side={side} sideOffset={sideOffset} className="z-50">
        <BasePopover.Popup
          ref={ref}
          className={cn(
            'z-50 origin-(--transform-origin) rounded-xl border bg-fd-popover text-sm text-fd-popover-foreground shadow-lg focus-visible:outline-none data-closed:animate-fd-popover-out data-open:animate-fd-popover-in',
            className,
          )}
          {...props}
        />
      </BasePopover.Positioner>
    </BasePopover.Portal>
  );
});

// Command: cmdk (what shadcn's command wraps), in Fumadocs tokens.

export function Command({ className, ...props }: ComponentProps<typeof CommandPrimitive>) {
  return (
    <CommandPrimitive
      data-slot="command"
      className={cn('flex h-full w-full flex-col overflow-hidden rounded-xl text-fd-popover-foreground', className)}
      {...props}
    />
  );
}

export function CommandInput({ className, ...props }: ComponentProps<typeof CommandPrimitive.Input>) {
  return (
    <div data-slot="command-input-wrapper" className="flex h-9 items-center gap-2 border-b px-3">
      <SearchIcon className="size-4 shrink-0 opacity-50" />
      <CommandPrimitive.Input
        data-slot="command-input"
        className={cn(
          'flex h-10 w-full bg-transparent py-3 text-sm outline-none placeholder:text-fd-muted-foreground',
          className,
        )}
        {...props}
      />
    </div>
  );
}

export function CommandList({ className, ...props }: ComponentProps<typeof CommandPrimitive.List>) {
  return (
    <CommandPrimitive.List
      data-slot="command-list"
      className={cn('max-h-[300px] scroll-py-1 overflow-x-hidden overflow-y-auto', className)}
      {...props}
    />
  );
}

export function CommandEmpty(props: ComponentProps<typeof CommandPrimitive.Empty>) {
  return <CommandPrimitive.Empty data-slot="command-empty" className="py-6 text-center text-sm" {...props} />;
}

export function CommandGroup({ className, ...props }: ComponentProps<typeof CommandPrimitive.Group>) {
  return (
    <CommandPrimitive.Group
      data-slot="command-group"
      className={cn(
        'overflow-hidden p-1 text-fd-foreground [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-fd-muted-foreground',
        className,
      )}
      {...props}
    />
  );
}

export function CommandSeparator({ className, ...props }: ComponentProps<typeof CommandPrimitive.Separator>) {
  return (
    <CommandPrimitive.Separator data-slot="command-separator" className={cn('-mx-1 h-px bg-fd-border', className)} {...props} />
  );
}

export function CommandItem({ className, ...props }: ComponentProps<typeof CommandPrimitive.Item>) {
  return (
    <CommandPrimitive.Item
      data-slot="command-item"
      className={cn(
        "relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none data-[disabled=true]:pointer-events-none data-[disabled=true]:opacity-50 data-[selected=true]:bg-fd-accent data-[selected=true]:text-fd-accent-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        className,
      )}
      {...props}
    />
  );
}
