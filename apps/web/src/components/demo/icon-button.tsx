import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import { forwardRef, type ComponentPropsWithoutRef } from 'react';
import { cn } from '@/lib/cn';

type IconButtonProps = ComponentPropsWithoutRef<'button'> & {
  /** The button's name, shown on hover and read by screen readers. */
  label: string;
  variant?: 'ghost' | 'outline' | 'primary' | 'secondary';
  size?: 'icon' | 'icon-sm' | 'icon-xs';
};

/** Fumadocs' icon button, named for hover and screen readers. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ label, variant = 'ghost', size = 'icon-xs', className, children, ...props }, ref) => (
    <button
      ref={ref}
      type="button"
      title={label}
      aria-label={label}
      className={cn(buttonVariants({ variant, size }), 'text-fd-muted-foreground', className)}
      {...props}
    >
      {children}
    </button>
  ),
);
IconButton.displayName = 'IconButton';
