import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@shared/lib/cn'

export const buttonVariants = cva(
  'inline-flex items-center justify-center gap-1.5 font-semibold rounded-xl border select-none transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/40',
  {
    variants: {
      variant: {
        primary:
          'bg-indigo-600 text-white border-indigo-600 hover:bg-indigo-700 hover:border-indigo-700',
        ghost:
          'bg-white text-indigo-600 border-indigo-200/50 hover:bg-indigo-600 hover:text-white hover:border-indigo-600',
        danger:
          'bg-white text-red-500 border-red-200/50 hover:bg-red-500 hover:text-white hover:border-red-500',
        muted:
          'bg-transparent text-slate-400 border-transparent hover:bg-white/70 hover:text-slate-700',
        stop:
          'bg-red-50 text-red-400 border-red-200/50 hover:bg-red-500 hover:text-white hover:border-red-500',
        icon:
          'bg-transparent text-slate-400 border-transparent hover:bg-white/70 hover:text-slate-600 rounded-lg',
      },
      size: {
        xs: 'px-2 py-1 text-[9px]',
        sm: 'px-2.5 py-1.5 text-[10px]',
        md: 'px-3.5 py-2 text-[11px]',
        lg: 'px-4 py-2.5 text-sm',
        icon: 'w-7 h-7 p-0 rounded-lg',
      },
    },
    defaultVariants: { variant: 'ghost', size: 'md' },
  },
)

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  ),
)
Button.displayName = 'Button'
