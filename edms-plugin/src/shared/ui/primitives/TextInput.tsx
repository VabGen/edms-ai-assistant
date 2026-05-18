import { forwardRef, type InputHTMLAttributes } from 'react'
import { cn } from '@shared/lib/cn'

interface TextInputProps extends InputHTMLAttributes<HTMLInputElement> {
  mono?: boolean
}

export const TextInput = forwardRef<HTMLInputElement, TextInputProps>(
  ({ mono, className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        'w-full px-3 py-2 rounded-xl text-[11px] bg-white focus:outline-none transition-all duration-200',
        'border border-black/6 text-slate-900 shadow-[0_1px_2px_rgba(0,0,0,0.03)]',
        'focus:border-indigo-500/30 focus:shadow-[0_0_0_2px_rgba(99,102,241,0.08),0_1px_2px_rgba(0,0,0,0.03)]',
        mono && 'font-mono',
        className,
      )}
      {...props}
    />
  ),
)
TextInput.displayName = 'TextInput'
