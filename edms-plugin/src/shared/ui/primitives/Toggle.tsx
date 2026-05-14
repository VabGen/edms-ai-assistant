import { type ButtonHTMLAttributes } from 'react'
import { cn } from '@shared/lib/cn'

interface ToggleProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'onChange'> {
  checked: boolean
  onChange: (checked: boolean) => void
  label?: string
}

export function Toggle({ checked, onChange, label, className, ...props }: ToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative shrink-0 w-9 h-5 rounded-full transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/40',
        checked ? 'bg-indigo-500' : 'bg-slate-200',
        className,
      )}
      {...props}
    >
      <span
        className={cn(
          'absolute top-[3px] w-[14px] h-[14px] rounded-full bg-white shadow-sm transition-all duration-200',
          checked ? 'left-[18px]' : 'left-[3px]',
        )}
        style={{
          boxShadow: checked
            ? '0 1px 4px rgba(99,102,241,0.30)'
            : '0 1px 2px rgba(0,0,0,0.10)',
        }}
      />
    </button>
  )
}
