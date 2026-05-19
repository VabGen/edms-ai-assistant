import React from 'react'
import { cn } from '@shared/lib/cn'

interface ProgressBarProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number // 0 to 100
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error'
}

export const ProgressBar = ({
  value,
  variant = 'primary',
  className,
  ...props
}: ProgressBarProps) => {
  const variants = {
    default: 'bg-zinc-500',
    primary: 'bg-blue-500',
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    error: 'bg-rose-500',
  }

  return (
    <div
      className={cn(
        'h-1.5 w-full bg-zinc-100  rounded-full overflow-hidden',
        className
      )}
      {...props}
    >
      <div
        className={cn(
          'h-full transition-all duration-500 ease-out rounded-full',
          variants[variant]
        )}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}
