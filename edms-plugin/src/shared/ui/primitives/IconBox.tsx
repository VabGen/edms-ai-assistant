import React from 'react'
import { LucideIcon } from 'lucide-react'
import { cn } from '@shared/lib/cn'

interface IconBoxProps extends React.HTMLAttributes<HTMLDivElement> {
  icon: LucideIcon
  size?: 'sm' | 'md' | 'lg'
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'zinc'
}

export const IconBox = ({
  icon: Icon,
  size = 'md',
  variant = 'default',
  className,
  ...props
}: IconBoxProps) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  }

  const containers = {
    sm: 'p-1.5 rounded-lg',
    md: 'p-2 rounded-xl',
    lg: 'p-2.5 rounded-2xl',
  }

  const variants = {
    default: 'bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400',
    primary: 'bg-blue-50 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400',
    success: 'bg-emerald-50 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400',
    warning: 'bg-amber-50 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400',
    error: 'bg-rose-50 text-rose-600 dark:bg-rose-900/30 dark:text-rose-400',
    zinc: 'bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300',
  }

  return (
    <div
      className={cn(
        'flex items-center justify-center shrink-0',
        containers[size],
        variants[variant],
        className
      )}
      {...props}
    >
      <Icon className={sizes[size]} />
    </div>
  )
}
