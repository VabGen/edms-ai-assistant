import React, { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@shared/lib/cn'

interface CollapsibleSectionProps {
  title: React.ReactNode
  children: React.ReactNode
  defaultOpen?: boolean
  className?: string
  icon?: React.ReactNode
}

export const CollapsibleSection = ({
  title,
  children,
  defaultOpen = false,
  className,
  icon,
}: CollapsibleSectionProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className={cn('border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden bg-white dark:bg-zinc-900', className)}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full p-3 text-left transition-colors hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2.5">
          {icon}
          <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</span>
        </div>
        <ChevronDown
          className={cn(
            'w-4 h-4 text-zinc-500 transition-transform duration-200',
            isOpen && 'rotate-180'
          )}
        />
      </button>
      <div
        className={cn(
          'overflow-hidden transition-all duration-200 ease-in-out',
          isOpen ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'
        )}
      >
        <div className="p-3 pt-0 border-t border-zinc-100 dark:border-zinc-800/50 mt-1">
          {children}
        </div>
      </div>
    </div>
  )
}
