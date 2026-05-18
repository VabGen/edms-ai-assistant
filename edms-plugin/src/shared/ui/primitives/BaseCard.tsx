import React, { type ReactNode } from 'react'
import { cn } from '@shared/lib/cn'

interface BaseCardProps {
  children: ReactNode
  onClick?: (() => void) | null
  isClickable?: boolean
  isSelected?: boolean
  className?: string | undefined
  style?: React.CSSProperties | undefined
  title?: string | undefined
}

export function BaseCard({
  children,
  onClick,
  isClickable = false,
  isSelected = false,
  className,
  style,
  title,
}: BaseCardProps) {
  const clickable = isClickable || !!onClick

  return (
    <div
      onClick={onClick || undefined}
      title={title}
      className={cn(
        'relative flex flex-col gap-1 transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)]',
        'bg-white border rounded-[16px] px-[14px] py-[12px] mb-[6px]',
        isSelected ? 'border-indigo-500 bg-indigo-50/30' : 'border-black/5',
        clickable ? 'cursor-pointer hover:bg-slate-50/50 hover:border-indigo-500/20 hover:shadow-lg hover:shadow-indigo-500/5 hover:-translate-y-[1px]' : 'cursor-default',
        className
      )}
      style={{
        boxShadow: isSelected
          ? '0 4px 12px rgba(99, 102, 241, 0.15)'
          : '0 1px 3px rgba(0,0,0,0.03)',
        ...style
      }}
    >
      {children}
    </div>
  )
}
