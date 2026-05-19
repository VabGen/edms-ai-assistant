import React, { useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight, Sparkles, LucideIcon } from 'lucide-react'
import { CardFooter as BaseCardFooter, IconBox } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'

export const THEME_PALETTE = {
    default: { variant: 'zinc' as const, bg: 'bg-zinc-50', text: 'text-zinc-600', border: 'border-zinc-200' },
    blue: { variant: 'primary' as const, bg: 'bg-blue-50', text: 'text-blue-600', border: 'border-blue-200' },
    indigo: { variant: 'primary' as const, bg: 'bg-indigo-50', text: 'text-indigo-600', border: 'border-indigo-200' },
    violet: { variant: 'primary' as const, bg: 'bg-violet-50', text: 'text-violet-600', border: 'border-violet-200' },
    green: { variant: 'success' as const, bg: 'bg-emerald-50', text: 'text-emerald-600', border: 'border-emerald-200' },
    amber: { variant: 'warning' as const, bg: 'bg-amber-50', text: 'text-amber-600', border: 'border-amber-200' },
    red: { variant: 'error' as const, bg: 'bg-rose-50', text: 'text-rose-600', border: 'border-rose-200' },
    rose: { variant: 'error' as const, bg: 'bg-rose-50', text: 'text-rose-600', border: 'border-rose-200' },
    cyan: { variant: 'primary' as const, bg: 'bg-cyan-50', text: 'text-cyan-600', border: 'border-cyan-200' },
}

export function CardFooter() {
    return (
        <BaseCardFooter className="flex justify-between items-center bg-zinc-50/50  border-t border-zinc-100  py-2.5">
            <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">Сгенерировано AI</span>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-white  border border-zinc-200  shadow-sm">
                <Sparkles size={10} className="text-blue-500"/>
                <span className="text-[10px] font-bold text-zinc-600 ">EDMS Assistant</span>
            </div>
        </BaseCardFooter>
    )
}

export function CollapsibleSection({
                                title,
                                children,
                                defaultOpen = true,
                                icon: Icon,
                                right,
                            }: {
    title: string
    children: ReactNode
    defaultOpen?: boolean
    icon?: LucideIcon
    right?: ReactNode
}) {
    const [open, setOpen] = useState(defaultOpen)
    return (
        <div className="mb-1 border-b border-zinc-100  last:border-0">
            <button
                onClick={() => setOpen(v => !v)}
                className="w-full flex items-center gap-3 py-3.5 px-1 text-left hover:bg-zinc-50/50  transition-all group"
            >
                <div className={cn(
                    "flex items-center justify-center w-5 h-5 rounded-md border border-zinc-200  transition-colors",
                    open ? "bg-zinc-100 " : "bg-white "
                )}>
                   {open ? <ChevronDown size={12} className="text-zinc-500" /> : <ChevronRight size={12} className="text-zinc-500" />}
                </div>

                {Icon && <Icon size={16} className="text-zinc-400 group-hover:text-zinc-600 transition-colors" />}
                <span className="flex-1 text-[13px] font-bold text-zinc-700  tracking-tight">{title}</span>
                {right}
            </button>
            <div className={cn(
                "overflow-hidden transition-all duration-300 ease-in-out",
                open ? "max-h-[5000px] opacity-100 mb-4" : "max-h-0 opacity-0"
            )}>
                <div className="pl-8 pr-1 py-1">
                    {children}
                </div>
            </div>
        </div>
    )
}
