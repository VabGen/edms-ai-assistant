import { ListChecks, User, Clock, Quote, Zap, Flame } from 'lucide-react'
import type { ActionItemsData } from '@/entities/message/model/types'
import { CardFooter } from './common'
import { Card, CardHeader, CardTitle, IconBox } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'

const PRIORITY_CONFIG: Record<string, { variant: 'error' | 'warning' | 'zinc'; icon: any; label: string }> = {
    high: {
        variant: 'error',
        icon: Flame,
        label: 'Высокий',
    },
    medium: {
        variant: 'warning',
        icon: Zap,
        label: 'Средний',
    },
    low: {
        variant: 'zinc',
        icon: Clock,
        label: 'Низкий',
    },
}

const DEFAULT_PRIORITY = PRIORITY_CONFIG['medium']!

function formatDate(raw: string): string {
    try {
        const d = new Date(raw)
        if (isNaN(d.getTime())) return raw
        return d.toLocaleDateString('ru-RU', {day: 'numeric', month: 'short', year: 'numeric'})
    } catch {
        return raw
    }
}

export function ActionItemsResult({data}: { data: ActionItemsData }) {
    const sorted = [...data.action_items].sort((a, b) => {
        const order: Record<string, number> = {high: 0, medium: 1, low: 2}
        return (order[a.priority] ?? 1) - (order[b.priority] ?? 1)
    })

    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 dark:border-zinc-800">
            <CardHeader className="flex-row items-center gap-4 p-4 space-y-0 bg-gradient-to-br from-amber-50/50 to-rose-50/50 dark:from-amber-900/10 dark:to-rose-900/10 border-b border-zinc-100 dark:border-zinc-800">
                <IconBox
                    icon={ListChecks}
                    variant="warning"
                    size="md"
                />
                <div className="flex-1 min-w-0">
                    <CardTitle className="text-base font-bold leading-snug">
                        Задачи и действия
                    </CardTitle>
                    {data.document_context && (
                        <div className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400 mt-1 line-clamp-1">
                            {data.document_context}
                        </div>
                    )}
                </div>
                <div className="px-2 py-1 rounded-lg bg-amber-100 dark:bg-amber-900/40 text-amber-600 dark:text-amber-400 text-[10px] font-bold uppercase tracking-wider border border-amber-200 dark:border-amber-800">
                    {data.action_items.length} задач
                </div>
            </CardHeader>

            <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
                {sorted.map((item, i) => {
                    const cfg = PRIORITY_CONFIG[item.priority] ?? DEFAULT_PRIORITY
                    const Icon = cfg.icon
                    return (
                        <div key={i} className="p-4 hover:bg-zinc-50/30 dark:hover:bg-zinc-800/20 transition-all group">
                            <div className="flex items-start gap-4">
                                <div className={cn(
                                    "w-1.5 h-1.5 rounded-full mt-2 shrink-0",
                                    item.priority === 'high' ? "bg-rose-500" : (item.priority === 'medium' ? "bg-amber-500" : "bg-zinc-300")
                                )} />

                                <div className="flex-1 min-w-0">
                                    <div className="text-[14px] font-bold text-zinc-900 dark:text-zinc-100 leading-relaxed mb-3">
                                        {item.task}
                                    </div>

                                    <div className="flex flex-wrap gap-2 mb-3">
                                        <div className={cn(
                                            "flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-tight border",
                                            cfg.variant === 'error' && "bg-rose-50 border-rose-100 text-rose-600",
                                            cfg.variant === 'warning' && "bg-amber-50 border-amber-100 text-amber-600",
                                            cfg.variant === 'zinc' && "bg-zinc-50 border-zinc-200 text-zinc-500",
                                        )}>
                                            <Icon size={12} /> {cfg.label}
                                        </div>

                                        {item.owner && (
                                            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-violet-50 dark:bg-violet-900/30 border border-violet-100 dark:border-violet-800/50 text-violet-600 dark:text-violet-400 text-[10px] font-bold uppercase tracking-tight">
                                                <User size={12}/> {item.owner}
                                            </div>
                                        )}

                                        {item.deadline && (
                                            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-blue-50 dark:bg-blue-900/30 border border-blue-100 dark:border-blue-800/50 text-blue-600 dark:text-blue-400 text-[10px] font-bold uppercase tracking-tight">
                                                <Clock size={12}/> {formatDate(item.deadline)}
                                            </div>
                                        )}
                                    </div>

                                    {item.source_fragment && (
                                        <div className="p-3 bg-zinc-50 dark:bg-zinc-900 rounded-xl border border-zinc-100 dark:border-zinc-800 text-[12px] text-zinc-500 dark:text-zinc-400 italic leading-relaxed flex gap-2.5 items-start">
                                            <Quote size={14} className="shrink-0 mt-0.5 text-zinc-300 dark:text-zinc-600" />
                                            {item.source_fragment}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>

            <CardFooter/>
        </Card>
    )
}
