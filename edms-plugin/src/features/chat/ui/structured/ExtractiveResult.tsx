import {
    FileSearch, Calendar, User, Building2, Banknote, ShieldAlert, Hourglass,
    MapPin, Phone, Mail, FileText, Network, Briefcase, Hash, Activity,
    Scale, Link, Contact, AlertTriangle, Play, MoreHorizontal
} from 'lucide-react'
import type { ExtractiveData } from '@/entities/message/model/types'
import { CardFooter } from './common'
import { Card, CardHeader, CardTitle, IconBox } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'

const FACT_CONFIG: Record<string, { variant: 'primary' | 'success' | 'warning' | 'error' | 'zinc'; icon: any }> = {
    'ДАТА': { variant: 'primary', icon: Calendar },
    'ПЕРСОНА': { variant: 'primary', icon: User },
    'ОРГАНИЗАЦИЯ': { variant: 'success', icon: Building2 },
    'СУММА': { variant: 'warning', icon: Banknote },
    'ТРЕБОВАНИЕ': { variant: 'error', icon: ShieldAlert },
    'СРОК': { variant: 'error', icon: Hourglass },
    'АДРЕС': { variant: 'primary', icon: MapPin },
    'ТЕЛЕФОН': { variant: 'primary', icon: Phone },
    'ЭЛ. ПОЧТА': { variant: 'primary', icon: Mail },
    'ДОКУМЕНТ': { variant: 'zinc', icon: FileText },
    'ПОДРАЗДЕЛЕНИЕ': { variant: 'success', icon: Network },
    'ДОЛЖНОСТЬ': { variant: 'warning', icon: Briefcase },
    'НОМЕР': { variant: 'zinc', icon: Hash },
    'СТАТУС': { variant: 'primary', icon: Activity },
    'ЗАКОН': { variant: 'primary', icon: Scale },
    'ССЫЛКА': { variant: 'primary', icon: Link },
    'КОНТАКТ': { variant: 'primary', icon: Contact },
    'РИСК': { variant: 'error', icon: AlertTriangle },
    'ДЕЙСТВИЕ': { variant: 'warning', icon: Play },
    'ПРОЧЕЕ': { variant: 'zinc', icon: MoreHorizontal },
}

const DEFAULT_CONFIG = FACT_CONFIG['ПРОЧЕЕ']!

function getFactConfig(key: string) {
    const upper = key.toUpperCase()
        .replace(/EMAIL/g, 'ЭЛ. ПОЧТА')
        .replace(/E-?MAIL/g, 'ЭЛ. ПОЧТА')
        .replace(/ПОЧТА/g, 'ЭЛ. ПОЧТА')
        .replace(/PHONE/g, 'ТЕЛЕФОН')
        .replace(/TEL/g, 'ТЕЛЕФОН')
        .replace(/ТЕЛ\./g, 'ТЕЛЕФОН')
        .replace(/MOBILE/g, 'ТЕЛЕФОН')
        .replace(/ADDRESS/g, 'АДРЕС')
        .replace(/NUMBER/g, 'НОМЕР')
        .replace(/NUM/g, 'НОМЕР');

    return FACT_CONFIG[upper] ?? DEFAULT_CONFIG
}

export function ExtractiveResult({data}: { data: ExtractiveData }) {
    const categories = [...new Set(data.facts.map(f => f.category.toUpperCase()))]

    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 dark:border-zinc-800">
            <CardHeader className="flex-row items-center gap-4 p-4 space-y-0 bg-gradient-to-br from-blue-50/50 to-indigo-50/50 dark:from-blue-900/10 dark:to-indigo-900/10 border-b border-zinc-100 dark:border-zinc-800">
                <IconBox
                    icon={FileSearch}
                    variant="primary"
                    size="md"
                />
                <div className="flex-1 min-w-0">
                    <CardTitle className="text-base font-bold leading-snug">
                        Извлечённые факты
                    </CardTitle>
                    {data.document_summary && (
                        <div className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400 mt-1 line-clamp-1">
                            {data.document_summary}
                        </div>
                    )}
                </div>
                <div className="px-2 py-1 rounded-lg bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 text-[10px] font-bold uppercase tracking-wider border border-blue-200 dark:border-blue-800">
                    {data.facts.length} фактов
                </div>
            </CardHeader>

            {categories.length > 1 && (
                <div className="flex flex-wrap gap-2 p-4 bg-zinc-50/50 dark:bg-zinc-800/30 border-b border-zinc-100 dark:border-zinc-800">
                    {categories.map(cat => {
                        const cfg = getFactConfig(cat)
                        const count = data.facts.filter(f => f.category.toUpperCase() === cat).length
                        const Icon = cfg.icon
                        return (
                            <div key={cat} className={cn(
                                "flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[10px] font-bold uppercase tracking-tight",
                                cfg.variant === 'primary' && "bg-blue-50 border-blue-100 text-blue-600",
                                cfg.variant === 'success' && "bg-emerald-50 border-emerald-100 text-emerald-600",
                                cfg.variant === 'warning' && "bg-amber-50 border-amber-100 text-amber-600",
                                cfg.variant === 'error' && "bg-rose-50 border-rose-100 text-rose-600",
                                cfg.variant === 'zinc' && "bg-zinc-100 border-zinc-200 text-zinc-600",
                            )}>
                                <Icon size={12} /> {cat} ({count})
                            </div>
                        )
                    })}
                </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-4">
                {data.facts.map((fact, i) => {
                    const cfg = getFactConfig(fact.category)
                    const Icon = cfg.icon
                    return (
                        <div key={i} className={cn(
                            "p-3.5 rounded-xl border transition-all hover:shadow-md",
                            cfg.variant === 'primary' && "bg-blue-50/30 border-blue-100/50 hover:border-blue-200",
                            cfg.variant === 'success' && "bg-emerald-50/30 border-emerald-100/50 hover:border-emerald-200",
                            cfg.variant === 'warning' && "bg-amber-50/30 border-amber-100/50 hover:border-amber-200",
                            cfg.variant === 'error' && "bg-rose-50/30 border-rose-100/50 hover:border-rose-200",
                            cfg.variant === 'zinc' && "bg-zinc-50/30 border-zinc-200/50 hover:border-zinc-300",
                        )}>
                            <div className="flex items-center justify-between gap-2 mb-2.5">
                                <div className={cn(
                                    "flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider",
                                    cfg.variant === 'primary' && "text-blue-600",
                                    cfg.variant === 'success' && "text-emerald-600",
                                    cfg.variant === 'warning' && "text-amber-600",
                                    cfg.variant === 'error' && "text-rose-600",
                                    cfg.variant === 'zinc' && "text-zinc-500",
                                )}>
                                    <Icon size={12} /> {fact.label}
                                </div>
                                <div className="text-[9px] font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-widest opacity-60">
                                    {fact.category}
                                </div>
                            </div>
                            <div className="text-[13px] font-bold text-zinc-900 dark:text-zinc-100 break-words leading-relaxed">
                                {fact.value}
                            </div>
                        </div>
                    )
                })}
            </div>

            <CardFooter/>
        </Card>
    )
}
