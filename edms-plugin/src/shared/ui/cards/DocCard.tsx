import { useContext } from 'react'
import { FileText, ExternalLink, FilePieChart, FileSearch, FileSignature, FileClock, ShieldCheck, Briefcase, Users } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, IconBox } from '../primitives'
import { DocumentClickContext } from '../ChatContext'
import { normalizeUuid, isValidUuid } from '@/shared/lib/url'
import { cn } from '@shared/lib/cn'

const CATEGORY_MAP: Record<string, { variant: any; icon: any; label: string }> = {
    'INCOMING': { variant: 'primary', icon: FileSearch, label: 'Входящий' },
    'OUTGOING': { variant: 'zinc', icon: ExternalLink, label: 'Исходящий' },
    'INTERN': { variant: 'default', icon: FileText, label: 'Внутренний' },
    'APPEAL': { variant: 'warning', icon: FilePieChart, label: 'Обращение' },
    'CONTRACT': { variant: 'success', icon: FileSignature, label: 'Договор' },
    'MEETING': { variant: 'primary', icon: Users, label: 'Совещание' },
    'ORDER': { variant: 'error', icon: ShieldCheck, label: 'Приказ' },
    'CITIZEN': { variant: 'warning', icon: Briefcase, label: 'Гражданин' },
}

function getCategoryConfig(raw: string) {
    const upper = raw.toUpperCase().replace(/[()]/g, '').trim()
    for (const [key, val] of Object.entries(CATEGORY_MAP)) {
        if (upper.includes(key)) return val
    }
    return { variant: 'default', icon: FileText, label: raw }
}

interface DocCardProps {
    headers: string[]
    row: string[]
    index: number
}

export function DocCard({ headers, row, index }: DocCardProps) {
    const onDocumentClick = useContext(DocumentClickContext)

    const pairs = headers.map((h, i) => ({ key: h.trim(), value: (row[i] || '—').trim() }))

    const num = pairs.find(p => /^[№#]$/.test(p.key))?.value
    const regNum = pairs.find(p => /рег.*номер|reg.*num|^номер$/i.test(p.key))?.value
    const date = pairs.find(p => /^дата$|^date$|рег.*дата|reg.*date/i.test(p.key))?.value
    const category = pairs.find(p => /категор|category|тип|type/i.test(p.key))?.value
    const summary = pairs.find(p => /содержан|summary|краткое|описан/i.test(p.key))?.value
    const author = pairs.find(p => /автор|author/i.test(p.key))?.value
    const status = pairs.find(p => /статус|status/i.test(p.key))?.value

    const rawId = pairs.find(p => /^(id|uuid|идентификатор|doc.*id|document.*id)$/i.test(p.key))?.value ?? ''
    const docId = rawId ? normalizeUuid(rawId) : ''
    const isClickable = Boolean(onDocumentClick && docId && isValidUuid(docId))

    const config = category ? getCategoryConfig(category) : { variant: 'default' as const, icon: FileText, label: 'Документ' }

    return (
        <Card
            isClickable={isClickable}
            onClick={isClickable ? () => onDocumentClick!(docId) : undefined}
            className="mb-3 group"
        >
            <CardHeader className="flex-row items-start gap-4 p-4 space-y-0">
                <IconBox
                    icon={config.icon}
                    variant={config.variant}
                    size="md"
                    className="mt-0.5"
                />
                <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 mb-1">
                        <CardTitle className="truncate">
                            {regNum && regNum !== '—' ? regNum : `Документ #${num ?? index + 1}`}
                        </CardTitle>
                        {isClickable && (
                            <ExternalLink size={14} className="text-zinc-400 transition-colors group-hover:text-blue-500 shrink-0" />
                        )}
                    </div>
                    <CardDescription className="flex items-center gap-2 mb-2">
                        <span className={cn(
                            "px-1.5 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider",
                            config.variant === 'primary' && "bg-blue-50 text-blue-600",
                            config.variant === 'success' && "bg-emerald-50 text-emerald-600",
                            config.variant === 'warning' && "bg-amber-50 text-amber-600",
                            config.variant === 'error' && "bg-rose-50 text-rose-600",
                            config.variant === 'zinc' && "bg-zinc-100 text-zinc-600",
                            config.variant === 'default' && "bg-zinc-100 text-zinc-500",
                        )}>
                            {config.label}
                        </span>
                        {date && date !== '—' && (
                            <span className="flex items-center gap-1 text-[11px] text-zinc-400">
                                <FileClock size={12} />
                                {date}
                            </span>
                        )}
                    </CardDescription>

                    {summary && summary !== '—' && (
                        <p className="text-[13px] text-zinc-600 dark:text-zinc-400 leading-relaxed line-clamp-2 m-0 mb-2">
                            {summary}
                        </p>
                    )}

                    <div className="flex flex-wrap gap-2 items-center">
                        {author && author !== '—' && (
                            <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-zinc-50 dark:bg-zinc-800/50 border border-zinc-100 dark:border-zinc-800 text-[11px] text-zinc-500">
                                <Users size={12} />
                                {author}
                            </div>
                        )}
                        {status && status !== '—' && (
                            <div className="px-2 py-1 rounded-lg bg-zinc-50 dark:bg-zinc-800/50 border border-zinc-100 dark:border-zinc-800 text-[10px] font-semibold text-zinc-500 uppercase tracking-tight">
                                {status}
                            </div>
                        )}
                    </div>
                </div>
            </CardHeader>
        </Card>
    )
}
