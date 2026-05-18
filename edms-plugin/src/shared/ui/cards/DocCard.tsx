import { useContext } from 'react'
import { User, ExternalLink } from 'lucide-react'
import { BaseCard } from '../primitives/BaseCard'
import { DocumentClickContext } from '../ChatContext'
import { normalizeUuid, isValidUuid } from '@/shared/lib/url'

const CATEGORY_COLORS: Record<string, { bg: string; text: string; label: string }> = {
    'INCOMING': {bg: 'rgba(59,130,246,0.08)', text: '#1d4ed8', label: 'Входящий'},
    'OUTGOING': {bg: 'rgba(16,185,129,0.08)', text: '#065f46', label: 'Исходящий'},
    'INTERN': {bg: 'rgba(139,92,246,0.08)', text: '#5b21b6', label: 'Внутренний'},
    'APPEAL': {bg: 'rgba(245,158,11,0.08)', text: '#92400e', label: 'Обращение'},
    'CONTRACT': {bg: 'rgba(239,68,68,0.08)', text: '#991b1b', label: 'Договор'},
    'MEETING': {bg: 'rgba(99,102,241,0.08)', text: '#3730a3', label: 'Совещание'},
    'ORDER': {bg: 'rgba(244,63,94,0.08)', text: '#9f1239', label: 'Приказ'},
    'CITIZEN': {bg: 'rgba(245,158,11,0.08)', text: '#92400e', label: 'Гражданин'},
}

function getCategoryStyle(raw: string) {
    const upper = raw.toUpperCase().replace(/[()]/g, '').trim()
    for (const [key, val] of Object.entries(CATEGORY_COLORS)) {
        if (upper.includes(key)) return val
    }
    return {bg: 'rgba(100,116,139,0.08)', text: '#334155', label: raw}
}

interface DocCardProps {
  headers: string[]
  row: string[]
  index: number
}

export function DocCard({ headers, row, index }: DocCardProps) {
    const onDocumentClick = useContext(DocumentClickContext)

    const pairs = headers.map((h, i) => ({key: h.trim(), value: (row[i] || '—').trim()}))

    const num = pairs.find(p => /^[№#]$/.test(p.key))?.value
    const regNum = pairs.find(p => /рег.*номер|reg.*num|^номер$/i.test(p.key))?.value
    const date = pairs.find(p => /^дата$|^date$|рег.*дата|reg.*date/i.test(p.key))?.value
    const category = pairs.find(p => /категор|category|тип|type/i.test(p.key))?.value
    const summary = pairs.find(p => /содержан|summary|краткое|описан/i.test(p.key))?.value
    const author = pairs.find(p => /автор|author/i.test(p.key))?.value
    const status = pairs.find(p => /статус|status/i.test(p.key))?.value
    const address = pairs.find(p => /адрес|address/i.test(p.key))?.value

    const rawId = pairs.find(p => /^id$/i.test(p.key))?.value ?? ''
    const docId = rawId ? normalizeUuid(rawId) : ''
    const isClickable = Boolean(onDocumentClick && docId && isValidUuid(docId))

    const _skipKeys = /^[№#]$|^id$|рег.*номер|reg.*num|^номер$|^дата$|^date$|рег.*дата|reg.*date|категор|category|тип|type|содержан|summary|краткое|описан|автор|author|статус|status|адрес|address/i
    const extraPairs = pairs.filter(p => !_skipKeys.test(p.key) && p.value && p.value !== '—')

    const catStyle = category ? getCategoryStyle(category) : null

    return (
        <BaseCard
            onClick={isClickable ? () => onDocumentClick!(docId) : null}
            title={isClickable ? 'Открыть документ в новой вкладке' : undefined}
            isClickable={isClickable}
        >
            <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[10px] font-bold"
                    style={{
                      background: isClickable ? 'rgba(99,102,241,0.08)' : 'rgba(148,163,184,0.08)',
                      color: isClickable ? '#6366f1' : '#94a3b8',
                    }}
                >
                    {num ?? index + 1}
                </span>

                {regNum && regNum !== '—'
                    ? <span className="text-[12px] font-bold text-slate-900 flex-1">{regNum}</span>
                    : <span className="text-[12px] text-slate-400 flex-1">—</span>
                }

                {date && date !== '—' && (
                    <span className="text-[10px] text-slate-400 shrink-0">{date}</span>
                )}

                {isClickable && (
                    <span className="shrink-0 text-indigo-500 opacity-50 flex items-center">
                        <ExternalLink size={12}/>
                    </span>
                )}
            </div>

            {summary && summary !== '—' && (
                <p className="text-[12px] text-slate-600 leading-relaxed line-clamp-2 m-0">
                    {summary}
                </p>
            )}

            <div className="flex flex-wrap gap-1 items-center">
                {catStyle && (
                    <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
                        style={{ background: catStyle.bg, color: catStyle.text }}
                    >
                        {catStyle.label}
                    </span>
                )}
                {author && author !== '—' && (
                    <span className="text-[10px] text-slate-500 px-2 py-0.5 rounded-full bg-slate-100 flex items-center gap-1">
                        <User size={10} className="opacity-70"/>
                        {author}
                    </span>
                )}
                {status && status !== '—' && (
                    <span className="text-[9px] text-slate-500 font-semibold px-2 py-0.5 rounded-[4px] bg-slate-100 uppercase">
                        {status}
                    </span>
                )}

                {address && address !== '—' && (
                    <div className="w-full mt-0.5 text-[11px] text-slate-500 flex items-start gap-1.5 leading-tight">
                        <span className="shrink-0 mt-[1px]">📍</span>
                        <span className="break-all">{address}</span>
                    </div>
                )}

                {extraPairs.map(({key, value}) => {
                    const isContact = key.toLowerCase().includes('контакт') || key.toLowerCase().includes('contact')
                    if (isContact && value) {
                        const parts = value.split(/\s{2,}|\n/).filter(part => part.trim().length > 0)
                        return (
                            <div key={key} className="flex flex-wrap gap-1.5 items-center w-full mt-0.5">
                                {parts.map((part, i) => (
                                    <span key={i} className="text-[10px] text-slate-600 px-2 py-0.5 rounded-[6px] bg-slate-50 border border-black/5 whitespace-nowrap font-mono">
                                        {part.trim()}
                                    </span>
                                ))}
                            </div>
                        )
                    }
                    return (
                        <span key={key} className="text-[10px] text-slate-500 px-2 py-0.5 rounded-full bg-slate-100">
                          {key}: {value}
                        </span>
                    )
                })}
            </div>
        </BaseCard>
    )
}
