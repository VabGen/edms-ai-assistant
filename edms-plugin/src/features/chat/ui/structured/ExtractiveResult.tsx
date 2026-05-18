import { FileSearch } from 'lucide-react'
import type { ExtractiveData } from '@/entities/message/model/types'
import { CARD, CARD_HEADER, BADGE_BASE, CardFooter } from './common'

const STRICT_SVG_PROPS = {
    width: 12,
    height: 12,
    viewBox: '0 0 16 16',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.5,
    strokeLinecap: 'square' as const,
    strokeLinejoin: 'miter' as const,
};

const FACT_CATEGORY: Record<string, { bg: string; text: string; icon: React.ReactNode; border: string }> = {
    'ДАТА': {
        bg: 'rgba(59,130,246,0.08)',
        text: '#1d4ed8',
        border: 'rgba(59,130,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 4h12v10H2V4zM2 7h12M5 4V2M11 4V2"/>
        </svg>,
    },
    'ПЕРСОНА': {
        bg: 'rgba(139,92,246,0.08)',
        text: '#5b21b6',
        border: 'rgba(139,92,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 6a2 2 0 100-4 2 2 0 000 4zM3 14v-2a5 5 0 0110 0v2"/>
        </svg>,
    },
    'ОРГАНИЗАЦИЯ': {
        bg: 'rgba(16,185,129,0.08)',
        text: '#065f46',
        border: 'rgba(16,185,129,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 14V2h5v4h7v8H7M7 6v8"/>
        </svg>,
    },
    'СУММА': {
        bg: 'rgba(245,158,11,0.08)',
        text: '#92400e',
        border: 'rgba(245,158,11,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 4h12v8H2V4zM5 4v8M8 7h3M8 10h3"/>
        </svg>,
    },
    'ТРЕБОВАНИЕ': {
        bg: 'rgba(239,68,68,0.08)',
        text: '#991b1b',
        border: 'rgba(239,68,68,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 1l6 3v5c0 3.5-6 6-6 6s-6-2.5-6-6V4l6-3zM8 5v3M8 11.5v0.5"/>
        </svg>,
    },
    'СРОК': {
        bg: 'rgba(244,63,94,0.08)',
        text: '#9f1239',
        border: 'rgba(244,63,94,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M4 2h8v3L8 8l4 3v3H4v-3l4-3-4-3V2z"/>
        </svg>,
    },
    'АДРЕС': {
        bg: 'rgba(6,182,212,0.08)',
        text: '#155e75',
        border: 'rgba(6,182,212,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 14s-5-4-5-8a5 5 0 0110 0c0 4-5 8-5 8z"/>
        </svg>,
    },
    'ТЕЛЕФОН': {
        bg: 'rgba(139,92,246,0.08)',
        text: '#5b21b6',
        border: 'rgba(139,92,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M4 1h8v14H4V1zM7 13h2"/>
        </svg>,
    },
    'ЭЛ. ПОЧТА': {
        bg: 'rgba(59,130,246,0.08)',
        text: '#1d4ed8',
        border: 'rgba(59,130,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 3h12v10H2V3zM2 3l6 6 6-6"/>
        </svg>,
    },
    'ДОКУМЕНТ': {
        bg: 'rgba(99,102,241,0.08)',
        text: '#4338ca',
        border: 'rgba(99,102,241,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M3 2h7l3 3v9H3V2zM10 2v3h3M5 7h6M5 9h6M5 11h4"/>
        </svg>,
    },
    'ПОДРАЗДЕЛЕНИЕ': {
        bg: 'rgba(20,184,166,0.08)',
        text: '#115e59',
        border: 'rgba(20,184,166,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 2v4M4 6h8M4 6v3M12 6v3M4 9h2M12 9h2"/>
        </svg>,
    },
    'ДОЛЖНОСТЬ': {
        bg: 'rgba(249,115,22,0.08)',
        text: '#9a3412',
        border: 'rgba(249,115,22,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 2l2.5 5H5.5L8 2zM6 7v5M10 7v5M5 12h6"/>
        </svg>,
    },
    'НОМЕР': {
        bg: 'rgba(100,116,139,0.08)',
        text: '#334155',
        border: 'rgba(100,116,139,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M4 3h8M7 2v12M4 13h8"/>
        </svg>,
    },
    'СТАТУС': {
        bg: 'rgba(192,38,211,0.08)',
        text: '#86198f',
        border: 'rgba(192,38,211,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 2h12v12H2V2zM4 8l3 3 5-5"/>
        </svg>,
    },
    'ЗАКОН': {
        bg: 'rgba(14,165,233,0.08)',
        text: '#0369a1',
        border: 'rgba(14,165,233,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 4h12M4 4l2 8M12 4l-2 8M6 12h4M8 2v2"/>
        </svg>,
    },
    'ССЫЛКА': {
        bg: 'rgba(59,130,246,0.08)',
        text: '#1d4ed8',
        border: 'rgba(59,130,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M6 6L3 9l4 4 3-3M10 10l3-3-4-4-3 3"/>
        </svg>,
    },
    'КОНТАКТ': {
        bg: 'rgba(139,92,246,0.08)',
        text: '#5b21b6',
        border: 'rgba(139,92,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 6a2 2 0 100-4 2 2 0 000 4zM3 14v-2a5 5 0 0110 0v2"/>
        </svg>,
    },
    'РИСК': {
        bg: 'rgba(239,68,68,0.08)',
        text: '#991b1b',
        border: 'rgba(239,68,68,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 14L8 2l6 12H2zM8 6v3M8 11.5v0.5"/>
        </svg>,
    },
    'ДЕЙСТВИЕ': {
        bg: 'rgba(245,158,11,0.08)',
        text: '#92400e',
        border: 'rgba(245,158,11,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M4 2l9 6-9 6V2z"/>
        </svg>,
    },
    'ПРОЧЕЕ': {
        bg: 'rgba(100,116,139,0.08)',
        text: '#475569',
        border: 'rgba(100,116,139,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M3 4h10M3 8h10M3 12h10"/>
        </svg>,
    },
};

function factCat(key: string) {
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

    return (FACT_CATEGORY[upper] ?? FACT_CATEGORY['ПРОЧЕЕ'])!
}

export function ExtractiveResult({data}: { data: ExtractiveData }) {
    const categories = [...new Set(data.facts.map(f => f.category.toUpperCase()))]

    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(59,130,246,0.04), rgba(99,102,241,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(59,130,246,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <FileSearch size={16} style={{color: '#3b82f6'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Извлечённые факты
                    </div>
                    {data.document_summary && (
                        <div style={{fontSize: 11, color: '#64748b', marginTop: 1}}>
                            {data.document_summary}
                        </div>
                    )}
                </div>
                <span style={{
                    ...BADGE_BASE,
                    background: 'rgba(59,130,246,0.08)', color: '#1d4ed8',
                }}>
                    {data.facts.length} фактов
                </span>
            </div>

            {categories.length > 1 && (
                <div style={{
                    display: 'flex', flexWrap: 'wrap', gap: 4,
                    padding: '8px 16px',
                    borderBottom: '1px solid rgba(0,0,0,0.04)',
                    background: '#fafbfc',
                }}>
                    {categories.map(cat => {
                        const cfg = factCat(cat)
                        const count = data.facts.filter(f => f.category.toUpperCase() === cat).length
                        return (
                            <span key={cat} style={{
                                ...BADGE_BASE,
                                background: cfg.bg, color: cfg.text,
                            }}>
                                {cfg.icon} {cat} ({count})
                            </span>
                        )
                    })}
                </div>
            )}

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
                gap: 8,
                padding: '12px 16px',
            }}>
                {data.facts.map((fact, i) => {
                    const cfg = factCat(fact.category)
                    return (
                        <div key={i} style={{
                            padding: '10px 12px',
                            borderRadius: 10,
                            background: cfg.bg,
                            border: `1px solid ${cfg.border}`,
                        }}>
                            <div style={{
                                display: 'flex', justifyContent: 'space-between',
                                alignItems: 'center', marginBottom: 4,
                            }}>
                                <span style={{
                                    fontSize: 11,
                                    fontWeight: 600,
                                    color: cfg.text,
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: 4
                                }}>
                                    {cfg.icon} {fact.label}
                                </span>
                                <span style={{
                                    fontSize: 9, fontWeight: 600, textTransform: 'uppercase',
                                    color: cfg.text, opacity: 0.7,
                                    display: 'inline-flex', alignItems: 'center', gap: 3
                                }}>
                                   {cfg.icon} {fact.category}
                                </span>
                            </div>
                            <div style={{
                                fontSize: 12, color: '#1e293b',
                                wordBreak: 'break-word', lineHeight: 1.5,
                            }}>
                                {fact.value}
                            </div>
                        </div>
                    )
                })}
            </div>

            <CardFooter/>
        </div>
    )
}
