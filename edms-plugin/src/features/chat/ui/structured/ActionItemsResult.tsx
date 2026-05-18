import { ListChecks, User, Clock, Quote } from 'lucide-react'
import type { ActionItemsData } from '@/entities/message/model/types'
import { CARD, CARD_HEADER, BADGE_BASE, CardFooter } from './common'

const PRIORITY_CFG: Record<string, { bg: string; text: string; icon: React.ReactNode; label: string }> = {
    high: {
        bg: 'rgba(239,68,68,0.08)',
        text: '#b91c1c',
        icon: <span style={{fontSize: 10}}>🔥</span>,
        label: 'Высокий',
    },
    medium: {
        bg: 'rgba(245,158,11,0.08)',
        text: '#92400e',
        icon: <span style={{fontSize: 10}}>⚡</span>,
        label: 'Средний',
    },
    low: {
        bg: 'rgba(100,116,139,0.08)',
        text: '#475569',
        icon: <Clock size={10}/>,
        label: 'Низкий',
    },
}

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
        const order = {high: 0, medium: 1, low: 2}
        return (order[a.priority] ?? 1) - (order[b.priority] ?? 1)
    })

    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(245,158,11,0.04), rgba(239,68,68,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(245,158,11,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <ListChecks size={16} style={{color: '#f59e0b'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Задачи и действия
                    </div>
                    {data.document_context && (
                        <div style={{fontSize: 11, color: '#64748b', marginTop: 1}}>
                            {data.document_context}
                        </div>
                    )}
                </div>
                <span style={{
                    ...BADGE_BASE,
                    background: 'rgba(245,158,11,0.08)', color: '#92400e',
                }}>
                    {data.action_items.length} задач
                </span>
            </div>

            <div style={{padding: '8px 0'}}>
                {sorted.map((item, i) => {
                    const cfg = (PRIORITY_CFG[item.priority] ?? PRIORITY_CFG['medium'])!
                    return (
                        <div key={i} style={{
                            padding: '10px 16px',
                            borderBottom: i < sorted.length - 1
                                ? '1px solid rgba(0,0,0,0.04)' : 'none',
                            background: i % 2 === 0 ? 'transparent' : 'rgba(248,250,252,0.5)',
                        }}>
                            <div style={{
                                display: 'flex', alignItems: 'flex-start', gap: 10,
                            }}>
                                <div style={{
                                    minWidth: 6, height: 6, borderRadius: '50%',
                                    background: cfg.text,
                                    marginTop: 6, flexShrink: 0,
                                }}/>

                                <div style={{flex: 1}}>
                                    <div style={{
                                        fontSize: 12, fontWeight: 600, color: '#1e293b',
                                        lineHeight: 1.5,
                                    }}>
                                        {item.task}
                                    </div>

                                    <div style={{
                                        display: 'flex', flexWrap: 'wrap', gap: 6,
                                        marginTop: 5,
                                    }}>
                                        <span style={{
                                            ...BADGE_BASE,
                                            background: cfg.bg, color: cfg.text,
                                        }}>
                                            {cfg.icon} {cfg.label}
                                        </span>

                                        {item.owner && (
                                            <span style={{
                                                ...BADGE_BASE,
                                                background: 'rgba(139,92,246,0.06)',
                                                color: '#5b21b6',
                                            }}>
                                                <User size={9}/> {item.owner}
                                            </span>
                                        )}

                                        {item.deadline && (
                                            <span style={{
                                                ...BADGE_BASE,
                                                background: 'rgba(59,130,246,0.06)',
                                                color: '#1d4ed8',
                                            }}>
                                                <Clock size={9}/> {formatDate(item.deadline)}
                                            </span>
                                        )}
                                    </div>

                                    {item.source_fragment && (
                                        <div style={{
                                            marginTop: 5, padding: '4px 10px',
                                            background: '#f8fafc', borderRadius: 6,
                                            border: '1px solid rgba(0,0,0,0.04)',
                                            fontSize: 11, color: '#64748b',
                                            fontStyle: 'italic', lineHeight: 1.5,
                                            display: 'flex', gap: 5,
                                        }}>
                                            <Quote size={10} style={{
                                                flexShrink: 0, marginTop: 2, color: '#94a3b8',
                                            }}/>
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
        </div>
    )
}
