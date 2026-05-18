import { Brain, Clock } from 'lucide-react'
import type { DetailedNotesData } from '@/entities/message/model/types'
import { CARD, CARD_HEADER, BADGE_BASE, CardFooter, CollapsibleSection } from './common'

export function DetailedNotesResult({data}: { data: DetailedNotesData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(6,182,212,0.04), rgba(59,130,246,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(6,182,212,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <Brain size={16} style={{color: '#06b6d4'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Подробные заметки
                    </div>
                    <div style={{
                        display: 'flex', gap: 8, marginTop: 3, flexWrap: 'wrap',
                    }}>
                        <span style={{
                            ...BADGE_BASE,
                            background: 'rgba(6,182,212,0.08)', color: '#155e75',
                        }}>
                            {data.document_type}
                        </span>
                        {data.date_range && (
                            <span style={{
                                ...BADGE_BASE,
                                background: 'rgba(59,130,246,0.06)', color: '#1d4ed8',
                            }}>
                                <Clock size={9}/> {data.date_range}
                            </span>
                        )}
                    </div>
                </div>
            </div>

            <div style={{padding: '4px 0'}}>
                {data.sections.map((section, i) => (
                    <CollapsibleSection
                        key={i}
                        title={section.title}
                        defaultOpen={i === 0}
                        icon={
                            <span style={{
                                ...BADGE_BASE, borderRadius: 6, fontSize: 10,
                                background: 'rgba(6,182,212,0.08)', color: '#155e75',
                            }}>
                                {i + 1}
                            </span>
                        }
                    >
                        <div style={{
                            fontSize: 12, color: '#334155',
                            lineHeight: 1.7, marginBottom: 6,
                        }}>
                            {section.content}
                        </div>

                        {section.subsections && section.subsections.length > 0 && (
                            <div style={{
                                paddingLeft: 12,
                                borderLeft: '2px solid rgba(6,182,212,0.2)',
                                marginTop: 4,
                            }}>
                                {section.subsections.map((sub, j) => (
                                    <div key={j} style={{
                                        fontSize: 11, color: '#64748b',
                                        lineHeight: 1.5, padding: '2px 0',
                                    }}>
                                        → {sub}
                                    </div>
                                ))}
                            </div>
                        )}
                    </CollapsibleSection>
                ))}
            </div>

            {data.key_entities.length > 0 && (
                <div style={{
                    padding: '10px 16px',
                    borderTop: '1px solid rgba(0,0,0,0.04)',
                    background: '#fafbfc',
                }}>
                    <div style={{
                        fontSize: 10, fontWeight: 600, color: '#94a3b8',
                        textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6
                    }}>
                        Ключевые сущности
                    </div>
                    <div style={{display: 'flex', flexWrap: 'wrap', gap: 4}}>
                        {data.key_entities.map((entity, i) => (
                            <span key={i} style={{
                                ...BADGE_BASE,
                                background: 'rgba(100,116,139,0.06)',
                                color: '#475569',
                                fontFamily: 'ui-monospace, monospace',
                                fontSize: 10,
                            }}>
                                {entity}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            <CardFooter/>
        </div>
    )
}
