import { Target, Quote, ArrowRight } from 'lucide-react'
import type { ThesisPlanData } from '@/entities/message/model/types'
import { CARD, CARD_HEADER, BADGE_BASE, CardFooter, CollapsibleSection } from './common'

/** Alternating row background helper */
function idx_bg(i: number) {
    return i % 2 === 0 ? 'rgba(248,250,252,0.6)' : 'transparent'
}

export function ThesisPlanResult({data}: { data: ThesisPlanData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(99,102,241,0.04), rgba(139,92,246,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(99,102,241,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <Target size={16} style={{color: '#6366f1'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{
                        fontSize: 10, fontWeight: 600, color: '#6366f1',
                        textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2
                    }}>
                        Главный тезис
                    </div>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a', lineHeight: 1.45}}>
                        {data.main_argument}
                    </div>
                </div>
            </div>

            <div style={{padding: '4px 0'}}>
                {data.sections.map((section, si) => (
                    <CollapsibleSection
                        key={si}
                        title={section.title || `Раздел ${si + 1}`}
                        icon={<span style={{
                            ...BADGE_BASE,
                            background: 'rgba(99,102,241,0.08)',
                            color: '#4338ca',
                            borderRadius: 6,
                            fontSize: 10,
                        }}>{si + 1}</span>}
                        right={
                            <span style={{fontSize: 10, color: '#94a3b8'}}>
                                {section.points.length} тезисов
                            </span>
                        }
                    >
                        <div style={{
                            padding: '8px 12px', marginBottom: 8,
                            background: 'rgba(99,102,241,0.03)',
                            borderLeft: '3px solid #c7d2fe',
                            borderRadius: '0 8px 8px 0',
                            fontSize: 12, color: '#334155', lineHeight: 1.6,
                        }}>
                            {section.thesis}
                        </div>

                        {section.points.map((point, pi) => (
                            <div key={pi} style={{
                                padding: '8px 12px', marginBottom: 6,
                                background: idx_bg(pi),
                                borderRadius: 10,
                                border: '1px solid rgba(0,0,0,0.03)',
                            }}>
                                <div style={{
                                    display: 'flex', alignItems: 'flex-start', gap: 8,
                                }}>
                                    <span style={{
                                        minWidth: 20, height: 20, borderRadius: 6,
                                        background: 'rgba(99,102,241,0.08)',
                                        color: '#4338ca', fontSize: 10, fontWeight: 700,
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        flexShrink: 0, marginTop: 1,
                                    }}>
                                        {pi + 1}
                                    </span>
                                    <div style={{flex: 1}}>
                                        <div style={{
                                            fontSize: 12, fontWeight: 600, color: '#1e293b',
                                            lineHeight: 1.5,
                                        }}>
                                            {point.claim}
                                        </div>

                                        {point.evidence && (
                                            <div style={{
                                                marginTop: 4, padding: '5px 10px',
                                                background: '#f8fafc', borderRadius: 6,
                                                border: '1px solid rgba(0,0,0,0.04)',
                                                fontSize: 11, color: '#64748b',
                                                fontStyle: 'italic', lineHeight: 1.5,
                                                display: 'flex', gap: 5, alignItems: 'flex-start',
                                            }}>
                                                <Quote size={11} style={{
                                                    flexShrink: 0, marginTop: 2,
                                                    color: '#94a3b8',
                                                }}/>
                                                {point.evidence}
                                            </div>
                                        )}

                                        {point.sub_points && point.sub_points.length > 0 && (
                                            <div style={{
                                                marginTop: 4, paddingLeft: 8,
                                                borderLeft: '2px solid rgba(99,102,241,0.15)',
                                            }}>
                                                {point.sub_points.map((sp, spi) => (
                                                    <div key={spi} style={{
                                                        fontSize: 11, color: '#64748b',
                                                        lineHeight: 1.5, padding: '1px 0',
                                                    }}>
                                                        → {sp}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </CollapsibleSection>
                ))}
            </div>

            {data.conclusion && (
                <div style={{
                    margin: '0 16px 12px', padding: '10px 14px',
                    background: 'linear-gradient(135deg, rgba(16,185,129,0.04), rgba(6,182,212,0.04))',
                    border: '1px solid rgba(16,185,129,0.1)',
                    borderRadius: 10,
                    fontSize: 12, color: '#065f46', lineHeight: 1.6,
                    display: 'flex', gap: 8, alignItems: 'flex-start',
                }}>
                    <ArrowRight size={14} style={{flexShrink: 0, marginTop: 2, color: '#10b981'}}/>
                    <div>
                        <div style={{
                            fontSize: 10, fontWeight: 600, textTransform: 'uppercase',
                            letterSpacing: 0.5, color: '#059669', marginBottom: 2,
                        }}>
                            Вывод
                        </div>
                        {data.conclusion}
                    </div>
                </div>
            )}

            <CardFooter/>
        </div>
    )
}
