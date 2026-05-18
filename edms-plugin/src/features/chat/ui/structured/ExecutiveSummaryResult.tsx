import { Sparkles } from 'lucide-react'
import type { ExecutiveSummaryData } from '@/entities/message/model/types'
import { CARD, CARD_HEADER, CardFooter } from './common'

export function ExecutiveSummaryResult({data}: { data: ExecutiveSummaryData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(139,92,246,0.04), rgba(99,102,241,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(139,92,246,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <Sparkles size={16} style={{color: '#8b5cf6'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{
                        fontSize: 10, fontWeight: 600, color: '#8b5cf6',
                        textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2,
                    }}>
                        Резюме
                    </div>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a', lineHeight: 1.45}}>
                        {data.headline}
                    </div>
                </div>
            </div>

            {data.bullets.length > 0 && (
                <div style={{padding: '12px 16px'}}>
                    {data.bullets.map((bullet, i) => (
                        <div key={i} style={{
                            display: 'flex', alignItems: 'flex-start', gap: 10,
                            padding: '7px 0',
                            borderBottom: i < data.bullets.length - 1
                                ? '1px solid rgba(0,0,0,0.04)' : 'none',
                        }}>
                            <span style={{
                                minWidth: 22, height: 22, borderRadius: 7,
                                background: `rgba(139,92,246,${0.06 + i * 0.02})`,
                                color: '#5b21b6', fontSize: 10, fontWeight: 700,
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                flexShrink: 0, marginTop: 1,
                            }}>
                                {i + 1}
                            </span>
                            <span style={{fontSize: 12, color: '#334155', lineHeight: 1.6}}>
                                {bullet}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {data.recommendation && (
                <div style={{
                    margin: '0 16px 12px', padding: '10px 14px',
                    background: '#fffbeb', border: '1px solid #fcd34d',
                    borderRadius: 10, fontSize: 12, color: '#92400e',
                    display: 'flex', gap: 8, alignItems: 'flex-start', lineHeight: 1.5,
                }}>
                    <span style={{fontSize: 14, flexShrink: 0}}>💡</span>
                    <div>
                        <div style={{fontWeight: 600, marginBottom: 2}}>Рекомендация</div>
                        {data.recommendation}
                    </div>
                </div>
            )}

            <CardFooter/>
        </div>
    )
}
