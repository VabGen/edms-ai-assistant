import { BookOpen, Sparkles } from 'lucide-react'
import type { AbstractiveData } from '@/entities/message/model/types'
import { CARD, CARD_HEADER, BADGE_BASE, THEME_PALETTE, CardFooter } from './common'

export function AbstractiveResult({data}: { data: AbstractiveData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(16,185,129,0.04), rgba(6,182,212,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(16,185,129,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <BookOpen size={16} style={{color: '#10b981'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Краткое изложение
                    </div>
                    <div style={{fontSize: 11, color: '#64748b', marginTop: 1}}>
                        Абстрактивная суммаризация
                    </div>
                </div>
            </div>

            {data.key_themes.length > 0 && (
                <div style={{
                    display: 'flex', flexWrap: 'wrap', gap: 6,
                    padding: '10px 16px',
                    borderBottom: '1px solid rgba(0,0,0,0.04)',
                    background: '#fafbfc',
                }}>
                    {data.key_themes.map((theme, i) => {
                        const colors = Object.values(THEME_PALETTE)
                        const c = colors[(i + 1) % colors.length] ?? colors[0]!
                        return (
                            <span key={i} style={{
                                ...BADGE_BASE,
                                background: c.bg, color: c.text,
                                border: `1px solid ${c.border}`,
                            }}>
                                <Sparkles size={9}/>
                                {theme}
                            </span>
                        )
                    })}
                </div>
            )}

            <div style={{
                padding: '14px 16px',
                fontSize: 13, color: '#334155', lineHeight: 1.75,
            }}>
                {data.summary.split(/\n\n+/).map((para, i) => (
                    <p key={i} style={{
                        margin: 0, marginBottom: i === data.summary.split(/\n\n+/).length - 1 ? 0 : 10,
                    }}>
                        {para}
                    </p>
                ))}
            </div>

            <CardFooter/>
        </div>
    )
}
