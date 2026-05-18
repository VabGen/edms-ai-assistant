import { Globe, AlertCircle } from 'lucide-react'
import type { MultilingualData } from '@/entities/message/model/types'
import { CARD, CARD_HEADER, BADGE_BASE, CardFooter } from './common'

const LANG_NAMES: Record<string, string> = {
    ru: 'Русский', en: 'English', be: 'Белорусская',
    de: 'Deutsch', fr: 'Français', es: 'Español',
    zh: '中文', ja: '日本語', ko: '한국어',
}

function langName(code: string) {
    return LANG_NAMES[code] ?? code.toUpperCase()
}

export function MultilingualResult({data}: { data: MultilingualData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(244,63,94,0.04), rgba(139,92,246,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(244,63,94,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <Globe size={16} style={{color: '#f43f5e'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Многоязычная суммаризация
                    </div>
                    <div style={{
                        display: 'flex', gap: 6, marginTop: 3,
                    }}>
                        <span style={{
                            ...BADGE_BASE,
                            background: 'rgba(244,63,94,0.08)', color: '#9f1239',
                        }}>
                            {langName(data.detected_language)}
                        </span>
                        <span style={{
                            fontSize: 10, color: '#94a3b8', display: 'flex',
                            alignItems: 'center'
                        }}>
                            →
                        </span>
                        <span style={{
                            ...BADGE_BASE,
                            background: 'rgba(139,92,246,0.08)', color: '#5b21b6',
                        }}>
                            {langName(data.summary_language)}
                        </span>
                    </div>
                </div>
            </div>

            <div style={{
                padding: '14px 16px',
                fontSize: 13, color: '#334155', lineHeight: 1.75,
            }}>
                {data.summary.split(/\n\n+/).map((para, i) => (
                    <p key={i} style={{
                        margin: 0,
                        marginBottom: i === data.summary.split(/\n\n+/).length - 1 ? 0 : 10,
                    }}>
                        {para}
                    </p>
                ))}
            </div>

            {data.translation_notes && (
                <div style={{
                    margin: '0 16px 12px', padding: '10px 14px',
                    background: '#f8fafc', border: '1px solid rgba(0,0,0,0.06)',
                    borderRadius: 10, fontSize: 11, color: '#64748b',
                    display: 'flex', gap: 8, alignItems: 'flex-start', lineHeight: 1.5,
                }}>
                    <AlertCircle size={14} style={{flexShrink: 0, marginTop: 1, color: '#94a3b8'}}/>
                    <div>
                        <div style={{fontWeight: 600, marginBottom: 2, color: '#475569'}}>
                            Примечания к переводу
                        </div>
                        {data.translation_notes}
                    </div>
                </div>
            )}

            <CardFooter/>
        </div>
    )
}
