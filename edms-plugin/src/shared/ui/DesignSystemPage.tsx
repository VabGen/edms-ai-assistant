import { useState, useEffect, useCallback, memo, type ReactNode } from 'react'
import {
    MessageSquare, Paperclip, FileText, Search, List, Mic,
    CheckCircle, AlertTriangle, XCircle, HelpCircle, Settings,
    Send, StopCircle, History, X, RefreshCw, Zap, Sparkles,
    ChevronDown, ChevronRight, Clock, User, Quote, ArrowRight,
    Target, BookOpen, FileSearch, Brain, Globe, Copy, Check,
} from 'lucide-react'

/* ═══════════════════════════════════════════════════════════════════════════
   ДИЗАЙН-СИСТЕМА EDMS AI Chat
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Утилиты ────────────────────────────────────────────────────────────────

const C = {
    primary: '#6C63FF',
    primaryLight: '#A78BFA',
    primaryDark: '#534AB7',
    success: '#16a34a',
    successLight: '#22c55e',
    warning: '#d97706',
    warningLight: '#f59e0b',
    danger: '#dc2626',
    dangerLight: '#ef4444',
    tx: '#0f172a',
    tx2: '#475569',
    tx3: '#94a3b8',
    bg: '#f8fafc',
    surface: '#f1f5f9',
    card: '#ffffff',
    border: 'rgba(0,0,0,0.06)',
    border2: 'rgba(0,0,0,0.10)',
}

const SECTION_GAP = 36
const CARD_RADIUS = 16
const PILL_RADIUS = 20

function SectionDivider({ title }: { title: string }) {
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <span style={{
                fontSize: 10, fontWeight: 600, letterSpacing: 1,
                textTransform: 'uppercase', color: C.tx3, whiteSpace: 'nowrap',
            }}>{title}</span>
            <div style={{ flex: 1, height: 0.5, background: C.border }}/>
        </div>
    )
}

function Card({ children, style }: { children: ReactNode; style?: React.CSSProperties }) {
    return (
        <div style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: CARD_RADIUS,
            padding: 14,
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            ...style,
        }}>{children}</div>
    )
}

function Label({ children }: { children: ReactNode }) {
    return (
        <div style={{
            fontSize: 10, fontWeight: 600, letterSpacing: 0.5,
            textTransform: 'uppercase', color: C.tx3, marginBottom: 4,
        }}>{children}</div>
    )
}

// ── Badge ───────────────────────────────────────────────────────────────────

type BadgeVariant = 'purple' | 'green' | 'amber' | 'red' | 'gray'

const BADGE_CFG: Record<BadgeVariant, { bg: string; color: string; border: string; dot: string }> = {
    purple: { bg: 'rgba(108,99,255,0.12)', color: C.primary, border: 'rgba(108,99,255,0.25)', dot: C.primary },
    green:  { bg: 'rgba(34,197,94,0.10)',  color: '#15803d', border: 'rgba(34,197,94,0.2)', dot: '#16a34a' },
    amber:  { bg: 'rgba(251,191,36,0.10)', color: '#92400e', border: 'rgba(251,191,36,0.2)', dot: '#d97706' },
    red:    { bg: 'rgba(239,68,68,0.10)',  color: '#991b1b', border: 'rgba(239,68,68,0.2)', dot: '#dc2626' },
    gray:   { bg: C.surface, color: C.tx2, border: C.border2, dot: C.tx3 },
}

function Badge({ variant, children, dot }: { variant: BadgeVariant; children: ReactNode; dot?: boolean }) {
    const c = BADGE_CFG[variant]
    return (
        <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 5,
            padding: '3px 9px', borderRadius: PILL_RADIUS,
            fontSize: 11, fontWeight: 500,
            background: c.bg, color: c.color,
            border: `0.5px solid ${c.border}`,
        }}>
            {dot && <span style={{ width: 6, height: 6, borderRadius: '50%', background: c.dot, flexShrink: 0 }}/>}
            {children}
        </span>
    )
}

// ── Pill ────────────────────────────────────────────────────────────────────

function Pill({ active, children, onClick }: { active?: boolean; children: ReactNode; onClick?: () => void }) {
    const [hover, setHover] = useState(false)
    return (
        <button
            type="button"
            onClick={onClick}
            onMouseEnter={() => setHover(true)}
            onMouseLeave={() => setHover(false)}
            style={{
                padding: '5px 12px', borderRadius: PILL_RADIUS, fontSize: 12,
                cursor: 'pointer', border: `0.5px solid ${active || hover ? 'rgba(108,99,255,0.3)' : C.border2}`,
                background: active ? 'rgba(108,99,255,0.12)' : hover ? 'rgba(108,99,255,0.07)' : C.surface,
                color: active || hover ? C.primary : C.tx2,
                transition: 'all 0.15s', fontFamily: 'inherit',
            }}
        >{children}</button>
    )
}

// ── Avatar ──────────────────────────────────────────────────────────────────

type AvatarSize = 'xs' | 'sm' | 'md' | 'lg'
type AvatarTone = 'ai' | 'user' | 'green' | 'amber'

const AV_SIZE: Record<AvatarSize, { w: number; fs: number; r: number }> = {
    xs: { w: 22, fs: 9, r: 6 },
    sm: { w: 28, fs: 11, r: 8 },
    md: { w: 36, fs: 13, r: 10 },
    lg: { w: 44, fs: 15, r: 12 },
}

const AV_TONE: Record<AvatarTone, { bg: string; color: string; border: string }> = {
    ai:    { bg: 'rgba(108,99,255,0.12)', color: C.primary, border: 'rgba(108,99,255,0.25)' },
    user:  { bg: C.surface, color: C.tx2, border: C.border2 },
    green: { bg: 'rgba(34,197,94,0.1)', color: '#15803d', border: 'rgba(34,197,94,0.2)' },
    amber: { bg: 'rgba(251,191,36,0.1)', color: '#92400e', border: 'rgba(251,191,36,0.2)' },
}

function Avatar({ size, tone, text }: { size: AvatarSize; tone: AvatarTone; text: string }) {
    const s = AV_SIZE[size]
    const t = AV_TONE[tone]
    return (
        <div style={{
            width: s.w, height: s.w, borderRadius: s.r,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 500, fontSize: s.fs,
            background: t.bg, color: t.color, border: `0.5px solid ${t.border}`,
            flexShrink: 0,
        }}>{text}</div>
    )
}

// ── StatusIndicator ─────────────────────────────────────────────────────────

function StatusIndicator({ live, label, time }: { live?: boolean; label: string; time: string }) {
    const dotColor = live ? '#22c55e' : (time === '—' ? C.tx3 : '#f59e0b')
    return (
        <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '7px 10px', borderRadius: 12,
            background: C.surface, border: `0.5px solid ${C.border}`,
        }}>
            <span style={{
                width: 7, height: 7, borderRadius: '50%', background: dotColor,
                boxShadow: live ? '0 0 0 2px rgba(34,197,94,0.2)' : 'none',
                ...(live ? { animation: 'ds-pulse-dot 2s ease-in-out infinite' } : {}),
            }}/>
            <span style={{ fontSize: 12, color: C.tx2, flex: 1 }}>{label}</span>
            <span style={{ fontSize: 11, color: C.tx3 }}>{time}</span>
        </div>
    )
}

// ── Progress ────────────────────────────────────────────────────────────────

function ProgressBar({ label, value, color }: { label: string; value: number; color: string }) {
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 11, color: C.tx2, minWidth: 80 }}>{label}</span>
            <div style={{
                flex: 1, height: 5, background: C.surface,
                borderRadius: 10, overflow: 'hidden', border: `0.5px solid ${C.border}`,
            }}>
                <div style={{
                    width: `${value}%`, height: '100%', borderRadius: 10,
                    background: color, transition: 'width 0.5s',
                }}/>
            </div>
            <span style={{ fontSize: 11, color: C.tx3, minWidth: 30, textAlign: 'right' }}>{value}%</span>
        </div>
    )
}

// ── FileCard ────────────────────────────────────────────────────────────────

function FileCard({ ext, name, size, badge }: { ext: string; name: string; size: string; badge: ReactNode }) {
    const extCfg: Record<string, { bg: string; color: string }> = {
        PDF: { bg: 'rgba(239,68,68,0.1)', color: '#dc2626' },
        DOC: { bg: 'rgba(59,130,246,0.1)', color: '#1d4ed8' },
        XLS: { bg: 'rgba(34,197,94,0.1)', color: '#15803d' },
    }
    const c = extCfg[ext] ?? { bg: C.surface, color: C.tx2 }
    return (
        <div style={{
            background: C.surface, border: `0.5px solid ${C.border}`,
            borderRadius: 12, padding: '9px 11px', display: 'flex', alignItems: 'center', gap: 10,
        }}>
            <span style={{
                width: 30, height: 30, borderRadius: 8, background: c.bg, color: c.color,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 10, fontWeight: 700, flexShrink: 0,
            }}>{ext}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                    fontSize: 12, fontWeight: 500, color: C.tx,
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>{name}</div>
                <div style={{ fontSize: 11, color: C.tx3 }}>{size}</div>
            </div>
            {badge}
        </div>
    )
}

// ── Message Bubbles ─────────────────────────────────────────────────────────

function BubbleAI({ children }: { children: ReactNode }) {
    return (
        <div style={{
            padding: '10px 13px', borderRadius: '12px 12px 12px 3px',
            fontSize: 12, lineHeight: 1.55,
            background: C.surface, border: `0.5px solid ${C.border}`, color: C.tx,
        }}>{children}</div>
    )
}

function BubbleUser({ children }: { children: ReactNode }) {
    return (
        <div style={{
            padding: '10px 13px', borderRadius: '12px 12px 3px 12px',
            fontSize: 12, lineHeight: 1.55,
            background: 'rgba(108,99,255,0.1)', border: '0.5px solid rgba(108,99,255,0.2)',
            color: C.tx, marginLeft: 'auto', maxWidth: '80%',
        }}>{children}</div>
    )
}

function BubbleError({ title, children }: { title: string; children: ReactNode }) {
    return (
        <div style={{
            padding: '10px 13px', borderRadius: 12,
            fontSize: 12, lineHeight: 1.55,
            background: 'rgba(239,68,68,0.07)', border: '0.5px solid rgba(239,68,68,0.2)', color: C.tx,
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                <XCircle size={12} style={{ color: '#dc2626', flexShrink: 0 }}/>
                <span style={{ fontSize: 11, fontWeight: 500, color: '#dc2626' }}>{title}</span>
            </div>
            {children}
        </div>
    )
}

function BubbleInfo({ title, children }: { title: string; children: ReactNode }) {
    return (
        <div style={{
            padding: '10px 13px', borderRadius: 12,
            fontSize: 12, lineHeight: 1.55,
            background: 'rgba(108,99,255,0.06)', border: '0.5px solid rgba(108,99,255,0.15)', color: C.tx,
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                <HelpCircle size={12} style={{ color: C.primary, flexShrink: 0 }}/>
                <span style={{ fontSize: 11, fontWeight: 500, color: C.primary }}>{title}</span>
            </div>
            {children}
        </div>
    )
}

// ── Disambiguation ──────────────────────────────────────────────────────────

function DisambigCard({ idx, name, dept }: { idx: number; name: string; dept: string }) {
    const [hover, setHover] = useState(false)
    return (
        <div
            onMouseEnter={() => setHover(true)}
            onMouseLeave={() => setHover(false)}
            style={{
                background: hover ? 'rgba(108,99,255,0.06)' : C.surface,
                border: `0.5px solid ${hover ? 'rgba(108,99,255,0.3)' : C.border}`,
                borderRadius: 12, padding: '9px 12px',
                display: 'flex', alignItems: 'center', gap: 10,
                cursor: 'pointer', transition: 'all 0.15s',
            }}
        >
            <span style={{
                width: 20, height: 20, borderRadius: 6,
                background: 'rgba(108,99,255,0.1)', border: '0.5px solid rgba(108,99,255,0.2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 10, fontWeight: 500, color: C.primary, flexShrink: 0,
            }}>{idx}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                    fontSize: 12, fontWeight: 500, color: C.tx,
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>{name}</div>
                <div style={{ fontSize: 11, color: C.tx3 }}>{dept}</div>
            </div>
            <ChevronDown size={12} style={{ color: C.tx3, transform: 'rotate(-90deg)', flexShrink: 0 }}/>
        </div>
    )
}

// ── Toast ───────────────────────────────────────────────────────────────────

type ToastType = 'success' | 'warn' | 'error' | 'info'

const TOAST_CFG: Record<ToastType, { bg: string; border: string; iconBg: string; iconColor: string; icon: ReactNode }> = {
    success: {
        bg: 'rgba(34,197,94,0.07)', border: 'rgba(34,197,94,0.2)',
        iconBg: 'rgba(34,197,94,0.15)', iconColor: '#16a34a',
        icon: <CheckCircle size={9}/>,
    },
    warn: {
        bg: 'rgba(251,191,36,0.07)', border: 'rgba(251,191,36,0.2)',
        iconBg: 'rgba(251,191,36,0.15)', iconColor: '#d97706',
        icon: <AlertTriangle size={9}/>,
    },
    error: {
        bg: 'rgba(239,68,68,0.07)', border: 'rgba(239,68,68,0.2)',
        iconBg: 'rgba(239,68,68,0.15)', iconColor: '#dc2626',
        icon: <XCircle size={9}/>,
    },
    info: {
        bg: 'rgba(108,99,255,0.07)', border: 'rgba(108,99,255,0.2)',
        iconBg: 'rgba(108,99,255,0.15)', iconColor: C.primary,
        icon: <Sparkles size={9}/>,
    },
}

function Toast({ type, text }: { type: ToastType; text: string }) {
    const c = TOAST_CFG[type]
    return (
        <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '10px 13px', borderRadius: 12,
            border: `0.5px solid ${c.border}`, fontSize: 12,
            background: c.bg, color: C.tx,
        }}>
            <span style={{
                width: 16, height: 16, borderRadius: '50%', background: c.iconBg,
                color: c.iconColor, display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0, fontSize: 9, fontWeight: 700,
            }}>{c.icon}</span>
            <span style={{ flex: 1 }}>{text}</span>
            <span style={{ color: C.tx3, cursor: 'pointer', fontSize: 14, lineHeight: 1 }}>×</span>
        </div>
    )
}

// ── Metric Card ─────────────────────────────────────────────────────────────

function MetricCard({ label, value, delta, up }: { label: string; value: string; delta: string; up: boolean }) {
    return (
        <div style={{
            background: C.surface, borderRadius: 12, padding: '12px 14px',
        }}>
            <div style={{ fontSize: 11, color: C.tx3, marginBottom: 6 }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 500, color: C.tx, lineHeight: 1 }}>{value}</div>
            <div style={{ fontSize: 11, marginTop: 4, color: up ? '#16a34a' : '#dc2626' }}>{delta}</div>
        </div>
    )
}

// ── Color Ramp ──────────────────────────────────────────────────────────────

function ColorRamp({ label, colors }: { label: string; colors: string[] }) {
    return (
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <span style={{ fontSize: 10, color: C.tx3, minWidth: 72 }}>{label}</span>
            <div style={{ display: 'flex', gap: 3, flex: 1 }}>
                {colors.map((c, i) => (
                    <div key={i} style={{
                        flex: 1, height: i === 4 ? 36 : 28,
                        background: c, border: `0.5px solid ${C.border}`,
                        borderRadius: i === 0 ? '5px 0 0 5px' : i === colors.length - 1 ? '0 5px 5px 0' : 0,
                        position: 'relative', display: 'flex', alignItems: 'flex-end', justifyContent: 'center',
                        paddingBottom: i === 4 ? 4 : 0,
                    }}>
                        {i === 4 && <span style={{ fontSize: 9, fontWeight: 500, color: '#fff' }}>Main</span>}
                    </div>
                ))}
            </div>
        </div>
    )
}

// ── Animation Row ───────────────────────────────────────────────────────────

function AnimationDemo({ duration }: { duration: number }) {
    const [progress, setProgress] = useState(0)
    useEffect(() => {
        const id = setInterval(() => {
            setProgress(p => p >= 100 ? 0 : p + 4)
        }, duration / 25)
        return () => clearInterval(id)
    }, [duration])
    return (
        <div style={{
            height: 3, borderRadius: 10, background: 'rgba(108,99,255,0.2)',
            overflow: 'hidden', width: 80,
        }}>
            <div style={{
                height: '100%', borderRadius: 10, background: C.primary,
                width: `${progress}%`, transition: `width ${duration}ms ease-out`,
            }}/>
        </div>
    )
}

// ── Icon Cell ───────────────────────────────────────────────────────────────

function IconCell({ icon, name }: { icon: ReactNode; name: string }) {
    const [hover, setHover] = useState(false)
    return (
        <div
            onMouseEnter={() => setHover(true)}
            onMouseLeave={() => setHover(false)}
            style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5,
                padding: '8px 4px', borderRadius: 12, cursor: 'pointer',
                background: hover ? C.surface : 'transparent', transition: 'background 0.15s',
            }}
        >
            <div style={{
                width: 32, height: 32, borderRadius: 9, background: C.surface,
                border: `0.5px solid ${C.border}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: name === 'ИИ ✦' ? C.primary : C.tx2,
            }}>{icon}</div>
            <span style={{ fontSize: 9, color: C.tx3, textAlign: 'center' }}>{name}</span>
        </div>
    )
}

// ── Spacing / Radius visual ────────────────────────────────────────────────

function SpacingBox({ size }: { size: number }) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <div style={{
                width: size, height: size, borderRadius: Math.max(2, size / 8),
                background: 'rgba(108,99,255,0.12)', border: '0.5px solid rgba(108,99,255,0.25)',
                display: 'flex', alignItems: 'flex-end', justifyContent: 'center', paddingBottom: 4,
            }}>
                {size >= 16 && <span style={{ fontSize: 9, color: C.primary }}>{size}px</span>}
            </div>
            <span style={{ fontSize: 9, color: C.tx3 }}>{size}px</span>
        </div>
    )
}

function RadiusBox({ radius, label }: { radius: number; label: string }) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <div style={{
                width: 44, height: 44, borderRadius: radius,
                background: 'rgba(108,99,255,0.12)', border: '0.5px solid rgba(108,99,255,0.25)',
            }}/>
            <span style={{ fontSize: 9, color: C.tx3, textAlign: 'center' }}>{label}</span>
        </div>
    )
}


/* ═══════════════════════════════════════════════════════════════════════════
   ГЛАВНЫЙ КОМПОНЕНТ
   ═══════════════════════════════════════════════════════════════════════════ */

export function DesignSystemPage() {
    const [activePill, setActivePill] = useState('Суммаризация')

    return (
        <div style={{
            display: 'flex', flexDirection: 'column', gap: SECTION_GAP,
            padding: '20px 0 32px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            // Animations keyframes injected via style tag below
        }}>

            {/* Inject keyframes */}
            <style>{`
                @keyframes ds-pulse-dot {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.4; }
                }
            `}</style>

            {/* ── ТИПОГРАФИКА ── */}
            <div>
                <SectionDivider title="Типографика"/>
                <Card style={{ gap: 0 }}>
                    {[
                        { meta: '18px / 500 — Заголовок H1', sample: 'Анализ договора поставки', fs: 18, fw: 500 },
                        { meta: '14px / 500 — Заголовок H2', sample: 'Ключевые условия и сроки', fs: 14, fw: 500 },
                        { meta: '13px / 400 — Тело сообщения', sample: 'Стороны договора: ООО «Альфа» и АО «Бета». Срок поставки — до 15 июля 2026 года.', fs: 13, fw: 400, color: C.tx2 },
                        { meta: '12px / 400 — Подпись / мета', sample: 'Сегодня, 09:42 · Договор · 2 вложения', fs: 12, fw: 400, color: C.tx3 },
                        { meta: '11px mono — Код / ID', sample: 'gpt-oss:120b-cloud · http://127.0.0.1:11434', fs: 11, fw: 400, mono: true, color: C.primary },
                        { meta: '10px / 500 uppercase — Лейбл', sample: 'История · Документы · Настройки', fs: 10, fw: 500, upper: true, color: C.tx3 },
                    ].map((row, i) => (
                        <div key={i} style={{
                            display: 'flex', alignItems: 'baseline', gap: 16,
                            padding: 6, borderBottom: i < 5 ? `0.5px solid ${C.border}` : 'none',
                        }}>
                            <span style={{ fontSize: 11, color: C.tx3, minWidth: 180, flexShrink: 0 }}>{row.meta}</span>
                            <span style={{
                                fontSize: row.fs, fontWeight: row.fw as any, color: (row as any).color ?? C.tx,
                                textTransform: (row as any).upper ? 'uppercase' : undefined,
                                letterSpacing: (row as any).upper ? 1 : undefined,
                                fontFamily: (row as any).mono ? 'ui-monospace, monospace' : undefined,
                            }}>{row.sample}</span>
                        </div>
                    ))}
                </Card>
            </div>

            {/* ── ЦВЕТОВАЯ ПАЛИТРА ── */}
            <div>
                <SectionDivider title="Цветовая палитра"/>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <ColorRamp label="Primary" colors={['#EEEDFE','#C4B5FD','#A78BFA','#8B5CF6','#6C63FF','#534AB7','#3C3489']}/>
                    <ColorRamp label="Success" colors={['#EAF3DE','#C0DD97','#97C459','#639922','#3B6D11','#27500A','#173404']}/>
                    <ColorRamp label="Warning" colors={['#FAEEDA','#FAC775','#EF9F27','#BA7517','#854F0B','#633806','#412402']}/>
                    <ColorRamp label="Danger"  colors={['#FCEBEB','#F7C1C1','#F09595','#E24B4A','#A32D2D','#791F1F','#501313']}/>
                    <ColorRamp label="Neutral" colors={['#F1EFE8','#D3D1C7','#B4B2A9','#888780','#5F5E5A','#444441','#2C2C2A']}/>
                </div>
            </div>

            {/* ── КОМПОНЕНТЫ ── */}
            <div>
                <SectionDivider title="Компоненты"/>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>

                    {/* Badges */}
                    <Card>
                        <Label>Бейджи / статусы</Label>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            <Badge variant="purple" dot>Активен</Badge>
                            <Badge variant="green" dot>Выполнен</Badge>
                            <Badge variant="amber" dot>На контроле</Badge>
                            <Badge variant="red" dot>Просрочен</Badge>
                            <Badge variant="gray">Черновик</Badge>
                            <Badge variant="purple">PDF</Badge>
                            <Badge variant="green">DOCX</Badge>
                            <Badge variant="amber">XLSX</Badge>
                        </div>
                    </Card>

                    {/* Quick Pills */}
                    <Card>
                        <Label>Быстрые действия</Label>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {['Суммаризация','Поиск','Тезисы','Поручение','Автозаполнение','Сравнение'].map(p => (
                                <Pill key={p} active={activePill === p} onClick={() => setActivePill(p)}>{p}</Pill>
                            ))}
                        </div>
                    </Card>

                    {/* Avatars */}
                    <Card>
                        <Label>Аватары</Label>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                            {(['xs','sm','md','lg'] as AvatarSize[]).map(s => (
                                <div key={s} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                                    <Avatar size={s} tone="ai" text="ИИ"/>
                                    <span style={{ fontSize: 9, color: C.tx3 }}>{s}</span>
                                </div>
                            ))}
                            <div style={{ width: 0.5, height: 40, background: C.border, margin: '0 4px' }}/>
                            <Avatar size="md" tone="user" text="ЮА"/>
                            <Avatar size="md" tone="green" text="АК"/>
                            <Avatar size="md" tone="amber" text="ИП"/>
                        </div>
                    </Card>

                    {/* Status */}
                    <Card>
                        <Label>Состояние системы</Label>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            <StatusIndicator live label="Модель активна" time="0 мс"/>
                            <StatusIndicator label="Redis — высокая нагрузка" time="340 мс"/>
                            <StatusIndicator label="Трассировка отключена" time="—"/>
                        </div>
                    </Card>

                    {/* Progress */}
                    <Card>
                        <Label>Прогресс</Label>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            <ProgressBar label="Обработка" value={72} color={C.primary}/>
                            <ProgressBar label="Загрузка" value={45} color="#16a34a"/>
                            <ProgressBar label="Кэш" value={90} color="#d97706"/>
                        </div>
                    </Card>

                    {/* File Cards */}
                    <Card>
                        <Label>Карточки файлов</Label>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            <FileCard ext="PDF" name="Договор_поставки.pdf" size="2.3 МБ · загружен 08.05"
                                      badge={<Badge variant="purple">✦ AI</Badge>}/>
                            <FileCard ext="DOC" name="Обращение_гражданина.docx" size="540 КБ · загружен вчера"
                                      badge={<Badge variant="green">Готово</Badge>}/>
                        </div>
                    </Card>
                </div>
            </div>

            {/* ── СОСТОЯНИЯ СООБЩЕНИЙ ── */}
            <div>
                <SectionDivider title="Состояния сообщений"/>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        <Label>Типы пузырей</Label>
                        <BubbleAI>Анализ завершён. Найдено 3 ключевых условия.</BubbleAI>
                        <BubbleUser>Проанализируй этот договор</BubbleUser>
                        <BubbleError title="Ошибка подключения">Нет соединения с сервером. Проверьте подключение.</BubbleError>
                        <BubbleInfo title="Уточнение">Найдено несколько совпадений. Выберите нужного сотрудника.</BubbleInfo>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        <Label>Disambiguation</Label>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                            <DisambigCard idx={1} name="Иванов Алексей Петрович" dept="Отдел закупок · Ведущий специалист"/>
                            <DisambigCard idx={2} name="Иванов Дмитрий Сергеевич" dept="Юридический · Старший юрист"/>
                            <DisambigCard idx={3} name="Иванов Роман Игоревич" dept="ИТ-отдел · Сисадмин"/>
                        </div>
                    </div>
                </div>
            </div>

            {/* ── ПОЛЕ ВВОДА + ТОСТЫ ── */}
            <div>
                <SectionDivider title="Поле ввода — состояния"/>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        <Label>Пустое / с текстом</Label>
                        <InputPreview empty/>
                        <InputPreview/>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        <Label>Уведомления (тосты)</Label>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                            <Toast type="success" text="Ознакомление добавлено успешно"/>
                            <Toast type="warn" text="Найдено несколько совпадений"/>
                            <Toast type="error" text="Ошибка: нет соединения"/>
                        </div>
                    </div>
                </div>
            </div>

            {/* ── МЕТРИКИ ── */}
            <div>
                <SectionDivider title="Метрики"/>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                    <MetricCard label="Запросов сегодня" value="1 248" delta="↑ 12% к вчера" up/>
                    <MetricCard label="Кэш hit rate" value="73%" delta="↑ 5% к норме" up/>
                    <MetricCard label="Ср. латентность" value="1.4 с" delta="↑ 0.2 с к норме" up={false}/>
                    <MetricCard label="Стоимость за день" value="$0.87" delta="↓ $0.12 к вчера" up/>
                </div>
            </div>

            {/* ── ОТСТУПЫ И СКРУГЛЕНИЯ ── */}
            <div>
                <SectionDivider title="Отступы и скругления"/>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                    <div>
                        <Label>Spacing scale</Label>
                        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, flexWrap: 'wrap' }}>
                            {[4, 8, 12, 16, 24, 32].map(s => <SpacingBox key={s} size={s}/>)}
                        </div>
                    </div>
                    <div>
                        <Label>Border radius</Label>
                        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                            {[
                                { r: 4, l: '4px' }, { r: 8, l: '8px md' }, { r: 12, l: '12px lg' },
                                { r: 16, l: '16px xl' }, { r: 24, l: '24px 2xl' }, { r: 22, l: 'full' },
                            ].map(({ r, l }) => <RadiusBox key={l} radius={r} label={l}/>)}
                        </div>
                    </div>
                </div>
            </div>

            {/* ── АНИМАЦИИ ── */}
            <div>
                <SectionDivider title="Анимации"/>
                <Card style={{ padding: 0, overflow: 'hidden' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                        <thead>
                            <tr>
                                {['Название','Длительность','Easing','Применение','Превью'].map(h => (
                                    <th key={h} style={{
                                        textAlign: 'left', padding: '7px 10px',
                                        fontSize: 10, fontWeight: 500, color: C.tx3,
                                        textTransform: 'uppercase', letterSpacing: 0.5,
                                        borderBottom: `0.5px solid ${C.border}`,
                                    }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {[
                                { name: 'Fade Up', dur: 250, ease: 'ease-out', use: 'Появление сообщений' },
                                { name: 'Micro',   dur: 150, ease: 'ease',     use: 'Hover на кнопках' },
                                { name: 'Spring',  dur: 350, ease: 'cubic-bezier(.22,1,.36,1)', use: 'Открытие виджета' },
                                { name: 'Bounce',  dur: 1200,ease: 'ease-in-out', use: 'Индикатор печати' },
                                { name: 'Slide',   dur: 300, ease: 'ease-out', use: 'Sidebar открытие' },
                            ].map((row, i) => (
                                <tr key={i}>
                                    <td style={{ padding: '7px 10px', fontWeight: 500, color: C.tx, borderBottom: `0.5px solid ${C.border}` }}>{row.name}</td>
                                    <td style={{ padding: '7px 10px', color: C.tx2, borderBottom: `0.5px solid ${C.border}` }}>{row.dur}ms</td>
                                    <td style={{ padding: '7px 10px', fontFamily: 'ui-monospace, monospace', fontSize: 11, color: C.tx2, borderBottom: `0.5px solid ${C.border}` }}>{row.ease}</td>
                                    <td style={{ padding: '7px 10px', color: C.tx2, borderBottom: `0.5px solid ${C.border}` }}>{row.use}</td>
                                    <td style={{ padding: '7px 10px', borderBottom: `0.5px solid ${C.border}` }}>
                                        <AnimationDemo duration={row.dur}/>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </Card>
            </div>

            {/* ── ИКОНКИ UI ── */}
            <div>
                <SectionDivider title="Иконки UI"/>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 6 }}>
                    {[
                        { icon: <MessageSquare size={16}/>, name: 'Чат' },
                        { icon: <Paperclip size={16}/>, name: 'Файл' },
                        { icon: <Mic size={16}/>, name: 'Голос' },
                        { icon: <Search size={16}/>, name: 'Поиск' },
                        { icon: <Sparkles size={16} style={{ color: C.primary }}/> , name: 'ИИ ✦' },
                        { icon: <FileText size={16}/>, name: 'Документ' },
                        { icon: <Zap size={16}/>, name: 'Статус' },
                        { icon: <Settings size={16}/>, name: 'Настройки' },
                    ].map(item => (
                        <IconCell key={item.name} icon={item.icon} name={item.name}/>
                    ))}
                </div>
            </div>

        </div>
    )
}

/* ── InputPreview ─────────────────────────────────────────────────────────── */

function InputPreview({ empty }: { empty?: boolean }) {
    return (
        <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            background: C.surface, border: `0.5px solid ${empty ? C.border2 : 'rgba(108,99,255,0.4)'}`,
            borderRadius: 28, padding: '8px 11px',
        }}>
            <Paperclip size={14} style={{ color: empty ? C.tx3 : C.primary, flexShrink: 0 }}/>
            <span style={{
                flex: 1, fontSize: 12,
                color: empty ? C.tx3 : C.tx,
            }}>{empty ? 'Спросите AI...' : 'Проанализируй договор'}</span>
            <div style={{
                width: 28, height: 28, borderRadius: 8,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: empty ? C.surface : C.primary,
                border: empty ? `0.5px solid ${C.border}` : 'none',
                color: empty ? C.tx3 : '#fff',
                cursor: 'pointer', flexShrink: 0,
            }}>
                <Send size={12} style={{ marginLeft: 1 }}/>
            </div>
        </div>
    )
}