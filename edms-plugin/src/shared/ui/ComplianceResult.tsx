import {useState, memo, useCallback} from 'react'
import {CheckCircle, AlertTriangle, HelpCircle, RefreshCw, Zap} from 'lucide-react'
import {extractDocIdFromUrl} from '@/shared/lib/url'

export interface ComplianceField {
    field_key: string
    label: string
    card_value: string
    file_value: string | null
    correct_value: string | null
    status: 'ok' | 'mismatch' | 'not_found'
    update_field: string
    recommendation: string | null
}

export interface ComplianceData {
    overall: 'ok' | 'has_mismatches' | 'cannot_verify'
    summary: string
    document_id?: string
    fields: ComplianceField[]
    stats?: {
        total?: number
        ok?: number
        mismatches?: number
        not_found?: number
    }
    fix_hint?: string | null
}

interface Props {
    data: ComplianceData
    threadId: string | null
    onFieldFixed: (fieldKey: string, newValue: string) => void
    onAllFixed: (fixedFields: Array<{ fieldKey: string; label: string; newValue: string }>) => void
}

const STATUS_CFG = {
    ok: {
        Icon: CheckCircle,
        color: '#16a34a',
        bg: 'rgba(22,163,74,0.07)',
        border: 'rgba(22,163,74,0.18)',
        badge: 'OK',
        badgeBg: 'rgba(22,163,74,0.10)',
        badgeColor: '#15803d',
    },
    mismatch: {
        Icon: AlertTriangle,
        color: '#d97706',
        bg: 'rgba(217,119,6,0.07)',
        border: 'rgba(217,119,6,0.22)',
        badge: 'Расхождение',
        badgeBg: 'rgba(217,119,6,0.10)',
        badgeColor: '#b45309',
    },
    not_found: {
        Icon: HelpCircle,
        color: '#94a3b8',
        bg: 'rgba(148,163,184,0.06)',
        border: 'rgba(148,163,184,0.16)',
        badge: 'Не в файле',
        badgeBg: 'rgba(148,163,184,0.10)',
        badgeColor: '#64748b',
    },
}

const OVERALL_CFG = {
    ok: {
        Icon: CheckCircle,
        color: '#16a34a',
        bg: 'rgba(22,163,74,0.07)',
        border: 'rgba(22,163,74,0.22)',
        title: 'Всё заполнено корректно',
    },
    has_mismatches: {
        Icon: AlertTriangle,
        color: '#d97706',
        bg: 'rgba(217,119,6,0.07)',
        border: 'rgba(217,119,6,0.28)',
        title: 'Найдены расхождения',
    },
    cannot_verify: {
        Icon: HelpCircle,
        color: '#6366f1',
        bg: 'rgba(99,102,241,0.07)',
        border: 'rgba(99,102,241,0.22)',
        title: 'Частичная проверка',
    },
}

// ── Field card ────────────────────────────────────────────────────────────

interface FieldCardProps {
    field: ComplianceField
    documentId: string
    threadId: string | null
    onFixed: (fieldKey: string, newValue: string) => void
    disabled: boolean
}

const FieldCard = memo(({field, documentId, threadId, onFixed, disabled}: FieldCardProps) => {
    const effectiveStatus = field.status
    const cfg = STATUS_CFG[effectiveStatus]
    const {Icon} = cfg

    const canFix = field.status === 'mismatch' && !!field.correct_value && !disabled

    const handleClick = useCallback(() => {
        if (!canFix) return
        // ★ Parent (AssistantWidget) handles the actual SSE stream + optimistic update
        onFixed(field.field_key, field.correct_value!)
    }, [canFix, field.field_key, field.correct_value, onFixed])

    return (
        <div
            onClick={handleClick}
            style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 10,
                padding: '9px 12px',
                borderRadius: 10,
                background: cfg.bg,
                border: `1px solid ${cfg.border}`,
                cursor: canFix ? 'pointer' : 'default',
                transition: 'all 0.15s',
            }}
            onMouseEnter={canFix ? e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.background = 'rgba(217,119,6,0.13)'
                el.style.borderColor = 'rgba(217,119,6,0.38)'
            } : undefined}
            onMouseLeave={canFix ? e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.background = cfg.bg
                el.style.borderColor = cfg.border
            } : undefined}
        >
            <span style={{color: cfg.color, flexShrink: 0, marginTop: 1, display: 'flex'}}>
                <Icon size={14}/>
            </span>

            <div style={{flex: 1, minWidth: 0}}>
                <div style={{
                    display: 'flex', alignItems: 'center',
                    justifyContent: 'space-between', gap: 8, marginBottom: 3,
                }}>
                    <span style={{fontSize: 11, fontWeight: 600, color: '#1e293b'}}>
                        {field.label}
                    </span>
                    <span style={{
                        flexShrink: 0, fontSize: 9, fontWeight: 700,
                        padding: '1px 7px', borderRadius: 20,
                        background: cfg.badgeBg, color: cfg.badgeColor,
                        textTransform: 'uppercase', letterSpacing: '0.04em',
                    }}>
                        {cfg.badge}
                    </span>
                </div>

                <div style={{fontSize: 11, color: '#475569', lineHeight: 1.4}}>
                    <span style={{color: '#94a3b8', fontSize: 10}}>В карточке: </span>
                    {field.card_value}
                </div>

                {field.status === 'mismatch' && field.correct_value && (
                    <div style={{fontSize: 11, color: '#d97706', lineHeight: 1.4, marginTop: 2}}>
                        <span style={{color: '#94a3b8', fontSize: 10}}>В файле: </span>
                        <strong>{field.correct_value}</strong>
                    </div>
                )}

                {canFix && (
                    <div style={{marginTop: 4, fontSize: 9, color: '#d97706', opacity: 0.65}}>
                        Нажмите чтобы применить исправление
                    </div>
                )}
            </div>

            {canFix && (
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
                     stroke="#d97706" strokeWidth={2.5} strokeLinecap="round"
                     style={{flexShrink: 0, marginTop: 2, opacity: 0.55}}>
                    <path d="M9 18l6-6-6-6"/>
                </svg>
            )}
        </div>
    )
})
FieldCard.displayName = 'FieldCard'

// ── Main component ────────────────────────────────────────────────────────

export const ComplianceResult = memo(({data, threadId, onFieldFixed, onAllFixed}: Props) => {
    const [fixingAll, setFixingAll] = useState(false)

    const documentId = data.document_id ?? extractDocIdFromUrl() ?? ''
    const overallCfg = OVERALL_CFG[data.overall]
    const {Icon: OverallIcon} = overallCfg

    const mismatchFields = data.fields.filter(f => f.status === 'mismatch')
    const notFoundFields = data.fields.filter(f => f.status === 'not_found')
    const okFields = data.fields.filter(f => f.status === 'ok')

    const pendingMismatches = mismatchFields.filter(f => !!f.correct_value)

    const handleFixAll = useCallback(() => {
        if (fixingAll || pendingMismatches.length === 0) return
        setFixingAll(true)

        const fixedFields = pendingMismatches.map(field => ({
            fieldKey: field.field_key,
            label: field.label,
            newValue: field.correct_value!,
        }))

        // ★ Parent (AssistantWidget) handles the actual SSE stream + optimistic update
        onAllFixed(fixedFields)

        // Reset local state after a short delay (parent updates optimistically)
        setTimeout(() => setFixingAll(false), 600)
    }, [fixingAll, pendingMismatches, onAllFixed])

    return (
        <div style={{display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4}}>

            {/* Общий статус */}
            <div style={{
                display: 'flex', alignItems: 'flex-start', gap: 9,
                padding: '10px 12px', borderRadius: 10,
                background: overallCfg.bg, border: `1px solid ${overallCfg.border}`,
            }}>
                <span style={{color: overallCfg.color, flexShrink: 0, marginTop: 1, display: 'flex'}}>
                    <OverallIcon size={15}/>
                </span>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 12, fontWeight: 700, color: overallCfg.color}}>
                        {overallCfg.title}
                    </div>
                    <div style={{fontSize: 11, color: '#64748b', marginTop: 2, lineHeight: 1.5}}>
                        {data.summary}
                    </div>
                </div>
                {((data.stats?.mismatches ?? data.fields.filter(f => f.status === 'mismatch').length) > 0) && pendingMismatches.length > 0 && (
                    <span style={{
                        flexShrink: 0, fontSize: 11, fontWeight: 700,
                        color: '#d97706', background: 'rgba(217,119,6,0.10)',
                        padding: '2px 8px', borderRadius: 20,
                    }}>
                        {pendingMismatches.length} расх.
                    </span>
                )}
            </div>

            {/* Кнопка «Исправить все» */}
            {pendingMismatches.length > 1 && (
                <button
                    type="button"
                    onClick={handleFixAll}
                    disabled={fixingAll}
                    style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        gap: 7, width: '100%', padding: '9px 14px', borderRadius: 10,
                        background: 'rgba(99,102,241,0.09)',
                        border: '1px solid rgba(99,102,241,0.25)',
                        color: '#4f46e5', fontSize: 12, fontWeight: 600,
                        cursor: fixingAll ? 'not-allowed' : 'pointer',
                        transition: 'all 0.15s', opacity: fixingAll ? 0.6 : 1,
                    }}
                    onMouseEnter={!fixingAll ? e => {
                        const el = e.currentTarget as HTMLButtonElement
                        el.style.background = 'rgba(99,102,241,0.15)'
                        el.style.borderColor = 'rgba(99,102,241,0.40)'
                    } : undefined}
                    onMouseLeave={!fixingAll ? e => {
                        const el = e.currentTarget as HTMLButtonElement
                        el.style.background = 'rgba(99,102,241,0.09)'
                        el.style.borderColor = 'rgba(99,102,241,0.25)'
                    } : undefined}
                >
                    {fixingAll
                        ? <RefreshCw size={13} style={{animation: 'spin 0.8s linear infinite'}}/>
                        : <Zap size={13}/>
                    }
                    {fixingAll
                        ? 'Исправляю...'
                        : `Исправить все расхождения (${pendingMismatches.length})`
                    }
                </button>
            )}

            {/* Расхождения */}
            {mismatchFields.length > 0 && (
                <div style={{display: 'flex', flexDirection: 'column', gap: 5}}>
                    <div style={{
                        fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                        letterSpacing: '0.08em', color: '#94a3b8', paddingLeft: 2,
                    }}>
                        Расхождения — нажмите для исправления
                    </div>
                    {mismatchFields.map(field => (
                        <FieldCard key={field.field_key} field={field}
                                   documentId={documentId} threadId={threadId}
                                   onFixed={onFieldFixed} disabled={fixingAll}/>
                    ))}
                </div>
            )}

            {/* Не найдено в файле */}
            {notFoundFields.length > 0 && (
                <div style={{display: 'flex', flexDirection: 'column', gap: 4}}>
                    <div style={{
                        fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                        letterSpacing: '0.08em', color: '#94a3b8', paddingLeft: 2,
                    }}>
                        Не найдено в файле
                    </div>
                    {notFoundFields.map(field => (
                        <FieldCard key={field.field_key} field={field}
                                   documentId={documentId} threadId={threadId}
                                   onFixed={onFieldFixed} disabled={fixingAll}/>
                    ))}
                </div>
            )}

            {/* OK поля — только если нет расхождений */}
            {data.overall === 'ok' && okFields.length > 0 && (
                <div style={{display: 'flex', flexDirection: 'column', gap: 4}}>
                    <div style={{
                        fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                        letterSpacing: '0.08em', color: '#94a3b8', paddingLeft: 2,
                    }}>
                        Проверено
                    </div>
                    {okFields.map(field => (
                        <FieldCard key={field.field_key} field={field}
                                   documentId={documentId} threadId={threadId}
                                   onFixed={onFieldFixed} disabled={false}/>
                    ))}
                </div>
            )}
        </div>
    )
})
ComplianceResult.displayName = 'ComplianceResult'