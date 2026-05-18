import {useState, memo, useCallback} from 'react'
import {CheckCircle, AlertTriangle, HelpCircle, RefreshCw, Zap} from 'lucide-react'
import {extractDocIdFromUrl} from '@/shared/lib/url'
import type {ComplianceData, ComplianceField, RefreshMeta} from '@entities/message/model/types'


interface Props {
    data: ComplianceData
    threadId: string | null
    refreshMeta?: RefreshMeta | null | undefined
    onFieldFixed: (fieldKey: string, newValue: string) => void
    onAllFixed: (fixedFields: Array<{ fieldKey: string; label: string; newValue: string }>) => void
    onRefreshDocument?: () => void
    onSendMessage?: (text: string) => void
}

const STATUS_CFG: Record<string, any> = {
    ok: {
        Icon: CheckCircle,
        color: '#16a34a',
        bg: 'rgba(22,163,74,0.07)',
        border: 'rgba(22,163,74,0.18)',
        badge: 'OK',
        badgeBg: 'rgba(22,163,74,0.10)',
        badgeColor: '#15803d'
    },
    mismatch: {
        Icon: AlertTriangle,
        color: '#d97706',
        bg: 'rgba(217,119,6,0.07)',
        border: 'rgba(217,119,6,0.22)',
        badge: 'Расхождение',
        badgeBg: 'rgba(217,119,6,0.10)',
        badgeColor: '#b45309'
    },
    not_found: {
        Icon: HelpCircle,
        color: '#94a3b8',
        bg: 'rgba(148,163,184,0.06)',
        border: 'rgba(148,163,184,0.16)',
        badge: 'Не в файле',
        badgeBg: 'rgba(148,163,184,0.10)',
        badgeColor: '#64748b'
    },
    missing: {
        Icon: HelpCircle,
        color: '#94a3b8',
        bg: 'rgba(148,163,184,0.06)',
        border: 'rgba(148,163,184,0.16)',
        badge: 'Отсутствует',
        badgeBg: 'rgba(148,163,184,0.10)',
        badgeColor: '#64748b'
    },
    warning: {
        Icon: AlertTriangle,
        color: '#d97706',
        bg: 'rgba(217,119,6,0.07)',
        border: 'rgba(217,119,6,0.22)',
        badge: 'Внимание',
        badgeBg: 'rgba(217,119,6,0.10)',
        badgeColor: '#b45309'
    },
}

const OVERALL_CFG: Record<string, any> = {
    ok: {
        Icon: CheckCircle,
        color: '#16a34a',
        bg: 'rgba(22,163,74,0.07)',
        border: 'rgba(22,163,74,0.22)',
        title: 'Всё заполнено корректно'
    },
    has_mismatches: {
        Icon: AlertTriangle,
        color: '#d97706',
        bg: 'rgba(217,119,6,0.07)',
        border: 'rgba(217,119,6,0.28)',
        title: 'Найдены расхождения'
    },
    cannot_verify: {
        Icon: HelpCircle,
        color: '#6366f1',
        bg: 'rgba(99,102,241,0.07)',
        border: 'rgba(99,102,241,0.22)',
        title: 'Частичная проверка'
    },
}

interface FieldCardProps {
    field: ComplianceField
    onFixed: (fieldKey: string, newValue: string) => void
    disabled: boolean
}

const FieldCard = memo(({field, onFixed, disabled}: FieldCardProps) => {
    const cfg = STATUS_CFG[field.status] || STATUS_CFG.not_found
    const {Icon} = cfg
    const canFix = field.status === 'mismatch' && !!field.correct_value && !disabled

    const handleClick = useCallback(() => {
        if (!canFix || !field.correct_value) return
        onFixed(field.field_key, field.correct_value)
    }, [canFix, field.field_key, field.correct_value, onFixed])

    return (
        <div onClick={handleClick} style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
            padding: '9px 12px',
            borderRadius: 10,
            background: cfg.bg,
            border: `1px solid ${cfg.border}`,
            cursor: canFix ? 'pointer' : 'default',
            transition: 'all 0.15s'
        }}>
            <span style={{color: cfg.color, flexShrink: 0, marginTop: 1, display: 'flex'}}><Icon size={14}/></span>
            <div style={{flex: 1, minWidth: 0}}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 8,
                    marginBottom: 3
                }}>
                    <span style={{fontSize: 11, fontWeight: 600, color: '#1e293b'}}>{field.label}</span>
                    <span style={{
                        flexShrink: 0,
                        fontSize: 9,
                        fontWeight: 700,
                        padding: '1px 7px',
                        borderRadius: 20,
                        background: cfg.badgeBg,
                        color: cfg.badgeColor,
                        textTransform: 'uppercase',
                        letterSpacing: '0.04em'
                    }}>{cfg.badge}</span>
                </div>
                {field.card_value && <div style={{fontSize: 11, color: '#475569', lineHeight: 1.4}}><span
                    style={{color: '#94a3b8', fontSize: 10}}>В карточке: </span>{field.card_value}</div>}
                {field.status === 'mismatch' && field.correct_value &&
                    <div style={{fontSize: 11, color: '#d97706', lineHeight: 1.4, marginTop: 2}}><span
                        style={{color: '#94a3b8', fontSize: 10}}>В файле: </span><strong>{field.correct_value}</strong>
                    </div>}
                {canFix &&
                    <div style={{marginTop: 4, fontSize: 9, color: '#d97706', opacity: 0.65}}>Нажмите чтобы применить
                        исправление</div>}
            </div>
        </div>
    )
})
FieldCard.displayName = 'FieldCard'

export const ComplianceResult = memo(({
                                          data,
                                          threadId,
                                          refreshMeta,
                                          onFieldFixed,
                                          onAllFixed,
                                          onRefreshDocument,
                                          onSendMessage
                                      }: Props) => {
    const [fixingAll, setFixingAll] = useState(false)
    const overallCfg = OVERALL_CFG[data.overall] || OVERALL_CFG.cannot_verify
    const {Icon: OverallIcon} = overallCfg

    const mismatchFields = data.fields.filter(f => f.status === 'mismatch')
    const notFoundFields = data.fields.filter(f => f.status === 'not_found' || f.status === 'missing')
    const okFields = data.fields.filter(f => f.status === 'ok')
    const pendingMismatches = mismatchFields.filter(f => !!f.correct_value)

    const handleFixAll = useCallback(() => {
        if (fixingAll || pendingMismatches.length === 0) return
        setFixingAll(true)
        const fixedFields = pendingMismatches.map(field => ({
            fieldKey: field.field_key,
            label: field.label,
            newValue: field.correct_value!
        }))
        onAllFixed(fixedFields)
        setTimeout(() => setFixingAll(false), 600)
    }, [fixingAll, pendingMismatches, onAllFixed])

    const handleRefresh = useCallback(() => {
        if (onRefreshDocument) onRefreshDocument()
        if (onSendMessage) onSendMessage('Перепроверь соответствие файла и карточки документа')
    }, [onRefreshDocument, onSendMessage])

    return (
        <div style={{display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4}}>
            <div style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 9,
                padding: '10px 12px',
                borderRadius: 10,
                background: overallCfg.bg,
                border: `1px solid ${overallCfg.border}`
            }}>
                <span style={{color: overallCfg.color, flexShrink: 0, marginTop: 1, display: 'flex'}}><OverallIcon
                    size={15}/></span>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 12, fontWeight: 700, color: overallCfg.color}}>{overallCfg.title}</div>
                    {data.summary && <div
                        style={{fontSize: 11, color: '#64748b', marginTop: 2, lineHeight: 1.5}}>{data.summary}</div>}
                </div>
                {pendingMismatches.length > 0 && (
                    <span style={{
                        flexShrink: 0,
                        fontSize: 11,
                        fontWeight: 700,
                        color: '#d97706',
                        background: 'rgba(217,119,6,0.10)',
                        padding: '2px 8px',
                        borderRadius: 20
                    }}>{pendingMismatches.length} расх.</span>
                )}
            </div>

            {refreshMeta && (
                <button type="button" onClick={handleRefresh}
                        className="flex items-center justify-center gap-2 w-full py-2 rounded-xl bg-slate-50 border border-slate-200 text-slate-600 text-xs font-medium hover:bg-slate-100 active:scale-95 transition-all">
                    <RefreshCw size={12}/> Данные изменены? Перепроверить
                </button>
            )}

            {pendingMismatches.length > 1 && (
                <button type="button" onClick={handleFixAll} disabled={fixingAll}
                        className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-600 text-xs font-semibold hover:bg-indigo-100 active:scale-95 transition-all">
                    {fixingAll ? <RefreshCw size={13} className="animate-spin"/> : <Zap size={13}/>}
                    {fixingAll ? 'Исправляю...' : `Исправить все расхождения (${pendingMismatches.length})`}
                </button>
            )}

            {mismatchFields.length > 0 && (
                <div className="flex flex-col gap-1.5">
                    <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400 pl-0.5">Расхождения —
                        нажмите для исправления
                    </div>
                    {mismatchFields.map(field => <FieldCard key={field.field_key} field={field}
                                                            onFixed={onFieldFixed}
                                                            disabled={fixingAll}/>)}
                </div>
            )}
            {notFoundFields.length > 0 && (
                <div className="flex flex-col gap-1.5">
                    <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400 pl-0.5">Не найдено в
                        файле
                    </div>
                    {notFoundFields.map(field => <FieldCard key={field.field_key} field={field}
                                                            onFixed={onFieldFixed}
                                                            disabled={fixingAll}/>)}
                </div>
            )}
            {data.overall === 'ok' && okFields.length > 0 && (
                <div className="flex flex-col gap-1.5">
                    <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400 pl-0.5">Проверено</div>
                    {okFields.map(field => <FieldCard key={field.field_key} field={field}
                                                      onFixed={onFieldFixed} disabled={false}/>)}
                </div>
            )}
        </div>
    )
})
ComplianceResult.displayName = 'ComplianceResult'