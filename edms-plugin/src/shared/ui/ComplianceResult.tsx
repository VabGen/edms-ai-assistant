import {useState, memo, useCallback} from 'react'
import {CheckCircle, AlertTriangle, HelpCircle, RefreshCw, Zap, Lightbulb, Info, XCircle} from 'lucide-react'
import type {ComplianceData, ComplianceField, RefreshMeta} from '@entities/message/model/types'
import {Card, CardHeader, CardTitle, IconBox} from './primitives'
import {cn} from '@shared/lib/cn'

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
        variant: 'success',
        label: 'OK',
    },
    mismatch: {
        Icon: XCircle,
        variant: 'error',
        label: 'Расхождение',
    },
    not_found: {
        Icon: HelpCircle,
        variant: 'zinc',
        label: 'Не в файле',
    },
    missing: {
        Icon: HelpCircle,
        variant: 'zinc',
        label: 'Отсутствует',
    },
    warning: {
        Icon: AlertTriangle,
        variant: 'warning',
        label: 'Внимание',
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
        <div
          onClick={handleClick}
          className={cn(
            "p-4 transition-all group relative",
            canFix ? "cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-800/30" : "cursor-default"
          )}
        >
            <div className="flex items-center justify-between gap-3 mb-3">
                <div className="flex items-center gap-2.5">
                    <IconBox icon={Icon} variant={cfg.variant} size="sm" />
                    <span className="text-[13px] font-bold text-zinc-800 dark:text-zinc-200 leading-none">{field.label}</span>
                </div>
                <span className={cn(
                    "px-1.5 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider",
                    cfg.variant === 'success' && "bg-emerald-50 text-emerald-600",
                    cfg.variant === 'error' && "bg-rose-50 text-rose-600",
                    cfg.variant === 'warning' && "bg-amber-50 text-amber-600",
                    cfg.variant === 'zinc' && "bg-zinc-100 text-zinc-500",
                )}>
                    {cfg.label}
                </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div className="bg-zinc-50 dark:bg-zinc-800/40 p-2.5 rounded-lg border border-zinc-100 dark:border-zinc-800">
                    <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1">В карточке</div>
                    <div className="text-[12px] font-medium text-zinc-700 dark:text-zinc-300 break-words leading-relaxed">{field.card_value}</div>
                </div>
                <div className={cn(
                    "p-2.5 rounded-lg border",
                    field.status === 'mismatch' ? "bg-rose-50/30 border-rose-100/50" : "bg-zinc-50 dark:bg-zinc-800/40 border-zinc-100 dark:border-zinc-800"
                )}>
                    <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1">В файле</div>
                    <div className={cn(
                        "text-[12px] font-bold break-words leading-relaxed",
                        field.status === 'mismatch' ? "text-rose-600" : "text-zinc-700 dark:text-zinc-300",
                        !field.correct_value && "text-zinc-300 italic font-normal"
                    )}>
                        {field.correct_value || '—'}
                    </div>
                </div>
            </div>

            {canFix && (
                <div className="mt-3 flex items-center gap-1.5 text-[10px] font-bold text-blue-500 uppercase tracking-tight animate-pulse">
                    <Zap size={10} /> Нажмите, чтобы исправить
                </div>
            )}
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
    const isError = data.overall === 'has_mismatches'
    const isWarning = data.overall === 'cannot_verify'

    const StatusIcon = isError ? AlertTriangle : (isWarning ? HelpCircle : CheckCircle)
    const statusVariant = isError ? 'warning' : (isWarning ? 'primary' : 'success')
    const statusTitle = isError ? 'Найдены расхождения' : (isWarning ? 'Частичная проверка' : 'Всё заполнено корректно')

    const mismatchFields = data.fields.filter(f => f.status === 'mismatch')
    const otherFields = data.fields.filter(f => f.status !== 'mismatch')
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
        setTimeout(() => setFixingAll(false), 800)
    }, [fixingAll, pendingMismatches, onAllFixed])

    const handleRefresh = useCallback(() => {
        if (onRefreshDocument) onRefreshDocument()
        if (onSendMessage) onSendMessage('Перепроверь соответствие файла и карточки документа')
    }, [onRefreshDocument, onSendMessage])

    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 dark:border-zinc-800 mt-2">
            <CardHeader className={cn(
                "flex-row items-start gap-4 p-4 space-y-0 border-b transition-colors",
                isError ? "bg-amber-50/50 border-amber-100" : "bg-emerald-50/50 border-emerald-100"
            )}>
                <IconBox icon={StatusIcon} variant={statusVariant} size="md" className="mt-1" />
                <div className="flex-1">
                    <CardTitle className={cn(
                        "text-base font-bold",
                        isError ? "text-amber-700" : "text-emerald-700"
                    )}>
                        {statusTitle}
                    </CardTitle>
                    {data.summary && <div className="text-[12px] text-zinc-500 font-medium mt-1 leading-relaxed">{data.summary}</div>}
                </div>
                {pendingMismatches.length > 0 && (
                    <div className="px-2 py-1 rounded-full bg-rose-100 text-rose-600 text-[10px] font-bold uppercase tracking-wider border border-rose-200">
                        {pendingMismatches.length} расх.
                    </div>
                )}
            </CardHeader>

            <div className="p-2 bg-zinc-50/50 border-b border-zinc-100 space-y-1">
                {refreshMeta && (
                    <button type="button" onClick={handleRefresh}
                            className="flex items-center justify-center gap-2 w-full py-2 rounded-xl bg-white border border-zinc-200 text-zinc-600 text-[11px] font-bold hover:bg-zinc-50 active:scale-95 transition-all shadow-sm">
                        <RefreshCw size={12} className="text-zinc-400"/> Обновить данные
                    </button>
                )}

                {pendingMismatches.length > 1 && (
                    <button type="button" onClick={handleFixAll} disabled={fixingAll}
                            className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl bg-blue-600 text-white text-[11px] font-bold hover:bg-blue-700 active:scale-95 transition-all shadow-md shadow-blue-200">
                        {fixingAll ? <RefreshCw size={13} className="animate-spin"/> : <Zap size={13}/>}
                        {fixingAll ? 'Исправляю...' : `Исправить всё (${pendingMismatches.length})`}
                    </button>
                )}
            </div>

            <div className="divide-y divide-zinc-100">
                {mismatchFields.length > 0 && (
                    <div className="bg-rose-50/10">
                        {mismatchFields.map(field => <FieldCard key={field.field_key} field={field} onFixed={onFieldFixed} disabled={fixingAll}/>)}
                    </div>
                )}
                {otherFields.map(field => <FieldCard key={field.field_key} field={field} onFixed={onFieldFixed} disabled={fixingAll}/>)}
            </div>

            <div className="p-3 bg-zinc-50/50 border-t border-zinc-100 text-[11px] text-zinc-400 font-medium flex items-center gap-2">
                <Info size={12} className="text-zinc-300" />
                Результаты проверки соответствия
            </div>
        </Card>
    )
})
ComplianceResult.displayName = 'ComplianceResult'
