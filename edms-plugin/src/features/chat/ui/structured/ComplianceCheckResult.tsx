import { FileText, XCircle, AlertTriangle, CheckCircle, Info, Lightbulb } from 'lucide-react'
import type { ComplianceData, ComplianceField } from '@/entities/message/model/types'
import { Card, CardHeader, CardTitle, IconBox } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'

export function ComplianceCheckResult({data}: { data: ComplianceData }) {
    const isError = data.overall === 'has_mismatches'
    const isWarning = data.overall === 'cannot_verify'

    const statusVariant = isError ? 'error' as const : (isWarning ? 'warning' as const : 'success' as const)
    const StatusIcon = isError ? XCircle : (isWarning ? AlertTriangle : CheckCircle)
    const statusText = isError
        ? 'Найдены расхождения'
        : (isWarning ? 'Требуется проверка' : 'Проверка пройдена')

    const okCount = data.fields.filter((f: ComplianceField) => f.status === 'ok').length
    const errCount = data.fields.filter((f: ComplianceField) => f.status === 'mismatch').length
    const naCount = data.fields.filter((f: ComplianceField) => f.status === 'not_found').length

    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 ">
            <CardHeader className={cn(
                "flex-row items-start gap-4 p-4 space-y-0 border-b",
                isError && "bg-rose-50/50  border-rose-100/50 ",
                isWarning && "bg-amber-50/50  border-amber-100/50 ",
                !isError && !isWarning && "bg-emerald-50/50  border-emerald-100/50 "
            )}>
                <IconBox
                    icon={StatusIcon}
                    variant={statusVariant}
                    size="md"
                    className="mt-1"
                />
                <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 mb-1">
                        <CardTitle className={cn(
                            "text-base font-bold flex items-center gap-2",
                            isError && "text-rose-600 ",
                            isWarning && "text-amber-600 ",
                            !isError && !isWarning && "text-emerald-600 "
                        )}>
                            {statusText}
                        </CardTitle>
                    </div>
                    <div className="text-[13px] text-zinc-600  leading-relaxed font-medium">
                        {data.summary}
                    </div>
                </div>
            </CardHeader>

            <div className="flex items-center gap-4 px-4 py-3 bg-zinc-50/50  border-b border-zinc-100 ">
                {okCount > 0 && (
                    <div className="flex items-center gap-1.5 text-[11px] font-bold text-emerald-600  uppercase tracking-tight">
                        <CheckCircle size={12} /> {okCount} ок
                    </div>
                )}
                {errCount > 0 && (
                    <div className="flex items-center gap-1.5 text-[11px] font-bold text-rose-600  uppercase tracking-tight">
                        <XCircle size={12} /> {errCount} ошибка
                    </div>
                )}
                {naCount > 0 && (
                    <div className="flex items-center gap-1.5 text-[11px] font-bold text-zinc-400  uppercase tracking-tight">
                        <Info size={12} /> {naCount} не найдено
                    </div>
                )}
            </div>

            <div className="divide-y divide-zinc-100 ">
                {data.fields.map((field: ComplianceField, idx: number) => {
                    const isFieldError = field.status === 'mismatch'
                    const isFieldOk = field.status === 'ok'

                    return (
                        <div key={idx} className="p-4 hover:bg-zinc-50/30  transition-all group">
                            <div className="flex items-center justify-between gap-3 mb-3">
                                <div className="flex items-center gap-2.5">
                                    <div className={cn(
                                        "w-1.5 h-1.5 rounded-full shrink-0",
                                        isFieldError ? "bg-rose-500" : (isFieldOk ? "bg-emerald-500" : "bg-zinc-300")
                                    )} />
                                    <span className="text-[13px] font-bold text-zinc-800  leading-none">{field.label}</span>
                                </div>
                                <span className={cn(
                                    "px-1.5 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider",
                                    isFieldError ? "bg-rose-50 text-rose-600  " :
                                    (isFieldOk ? "bg-emerald-50 text-emerald-600  " :
                                    "bg-zinc-100 text-zinc-500  ")
                                )}>
                                    {isFieldError ? 'Ошибка' : (isFieldOk ? 'OK' : 'Пропуск')}
                                </span>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-zinc-50  p-2.5 rounded-lg border border-zinc-100 ">
                                    <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1">В карточке</div>
                                    <div className="text-[12px] font-medium text-zinc-700  break-words leading-relaxed">{field.card_value}</div>
                                </div>
                                <div className={cn(
                                    "p-2.5 rounded-lg border",
                                    isFieldError ? "bg-rose-50/30  border-rose-100/50 " : "bg-zinc-50  border-zinc-100 "
                                )}>
                                    <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1">В файле</div>
                                    <div className={cn(
                                        "text-[12px] font-bold break-words leading-relaxed",
                                        isFieldError ? "text-rose-600 " : "text-zinc-700 ",
                                        !field.file_value && "text-zinc-300  italic font-normal"
                                    )}>
                                        {field.file_value || '—'}
                                    </div>
                                </div>
                            </div>

                            {field.recommendation && (
                                <div className="mt-3 p-3 bg-amber-50/50  border border-amber-100  rounded-xl text-[12px] text-amber-700  flex gap-2.5 items-start font-medium leading-relaxed">
                                    <Lightbulb size={14} className="shrink-0 mt-0.5 text-amber-500" />
                                    {field.recommendation}
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            <div className="p-3 bg-zinc-50/50  border-t border-zinc-100  text-[11px] text-zinc-400 font-medium flex items-center gap-2">
                <Info size={12} className="text-zinc-300" />
                Проверено AI. Результат добавлен в карточку.
            </div>
        </Card>
    )
}
