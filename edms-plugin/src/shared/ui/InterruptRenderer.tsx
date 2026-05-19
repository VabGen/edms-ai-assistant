import {useState} from 'react'
import {ExternalLink, User, FileText, ChevronRight} from 'lucide-react'
import type {InterruptPayload, ResumeValue} from '@entities/interrupt/model/types'
import {sendMessage} from '@shared/api/messaging'
import {Card, CardHeader, CardTitle, CardDescription, IconBox} from './primitives'
import {cn} from '@shared/lib/cn'
import {getCategoryConfig} from './cards/DocCard'

interface Props {
    payload: InterruptPayload
    onReply: (resume: ResumeValue) => void
}

export function InterruptRenderer({payload, onReply}: Props) {
    const [selectedId, setSelectedId] = useState<string | null>(null)

    const handleSelect = (id: string, resume: ResumeValue) => {
        setSelectedId(id)
        onReply(resume)
    }

    // ── card_select ────────────────────────────────────────────────────────
    if (payload.kind === 'card_select') {
        return (
            <div className="flex flex-col gap-2">
                {payload.prompt && (
                    <p className="text-[13px] font-medium text-zinc-500 dark:text-zinc-400 mb-1 ml-1 px-1">
                        {payload.prompt}
                    </p>
                )}
                {payload.cards.map((card, idx) => {
                    const isSelected = selectedId === card.id
                    const cardUrl = typeof card.metadata?.['url'] === 'string'
                        ? (card.metadata['url'] as string)
                        : null
                    const isEmployee = card.badges?.some(b => b.toLowerCase().includes('сотрудник') || b.toLowerCase().includes('физлицо'));
                    const docCategory = card.metadata?.['category'] as string | undefined;
                    const docConfig = docCategory ? getCategoryConfig(docCategory) : null;

                    return (
                        <div key={card.id} className="flex items-stretch gap-2 group/row">
                            <Card
                                isSelected={isSelected}
                                isClickable={true}
                                onClick={() =>
                                    handleSelect(card.id, {
                                        kind: 'card_select',
                                        selected_ids: [card.id],
                                    })
                                }
                                className={cn(
                                    "flex-1 min-w-0 transition-all duration-300",
                                    isSelected && "border-indigo-500 bg-indigo-50/30",
                                    !isSelected && docConfig && (
                                        docConfig.variant === 'primary' ? "hover:border-indigo-200" :
                                        docConfig.variant === 'success' ? "hover:border-emerald-200" :
                                        docConfig.variant === 'warning' ? "hover:border-amber-200" :
                                        docConfig.variant === 'error' ? "hover:border-rose-200" :
                                        "hover:border-zinc-300"
                                    )
                                )}
                            >
                                <CardHeader className="flex-row items-center gap-3 p-4 space-y-0">
                                    <div className={cn(
                                        "w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[10px] font-bold border transition-all",
                                        isSelected
                                          ? "bg-indigo-600 border-indigo-400 text-white"
                                          : "bg-zinc-100 border-zinc-200 text-zinc-500 group-hover/row:bg-zinc-200"
                                    )}>
                                        {idx + 1}
                                    </div>

                                    {card.image_url ? (
                                        <div className="shrink-0 w-8 h-8 rounded-full overflow-hidden border border-zinc-100">
                                            <img src={card.image_url} alt="" className="w-full h-full object-cover" />
                                        </div>
                                    ) : (
                                        <IconBox
                                            icon={isEmployee ? User : (docConfig?.icon || FileText)}
                                            variant={isSelected ? 'primary' : (docConfig?.variant || 'zinc')}
                                            size="sm"
                                        />
                                    )}

                                    <div className="flex-1 min-w-0">
                                        <CardTitle className={cn(
                                            "text-[14px] font-bold truncate transition-colors",
                                            isSelected && "text-indigo-700"
                                        )}>
                                            {card.label}
                                        </CardTitle>
                                        {card.description && (
                                            <CardDescription className={cn(
                                                "text-[12px] line-clamp-2 mt-0.5",
                                                isSelected && "text-indigo-600/70"
                                            )}>
                                                {card.description}
                                            </CardDescription>
                                        )}
                                        <div className="flex flex-wrap gap-2 mt-2">
                                            {docConfig && (
                                                <div className={cn(
                                                    "px-1.5 py-0.5 rounded-md text-[9px] font-bold uppercase tracking-wider",
                                                    docConfig.variant === 'primary' && "bg-indigo-50 text-indigo-600",
                                                    docConfig.variant === 'success' && "bg-emerald-50 text-emerald-600",
                                                    docConfig.variant === 'warning' && "bg-amber-50 text-amber-600",
                                                    docConfig.variant === 'error' && "bg-rose-50 text-rose-600",
                                                    docConfig.variant === 'zinc' && "bg-zinc-100 text-zinc-600",
                                                    docConfig.variant === 'default' && "bg-zinc-100 text-zinc-500",
                                                )}>
                                                    {docConfig.label}
                                                </div>
                                            )}
                                            {card.badges?.map(badge => (
                                                <div key={badge} className="px-1.5 py-0.5 rounded-md bg-zinc-100 dark:bg-zinc-800 text-zinc-500 text-[9px] font-bold uppercase tracking-wider">
                                                    {badge}
                                                </div>
                                            ))}
                                            {Object.entries(card.primary_attrs ?? {}).map(([k, v]) => (
                                                <div key={k} className={cn(
                                                    "text-[10px] font-bold px-1.5 py-0.5 rounded-md border tracking-tight uppercase",
                                                    isSelected
                                                      ? "bg-blue-100/50 border-blue-200/50 text-blue-600"
                                                      : "bg-zinc-50 border-zinc-100 text-zinc-400"
                                                )}>
                                                    <span className="opacity-60">{k}:</span> {v}
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <ChevronRight className={cn(
                                        "w-4 h-4 transition-all",
                                        isSelected ? "text-blue-500 transform translate-x-1" : "text-zinc-300"
                                    )} />
                                </CardHeader>
                            </Card>

                            {cardUrl && (
                                <button
                                    type="button"
                                    title="Открыть в новой вкладке"
                                    onClick={(e) => {
                                        e.stopPropagation()
                                        void sendMessage('navigateTo', {url: cardUrl, newTab: true})
                                    }}
                                    className="shrink-0 w-12 border border-zinc-200 dark:border-zinc-800 rounded-xl bg-white dark:bg-zinc-900 text-zinc-400 hover:text-blue-500 hover:border-blue-200 dark:hover:border-blue-900/50 hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-all flex items-center justify-center shadow-sm"
                                >
                                    <ExternalLink size={18}/>
                                </button>
                            )}
                        </div>
                    )
                })}
            </div>
        )
    }

    // ── disambiguation ─────────────────────────────────────────────────────
    if (payload.kind === 'disambiguation') {
        return (
            <div className="flex flex-col gap-2">
                {payload.prompt && (
                    <p className="text-[13px] font-medium text-zinc-500 dark:text-zinc-400 mb-1 ml-1 px-1">
                        {payload.prompt}
                    </p>
                )}
                {payload.options.map((opt) => {
                    const isSelected = selectedId === opt.id
                    return (
                        <Card
                            key={opt.id}
                            isSelected={isSelected}
                            isClickable={true}
                            onClick={() =>
                                handleSelect(opt.id, {
                                    kind: 'disambiguation',
                                    selected_ids: [opt.id],
                                })
                            }
                            className="p-4 transition-all duration-300"
                        >
                            <div className="flex items-center justify-between gap-4">
                                <div className="flex-1 min-w-0">
                                    <CardTitle className={cn(
                                        "text-[14px] font-bold mb-1",
                                        isSelected && "text-blue-700 dark:text-blue-300"
                                    )}>
                                        {opt.label}
                                    </CardTitle>
                                    {opt.description && (
                                        <CardDescription className="text-[12px] line-clamp-1">
                                            {opt.description}
                                        </CardDescription>
                                    )}
                                </div>
                                <div className={cn(
                                    "w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all",
                                    isSelected ? "border-blue-500 bg-blue-500" : "border-zinc-200 bg-white"
                                )}>
                                    {isSelected && <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />}
                                </div>
                            </div>
                        </Card>
                    )
                })}
            </div>
        )
    }

    // ── select ─────────────────────────────────────────────────────────────
    if (payload.kind === 'select') {
        return (
            <Card className="overflow-hidden">
                {payload.prompt && (
                    <div className="p-3.5 bg-zinc-50 dark:bg-zinc-800/50 border-b border-zinc-100 dark:border-zinc-800">
                        <p className="text-[13px] font-bold text-zinc-600 dark:text-zinc-300 uppercase tracking-tight">
                            {payload.prompt}
                        </p>
                    </div>
                )}
                <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
                    {payload.options.map((opt) => {
                        const isSelected = selectedId === opt.id
                        return (
                            <button
                                key={opt.id}
                                type="button"
                                onClick={() =>
                                    handleSelect(opt.id, {
                                        kind: 'select',
                                        selected_id: opt.id,
                                    })
                                }
                                className={cn(
                                    "w-full px-4 py-3 text-left transition-all flex items-center justify-between group",
                                    isSelected ? "bg-blue-50 dark:bg-blue-900/20" : "hover:bg-zinc-50/50 dark:hover:bg-zinc-800/30"
                                )}
                            >
                                <div className="flex-1 min-w-0">
                                    <div className={cn(
                                        "text-[13px] font-bold transition-colors",
                                        isSelected ? "text-blue-600 dark:text-blue-400" : "text-zinc-700 dark:text-zinc-200"
                                    )}>
                                        {opt.label}
                                    </div>
                                    {opt.description && (
                                        <div className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-0.5">{opt.description}</div>
                                    )}
                                </div>
                                <ChevronRight className={cn(
                                    "w-4 h-4 transition-all",
                                    isSelected ? "text-blue-500 opacity-100" : "text-zinc-200 opacity-0 group-hover:opacity-100"
                                )} />
                            </button>
                        )
                    })}
                </div>
            </Card>
        )
    }

    // ── confirmation ───────────────────────────────────────────────────────
    if (payload.kind === 'confirmation') {
        return (
            <Card className="p-4 border-l-4 border-l-blue-500 dark:border-l-blue-600">
                {payload.prompt && (
                    <p className="text-[14px] font-bold text-zinc-800 dark:text-zinc-200 mb-4 leading-relaxed px-1">
                        {payload.prompt}
                    </p>
                )}
                <div className="flex gap-2">
                    <button
                        type="button"
                        onClick={() => {
                            setSelectedId('confirm')
                            onReply({kind: 'confirmation', confirmed: true})
                        }}
                        className={cn(
                            "flex-1 px-4 py-2.5 rounded-xl text-[13px] font-bold transition-all shadow-sm active:scale-95",
                            payload.danger
                                ? "bg-rose-500 hover:bg-rose-600 text-white shadow-rose-200 dark:shadow-none"
                                : "bg-blue-600 hover:bg-blue-700 text-white shadow-blue-200 dark:shadow-none"
                        )}
                    >
                        {payload.confirm_label ?? 'Подтвердить'}
                    </button>
                    <button
                        type="button"
                        onClick={() => {
                            setSelectedId('cancel')
                            onReply({kind: 'confirmation', confirmed: false})
                        }}
                        className="px-6 py-2.5 rounded-xl text-[13px] font-bold text-zinc-600 dark:text-zinc-400 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-all shadow-sm active:scale-95"
                    >
                        {payload.cancel_label ?? 'Отмена'}
                    </button>
                </div>
            </Card>
        )
    }

    if (payload.kind === 'text_input') {
        return <TextInputInterruptForm payload={payload} onReply={onReply}/>
    }

    return null
}

function TextInputInterruptForm({
                                    payload,
                                    onReply,
                                }: {
    payload: Extract<InterruptPayload, { kind: 'text_input' }>
    onReply: (resume: ResumeValue) => void
}) {
    const [value, setValue] = useState('')
    const [submitted, setSubmitted] = useState(false)

    const handleSubmit = () => {
        if (!value.trim() || submitted) return
        setSubmitted(true)
        onReply({kind: 'text_input', value: value.trim()})
    }

    return (
        <Card className="p-4 border-l-4 border-l-blue-500">
            {payload.prompt && (
                <p className="text-[14px] font-bold text-zinc-800 dark:text-zinc-200 mb-4 px-1">
                    {payload.prompt}
                </p>
            )}
            <div className="flex gap-2">
                <input
                    type={payload.secret ? 'password' : 'text'}
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                    placeholder={payload.placeholder ?? 'Введите значение...'}
                    disabled={submitted}
                    className="flex-1 px-4 py-2.5 rounded-xl border border-zinc-200 dark:border-zinc-700 bg-zinc-50/50 dark:bg-zinc-800/50 text-[14px] outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 transition-all disabled:opacity-50"
                />
                <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={!value.trim() || submitted}
                    className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-300 dark:disabled:bg-zinc-800 text-white text-[13px] font-bold transition-all shadow-sm shadow-blue-200 dark:shadow-none active:scale-95"
                >
                    OK
                </button>
            </div>
        </Card>
    )
}
