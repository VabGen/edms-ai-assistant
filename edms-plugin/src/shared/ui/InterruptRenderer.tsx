import {useState} from 'react'
import {ExternalLink, User, FileText, ChevronRight} from 'lucide-react'
import type {InterruptPayload, ResumeValue} from '@entities/interrupt/model/types'
import {sendMessage} from '@shared/api/messaging'
import {BaseCard} from './primitives/BaseCard'

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
            <div style={{display: 'flex', flexDirection: 'column', gap: 6}}>
                {payload.prompt && (
                    <p style={{fontSize: 12, color: '#64748b', margin: 0, lineHeight: 1.5}}>
                        {payload.prompt}
                    </p>
                )}
                {payload.cards.map((card, idx) => {
                    const isSelected = selectedId === card.id
                    const cardUrl = typeof card.metadata?.['url'] === 'string'
                        ? (card.metadata['url'] as string)
                        : null
                    const isEmployee = card.description?.toLowerCase().includes('подразделение') ||
                        card.badges?.some(b => b.toLowerCase().includes('сотрудник') || b.toLowerCase().includes('физлицо'));

                    return (
                        <div key={card.id} className="flex items-stretch gap-1.5">
                            <BaseCard
                                isSelected={isSelected}
                                onClick={() =>
                                    handleSelect(card.id, {
                                        kind: 'card_select',
                                        selected_ids: [card.id],
                                    })
                                }
                                className="flex-1 min-w-0 flex-row items-center gap-3 py-3"
                                style={isSelected ? { background: '#7c3aed' } : {}}
                            >
                                <div style={{
                                    width: 32,
                                    height: 32,
                                    borderRadius: '50%',
                                    background: isSelected ? 'rgba(255,255,255,0.2)' : 'rgba(99,102,241,0.06)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    flexShrink: 0,
                                    fontSize: 12,
                                    fontWeight: 700,
                                    color: isSelected ? '#ffffff' : '#6366f1',
                                }}>
                                    {idx + 1}
                                </div>

                                {isEmployee ? (
                                    <User size={18} style={{
                                        color: isSelected ? '#ffffff' : '#6366f1',
                                        opacity: isSelected ? 0.9 : 0.7,
                                        flexShrink: 0
                                    }}/>
                                ) : (
                                    <FileText size={18} style={{
                                        color: isSelected ? '#ffffff' : '#6366f1',
                                        opacity: isSelected ? 0.9 : 0.7,
                                        flexShrink: 0
                                    }}/>
                                )}

                                <div style={{flex: 1, minWidth: 0}}>
                                    <div style={{
                                        fontSize: 13,
                                        fontWeight: 600,
                                        color: isSelected ? '#ffffff' : '#0f172a',
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis'
                                    }}>
                                        {card.label}
                                    </div>
                                    {card.description && (
                                        <div style={{
                                            fontSize: 11,
                                            color: isSelected ? 'rgba(255,255,255,0.8)' : '#64748b',
                                            marginTop: 1,
                                            whiteSpace: 'nowrap',
                                            overflow: 'hidden',
                                            textOverflow: 'ellipsis'
                                        }}>
                                            {card.description}
                                        </div>
                                    )}
                                    {Object.keys(card.primary_attrs ?? {}).length > 0 && (
                                        <div style={{display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 4}}>
                                            {Object.entries(card.primary_attrs).map(([k, v]) => (
                                                <div key={k} style={{
                                                    fontSize: 10,
                                                    color: isSelected ? 'rgba(255,255,255,0.7)' : '#475569',
                                                    display: 'flex',
                                                    gap: 3
                                                }}>
                                                    <span style={{opacity: 0.7}}>{k}:</span>
                                                    <span>{v}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                <ChevronRight size={16} style={{
                                    color: isSelected ? '#ffffff' : '#cbd5e1',
                                    flexShrink: 0,
                                    marginLeft: 'auto'
                                }}/>
                            </BaseCard>

                            {cardUrl && (
                                <button
                                    type="button"
                                    title="Открыть в новой вкладке"
                                    onClick={(e) => {
                                        e.stopPropagation()
                                        void sendMessage('navigateTo', {url: cardUrl, newTab: true})
                                    }}
                                    style={{
                                        flexShrink: 0,
                                        width: 42,
                                        border: '1px solid rgba(0,0,0,0.08)',
                                        borderRadius: 16,
                                        background: '#ffffff',
                                        color: '#6366f1',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        transition: 'all 0.2s',
                                        boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.background = 'rgba(99,102,241,0.08)'
                                        e.currentTarget.style.borderColor = 'rgba(99,102,241,0.35)'
                                        e.currentTarget.style.transform = 'translateY(-1px)'
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.background = '#ffffff'
                                        e.currentTarget.style.borderColor = 'rgba(0,0,0,0.08)'
                                        e.currentTarget.style.transform = 'translateY(0)'
                                    }}
                                >
                                    <ExternalLink size={16}/>
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
            <div style={{display: 'flex', flexDirection: 'column', gap: 6}}>
                {payload.prompt && (
                    <p style={{fontSize: 12, color: '#64748b', margin: 0, lineHeight: 1.5}}>
                        {payload.prompt}
                    </p>
                )}
                {payload.options.map((opt) => {
                    const isSelected = selectedId === opt.id
                    return (
                        <button
                            key={opt.id}
                            type="button"
                            onClick={() =>
                                handleSelect(opt.id, {
                                    kind: 'disambiguation',
                                    selected_ids: [opt.id],
                                })
                            }
                            style={{
                                padding: '10px 14px',
                                borderRadius: 12,
                                border: `1px solid ${isSelected ? '#6366f1' : 'rgba(0,0,0,0.08)'}`,
                                background: isSelected ? 'rgba(99,102,241,0.08)' : '#ffffff',
                                cursor: 'pointer',
                                textAlign: 'left',
                                transition: 'all 0.15s',
                                boxShadow: isSelected
                                    ? '0 0 0 2px rgba(99,102,241,0.2)'
                                    : '0 1px 3px rgba(0,0,0,0.04)',
                                width: '100%',
                                maxWidth: '100%',
                                overflow: 'hidden',
                                whiteSpace: 'normal',
                                wordBreak: 'break-word',
                                overflowWrap: 'anywhere',
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(99,102,241,0.35)'
                                e.currentTarget.style.background = 'rgba(99,102,241,0.04)'
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = isSelected ? '#6366f1' : 'rgba(0,0,0,0.08)'
                                e.currentTarget.style.background = isSelected ? 'rgba(99,102,241,0.08)' : '#ffffff'
                            }}
                        >
                            <div style={{fontSize: 13, fontWeight: 600, color: '#0f172a', whiteSpace: 'normal', wordBreak: 'break-word', overflowWrap: 'anywhere'}}>
                                {opt.label}
                            </div>
                            {opt.description && (
                                <div style={{fontSize: 11, color: '#64748b', marginTop: 2, lineHeight: 1.4}}>
                                    {opt.description}
                                </div>
                            )}
                        </button>
                    )
                })}
            </div>
        )
    }

    // ── select ─────────────────────────────────────────────────────────────
    if (payload.kind === 'select') {
        return (
            <div style={{display: 'flex', flexDirection: 'column', gap: 6}}>
                {payload.prompt && (
                    <p style={{fontSize: 12, color: '#64748b', margin: 0, lineHeight: 1.5}}>
                        {payload.prompt}
                    </p>
                )}
                <div style={{display: 'flex', flexDirection: 'column', gap: 4}}>
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
                                style={{
                                    padding: '8px 14px',
                                    borderRadius: 10,
                                    border: `1px solid ${isSelected ? '#6366f1' : 'rgba(0,0,0,0.08)'}`,
                                    background: isSelected ? 'rgba(99,102,241,0.08)' : '#ffffff',
                                cursor: 'pointer',
                                    textAlign: 'left',
                                    fontSize: 12,
                                    fontWeight: isSelected ? 600 : 400,
                                    color: isSelected ? '#4338ca' : '#334155',
                                    transition: 'all 0.15s',
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.borderColor = 'rgba(99,102,241,0.35)'
                                    e.currentTarget.style.background = 'rgba(99,102,241,0.04)'
                                }}
                                onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = isSelected ? '#6366f1' : 'rgba(0,0,0,0.08)'
                                e.currentTarget.style.background = isSelected ? 'rgba(99,102,241,0.08)' : '#ffffff'
                                }}
                            >
                                {opt.label}
                                {opt.description && (
                                    <span style={{color: '#94a3b8', fontSize: 10, marginLeft: 6}}>
                    {opt.description}
                  </span>
                                )}
                            </button>
                        )
                    })}
                </div>
            </div>
        )
    }

    // ── confirmation ───────────────────────────────────────────────────────
    if (payload.kind === 'confirmation') {
        return (
            <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
                {payload.prompt && (
                    <p style={{fontSize: 12, color: '#475569', margin: 0, lineHeight: 1.5}}>
                        {payload.prompt}
                    </p>
                )}
                <div style={{display: 'flex', gap: 8}}>
                    <button
                        type="button"
                        disabled={!!selectedId}
                        onClick={() => {
                            setSelectedId('confirm')
                            onReply({kind: 'confirmation', confirmed: true})
                        }}
                        style={{
                            padding: '8px 16px',
                            borderRadius: 10,
                            border: 'none',
                            background:
                                selectedId === 'confirm'
                                    ? '#4f46e5'
                                    : payload.danger
                                        ? '#ef4444'
                                        : '#6366f1',
                            color: '#fff',
                            fontSize: 12,
                            fontWeight: 600,
                            cursor: selectedId ? 'default' : 'pointer',
                            opacity: selectedId && selectedId !== 'confirm' ? 0.5 : 1,
                            transition: 'all 0.15s',
                        }}
                    >
                        {payload.confirm_label ?? 'Подтвердить'}
                    </button>
                    <button
                        type="button"
                        disabled={!!selectedId}
                        onClick={() => {
                            setSelectedId('cancel')
                            onReply({kind: 'confirmation', confirmed: false})
                        }}
                        style={{
                            padding: '8px 16px',
                            borderRadius: 10,
                            border: '1px solid rgba(0,0,0,0.1)',
                            background: selectedId === 'cancel' ? 'rgba(0,0,0,0.05)' : '#fff',
                            fontSize: 12,
                            cursor: selectedId ? 'default' : 'pointer',
                            opacity: selectedId && selectedId !== 'cancel' ? 0.5 : 1,
                            transition: 'all 0.15s',
                        }}
                    >
                        {payload.cancel_label ?? 'Отмена'}
                    </button>
                </div>
            </div>
        )
    }

    if (payload.kind === 'text_input') {
        return <TextInputInterruptForm payload={payload} onReply={onReply}/>
    }

    return null
}

// ── Отдельный компонент для text_input ────────────────────────────────────

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
        <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
            {payload.prompt && (
                <p style={{fontSize: 12, color: '#475569', margin: 0, lineHeight: 1.5}}>
                    {payload.prompt}
                </p>
            )}
            <div style={{display: 'flex', gap: 8}}>
                <input
                    type={payload.secret ? 'password' : 'text'}
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                    placeholder={payload.placeholder ?? ''}
                    disabled={submitted}
                    style={{
                        flex: 1,
                        padding: '8px 12px',
                        borderRadius: 10,
                        border: '1px solid rgba(0,0,0,0.1)',
                        fontSize: 12,
                        outline: 'none',
                        background: submitted ? '#f8fafc' : '#fff',
                    }}
                />
                <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={!value.trim() || submitted}
                    style={{
                        padding: '8px 14px',
                        borderRadius: 10,
                        border: 'none',
                        background: value.trim() && !submitted ? '#6366f1' : 'rgba(0,0,0,0.06)',
                        color: value.trim() && !submitted ? '#fff' : '#94a3b8',
                        fontSize: 12,
                        fontWeight: 600,
                        cursor: value.trim() && !submitted ? 'pointer' : 'not-allowed',
                        transition: 'all 0.15s',
                    }}
                >
                    OK
                </button>
            </div>
        </div>
    )
}