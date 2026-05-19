import {
    useRef,
    useState,
    useEffect,
    useCallback,
    type KeyboardEvent,
} from 'react'
import {Paperclip, X, Mic, Send, Square, MessageSquare, RefreshCw, Sparkles} from 'lucide-react'
import {useChatStore} from '@features/chat/model/useChatStore'
import {useStreamChat} from '@features/chat/model/useStreamChat'
import {useSpeechRecognition} from '@features/voice/model/useSpeechRecognition'
import {useApplyPreferences} from '@features/settings/model/useApplyPreferences'
import {sendMessage} from '@shared/api/messaging'
import {getAuthToken} from '@/shared/lib/auth'
import {extractDocIdFromUrl} from '@/shared/lib/url'
import {toast} from '@/shared/lib/toast'
import {ChatMessage} from '@/shared/ui/ChatMessage'
import {InterruptRenderer} from '@/shared/ui/InterruptRenderer'
import {ComplianceResult} from '@/shared/ui/ComplianceResult'
import {cn} from '@shared/lib/cn'
import type {ChatMessage as ChatMessageType} from '@entities/message/model/types'
import type {InterruptPayload, ResumeValue} from '@entities/interrupt/model/types'

function SoundWave({active}: { active: boolean }) {
    return (
        <span style={{display: 'flex', alignItems: 'center', gap: 2, height: 14}}>
            {[0, 1, 2, 3, 4].map((i) => (
                <span
                    key={i}
                    style={{
                        display: 'block',
                        width: 2.5,
                        height: 8,
                        borderRadius: 2,
                        background: 'rgba(99,102,241,0.60)',
                        transformOrigin: 'bottom',
                        animation: active ? `edms-soundbar 0.7s ease-in-out ${i * 90}ms infinite` : 'none',
                    }}
                />
            ))}
        </span>
    )
}

function TypingDots() {
    return (
        <span style={{display: 'inline-flex', alignItems: 'center', gap: 5, padding: '6px 4px'}}>
            {[0, 1, 2].map((i) => (
                <span
                    key={i}
                    style={{
                        display: 'block',
                        width: 7,
                        height: 7,
                        borderRadius: '50%',
                        background: '#94a3b8',
                        animation: `edms-wave 1.2s ease-in-out ${i * 0.18}s infinite`,
                    }}
                />
            ))}
        </span>
    )
}

function normalizeSpeechText(text: string): string {
    if (!text) return ''
    const trimmed = text.trim()
    return trimmed.charAt(0).toUpperCase() + trimmed.slice(1)
}

export function WidgetChat() {
    const {messages, loading, threadId, setThreadId, updateMessage} = useChatStore()
    const {startStream, abort} = useStreamChat()
    const {prefs} = useApplyPreferences()

    const [input, setInput] = useState('')
    const [attachedFile, setAttachedFile] = useState<{ path: string; name: string } | null>(null)
    const [isFocused, setIsFocused] = useState(false)
    const [pendingResume, setPendingResume] = useState<ResumeValue | null>(null)

    const bottomRef = useRef<HTMLDivElement>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const streamHandleRef = useRef<ReturnType<typeof startStream> | null>(null)
    const sendRef = useRef<() => void>(() => undefined)
    const handsFreeRef = useRef(prefs.voice.handsFreeEnabled)
    handsFreeRef.current = prefs.voice.handsFreeEnabled
    const prevLoadingRef = useRef(false)
    const toggleMicRef = useRef<() => void>(() => undefined)

    const {
        isListening,
        interimTranscript,
        isSupported: isSpeechSupported,
        toggle: toggleMic,
        stop: stopMic,
    } = useSpeechRecognition({
        lang: prefs.voice.sttLanguage,
        silenceMs: prefs.voice.handsFreeEnabled ? prefs.voice.autoSendPauseMs + 1000 : 999999,
        autoSend: prefs.voice.handsFreeEnabled,
        autoSendMs: prefs.voice.autoSendPauseMs,
        onFinalResult: (delta) => {
            const processed = normalizeSpeechText(delta)
            setInput((prev) => (prev.trimEnd() ? `${prev.trimEnd()} ` : '') + processed)
        },
        onAutoSend: () => {
            if (handsFreeRef.current) sendRef.current()
        },
    })

    const displayInput =
        isListening && interimTranscript
            ? input + (input ? ' ' : '') + interimTranscript
            : input

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto'
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`
        }
    }, [displayInput])

    useEffect(() => {
        bottomRef.current?.scrollIntoView({behavior: 'smooth'})
    }, [messages])

    const getOrCreateThreadId = useCallback(async (): Promise<string> => {
        if (threadId) return threadId
        const token = getAuthToken()
        if (!token) throw new Error('Войдите в систему')
        const res = await sendMessage('createNewChat', {user_token: token})
        setThreadId(res.thread_id)
        return res.thread_id
    }, [threadId, setThreadId])

    const send = useCallback(async () => {
        const text = input.trim()
        if ((!text && !attachedFile) || loading) return

        stopMic()
        setInput('')

        const token = getAuthToken()
        if (!token) {
            toast.error('Войдите в систему', 'Авторизация')
            return
        }

        let tid: string
        try {
            tid = await getOrCreateThreadId()
        } catch (e) {
            toast.error(e instanceof Error ? e.message : 'Ошибка', 'Ошибка')
            return
        }

        let filePath: string | null = attachedFile?.path ?? null
        if (attachedFile && !filePath?.startsWith('/')) {
            try {
                const uploadRes = await sendMessage('uploadFile', {
                    user_token: token,
                    thread_id: tid,
                    file_data: attachedFile.path,
                    file_name: attachedFile.name,
                    context_ui_id: extractDocIdFromUrl(),
                })
                if (uploadRes.success && uploadRes.file_path) {
                    filePath = uploadRes.file_path
                }
            } catch {
                toast.error('Ошибка загрузки файла', 'Ошибка')
                return
            }
        }

        setAttachedFile(null)

        streamHandleRef.current = startStream({
            message: text,
            userToken: token,
            threadId: tid,
            contextUiId: extractDocIdFromUrl(),
            filePath,
            resumeValue: pendingResume,
        })
        setPendingResume(null)
    }, [input, attachedFile, loading, pendingResume, stopMic, getOrCreateThreadId, startStream])

    sendRef.current = () => {
        void send()
    }
    toggleMicRef.current = toggleMic

    useEffect(() => {
        const wasLoading = prevLoadingRef.current
        prevLoadingRef.current = loading
        if (!wasLoading || loading || !handsFreeRef.current || !isSpeechSupported) return undefined
        const t = setTimeout(() => {
            if (handsFreeRef.current) toggleMicRef.current()
        }, 600)
        return () => clearTimeout(t)
    }, [loading, isSpeechSupported])

    const handleStop = useCallback(() => {
        streamHandleRef.current?.abort()
        streamHandleRef.current = null
        abort()
    }, [abort])

    const handleInterruptReply = useCallback(
        async (resume: ResumeValue) => {
            if (loading) return

            const token = getAuthToken()
            if (!token) {
                toast.error('Войдите в систему', 'Авторизация')
                return
            }

            let tid: string
            try {
                tid = await getOrCreateThreadId()
            } catch {
                return
            }

            streamHandleRef.current = startStream({
                message: '',
                userToken: token,
                threadId: tid,
                contextUiId: extractDocIdFromUrl(),
                filePath: null,
                resumeValue: resume,
            })
        },
        [loading, getOrCreateThreadId, startStream],
    )

    const applyOptimisticFix = useCallback(
        (messageId: string, fixed: Array<{ fieldKey: string; newValue: string }>) => {
            updateMessage(messageId, (m) => {
                if (!m.compliance) return m
                const keys = new Set(fixed.map((f) => f.fieldKey))
                const valueByKey = new Map(fixed.map((f) => [f.fieldKey, f.newValue]))
                const fields = m.compliance.fields.map((f) =>
                    keys.has(f.field_key)
                        ? {
                              ...f,
                              status: 'ok' as const,
                              card_value: valueByKey.get(f.field_key) ?? f.card_value,
                              correct_value: null,
                          }
                        : f,
                )
                const remainingMismatches = fields.filter((f) => f.status === 'mismatch').length
                return {
                    ...m,
                    compliance: {
                        ...m.compliance,
                        fields,
                        overall: remainingMismatches === 0 ? ('ok' as const) : m.compliance.overall,
                    },
                }
            })
            // Give the agent ~3s to call doc_update_field before reloading the page.
            window.setTimeout(() => {
                void sendMessage('reloadActiveTab', undefined)
            }, 3000)
        },
        [updateMessage],
    )

    const handleRefreshSummary = useCallback(
        async (messageId: string, meta: NonNullable<ChatMessageType['refreshMeta']>) => {
            const summaryType = meta.cache_summary_type ?? 'thesis'
            const filePath = meta.cache_file_path ?? meta.cache_file_identifier ?? null
            const fileName = meta.cache_file_identifier ?? 'Документ'
            if (!filePath) {
                toast.error('Нет идентификатора файла', 'Не удалось обновить')
                return
            }
            await (async () => {
                const token = getAuthToken()
                if (!token) {
                    toast.error('Войдите в систему', 'Авторизация')
                    return
                }
                try {
                    if (meta.cache_file_identifier) {
                        try {
                            await sendMessage('deleteCache', {
                                user_token: token,
                                thread_id: threadId ?? '',
                                file_identifier: meta.cache_file_identifier,
                                summary_type: meta.cache_summary_type ?? null,
                                context_id: meta.cache_context_id ?? null,
                                file_path: filePath,
                            })
                        } catch {
                            // non-fatal: cache may not exist yet
                        }
                    }
                    const res = await sendMessage('summarizeDocument', {
                        message: fileName,
                        user_token: token,
                        context_ui_id: meta.cache_context_id ?? null,
                        file_path: filePath,
                        preferred_summary_format: summaryType,
                    })
                    if (res.success) {
                        const newText = res.data?.response
                        if (newText) {
                            updateMessage(messageId, (m) => ({...m, content: newText}))
                            toast.success('Анализ обновлён', 'Готово')
                        }
                    } else {
                        toast.error(res.error ?? 'Неизвестная ошибка', 'Не удалось обновить')
                    }
                } catch (err) {
                    toast.error(err instanceof Error ? err.message : 'Ошибка', 'Не удалось обновить')
                }
            })()
        },
        [updateMessage, threadId],
    )

    const handleFieldFixed = useCallback(
        (messageId: string, fieldKey: string, newValue: string) => {
            applyOptimisticFix(messageId, [{fieldKey, newValue}])
            void (async () => {
                const token = getAuthToken()
                if (!token) return
                let tid: string
                try {
                    tid = await getOrCreateThreadId()
                } catch {
                    return
                }
                streamHandleRef.current = startStream({
                    message: `Исправь поле ${fieldKey} на значение: ${newValue}`,
                    userToken: token,
                    threadId: tid,
                    contextUiId: extractDocIdFromUrl(),
                    filePath: null,
                    resumeValue: null,
                })
            })()
        },
        [applyOptimisticFix, getOrCreateThreadId, startStream],
    )

    const handleAllFixed = useCallback(
        (
            messageId: string,
            fixedFields: Array<{ fieldKey: string; label: string; newValue: string }>,
        ) => {
            applyOptimisticFix(messageId, fixedFields)
            void (async () => {
                const token = getAuthToken()
                if (!token) return
                let tid: string
                try {
                    tid = await getOrCreateThreadId()
                } catch {
                    return
                }
                const corrections = fixedFields
                    .map((f) => `${f.label}: ${f.newValue}`)
                    .join(', ')
                streamHandleRef.current = startStream({
                    message: `Исправь следующие поля: ${corrections}`,
                    userToken: token,
                    threadId: tid,
                    contextUiId: extractDocIdFromUrl(),
                    filePath: null,
                    resumeValue: null,
                })
            })()
        },
        [applyOptimisticFix, getOrCreateThreadId, startStream],
    )

    const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return
        const dataUrl = await new Promise<string>((resolve) => {
            const reader = new FileReader()
            reader.onload = () => resolve(reader.result as string)
            reader.readAsDataURL(file)
        })
        setAttachedFile({path: dataUrl, name: file.name})
        if (fileInputRef.current) fileInputRef.current.value = ''
    }, [])

    const handleSubmit = useCallback(
        (e: { preventDefault(): void }) => {
            e.preventDefault()
            void send()
        },
        [send],
    )

    const handleKeyDown = useCallback(
        (e: KeyboardEvent<HTMLTextAreaElement>) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send()
            }
        },
        [send],
    )

    return (
        <div className="flex flex-col flex-1 min-w-0 min-h-0">
            <div
                className={cn(
                    'flex-1 overflow-y-auto overflow-x-hidden px-3 py-3 min-h-0 scrollbar-none',
                    messages.length === 0 ? 'flex flex-col' : 'space-y-3',
                )}
            >
                {messages.length === 0 && (
                    <EmptyState
                        onAction={(text) => setInput(text)}
                        showQuickActions={prefs.documents.showQuickActionHints}
                    />
                )}

                {messages.map((msg, idx) => {
                    if (
                        loading &&
                        idx === messages.length - 1 &&
                        msg.role === 'assistant' &&
                        msg.content === ''
                    )
                        return null

                    return (
                        <MessageRow
                            key={msg.id}
                            msg={msg}
                            threadId={threadId}
                            onInterruptReply={handleInterruptReply}
                            onFieldFixed={handleFieldFixed}
                            onAllFixed={handleAllFixed}
                            onRefreshSummary={handleRefreshSummary}
                        />
                    )
                })}

                {loading && messages[messages.length - 1]?.content === '' && (
                    <div className="flex justify-start pl-2">
                        <TypingDots/>
                    </div>
                )}
                <div ref={bottomRef}/>
            </div>

            {attachedFile && (
                <div
                    className="mx-3 mb-1 px-3 py-1.5 rounded-xl flex items-center gap-2 text-[11px]"
                    style={{
                        background: 'rgba(99,102,241,0.06)',
                        border: '1px solid rgba(99,102,241,0.12)',
                    }}
                >
                    <Paperclip size={11} style={{color: '#6366f1', flexShrink: 0}}/>
                    <span className="flex-1 text-indigo-700 truncate">{attachedFile.name}</span>
                    <button
                        type="button"
                        onClick={() => setAttachedFile(null)}
                        className="text-slate-400 hover:text-slate-600 transition-colors"
                    >
                        <X size={12}/>
                    </button>
                </div>
            )}

            <form onSubmit={handleSubmit} className="shrink-0 px-3 pb-2.5">
                <div
                    className={cn(
                        'flex items-center gap-1.5 rounded-full px-3 py-1.5 transition-all duration-200',
                        isFocused
                            ? 'shadow-[0_0_0_2px_rgba(99,102,241,0.15),0_2px_8px_rgba(0,0,0,0.05)]'
                            : 'shadow-[0_2px_8px_rgba(0,0,0,0.05)]',
                    )}
                    style={{
                        background: '#ffffff',
                        border: `1px solid ${isFocused ? 'rgba(99,102,241,0.25)' : 'rgba(0,0,0,0.07)'}`,
                    }}
                >
                    <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        title="Прикрепить файл"
                        className="shrink-0 w-8 h-8 flex items-center justify-center rounded-lg text-slate-400 hover:text-indigo-500 hover:bg-indigo-50 transition-all"
                    >
                        <Paperclip size={15}/>
                    </button>

                    <div className="w-px h-4 bg-zinc-200 mx-0.5 shrink-0" />

                    <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        onChange={handleFileChange}
                        accept=".pdf,.doc,.docx,.txt,.xlsx,.xls,.csv,.png,.jpg,.jpeg"
                    />

                    <textarea
                        ref={textareaRef}
                        rows={1}
                        value={displayInput}
                        onChange={(e) => !isListening && setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setIsFocused(false)}
                        placeholder={
                            isListening
                                ? 'Слушаю…'
                                : attachedFile
                                    ? 'Комментарий...'
                                    : 'Спросите помощника...'
                        }
                        disabled={loading && !isListening}
                        className={cn(
                            'flex-1 resize-none bg-transparent text-slate-900 placeholder:text-slate-400',
                            'focus:outline-none text-[13px] leading-snug py-1 min-h-[22px] max-h-[120px] overflow-y-auto text-center scrollbar-none',
                            isListening && 'text-indigo-700 font-medium',
                        )}
                    />

                    {isSpeechSupported && (
                        <button
                            type="button"
                            onClick={toggleMic}
                            title={isListening ? 'Остановить' : 'Голосовой ввод'}
                            className={cn(
                                'shrink-0 w-8 h-8 flex items-center justify-center rounded-lg transition-all',
                                isListening
                                    ? 'text-indigo-500 bg-indigo-50'
                                    : 'text-slate-400 hover:text-indigo-500 hover:bg-indigo-50',
                            )}
                        >
                            {isListening ? <SoundWave active/> : <Mic size={16}/>}
                        </button>
                    )}

                    {loading ? (
                        <button
                            type="button"
                            onClick={handleStop}
                            title="Остановить"
                            style={{
                                width: 32,
                                height: 32,
                                borderRadius: '50%',
                                background: '#ef4444',
                                border: 'none',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: 'white',
                                flexShrink: 0,
                            }}
                        >
                            <Square size={12} fill="white" strokeWidth={0}/>
                        </button>
                    ) : (
                        <button
                            type="submit"
                            disabled={!displayInput.trim() && !attachedFile}
                            title="Отправить"
                            style={{
                                width: 32,
                                height: 32,
                                borderRadius: '50%',
                                background:
                                    displayInput.trim() || attachedFile
                                        ? '#6366f1'
                                        : 'rgba(0,0,0,0.06)',
                                border: 'none',
                                cursor: displayInput.trim() || attachedFile ? 'pointer' : 'not-allowed',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: displayInput.trim() || attachedFile ? 'white' : '#cbd5e1',
                                flexShrink: 0,
                                transition: 'background 0.15s',
                            }}
                        >
                            <Send size={13}/>
                        </button>
                    )}
                </div>
            </form>

            <p
                className="shrink-0 text-center px-3 pb-2"
                style={{fontSize: 10, color: '#cbd5e1', lineHeight: 1.4}}
            >
                EDMS Assistant — ИИ. Может ошибаться и давать неверную информацию.
            </p>
        </div>
    )
}

interface MessageRowProps {
    msg: ChatMessageType
    threadId: string | null
    onInterruptReply: (resume: ResumeValue) => void
    onFieldFixed: (messageId: string, fieldKey: string, newValue: string) => void
    onAllFixed: (
        messageId: string,
        fixedFields: Array<{ fieldKey: string; label: string; newValue: string }>,
    ) => void
    onRefreshSummary: (
        messageId: string,
        meta: NonNullable<ChatMessageType['refreshMeta']>,
    ) => Promise<void>
}

const SUMMARY_TYPE_LABELS: Record<string, string> = {
    extractive: 'Факты',
    abstractive: 'Пересказ',
    thesis: 'Тезисы',
}

function MessageRow({msg, threadId, onInterruptReply, onFieldFixed, onAllFixed, onRefreshSummary}: MessageRowProps) {
    const handleFieldFixed = useCallback(
        (fieldKey: string, newValue: string) => onFieldFixed(msg.id, fieldKey, newValue),
        [msg.id, onFieldFixed],
    )
    const handleAllFixed = useCallback(
        (fixedFields: Array<{ fieldKey: string; label: string; newValue: string }>) =>
            onAllFixed(msg.id, fixedFields),
        [msg.id, onAllFixed],
    )
    const [isRefreshing, setIsRefreshing] = useState(false)
    const handleRefreshClick = useCallback(() => {
        if (!msg.refreshMeta || isRefreshing) return
        setIsRefreshing(true)
        void onRefreshSummary(msg.id, msg.refreshMeta).finally(() => setIsRefreshing(false))
    }, [msg.id, msg.refreshMeta, onRefreshSummary, isRefreshing])
    const refreshLabel = msg.refreshMeta?.cache_summary_type
        ? (SUMMARY_TYPE_LABELS[msg.refreshMeta.cache_summary_type] ?? msg.refreshMeta.cache_summary_type)
        : null
    return (
        <div className={cn('flex flex-col', msg.role === 'user' ? 'items-end' : 'items-start')}>
            <div className={cn(msg.role === 'user' ? 'max-w-[75%]' : 'w-full max-w-[85%]')}>
                {msg.content && (
                    <ChatMessage
                        content={msg.content}
                        role={msg.role}
                        timestamp={msg.timestamp}
                        isError={msg.isError === true}
                    />
                )}

                {msg.compliance != null && msg.role === 'assistant' && (
                    <div style={{marginTop: msg.content ? 8 : 0}}>
                        <ComplianceResult
                            data={msg.compliance}
                            threadId={threadId}
                            refreshMeta={msg.refreshMeta}
                            onFieldFixed={handleFieldFixed}
                            onAllFixed={handleAllFixed}
                        />
                    </div>
                )}

                {msg.refreshMeta != null && msg.compliance == null && msg.role === 'assistant' && refreshLabel && (
                    <div style={{marginTop: msg.content ? 6 : 0}}>
                        <button
                            type="button"
                            onClick={handleRefreshClick}
                            disabled={isRefreshing}
                            title={`Пересчитать анализ «${refreshLabel}»`}
                            className={cn(
                                'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-[11px] font-medium transition-all',
                                isRefreshing
                                    ? 'bg-indigo-100 border-indigo-200 text-indigo-700 cursor-wait'
                                    : 'bg-indigo-50 border-indigo-100 text-indigo-600 hover:bg-indigo-100 active:scale-95',
                            )}
                        >
                            <RefreshCw size={11} className={isRefreshing ? 'animate-spin' : undefined}/>
                            <span>{isRefreshing ? `Обновляю «${refreshLabel}»…` : `Обновить «${refreshLabel}»`}</span>
                        </button>
                    </div>
                )}

                {msg.interrupt != null && msg.role === 'assistant' && (
                    <div style={{marginTop: msg.content ? 8 : 0}}>
                        <InterruptRenderer
                            payload={msg.interrupt as InterruptPayload}
                            onReply={onInterruptReply}
                        />
                    </div>
                )}
            </div>
        </div>
    )
}

// ── EmptyState ────────────────────────────────────────────────────────────

const QUICK_ACTIONS = ['Суммаризация', 'Поиск', 'Тезисы'] as const

function EmptyState({
                        onAction,
                        showQuickActions,
                    }: {
    onAction: (text: string) => void
    showQuickActions: boolean
}) {
    return (
        <div className="flex flex-col items-center justify-center flex-1 gap-4 px-6">
            <div
                style={{
                    width: 64,
                    height: 64,
                    borderRadius: 20,
                    background: 'rgba(99,102,241,0.08)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}
            >
                <Sparkles size={32} strokeWidth={1.5} color="#6366f1"/>
            </div>
            <div className="text-center">
                <p style={{fontSize: 16, fontWeight: 600, color: '#1e293b', marginBottom: 4}}>
                    Чем могу помочь?
                </p>
                <p style={{fontSize: 13, color: '#94a3b8'}}>Задайте вопрос ...</p>
            </div>
            {showQuickActions && (
                <div className="flex items-center gap-2 flex-wrap justify-center">
                    {QUICK_ACTIONS.map((action) => (
                        <button
                            key={action}
                            type="button"
                            onClick={() => onAction(action)}
                            style={{
                                padding: '6px 16px',
                                borderRadius: 999,
                                background: 'white',
                                border: '1px solid rgba(0,0,0,0.10)',
                                fontSize: 12,
                                color: '#334155',
                                cursor: 'pointer',
                                fontWeight: 500,
                                transition: 'border-color 0.15s, color 0.15s',
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(99,102,241,0.4)'
                                e.currentTarget.style.color = '#6366f1'
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(0,0,0,0.10)'
                                e.currentTarget.style.color = '#334155'
                            }}
                        >
                            {action}
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}