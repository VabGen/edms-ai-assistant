import {
    useRef,
    useState,
    useEffect,
    useCallback,
    type KeyboardEvent,
} from 'react'
import {Paperclip, X, Mic, Send, Square, MessageSquare, RefreshCw, Loader2, Sparkles} from 'lucide-react'
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
import { Card, CardHeader, CardTitle, IconBox, Button } from '@/shared/ui/primitives'

function SoundWave({active}: { active: boolean }) {
    return (
        <div className="flex items-center gap-0.5 h-3.5 px-1">
            {[0, 1, 2, 3, 4].map((i) => (
                <div
                    key={i}
                    className={cn(
                        "w-0.5 rounded-full transition-all duration-300 bg-blue-500",
                        active ? "h-full" : "h-1"
                    )}
                    style={{
                        animation: active ? `edms-soundbar 0.7s ease-in-out ${i * 100}ms infinite` : 'none',
                    }}
                />
            ))}
        </div>
    )
}

function TypingDots() {
    return (
        <div className="flex items-center gap-1 px-3 py-2 bg-zinc-50 dark:bg-zinc-800 rounded-2xl w-fit">
            {[0, 1, 2].map((i) => (
                <div
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-zinc-300 dark:bg-zinc-600 animate-bounce"
                    style={{ animationDelay: `${i * 150}ms` }}
                />
            ))}
        </div>
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

    const send = useCallback(async (customText?: string) => {
        const text = customText ?? input.trim()
        if ((!text && !attachedFile) || loading) return

        stopMic()
        if (!customText) setInput('')

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
            // Remove loading check to allow switching selection
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
        [getOrCreateThreadId, startStream],
    )

    const applyOptimisticFix = useCallback(
        (messageId: string, fixed: Array<{ fieldKey: string; newValue: string }>) => {
            void sendMessage('reloadActiveTab', undefined)

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
                            // non-fatal
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

    const handleDocumentClick = useCallback((docId: string) => {
        void sendMessage('navigateTo', {url: `/document-form/${docId}`, newTab: true})
    }, [])

    const handleAttachmentClick = useCallback(
        async (fileName: string) => {
            // Remove loading check to allow switching attachments
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

            // Enhanced prompt for attachment analysis, prioritizing facts if that's what's failing
            const prefFormat = prefs.documents.defaultSummaryFormat
            let prompt = `Проанализируй вложение: ${fileName}`
            if (prefFormat === 'extractive') prompt = `Извлеки основные факты из вложения: ${fileName}`
            if (prefFormat === 'thesis') prompt = `Подготовь тезисный план вложения: ${fileName}`

            streamHandleRef.current = startStream({
                message: prompt,
                userToken: token,
                threadId: tid,
                contextUiId: extractDocIdFromUrl(),
                filePath: null,
                resumeValue: null,
            })
        },
        [getOrCreateThreadId, startStream, prefs.documents.defaultSummaryFormat],
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
        <div className="flex flex-col flex-1 min-w-0 min-h-0 bg-white dark:bg-zinc-900">
            <div
                className={cn(
                    'flex-1 overflow-y-auto overflow-x-hidden px-4 py-4 min-h-0 scrollbar-thin',
                    messages.length === 0 ? 'flex flex-col' : 'space-y-4',
                )}
            >
                {messages.length === 0 && (
                    <EmptyState
                        onAction={(text) => send(text)}
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
                            onDocumentClick={handleDocumentClick}
                            onAttachmentClick={handleAttachmentClick}
                            onSendMessage={(text) => send(text)}
                        />
                    )
                })}

                {loading && messages[messages.length - 1]?.content === '' && (
                    <div className="flex justify-start px-2 animate-edms-fade-in">
                        <TypingDots/>
                    </div>
                )}
                <div ref={bottomRef} className="h-4 shrink-0" />
            </div>

            <div className="px-4 pb-4 shrink-0">
                {attachedFile && (
                    <div className="mb-2 p-2.5 rounded-xl flex items-center gap-3 bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 animate-edms-slide-up shadow-sm">
                        <div className="p-1.5 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
                            <Paperclip size={14} className="text-blue-500"/>
                        </div>
                        <span className="flex-1 text-[13px] font-bold text-zinc-700 dark:text-zinc-200 truncate">{attachedFile.name}</span>
                        <button
                            type="button"
                            onClick={() => setAttachedFile(null)}
                            className="p-1.5 hover:bg-zinc-200 dark:hover:bg-zinc-700 rounded-lg text-zinc-400 transition-colors"
                        >
                            <X size={14}/>
                        </button>
                    </div>
                )}

                <form onSubmit={handleSubmit}>
                    <div
                        className={cn(
                            'flex items-end gap-2 rounded-[28px] p-2 transition-all duration-500 border bg-zinc-50/50',
                            isFocused
                                ? 'border-indigo-500 ring-4 ring-indigo-500/10 shadow-xl bg-white'
                                : 'border-zinc-200 shadow-sm',
                        )}
                    >
                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className="shrink-0 w-12 h-12 flex items-center justify-center rounded-full text-zinc-400 hover:text-indigo-600 hover:bg-indigo-50 transition-all duration-300"
                            title="Прикрепить"
                        >
                            <Paperclip size={20}/>
                        </button>
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
                                        ? 'Добавьте комментарий...'
                                        : 'Спросите помощника...'
                            }
                            disabled={loading && !isListening}
                            className={cn(
                                'flex-1 resize-none bg-transparent text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400',
                                'focus:outline-none text-[15px] leading-relaxed py-2.5 min-h-[40px] max-h-[200px] overflow-y-auto scrollbar-none',
                                isListening && 'text-blue-600 font-medium',
                            )}
                        />

                        <div className="flex items-center gap-1 shrink-0 pb-1">
                            {isSpeechSupported && (
                                <button
                                    type="button"
                                    onClick={toggleMic}
                                    className={cn(
                                        'w-12 h-12 flex items-center justify-center rounded-full transition-all duration-300',
                                        isListening
                                            ? 'text-indigo-600 bg-indigo-50'
                                            : 'text-zinc-400 hover:text-indigo-600 hover:bg-indigo-50',
                                    )}
                                    title={isListening ? 'Остановить' : 'Голос'}
                                >
                                    {isListening ? <SoundWave active/> : <Mic size={20}/>}
                                </button>
                            )}

                            {loading ? (
                                <button
                                    type="button"
                                    onClick={handleStop}
                                    className="w-12 h-12 flex items-center justify-center rounded-full bg-rose-500 text-white shadow-lg shadow-rose-100 hover:bg-rose-600 active:scale-90 transition-all duration-300"
                                    title="Остановить"
                                >
                                    <Square size={16} fill="currentColor" />
                                </button>
                            ) : (
                                <button
                                    type="submit"
                                    disabled={!displayInput.trim() && !attachedFile}
                                    className={cn(
                                        'w-12 h-12 flex items-center justify-center rounded-full transition-all duration-300 shadow-lg active:scale-90',
                                        displayInput.trim() || attachedFile
                                            ? 'bg-indigo-600 text-white shadow-indigo-100 hover:bg-indigo-700'
                                            : 'bg-zinc-100 text-zinc-300 shadow-none cursor-not-allowed',
                                    )}
                                    title="Отправить"
                                >
                                    <Send size={18} className={displayInput.trim() || attachedFile ? "ml-0.5" : ""} />
                                </button>
                            )}
                        </div>
                    </div>
                </form>
            </div>
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
    onDocumentClick: (docId: string) => void
    onAttachmentClick: (fileName: string) => void
    onSendMessage: (text: string) => void
}

const SUMMARY_TYPE_LABELS: Record<string, string> = {
    extractive: 'Факты',
    abstractive: 'Пересказ',
    thesis: 'Тезисы',
}

function MessageRow({
                        msg,
                        threadId,
                        onInterruptReply,
                        onFieldFixed,
                        onAllFixed,
                        onRefreshSummary,
                        onDocumentClick,
                        onAttachmentClick,
                        onSendMessage,
                    }: MessageRowProps) {
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
            <div className={cn(msg.role === 'user' ? 'max-w-[85%]' : 'w-full max-w-[92%]')}>
                {msg.content && (
                    <ChatMessage
                        content={msg.content}
                        role={msg.role}
                        timestamp={msg.timestamp}
                        isError={msg.isError === true}
                        onDocumentClick={onDocumentClick}
                        onAttachmentClick={onAttachmentClick}
                    />
                )}

                {msg.compliance != null && msg.role === 'assistant' && (
                    <ComplianceResult
                        data={msg.compliance}
                        threadId={threadId}
                        refreshMeta={msg.refreshMeta}
                        onFieldFixed={handleFieldFixed}
                        onAllFixed={handleAllFixed}
                        onSendMessage={(text) => onSendMessage(text)}
                    />
                )}

                {msg.refreshMeta != null && msg.compliance == null && msg.role === 'assistant' && refreshLabel && (
                    <div className="mt-2 pl-2">
                        <button
                            type="button"
                            onClick={handleRefreshClick}
                            disabled={isRefreshing}
                            className={cn(
                                'inline-flex items-center gap-2 px-3.5 py-2 rounded-xl border text-[11px] font-bold transition-all shadow-sm',
                                isRefreshing
                                    ? 'bg-zinc-100 border-zinc-200 text-zinc-400 cursor-wait'
                                    : 'bg-white border-zinc-200 text-blue-600 hover:bg-zinc-50 active:scale-95',
                            )}
                        >
                            {isRefreshing ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                            <span>{isRefreshing ? `Обновляю ${refreshLabel}...` : `Обновить ${refreshLabel}`}</span>
                        </button>
                    </div>
                )}

                {msg.interrupt != null && msg.role === 'assistant' && (
                    <div className="mt-2">
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

const QUICK_ACTIONS = [
  { label: 'Суммаризация', icon: Sparkles },
  { label: 'Поиск', icon: Paperclip },
  { label: 'Тезисы', icon: MessageSquare }
]

function EmptyState({
                        onAction,
                        showQuickActions,
                    }: {
    onAction: (text: string) => void
    showQuickActions: boolean
}) {
    return (
        <div className="flex flex-col items-center justify-center flex-1 gap-8 px-10 py-12">
            <div className="w-24 h-24 rounded-[32px] bg-indigo-50 border border-indigo-100 flex items-center justify-center shadow-xl shadow-indigo-100/50 animate-edms-slide-up">
                <Sparkles size={44} className="text-indigo-600" />
            </div>
            <div className="text-center space-y-3 animate-edms-slide-up" style={{ animationDelay: '100ms' }}>
                <h1 className="text-2xl font-bold text-zinc-900 tracking-tight">Чем я могу помочь?</h1>
                <p className="text-[15px] text-zinc-500 font-medium leading-relaxed max-w-[280px] mx-auto">
                    Задайте вопрос о документах или выберите действие ниже
                </p>
            </div>
            {showQuickActions && (
                <div className="flex flex-wrap items-center gap-2.5 justify-center mt-2 animate-edms-slide-up" style={{ animationDelay: '200ms' }}>
                    {QUICK_ACTIONS.map((action) => {
                        const Icon = action.icon
                        return (
                            <button
                                key={action.label}
                                type="button"
                                onClick={() => onAction(action.label)}
                                className="flex items-center gap-2.5 px-5 py-3 rounded-2xl bg-white border border-zinc-100 text-[13px] font-bold text-zinc-700 hover:border-indigo-200 hover:text-indigo-600 transition-all shadow-sm hover:shadow-md hover:translate-y-[-1px] active:scale-95"
                            >
                                <Icon size={16} className="text-zinc-400 group-hover:text-indigo-500" />
                                {action.label}
                            </button>
                        )
                    })}
                </div>
            )}
        </div>
    )
}
