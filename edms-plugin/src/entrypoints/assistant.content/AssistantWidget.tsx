import {useState, useRef, useEffect, useCallback, memo, type FormEvent} from 'react'
import {
    Paperclip, X, Mic, Send, MessageSquare,
    StopCircle, FileText, Search, List, History, Settings,
} from 'lucide-react'
import dayjs from 'dayjs'
import 'dayjs/locale/ru'

import {ChatMessage} from '@/shared/ui/ChatMessage'
import {getAuthToken} from '@/shared/lib/auth'
import {extractDocIdFromUrl} from '@/shared/lib/url'
import {sendMsg} from '@/shared/lib/messaging'
import {toast} from '@/shared/lib/toast'
import {useSpeechRecognition} from '@/shared/hooks/useSpeechRecognition'
import {useApplyPreferences} from '@/shared/hooks/useApplyPreferences'
import {SettingsPanel} from '@/shared/ui/SettingsPanel'

dayjs.locale('ru')

interface Message {
    role: 'user' | 'assistant'
    content: string
    action_type?: string
    isError?: boolean
    id: string
    timestamp: number
    cacheFileIdentifier?: string | null
    cacheSummaryType?: string | null
    cacheFilePath?: string | null
    cacheContextId?: string | null
}

interface Thread {
    id: string
    preview: string
    date: string
}

const newMsgId = (): string =>
    typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `msg_${Date.now()}_${Math.random().toString(36).slice(2)}`

const MAX_THREADS = 20
const THREADS_STORAGE_KEY = 'edmsWidgetThreads'
const CHOICE_LABELS: Record<string, string> = {abstractive: 'Пересказ', extractive: 'Факты', thesis: 'Тезисы'}

function persistThreads(threads: Thread[]): void {
    chrome.storage.local.set({[THREADS_STORAGE_KEY]: threads})
}

function makeErrorMessage(err: unknown): string {
    if (err instanceof Error) return `__error__:${err.message}`
    if (err && typeof err === 'object') {
        const e = err as Record<string, unknown>
        if (typeof e.error === 'string') return `__error__:${e.error}`
        if (typeof e.message === 'string') return `__error__:${e.message}`
        if (typeof e.detail === 'string') return `__error__:${e.detail}`
    }
    return `__error__:${String(err)}`
}

function getToastErrorText(err: unknown): string {
    let rawMsg: string
    if (err instanceof Error) rawMsg = err.message
    else if (err && typeof err === 'object') {
        const e = err as Record<string, unknown>
        rawMsg = typeof e.error === 'string' ? e.error
            : typeof e.message === 'string' ? e.message
                : typeof e.detail === 'string' ? e.detail
                    : JSON.stringify(e)
    } else {
        rawMsg = String(err)
    }
    const raw = rawMsg.toLowerCase()
    if (raw.includes('failed to fetch') || raw.includes('networkerror')) return 'Нет соединения с сервером'
    if (raw.includes('timeout')) return 'Сервер не ответил вовремя'
    if (raw.includes('401') || raw.includes('unauthorized')) return 'Ошибка авторизации — обновите страницу'
    if (raw.includes('403')) return 'Доступ запрещён'
    return err instanceof Error ? err.message : String(err)
}

function buildThreadPreview(messages: Message[]): string {
    const userMsgs = messages.filter(m => m.role === 'user' && !m.isError)
    const meaningful = userMsgs.find(m => m.content.trim().length > 15) ?? userMsgs[0]
    return (meaningful?.content ?? 'Диалог').trim().slice(0, 50)
}

function SoundWave() {
    return (
        <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 3,
            height: 16,
            marginBottom: 6,
            padding: '0 4px',
        }}>
            {[0, 1, 2, 3, 4].map(i => (
                <div key={i} style={{
                    width: 2.5,
                    height: 8,
                    borderRadius: 2,
                    background: 'rgba(99,102,241,0.50)',
                    transformOrigin: 'bottom',
                    animation: `edms-soundbar 0.7s ease-in-out ${i * 90}ms infinite`,
                }}/>
            ))}
        </div>
    )
}

function TypingDots() {
    return (
        <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            padding: '12px 18px',
            // background: '#ffffff',
            borderRadius: 22,
            width: 'fit-content',
            // boxShadow: 'var(--edms-shadow-sm)',
            // border: '1px solid rgba(0,0,0,0.03)',
            animation: 'edms-fade-in-up .3s ease-out',
        }}>
            {[0, 180, 360].map(d => (
                <div key={d} style={{
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    background: '#a5b4fc',
                    animation: `edms-wave 1.6s ease-in-out ${d}ms infinite`,
                }}/>
            ))}
        </div>
    )
}

const CandidateIcon = memo(({type, className}: { type: string; className?: string }) => {
    const cls = `w-4 h-4 flex-shrink-0 ${className ?? ''}`
    if (type === 'employee') return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="12" cy="8" r="4"/>
            <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
        </svg>
    )
    if (type === 'docx') return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <line x1="10" y1="9" x2="8" y2="9"/>
        </svg>
    )
    if (type === 'xlsx') return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="8" y1="12" x2="16" y2="12"/>
            <line x1="8" y1="16" x2="16" y2="16"/>
            <line x1="12" y1="10" x2="12" y2="18"/>
        </svg>
    )
    if (type === 'pdf') return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <path d="M9 13h1.5a1.5 1.5 0 0 1 0 3H9v-3z"/>
        </svg>
    )
    return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
            <polyline points="13 2 13 9 20 9"/>
        </svg>
    )
})
CandidateIcon.displayName = 'CandidateIcon'

const typeAccent: Record<string, string> = {
    employee: 'text-violet-500 group-hover:text-white',
    docx: 'text-blue-500 group-hover:text-white',
    xlsx: 'text-emerald-500 group-hover:text-white',
    pdf: 'text-rose-500 group-hover:text-white',
    file: 'text-slate-400 group-hover:text-white',
}

function getCandidateType(name: string): 'employee' | 'docx' | 'xlsx' | 'pdf' | 'file' {
    const lower = name.toLowerCase()
    if (lower.endsWith('.docx') || lower.endsWith('.doc')) return 'docx'
    if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) return 'xlsx'
    if (lower.endsWith('.pdf')) return 'pdf'
    if (!lower.includes('.')) return 'employee'
    return 'file'
}

function parseDisambigCandidates(content: string): Array<{ id: string; name: string; dept: string }> {
    const m = content.match(/<!--CANDIDATES:(.+?)-->/)
    if (!m) return []
    try {
        return JSON.parse(m[1])
    } catch {
        return []
    }
}

function cleanDisambigMessage(content: string): string {
    return content.replace(/\n\n<!--CANDIDATES:.+?-->/, '').trimEnd()
}

interface ActionButtonsProps {
    msg: Message
    loading: boolean
    onSend: (choice: string, label: string) => void
    defaultSummaryFormat?: string
}

const ActionButtons = memo(({msg, loading, onSend, defaultSummaryFormat}: ActionButtonsProps) => {
    useEffect(() => {
        if (
            msg.action_type === 'summarize_selection' &&
            defaultSummaryFormat &&
            defaultSummaryFormat !== 'ask' &&
            !loading
        ) {
            const label = CHOICE_LABELS[defaultSummaryFormat] ?? defaultSummaryFormat
            const timer = setTimeout(() => onSend(defaultSummaryFormat, label), 150)
            return () => clearTimeout(timer)
        }
    }, [msg.id])

    if (msg.action_type !== 'summarize_selection') return null
    if (defaultSummaryFormat && defaultSummaryFormat !== 'ask') return null

    return (
        <div className="mt-2.5 flex flex-wrap gap-2" style={{animation: 'edms-fade-in-up .3s ease-out'}}>
            {[
                {id: 'abstractive', label: 'Пересказ', icon: <FileText size={13}/>},
                {id: 'extractive', label: 'Факты', icon: <Search size={13}/>},
                {id: 'thesis', label: 'Тезисы', icon: <List size={13}/>},
            ].map(btn => (
                <button
                    key={btn.id}
                    type="button"
                    disabled={loading}
                    onClick={e => {
                        e.stopPropagation();
                        onSend(btn.id, btn.label)
                    }}
                    className="edms-action-btn flex items-center gap-1.5 px-3.5 py-2 font-semibold rounded-xl border bg-white text-indigo-600 hover:bg-indigo-600 hover:text-white hover:border-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
                    style={{
                        borderColor: 'rgba(99,102,241,0.18)',
                        boxShadow: 'var(--edms-shadow-xs)',
                    }}
                    onMouseEnter={e => {
                        if (loading) return
                        const el = e.currentTarget as HTMLButtonElement
                        el.style.boxShadow = '0 4px 14px rgba(99,102,241,0.25)'
                        el.style.transform = 'translateY(-1px)'
                    }}
                    onMouseLeave={e => {
                        const el = e.currentTarget as HTMLButtonElement
                        el.style.boxShadow = 'var(--edms-shadow-xs)'
                        el.style.transform = 'translateY(0)'
                    }}
                >
                    {btn.icon}{btn.label}
                </button>
            ))}
        </div>
    )
})
ActionButtons.displayName = 'ActionButtons'

interface DisambButtonsProps {
    msg: Message
    loading: boolean
    onSend: (choice: string, label: string) => void
}

const DisambiguationButtons = memo(({msg, loading, onSend}: DisambButtonsProps) => {
    if (msg.action_type !== 'requires_disambiguation') return null
    const candidates = parseDisambigCandidates(msg.content)
    if (candidates.length === 0) return null
    const isEmployeeList = candidates.every(c => getCandidateType(c.name) === 'employee')
    const isFileList = candidates.every(c => getCandidateType(c.name) !== 'employee')
    return (
        <div className="mt-3" style={{animation: 'edms-fade-in-up .3s ease-out'}}>
            <p className="mb-2.5 px-1" style={{
                fontSize: 10,
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: '#94a3b8',
            }}>
                {isEmployeeList ? '👤 Выберите сотрудника' : isFileList ? '📎 Выберите вложение' : '✦ Выберите вариант'}
            </p>
            <div className="flex flex-col gap-1.5">
                {candidates.map((c, idx) => {
                    const ctype = getCandidateType(c.name)
                    const accent = typeAccent[ctype] ?? typeAccent.file
                    return (
                        <button
                            key={c.id}
                            type="button"
                            disabled={loading}
                            onClick={e => {
                                e.stopPropagation();
                                onSend(c.id, c.name)
                            }}
                            className="group flex items-center gap-3 w-full px-3.5 py-3 rounded-xl border bg-white text-left hover:bg-indigo-600 hover:border-indigo-600 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
                            style={{
                                borderColor: 'rgba(0,0,0,0.05)',
                                color: '#1e293b',
                                fontSize: 12,
                                boxShadow: 'var(--edms-shadow-xs)',
                            }}
                            onMouseEnter={e => {
                                if (loading) return
                                const el = e.currentTarget as HTMLButtonElement
                                el.style.boxShadow = '0 4px 16px rgba(99,102,241,0.22)'
                                el.style.transform = 'translateY(-1px)'
                            }}
                            onMouseLeave={e => {
                                const el = e.currentTarget as HTMLButtonElement
                                el.style.boxShadow = 'var(--edms-shadow-xs)'
                                el.style.transform = 'translateY(0)'
                            }}
                        >
                            <span
                                className="flex-shrink-0 w-6 h-6 rounded-lg text-[10px] font-bold flex items-center justify-center transition-colors duration-200"
                                style={{background: 'rgba(241,245,249,0.80)', color: '#64748b'}}>
                                {idx + 1}
                            </span>
                            <CandidateIcon type={ctype} className={accent}/>
                            <div className="flex-1 min-w-0">
                                <p className="font-semibold truncate leading-tight">{c.name}</p>
                                {c.dept &&
                                    <p className="truncate mt-0.5 leading-tight opacity-60 group-hover:opacity-80"
                                       style={{fontSize: 10}}>{c.dept}</p>}
                            </div>
                            <svg
                                className="w-3.5 h-3.5 flex-shrink-0 opacity-25 group-hover:opacity-80 transition-opacity"
                                viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                                <path d="M9 18l6-6-6-6"/>
                            </svg>
                        </button>
                    )
                })}
            </div>
        </div>
    )
})
DisambiguationButtons.displayName = 'DisambiguationButtons'

interface RefreshCacheButtonProps {
    msg: Message
    userToken: string
    onRefreshStart: (msgId: string, prevContent: string) => void
    onRefreshDone: (msgId: string, newContent: string, payload?: any) => void
    onRefreshError: (msgId: string) => void
}

const SUMMARY_TYPE_LABELS: Record<string, string> = {
    extractive: 'Факты',
    abstractive: 'Пересказ',
    thesis: 'Тезисы',
}

const RefreshCacheButton = memo(({
                                     msg,
                                     userToken,
                                     onRefreshStart,
                                     onRefreshDone,
                                     onRefreshError
                                 }: RefreshCacheButtonProps) => {
    const [refreshing, setRefreshing] = useState(false)

    if (msg.role !== 'assistant' || msg.action_type || msg.isError) return null
    if (!msg.cacheFileIdentifier) return null

    const effectiveFilePath = msg.cacheFilePath ?? msg.cacheFileIdentifier

    const typeLabel = msg.cacheSummaryType
        ? SUMMARY_TYPE_LABELS[msg.cacheSummaryType] ?? msg.cacheSummaryType
        : null

    const handleRefresh = async () => {
        if (refreshing) return
        setRefreshing(true)
        onRefreshStart(msg.id, msg.content)

        try {
            await sendMsg('deleteCache', {
                file_identifier: msg.cacheFileIdentifier,
                summary_type: msg.cacheSummaryType ?? undefined,
            })

            const summaryType = msg.cacheSummaryType ?? 'extractive'
            const res = await sendMsg<any>('summarizeDocument', {
                message: `Проанализируй вложение`,
                user_token: userToken,
                context_ui_id: msg.cacheContextId ?? null,
                file_path: effectiveFilePath,
                human_choice: summaryType,
            })

            const payload = (res && typeof res === 'object' && 'data' in res && (res as any).success)
                ? (res as any).data
                : res

            const newContent = payload?.response ?? payload?.content ?? payload?.message ?? 'Анализ завершён.'
            onRefreshDone(msg.id, newContent, payload)

            const label = typeLabel ? `«${typeLabel}»` : 'анализ'
            toast.success(`${label} обновлён.`, 'Готово')
        } catch {
            onRefreshError(msg.id)
            toast.error('Не удалось обновить анализ', 'Ошибка')
        } finally {
            setRefreshing(false)
        }
    }

    return (
        <button
            type="button"
            disabled={refreshing}
            onClick={handleRefresh}
            title={typeLabel ? `Пересчитать анализ «${typeLabel}»` : 'Пересчитать анализ'}
            className="mt-2 flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all duration-200 self-start"
            style={{
                color: '#94a3b8',
                background: 'transparent',
                border: '1px solid rgba(0,0,0,0.05)',
            }}
            onMouseEnter={e => {
                if (refreshing) return
                const el = e.currentTarget as HTMLButtonElement
                el.style.color = '#6366f1'
                el.style.borderColor = 'rgba(99,102,241,0.20)'
                el.style.background = 'rgba(99,102,241,0.04)'
            }}
            onMouseLeave={e => {
                const el = e.currentTarget as HTMLButtonElement
                el.style.color = '#94a3b8'
                el.style.borderColor = 'rgba(0,0,0,0.05)'
                el.style.background = 'transparent'
            }}
        >
            <svg
                width="11" height="11" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"
                style={refreshing ? {animation: 'spin 1s linear infinite'} : {}}
            >
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
                <path d="M21 3v5h-5"/>
                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
                <path d="M8 16H3v5"/>
            </svg>
            {refreshing
                ? 'Обновляю…'
                : typeLabel ? `Обновить «${typeLabel}»` : 'Обновить анализ'}
        </button>
    )
})
RefreshCacheButton.displayName = 'RefreshCacheButton'

export function AssistantWidget() {
    const [isEnabled, setIsEnabled] = useState(true)
    const [isOpen, setIsOpen] = useState(false)
    const [isSidebarOpen, setIsSidebarOpen] = useState(false)
    const [isSettingsOpen, setIsSettingsOpen] = useState(false)
    const [messages, setMessages] = useState<Message[]>([])
    const [threads, setThreads] = useState<Thread[]>([])
    const [threadId, setThreadId] = useState<string | null>(null)
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [attachedFile, setAttachedFile] = useState<{ path: string; name: string } | null>(null)
    const [isFocused, setIsFocused] = useState(false)
    const [userContext, setUserContext] = useState<Record<string, string>>({})
    const [pendingDelete, setPendingDelete] = useState<{ id: string; thread: Thread } | null>(null)
    const [handsFree, setHandsFree] = useState(false)

    const {rootClassName, rootStyle, prefs: userPrefs} = useApplyPreferences()

    useEffect(() => {
        setHandsFree(userPrefs.voice.handsFreeEnabled)
    }, [userPrefs.voice.handsFreeEnabled])

    const sendRef = useRef<() => void>(() => {
    })

    const {
        isListening,
        interimTranscript,
        isSupported: isSpeechSupported,
        autoSendPending,
        toggle: toggleMic,
        stop: stopMic,
    } = useSpeechRecognition({
        lang: userPrefs.voice.sttLanguage,
        silenceMs: 3000,
        autoSend: handsFree,
        autoSendMs: userPrefs.voice.autoSendPauseMs,
        onFinalResult: (deltaText) => {
            setInput(prev => (prev.trimEnd() ? `${prev.trimEnd()} ` : '') + deltaText)
        },
        onAutoSend: () => {
            stopMicRef.current()
            sendRef.current()
        },
    })

    const stopMicRef = useRef(stopMic)
    stopMicRef.current = stopMic

    const bottomRef = useRef<HTMLDivElement>(null)
    const fileRef = useRef<HTMLInputElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const requestIdRef = useRef<string | null>(null)
    const serverPathRef = useRef<string | null>(null)
    const filePersistRef = useRef<string | null>(null)
    const messagesRef = useRef<Message[]>(messages)
    const refreshPrevContentRef = useRef<Record<string, string>>({})
    messagesRef.current = messages

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto'
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`
        }
    }, [input])

    useEffect(() => {
        chrome.storage.local.get(['assistantEnabled'], r => {
            if (r.assistantEnabled !== undefined) setIsEnabled(r.assistantEnabled)
        })
        chrome.storage.local.get([THREADS_STORAGE_KEY], r => {
            const saved = r[THREADS_STORAGE_KEY]
            if (Array.isArray(saved) && saved.length > 0) setThreads(saved)
        })
        chrome.storage.local.get(['userContext'], r => {
            if (r.userContext && typeof r.userContext === 'object') setUserContext(r.userContext)
        })

        const onStorage = (changes: Record<string, chrome.storage.StorageChange>, area: string) => {
            if (area !== 'local') return
            if ('assistantEnabled' in changes) {
                const val = changes.assistantEnabled.newValue as boolean
                setIsEnabled(val)
                if (!val) setIsOpen(false)
            }
            if ('userContext' in changes) {
                const val = changes.userContext.newValue
                if (val && typeof val === 'object') setUserContext(val as Record<string, string>)
            }
        }
        chrome.storage.onChanged.addListener(onStorage)

        const onWindowMsg = (e: MessageEvent) => {
            if (e.data?.type !== 'REFRESH_CHAT_HISTORY') return
            const {
                messages: raw,
                thread_id,
                cache_file_identifier,
                cache_summary_type,
                cache_file_path,
                cache_context_id
            } = e.data as {
                messages: { type: string; content: string }[]
                thread_id?: string
                cache_file_identifier?: string | null
                cache_summary_type?: string | null
                cache_file_path?: string | null
                cache_context_id?: string | null
            }
            const now = Date.now()
            const mapped = raw.map((m, i) => ({
                role: (m.type === 'human' ? 'user' : 'assistant') as Message['role'],
                content: m.content,
                id: newMsgId(),
                timestamp: now + i,
                cacheFileIdentifier: (m.type === 'ai' && i === raw.length - 1)
                    ? (cache_file_identifier ?? null) : null,
                cacheSummaryType: (m.type === 'ai' && i === raw.length - 1)
                    ? (cache_summary_type ?? null) : null,
                cacheFilePath: (m.type === 'ai' && i === raw.length - 1)
                    ? (cache_file_path ?? null) : null,
                cacheContextId: (m.type === 'ai' && i === raw.length - 1)
                    ? (cache_context_id ?? null) : null,
            }))
            if (thread_id) {
                setThreadId(thread_id)
                setMessages(mapped)
            } else {
                setMessages(prev => [...prev, ...mapped])
            }
            setIsOpen(true)
            setIsSidebarOpen(false)
            setTimeout(() => bottomRef.current?.scrollIntoView({behavior: 'smooth'}), 120)
        }
        window.addEventListener('message', onWindowMsg)
        return () => {
            chrome.storage.onChanged.removeListener(onStorage)
            window.removeEventListener('message', onWindowMsg)
        }
    }, [])

    useEffect(() => {
        bottomRef.current?.scrollIntoView({behavior: 'smooth'})
    }, [messages, loading])

    useEffect(() => {
        chrome.storage.local.get(['edmsWidgetSnapshot'], r => {
            const snap = r?.edmsWidgetSnapshot
            if (snap) chrome.storage.local.remove('edmsWidgetSnapshot')
            if (!snap) return
            if (Date.now() - (snap.savedAt ?? 0) > 30_000) return
            if (Array.isArray(snap.messages) && snap.messages.length > 0) setMessages(snap.messages)
            if (snap.threadId) setThreadId(snap.threadId)
            setIsOpen(true)
            setTimeout(() => bottomRef.current?.scrollIntoView({behavior: 'smooth'}), 200)
        })
    }, [])

    useEffect(() => {
        if (!pendingDelete) return
        const timer = setTimeout(() => setPendingDelete(null), 3000)
        return () => clearTimeout(timer)
    }, [pendingDelete])

    const newChat = async () => {
        if (messages.length && threadId) {
            const preview = buildThreadPreview(messages)
            setThreads(prev => {
                if (prev.find(t => t.id === threadId)) return prev
                const updated = [{id: threadId, preview, date: dayjs().format('HH:mm')}, ...prev].slice(0, MAX_THREADS)
                persistThreads(updated)
                return updated
            })
        }
        setLoading(true)
        try {
            const res = await sendMsg<{
                thread_id: string
            }>('createNewChat', {user_token: getAuthToken() ?? 'no_token'})
            setThreadId(res.thread_id)
            setMessages([])
            setIsSidebarOpen(false)
            filePersistRef.current = null
            serverPathRef.current = null
            setAttachedFile(null)
            if (fileRef.current) fileRef.current.value = ''
        } catch (err) {
            toast.error(getToastErrorText(err), 'Не удалось создать диалог')
        } finally {
            setLoading(false)
        }
    }

    const loadHistory = async (id: string) => {
        setLoading(true)
        setIsSidebarOpen(false)
        try {
            const res = await sendMsg<{
                messages: { type: string; content: string }[]
            }>('getChatHistory', {thread_id: id})
            const now = Date.now()
            setMessages(res.messages.map((m, i) => ({
                role: (m.type === 'human' ? 'user' : 'assistant') as Message['role'],
                content: m.content,
                id: newMsgId(),
                timestamp: now - (res.messages.length - i) * 1000,
            })))
            setThreadId(id)
        } catch (err) {
            toast.error(getToastErrorText(err), 'Не удалось загрузить историю')
        } finally {
            setLoading(false)
        }
    }

    const deleteThread = useCallback((id: string, e: React.MouseEvent) => {
        e.stopPropagation()
        setThreads(prev => {
            const target = prev.find(t => t.id === id)
            if (!target) return prev
            setPendingDelete({id, thread: target})
            const updated = prev.filter(t => t.id !== id)
            persistThreads(updated)
            return updated
        })
        if (threadId === id) {
            setMessages([])
            setThreadId(null)
        }
    }, [threadId])

    const undoDelete = useCallback(() => {
        if (!pendingDelete) return
        setThreads(prev => {
            const updated = [pendingDelete.thread, ...prev].slice(0, MAX_THREADS)
            persistThreads(updated)
            return updated
        })
        setPendingDelete(null)
    }, [pendingDelete])

    const handleRefreshStart = useCallback((msgId: string, prevContent: string) => {
        refreshPrevContentRef.current[msgId] = prevContent
        setMessages(prev => prev.map(m =>
            m.id === msgId ? {...m, content: '…'} : m
        ))
    }, [])

    const handleRefreshDone = useCallback((msgId: string, newContent: string, newPayload?: any) => {
        delete refreshPrevContentRef.current[msgId]
        setMessages(prev => prev.map(m => {
            if (m.id !== msgId) return m
            return {
                ...m,
                content: newContent,
                cacheFileIdentifier: newPayload?.metadata?.cache_file_identifier ?? m.cacheFileIdentifier,
                cacheSummaryType: newPayload?.metadata?.cache_summary_type ?? m.cacheSummaryType,
                cacheContextId: newPayload?.metadata?.cache_context_ui_id ?? m.cacheContextId,
                cacheFilePath: m.cacheFilePath,
            }
        }))
    }, [])

    const handleRefreshError = useCallback((msgId: string) => {
        const prev = refreshPrevContentRef.current[msgId]
        if (prev) {
            setMessages(msgs => msgs.map(m =>
                m.id === msgId ? {...m, content: prev} : m
            ))
            delete refreshPrevContentRef.current[msgId]
        }
    }, [])

    const send = async (e?: FormEvent | React.KeyboardEvent, humanChoice?: string, humanChoiceLabel?: string) => {
        if (e) e.preventDefault()
        const isChoiceFlow = Boolean(humanChoice)
        const hasTextInput = input.trim().length > 0
        const hasFile = Boolean(attachedFile)
        if (!isChoiceFlow && !hasTextInput && !hasFile) return
        if (loading) return
        if (isListening) stopMic()

        const token = getAuthToken() ?? 'no_token'
        const docId = extractDocIdFromUrl()
        const reqId = Math.random().toString(36).slice(7)
        const tid = threadId ?? `${token.slice(0, 8)}_${docId}`
        requestIdRef.current = reqId
        if (!threadId) setThreadId(tid)

        const userLabel = isChoiceFlow
            ? (humanChoiceLabel ?? CHOICE_LABELS[humanChoice!] ?? humanChoice!)
            : hasFile ? `${input} (Файл: ${attachedFile!.name})`.trim()
                : input
        setMessages(prev => [...prev, {role: 'user', content: userLabel, id: newMsgId(), timestamp: Date.now()}])
        setLoading(true)
        const textToSend = isChoiceFlow ? (humanChoice ?? '') : input
        setInput('')

        try {
            if (hasFile && !isChoiceFlow) {
                const up = await sendMsg<{ file_path: string }>('uploadFile', {
                    fileData: attachedFile!.path,
                    fileName: attachedFile!.name,
                    user_token: token,
                })
                serverPathRef.current = up.file_path
                filePersistRef.current = up.file_path
            }

            const finalFilePath = isChoiceFlow && filePersistRef.current
                ? filePersistRef.current
                : serverPathRef.current

            const preferredFormat = userPrefs.documents.defaultSummaryFormat
            const resolvedHumanChoice = isChoiceFlow
                ? humanChoice!
                : (preferredFormat && preferredFormat !== 'ask' ? preferredFormat : undefined)

            const res = await sendMsg<any>('sendChatMessage', {
                message: isChoiceFlow ? humanChoice! : textToSend,
                user_token: token,
                requestId: reqId,
                thread_id: tid,
                context_ui_id: docId,
                file_path: finalFilePath,
                human_choice: resolvedHumanChoice,
                preferred_summary_format: preferredFormat !== 'ask' ? preferredFormat : undefined,
                user_context: Object.keys(userContext).length > 0 ? userContext : undefined,
            })

            if (!isChoiceFlow) {
                serverPathRef.current = null
                filePersistRef.current = null
            }

            const payload = (res && typeof res === 'object' && 'data' in res && res.success) ? (res as any).data : res
            const content = payload?.response ?? payload?.content ?? payload?.message
                ?? (Array.isArray(payload?.messages) ? payload.messages.at(-1)?.content : undefined)
                ?? 'Анализ завершён.'

            const cacheFileIdentifier: string | null = payload?.metadata?.cache_file_identifier ?? null
            const cacheSummaryType: string | null = payload?.metadata?.cache_summary_type ?? null

            const assistantMsg: Message = {
                role: 'assistant',
                content,
                action_type: payload?.action_type,
                id: newMsgId(),
                timestamp: Date.now(),
                cacheFileIdentifier,
                cacheSummaryType,
                cacheFilePath: finalFilePath ?? null,
                cacheContextId: docId ?? null,
            }
            setMessages(prev => [...prev, assistantMsg])

            if (payload?.navigate_url) {
                chrome.runtime.sendMessage(
                    {type: 'navigateTo', payload: {url: payload.navigate_url}},
                    (r) => {
                        if (!r?.success) {
                            try {
                                window.location.href = payload.navigate_url
                            } catch {
                                toast.info(`Документ создан. Перейдите по адресу: ${payload.navigate_url}`)
                            }
                        }
                    }
                )
            } else if (payload?.requires_reload) {
                refreshDocumentPage(docId, [...messagesRef.current, assistantMsg])
            }
        } catch (err: unknown) {
            if (!String(err).includes('aborted')) {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: makeErrorMessage(err),
                    isError: true,
                    id: newMsgId(),
                    timestamp: Date.now(),
                }])
            }
        } finally {
            setLoading(false)
            requestIdRef.current = null
            if (!isChoiceFlow) {
                setAttachedFile(null)
                if (fileRef.current) fileRef.current.value = ''
            }
        }
    }

    const sendWithLabel = (humanChoice: string, label: string) => send(undefined, humanChoice, label)
    sendRef.current = () => send()

    const refreshDocumentPage = async (_documentId: string | null, finalMessages?: Message[]) => {
        try {
            const [tab] = await chrome.tabs.query({active: true, currentWindow: true})
            if (tab?.id) {
                const results = await chrome.scripting.executeScript({
                    target: {tabId: tab.id},
                    func: () => {
                        const refresh = (window as any).__edms_refresh__
                        if (typeof refresh === 'function') {
                            refresh();
                            return true
                        }
                        return false
                    },
                })
                if (results?.[0]?.result === true) return
            }
        } catch {
        }
        const snapshot = {messages: finalMessages ?? messagesRef.current, threadId, isOpen: true, savedAt: Date.now()}
        try {
            chrome.storage.local.set({edmsWidgetSnapshot: snapshot}, () => {
                window.location.reload()
            })
        } catch {
            window.location.reload()
        }
    }

    const handleDocumentClick = useCallback((documentId: string) => {
        const normalizedId = documentId.replace(
            /[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00AD\uFE58\uFE63\uFF0D]/g,
            '-'
        ).trim()

        const url = `/document-form/${normalizedId}`

        chrome.runtime.sendMessage(
            {type: 'navigateTo', payload: {url, newTab: true}},
            (r) => {
                if (!r?.success) {
                    try {
                        window.open(url, '_blank', 'noopener,noreferrer')
                    } catch {
                        toast.info(`Откройте документ: ${url}`)
                    }
                }
            }
        )
    }, [])

    const abort = () => {
        if (!requestIdRef.current) return
        chrome.runtime.sendMessage({type: 'abortRequest', payload: {requestId: requestIdRef.current}})
        setLoading(false)
        requestIdRef.current = null
        setMessages(prev => [...prev, {
            role: 'assistant',
            content: '_Запрос отменён._',
            id: newMsgId(),
            timestamp: Date.now()
        }])
    }

    const closeWidget = () => {
        setIsOpen(false)
        setIsSidebarOpen(false)
        setIsSettingsOpen(false)
    }

    const canSend = input.trim().length > 0 || Boolean(attachedFile)

    // Прозрачность: 0 = непрозрачный, 0.5 = максимально прозрачный
    const glassOpacity = userPrefs.appearance.glassOpacity ?? 0
    const glassAlpha = 1 - glassOpacity

    if (!isEnabled) return null

    return (
        <div className={rootClassName} style={rootStyle}>

            {pendingDelete && (
                <div
                    className="pointer-events-auto mb-2 flex items-center gap-3 px-4 py-2.5 rounded-2xl text-white shadow-xl"
                    style={{
                        background: 'rgba(15,23,42,0.92)',
                        border: '1px solid rgba(255,255,255,0.06)',
                        animation: 'edms-fade-in .25s ease-out',
                        backdropFilter: 'blur(16px)',
                        fontSize: 12,
                    }}
                >
                    <span style={{opacity: 0.80}}>Диалог удалён</span>
                    <button type="button" onClick={undoDelete}
                            className="font-semibold transition-colors duration-200"
                            style={{color: '#a5b4fc'}}>Отменить
                    </button>
                </div>
            )}

            {!isOpen && (
                <button
                    type="button"
                    onClick={() => setIsOpen(true)}
                    className="pointer-events-auto relative w-[56px] h-[56px] rounded-full flex items-center justify-center glass hover:scale-110 active:scale-95 transition-transform duration-300 group"
                    style={{boxShadow: '0 4px 20px rgba(99,102,241,0.25)'}}
                >
                    <span style={{
                        position: 'absolute',
                        inset: -4,
                        borderRadius: '50%',
                        background: 'rgba(99,102,241,0.15)',
                        animation: 'edms-ripple 3s cubic-bezier(0.4,0,0.2,1) infinite',
                        pointerEvents: 'none',
                    }}/>
                    <MessageSquare size={24}
                                   className="text-indigo-500 group-hover:text-indigo-600 transition-colors z-10"
                                   strokeWidth={2}/>
                </button>
            )}

            {isOpen && (
                <div
                    className="pointer-events-auto flex flex-col w-[500px] max-w-[calc(100vw-32px)] overflow-hidden"
                    style={{
                        height: 'min(740px, calc(100vh - 64px))',
                        borderRadius: 24,
                        background: `rgba(255,255,255,${glassAlpha})`,
                        border: `1px solid rgba(255,255,255,${Math.min(glassAlpha + 0.12, 0.65)})`,
                        boxShadow: `0 8px 40px rgba(0,0,0,${0.04 + glassOpacity * 0.08}), 0 0 0 1px rgba(0,0,0,0.03)`,
                        backdropFilter: `blur(${Math.round(20 + glassOpacity * 20)}px) saturate(1.8)`,
                        WebkitBackdropFilter: `blur(${Math.round(20 + glassOpacity * 20)}px) saturate(1.8)`,
                        animation: 'edms-fade-in .35s cubic-bezier(.22,1,.36,1) forwards',
                    }}
                >
                    {/* ── Header ─────────────────────────────────────────── */}
                    <header
                        className="flex items-center justify-between px-4 py-3 shrink-0"
                        style={{
                            background: `rgba(255,255,255,${Math.min(glassAlpha + 0.15, 0.85)})`,
                            borderBottom: '1px solid rgba(0,0,0,0.05)',
                            backdropFilter: 'blur(12px)',
                        }}
                    >
                        <div className="flex items-center gap-2.5">
                            <button
                                type="button"
                                onClick={() => setIsSidebarOpen(v => !v)}
                                className="p-2 rounded-xl transition-all duration-200 flex flex-col justify-center items-center gap-[4px] w-8 h-8"
                                style={{
                                    background: isSidebarOpen ? 'rgba(99,102,241,0.08)' : 'transparent',
                                    color: isSidebarOpen ? '#6366f1' : '#94a3b8',
                                }}
                            >
                                <span
                                    className={`h-[1.5px] w-3.5 bg-current rounded-full transition-all duration-300 origin-center ${isSidebarOpen ? 'rotate-45 translate-y-[5px]' : ''}`}/>
                                <span
                                    className={`h-[1.5px] w-3.5 bg-current rounded-full transition-all duration-200 ${isSidebarOpen ? 'opacity-0 scale-x-0' : ''}`}/>
                                <span
                                    className={`h-[1.5px] w-3.5 bg-current rounded-full transition-all duration-300 origin-center ${isSidebarOpen ? '-rotate-45 -translate-y-[5px]' : ''}`}/>
                            </button>
                            <div className="flex items-center gap-2">
                                <div style={{
                                    width: 8, height: 8, borderRadius: '50%',
                                    background: '#6366f1',
                                    boxShadow: '0 0 8px rgba(99,102,241,0.40)',
                                }}/>
                                <h3 className="edms-header-title"
                                    style={{color: '#0f172a'}}>EDMS Assistant</h3>
                            </div>
                        </div>
                        <button
                            type="button"
                            onClick={closeWidget}
                            className="p-2 rounded-xl transition-all duration-200"
                            style={{color: '#cbd5e1'}}
                            onMouseEnter={e => {
                                const el = e.currentTarget as HTMLElement;
                                el.style.color = '#f87171';
                                el.style.background = 'rgba(254,226,226,0.60)'
                            }}
                            onMouseLeave={e => {
                                const el = e.currentTarget as HTMLElement;
                                el.style.color = '#cbd5e1';
                                el.style.background = 'transparent'
                            }}
                        >
                            <X size={17}/>
                        </button>
                    </header>

                    <div className="flex-1 flex overflow-hidden">
                        {/* ── Sidebar ────────────────────────────────────── */}
                        <aside
                            className={`shrink-0 flex flex-col transition-all duration-300 ease-out overflow-hidden ${isSidebarOpen ? 'w-[232px]' : 'w-0'}`}
                            style={{
                                background: `rgba(248,250,252,${Math.min(glassAlpha + 0.2, 0.92)})`,
                                borderRight: isSidebarOpen ? '1px solid rgba(0,0,0,0.05)' : '1px solid transparent',
                                backdropFilter: 'blur(16px)',
                            }}
                        >
                            <div className="p-3 w-[232px] flex flex-col h-full">
                                {isSettingsOpen ? (
                                    <SettingsPanel onClose={() => setIsSettingsOpen(false)}/>
                                ) : (
                                    <>
                                        <button
                                            type="button"
                                            onClick={newChat}
                                            disabled={loading}
                                            className="w-full py-2.5 px-3 rounded-xl text-white font-semibold transition-all duration-200 mb-4"
                                            style={{
                                                fontSize: 12,
                                                background: loading
                                                    ? 'rgba(99,102,241,0.40)'
                                                    : 'linear-gradient(135deg, #6366f1 0%, #818cf8 100%)',
                                                boxShadow: loading ? 'none' : '0 2px 10px rgba(99,102,241,0.30)',
                                            }}
                                            onMouseEnter={e => {
                                                if (loading) return
                                                const el = e.currentTarget as HTMLButtonElement
                                                el.style.boxShadow = '0 4px 16px rgba(99,102,241,0.40)'
                                                el.style.transform = 'translateY(-1px)'
                                            }}
                                            onMouseLeave={e => {
                                                const el = e.currentTarget as HTMLButtonElement
                                                el.style.boxShadow = loading ? 'none' : '0 2px 10px rgba(99,102,241,0.30)'
                                                el.style.transform = 'translateY(0)'
                                            }}
                                        >
                                            + Новый диалог
                                        </button>
                                        <div className="flex items-center gap-2 px-1 mb-2.5">
                                            <History size={10} style={{color: '#94a3b8'}}/>
                                            <span className="uppercase tracking-[0.12em] font-bold"
                                                  style={{color: '#94a3b8', fontSize: 9}}>История</span>
                                        </div>
                                        <div className="flex-1 overflow-y-auto scrollbar-thin flex flex-col gap-1">
                                            {threads.length === 0 ? (
                                                <p className="italic text-center py-8 rounded-xl border border-dashed"
                                                   style={{
                                                       color: '#94a3b8',
                                                       borderColor: 'rgba(0,0,0,0.06)',
                                                       fontSize: 11,
                                                   }}>
                                                    История пуста
                                                </p>
                                            ) : threads.map(t => (
                                                <div
                                                    key={t.id}
                                                    className="group relative flex items-stretch rounded-xl transition-all duration-150"
                                                    style={{
                                                        background: threadId === t.id ? 'rgba(255,255,255,0.90)' : 'transparent',
                                                        border: threadId === t.id ? '1px solid rgba(0,0,0,0.05)' : '1px solid transparent',
                                                        boxShadow: threadId === t.id ? 'var(--edms-shadow-xs)' : 'none',
                                                    }}
                                                >
                                                    <button type="button" onClick={() => loadHistory(t.id)}
                                                            className="flex-1 text-left p-2.5 min-w-0 transition-colors duration-150"
                                                            style={{borderRadius: threadId === t.id ? 12 : 0}}
                                                            onMouseEnter={e => {
                                                                if (threadId === t.id) return
                                                                e.currentTarget.style.background = 'rgba(255,255,255,0.60)'
                                                            }}
                                                            onMouseLeave={e => {
                                                                if (threadId === t.id) return
                                                                e.currentTarget.style.background = 'transparent'
                                                            }}
                                                    >
                                                        <p className="edms-thread-preview font-medium line-clamp-2 leading-relaxed pr-5"
                                                           style={{color: '#334155'}}>{t.preview}</p>
                                                        <span className="mt-1 block"
                                                              style={{color: '#94a3b8', fontSize: 9}}>{t.date}</span>
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={(e) => deleteThread(t.id, e)}
                                                        title="Удалить"
                                                        className="absolute top-1.5 right-1.5 p-1 rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-150"
                                                        style={{color: '#cbd5e1'}}
                                                        onMouseEnter={e => {
                                                            const el = e.currentTarget as HTMLElement;
                                                            el.style.color = '#f87171';
                                                            el.style.background = 'rgba(254,226,226,0.60)'
                                                        }}
                                                        onMouseLeave={e => {
                                                            const el = e.currentTarget as HTMLElement;
                                                            el.style.color = '#cbd5e1';
                                                            el.style.background = 'transparent'
                                                        }}
                                                    >
                                                        <X size={11}/>
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="mt-auto pt-3"
                                             style={{borderTop: '1px solid rgba(0,0,0,0.05)'}}>
                                            <button
                                                type="button"
                                                onClick={() => setIsSettingsOpen(true)}
                                                className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl font-medium transition-all duration-150 group"
                                                style={{color: '#64748b', fontSize: 11}}
                                                onMouseEnter={e => {
                                                    const el = e.currentTarget as HTMLButtonElement
                                                    el.style.background = 'rgba(255,255,255,0.70)'
                                                    el.style.color = '#334155'
                                                }}
                                                onMouseLeave={e => {
                                                    const el = e.currentTarget as HTMLButtonElement
                                                    el.style.background = 'transparent'
                                                    el.style.color = '#64748b'
                                                }}
                                            >
                                                <Settings size={13}
                                                          className="transition-colors duration-300"
                                                          style={{color: '#94a3b8'}}/>
                                                <span>Настройки</span>
                                            </button>
                                        </div>
                                    </>
                                )}
                            </div>
                        </aside>

                        {/* ── Main Chat Area ─────────────────────────────── */}
                        <main className="flex-1 flex flex-col min-w-0 overflow-hidden"
                              style={{background: `rgba(255,255,255,${glassAlpha * 0.6})`}}>
                            <div className="flex-1 p-4 overflow-y-auto scrollbar-thin flex flex-col gap-3">
                                {messages.length === 0 && !loading && (
                                    <div className="flex-1 flex flex-col items-center justify-center gap-4"
                                         style={{animation: 'edms-fade-in-up .5s ease-out'}}>
                                        <div style={{
                                            position: 'relative',
                                            width: 64, height: 64,
                                            borderRadius: 20,
                                            background: 'linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(129,140,248,0.04) 100%)',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                        }}>
                                            <MessageSquare size={28} strokeWidth={1.5}
                                                           style={{color: '#a5b4fc'}}/>
                                        </div>
                                        <div className="text-center">
                                            <p className="font-semibold"
                                               style={{color: '#334155', fontSize: 15, letterSpacing: '-0.01em'}}>
                                                Чем могу помочь?
                                            </p>
                                            <p className="mt-1"
                                               style={{color: '#94a3b8', fontSize: 12}}>
                                                Задайте вопрос о документе или загрузите файл
                                            </p>
                                        </div>
                                    </div>
                                )}
                                {messages.map(msg => {
                                    const displayContent = msg.action_type === 'requires_disambiguation'
                                        ? cleanDisambigMessage(msg.content)
                                        : msg.content
                                    return (
                                        <div key={msg.id} className="flex flex-col">
                                            <ChatMessage
                                                content={displayContent}
                                                role={msg.role}
                                                timestamp={msg.timestamp}
                                                isError={msg.isError}
                                                onAttachmentClick={(fileName) => {
                                                    setInput(prev => prev ? prev : `Проанализируй файл «${fileName}»`)
                                                    textareaRef.current?.focus()
                                                }}
                                                onDocumentClick={handleDocumentClick}
                                            />
                                            <ActionButtons
                                                msg={msg}
                                                loading={loading}
                                                onSend={sendWithLabel}
                                                defaultSummaryFormat={userPrefs.documents.defaultSummaryFormat}
                                            />
                                            <DisambiguationButtons msg={msg} loading={loading} onSend={sendWithLabel}/>
                                            <RefreshCacheButton
                                                msg={msg}
                                                userToken={getAuthToken() ?? ''}
                                                onRefreshStart={(id, prev) => handleRefreshStart(id, prev)}
                                                onRefreshDone={handleRefreshDone}
                                                onRefreshError={handleRefreshError}
                                            />
                                        </div>
                                    )
                                })}
                                {loading && <TypingDots/>}
                                <div ref={bottomRef}/>
                            </div>

                            {/* ── Input Area ─────────────────────────────── */}
                            <footer className="px-3 pb-3 pt-1 shrink-0">
                                {isListening && (
                                    <div className="flex items-center justify-between mb-1"
                                         style={{animation: 'edms-fade-in .2s ease-out'}}>
                                        <SoundWave/>
                                        {autoSendPending && (
                                            <span className="font-medium pr-1"
                                                  style={{
                                                      fontSize: 10,
                                                      color: '#6366f1',
                                                      animation: 'edms-pulse-soft 1.2s ease-in-out infinite',
                                                  }}>
                                                отправляю…
                                            </span>
                                        )}
                                    </div>
                                )}

                                {attachedFile && (
                                    <div
                                        className="flex items-center gap-2 mb-2 px-3 py-2 rounded-2xl w-fit"
                                        style={{
                                            background: 'rgba(241,245,249,0.90)',
                                            border: '1px solid rgba(0,0,0,0.05)',
                                            fontSize: 11,
                                            boxShadow: 'var(--edms-shadow-xs)',
                                            animation: 'edms-fade-in-up .2s ease-out',
                                        }}
                                    >
                                        <Paperclip size={12} style={{color: '#6366f1', flexShrink: 0}}/>
                                        <span className="font-medium truncate max-w-[180px]"
                                              style={{color: '#1e293b'}}>{attachedFile.name}</span>
                                        <button
                                            type="button"
                                            onClick={() => setAttachedFile(null)}
                                            className="ml-0.5 transition-colors duration-150"
                                            style={{color: '#cbd5e1'}}
                                            onMouseEnter={e => {
                                                (e.currentTarget as HTMLElement).style.color = '#f87171'
                                            }}
                                            onMouseLeave={e => {
                                                (e.currentTarget as HTMLElement).style.color = '#cbd5e1'
                                            }}
                                        >
                                            <X size={13}/>
                                        </button>
                                    </div>
                                )}

                                {/* ── Pill Input ─────────────────────────── */}
                                <div
                                    className={`edms-input-pill ${isFocused ? 'focused' : ''}`}
                                    style={{padding: '4px 4px 4px 2px'}}
                                >
                                    <div style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 0,
                                        width: '100%',
                                    }}>
                                        {/* Left: attachment + mic/send toggle */}
                                        <div style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: 0,
                                            flexShrink: 0,
                                        }}>
                                            <button
                                                type="button"
                                                onClick={() => fileRef.current?.click()}
                                                className="edms-icon-btn"
                                                title="Прикрепить файл"
                                            >
                                                <Paperclip size={18}/>
                                            </button>

                                            {isSpeechSupported && (
                                                <>
                                                    {/* Микрофон ↔ Отправить: при вводе текста микрофон превращается в отправку */}
                                                    <button
                                                        type="button"
                                                        onClick={canSend ? (e) => send(e) : toggleMic}
                                                        title={canSend ? 'Отправить' : isListening ? 'Остановить' : 'Голосовой ввод'}
                                                        className="edms-icon-btn"
                                                        style={canSend ? {
                                                            background: 'rgba(99,102,241,0.10)',
                                                            color: '#6366f1',
                                                        } : isListening ? {
                                                            background: 'rgba(239,68,68,0.08)',
                                                            color: '#ef4444',
                                                            animation: 'edms-pulse-soft 1.5s ease-in-out infinite',
                                                        } : {}}
                                                    >
                                                        {canSend
                                                            ? <Send size={17} style={{marginLeft: 1}}/>
                                                            : isListening
                                                                ? <StopCircle size={17}/>
                                                                : <Mic size={18}/>
                                                        }
                                                    </button>

                                                    {/* Hands-Free toggle */}
                                                    {(isListening || handsFree) && !canSend && (
                                                        <button
                                                            type="button"
                                                            onClick={() => setHandsFree(v => !v)}
                                                            className="edms-icon-btn"
                                                            title={handsFree ? 'Выключить автоотправку' : 'Включить автоотправку'}
                                                            style={handsFree ? {
                                                                background: 'rgba(99,102,241,0.10)',
                                                                color: '#6366f1',
                                                            } : {width: 28, height: 28}}
                                                        >
                                                            <svg width={handsFree ? 14 : 11}
                                                                 height={handsFree ? 14 : 11}
                                                                 viewBox="0 0 24 24" fill="none"
                                                                 stroke="currentColor" strokeWidth={2.5}
                                                                 strokeLinecap="round" strokeLinejoin="round">
                                                                <polygon
                                                                    points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                                                            </svg>
                                                        </button>
                                                    )}
                                                </>
                                            )}
                                        </div>

                                        <div className="flex-1 relative" style={{padding: '0 2px'}}>
                                            <textarea
                                                ref={textareaRef}
                                                rows={1}
                                                value={input}
                                                onFocus={() => setIsFocused(true)}
                                                onBlur={() => setIsFocused(false)}
                                                onChange={e => setInput(e.target.value)}
                                                onKeyDown={e => {
                                                    if (e.key === 'Enter' && !e.shiftKey) {
                                                        e.preventDefault()
                                                        send(e)
                                                    }
                                                }}
                                                placeholder={
                                                    isSidebarOpen
                                                        ? ''
                                                        : autoSendPending
                                                            ? 'Отправляю…'
                                                            : isListening
                                                                ? 'Слушаю…'
                                                                : attachedFile
                                                                    ? 'Добавьте комментарий…'
                                                                    : 'Спросите AI...'
                                                }
                                                className="edms-textarea w-full bg-transparent border-none outline-none resize-none scrollbar-thin"
                                                style={{
                                                    color: '#0f172a',
                                                    caretColor: '#6366f1',
                                                    padding: '9px 6px',
                                                    maxHeight: 150,
                                                    minWidth: 0,
                                                }}
                                            />
                                            {interimTranscript && (
                                                <span
                                                    aria-hidden="true"
                                                    className="absolute left-1 pointer-events-none"
                                                    style={{
                                                        top: '10px',
                                                        paddingLeft: input ? '0.5ch' : 0,
                                                        whiteSpace: 'pre-wrap',
                                                        wordBreak: 'break-word',
                                                        color: '#cbd5e1',
                                                        fontSize: 14,
                                                    }}
                                                >
                                                    {input ? `${input} ` : ''}{interimTranscript}
                                                </span>
                                            )}
                                        </div>

                                        {/* Right: только стоп при загрузке */}
                                        <div style={{flexShrink: 0}}>
                                            {loading && (
                                                <button
                                                    type="button"
                                                    onClick={abort}
                                                    className="edms-stop-btn"
                                                    title="Остановить"
                                                >
                                                    <StopCircle size={17}/>
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <p className="text-center mt-2 px-4 leading-tight"
                                   style={{color: '#cbd5e1', fontSize: 9.5}}>
                                    EDMS Assistant — ИИ. Может ошибаться и давать неверную информацию.
                                </p>
                            </footer>
                        </main>
                    </div>
                </div>
            )}

            <input
                type="file"
                ref={fileRef}
                className="hidden"
                onChange={e => {
                    const f = e.target.files?.[0]
                    if (!f) return
                    const r = new FileReader()
                    r.onload = () => setAttachedFile({path: r.result as string, name: f.name})
                    r.readAsDataURL(f)
                }}
            />
        </div>
    )
}