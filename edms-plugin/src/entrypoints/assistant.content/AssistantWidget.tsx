import {useState, useRef, useEffect, useCallback, memo, type FormEvent} from 'react'
import {
    Paperclip, X, Mic, Send, MessageSquare,
    Square, StopCircle, FileText, Search, List, History,
} from 'lucide-react'
import dayjs from 'dayjs'
import 'dayjs/locale/ru'

import {ChatMessage} from '../../shared/ui/ChatMessage'
import {LiquidGlassFilter} from '../../shared/ui/LiquidGlassFilter'
import {getAuthToken} from '../../shared/lib/auth'
import {extractDocIdFromUrl} from '../../shared/lib/url'
import {sendMsg} from '../../shared/lib/messaging'
import {toast} from '../../shared/lib/toast'

dayjs.locale('ru')

interface Message {
    role: 'user' | 'assistant'
    content: string
    action_type?: string
    isError?: boolean
    id: string
}

interface Thread {
    id: string
    preview: string
    date: string
}

const ANIM_STYLES = `
  @keyframes edms-ripple {
    0%   { transform: scale(0.8); opacity: 0.6; }
    100% { transform: scale(2.4); opacity: 0; }
  }
  @keyframes edms-fade-in {
    from { opacity: 0; transform: scale(0.93) translateY(8px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
  }
  @keyframes edms-wave {
    0%, 60%, 100% { transform: translateY(0);    opacity: 0.4; }
    30%            { transform: translateY(-6px); opacity: 1;   }
  }
  @keyframes edms-soundbar {
    0%, 100% { transform: scaleY(0.4); }
    50%      { transform: scaleY(1.2); }
  }
`

declare global {
    interface Window {
        webkitSpeechRecognition: any;
        SpeechRecognition: any;
    }
}

function SoundWave() {
    return (
        <>
            <style>{ANIM_STYLES}</style>
            <div style={{
                display: 'flex',
                alignItems: 'flex-end',
                justifyContent: 'center',
                gap: 3,
                height: 12,
                marginBottom: 8
            }}>
                {[0, 1, 2, 3, 4].map(i => (
                    <div key={i} style={{
                        width: 3, height: 10, borderRadius: 2,
                        background: 'rgba(99,102,241,0.7)',
                        transformOrigin: 'bottom',
                        animation: `edms-soundbar 0.6s ease-in-out ${i * 80}ms infinite`,
                    }}/>
                ))}
            </div>
        </>
    )
}

function TypingDots() {
    return (
        <>
            <style>{ANIM_STYLES}</style>
            <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '10px 14px',
                background: 'rgba(255,255,255,0.35)',
                borderRadius: 18,
                border: '1px solid rgba(255,255,255,0.4)',
                width: 'fit-content',
            }}>
                {[0, 160, 320].map(d => (
                    <div key={d} style={{
                        width: 7, height: 7, borderRadius: '50%',
                        background: '#818cf8',
                        animation: `edms-wave 1.4s ease-in-out ${d}ms infinite`,
                    }}/>
                ))}
            </div>
        </>
    )
}

function makeErrorMessage(err: unknown): string {
    if (err instanceof Error) return `__error__:${err.message}`
    // sendMsg бросает объект { success: false, error: "..." } из background.js
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
        rawMsg = (typeof e.error === 'string' ? e.error
            : typeof e.message === 'string' ? e.message
                : typeof e.detail === 'string' ? e.detail
                    : JSON.stringify(e))
    } else {
        rawMsg = String(err)
    }
    const raw = rawMsg.toLowerCase()
    if (raw.includes('failed to fetch') || raw.includes('networkerror'))
        return 'Нет соединения с сервером'
    if (raw.includes('timeout'))
        return 'Сервер не ответил вовремя'
    if (raw.includes('401') || raw.includes('unauthorized'))
        return 'Ошибка авторизации — обновите страницу'
    if (raw.includes('403'))
        return 'Доступ запрещён'
    return err instanceof Error ? err.message : String(err)
}

const CHOICE_LABELS: Record<string, string> = {
    abstractive: 'Пересказ',
    extractive: 'Факты',
    thesis: 'Тезисы',
}

let _msgIdCounter = 0
const newMsgId = () => `msg_${Date.now()}_${++_msgIdCounter}`

const MAX_THREADS = 20
const THREADS_STORAGE_KEY = 'edmsWidgetThreads'

/** Сохраняет список диалогов в chrome.storage.local. */
function persistThreads(threads: Thread[]): void {
    chrome.storage.local.set({[THREADS_STORAGE_KEY]: threads})
}

export function AssistantWidget() {
    const [isEnabled, setIsEnabled] = useState(true)
    const [isOpen, setIsOpen] = useState(false)
    const [isSidebarOpen, setIsSidebarOpen] = useState(false)
    const [messages, setMessages] = useState<Message[]>([])
    const [threads, setThreads] = useState<Thread[]>([])
    const [threadId, setThreadId] = useState<string | null>(null)
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [attachedFile, setAttachedFile] = useState<{ path: string; name: string } | null>(null)
    const [isListening, setIsListening] = useState(false)
    const [isFocused, setIsFocused] = useState(false)
    const [recognition, setRecognition] = useState<any>(null)
    const [userContext, setUserContext] = useState<Record<string, string>>({})

    const bottomRef = useRef<HTMLDivElement>(null)
    const fileRef = useRef<HTMLInputElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const requestIdRef = useRef<string | null>(null)
    const serverPathRef = useRef<string | null>(null)
    const messagesRef = useRef<Message[]>(messages)

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
            if (Array.isArray(saved) && saved.length > 0) {
                setThreads(saved)
            }
        })

        chrome.storage.local.get(['userContext'], r => {
            if (r.userContext && typeof r.userContext === 'object') {
                setUserContext(r.userContext)
            }
        })

        const onStorage = (
            changes: Record<string, chrome.storage.StorageChange>,
            area: string,
        ) => {
            if (area === 'local' && 'assistantEnabled' in changes) {
                const val = changes.assistantEnabled.newValue as boolean
                setIsEnabled(val)
                if (!val) setIsOpen(false)
            }
        }
        chrome.storage.onChanged.addListener(onStorage)

        const onWindowMsg = (e: MessageEvent) => {
            if (e.data?.type !== 'REFRESH_CHAT_HISTORY') return
            const {messages: raw, thread_id} = e.data as {
                messages: { type: string; content: string }[]
                thread_id?: string
            }
            const mapped = raw.map(m => ({
                role: (m.type === 'human' ? 'user' : 'assistant') as Message['role'],
                content: m.content,
                id: newMsgId(),
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

        const SR = window.SpeechRecognition || window.webkitSpeechRecognition
        if (SR) {
            const inst = new SR()
            inst.lang = 'ru-RU'
            inst.continuous = false
            inst.interimResults = false
            inst.onresult = (ev: any) => {
                const transcript = ev.results[0][0].transcript
                if (transcript) setInput(prev => (prev ? prev + ' ' : '') + transcript)
                setIsListening(false)
            }
            inst.onend = () => setIsListening(false)
            inst.onerror = () => setIsListening(false)
            setRecognition(inst)
        }

        return () => {
            chrome.storage.onChanged.removeListener(onStorage)
            window.removeEventListener('message', onWindowMsg)
        }
    }, [])

    useEffect(() => {
        bottomRef.current?.scrollIntoView({behavior: 'smooth'})
    }, [messages, loading])

    const newChat = async () => {
        if (messages.length && threadId) {
            const preview = (messages.find(m => m.role === 'user')?.content ?? 'Диалог').slice(0, 40)
            setThreads(prev => {
                if (prev.find(t => t.id === threadId)) return prev
                const updated = [
                    {id: threadId, preview, date: dayjs().format('HH:mm')},
                    ...prev,
                ].slice(0, MAX_THREADS)
                persistThreads(updated)
                return updated
            })
        }
        setLoading(true)
        try {
            const res = await sendMsg<{ thread_id: string }>('createNewChat', {
                user_token: getAuthToken() ?? 'no_token',
            })
            setThreadId(res.thread_id)
            setMessages([])
            setIsSidebarOpen(false)
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
            const res = await sendMsg<{ messages: { type: string; content: string }[] }>(
                'getChatHistory', {thread_id: id},
            )
            setMessages(res.messages.map(m => ({
                role: (m.type === 'human' ? 'user' : 'assistant') as Message['role'],
                content: m.content,
                id: newMsgId(),
            })))
            setThreadId(id)
        } catch (err) {
            toast.error(getToastErrorText(err), 'Не удалось загрузить историю')
        } finally {
            setLoading(false)
        }
    }

    const deleteThread = (id: string, e: React.MouseEvent) => {
        e.stopPropagation()
        setThreads(prev => {
            const updated = prev.filter(t => t.id !== id)
            persistThreads(updated)
            return updated
        })
        if (threadId === id) {
            setMessages([])
            setThreadId(null)
        }
    }

    const send = async (e?: FormEvent | React.KeyboardEvent, humanChoice?: string, humanChoiceLabel?: string) => {
        if (e) e.preventDefault()

        const isChoiceFlow = Boolean(humanChoice)
        const hasTextInput = input.trim().length > 0
        const hasFile = Boolean(attachedFile)

        if (!isChoiceFlow && !hasTextInput && !hasFile) return
        if (loading) return

        if (isListening) {
            recognition?.stop()
            setIsListening(false)
        }

        const token = getAuthToken() ?? 'no_token'
        const docId = extractDocIdFromUrl()
        const reqId = Math.random().toString(36).slice(7)
        const tid = threadId ?? `${token.slice(0, 8)}_${docId}`
        requestIdRef.current = reqId
        if (!threadId) setThreadId(tid)

        const userLabel = isChoiceFlow
            ? (humanChoiceLabel ?? CHOICE_LABELS[humanChoice!] ?? humanChoice!)
            : hasFile
                ? `${input} (Файл: ${attachedFile!.name})`.trim()
                : input

        setMessages(prev => [...prev, {role: 'user', content: userLabel, id: newMsgId()}])
        setLoading(true)

        const textToSend = isChoiceFlow
            ? (humanChoice ?? '')
            : input

        setInput('')

        try {
            if (hasFile && !isChoiceFlow) {
                const up = await sendMsg<{ file_path: string }>('uploadFile', {
                    fileData: attachedFile!.path,
                    fileName: attachedFile!.name,
                    user_token: token,
                })
                serverPathRef.current = up.file_path
            }

            const res = await sendMsg<any>('sendChatMessage', {
                message: isChoiceFlow ? humanChoice! : textToSend,
                user_token: token,
                requestId: reqId,
                thread_id: tid,
                context_ui_id: docId,
                file_path: serverPathRef.current,
                human_choice: isChoiceFlow ? humanChoice! : undefined,
                user_context: Object.keys(userContext).length > 0 ? userContext : undefined,
            })

            if (!isChoiceFlow) serverPathRef.current = null

            const payload = (res && typeof res === 'object' && 'data' in res && res.success)
                ? (res as any).data
                : res
            const content =
                payload?.response
                ?? payload?.content
                ?? payload?.message
                ?? (Array.isArray(payload?.messages) ? payload.messages.at(-1)?.content : undefined)
                ?? 'Анализ завершён.'

            if (payload?.action_type === 'requires_disambiguation') {
                console.log('[EDMS DEBUG] disambiguation payload:', JSON.stringify(payload, null, 2))
                console.log('[EDMS DEBUG] message field:', payload?.message?.substring(0, 500))
            }

            const newAssistantMsg: Message = {
                role: 'assistant',
                content,
                action_type: payload?.action_type,
                id: newMsgId(),
            }
            setMessages(prev => [...prev, newAssistantMsg])

            // ── Обновление данных EDMS после мутации (без перезагрузки страницы) ──
            if (payload?.requires_reload) {
                refreshDocumentPage(docId)
            }
        } catch (err: unknown) {
            if (!String(err).includes('aborted')) {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: makeErrorMessage(err),
                    isError: true,
                    id: newMsgId(),
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

    const sendWithLabel = (humanChoice: string, label: string) =>
        send(undefined, humanChoice, label)

    // ── Обновление данных EDMS после мутации ─────────────────────────────────
    const refreshDocumentPage = async (_documentId: string | null) => {
        try {
            const [tab] = await chrome.tabs.query({active: true, currentWindow: true})
            if (tab?.id) {
                const results = await chrome.scripting.executeScript({
                    target: {tabId: tab.id},
                    func: () => {
                        const refresh = (window as any).__edms_refresh__
                        if (typeof refresh === 'function') {
                            refresh()
                            return true
                        }
                        return false
                    },
                })
                if (results?.[0]?.result === true) {
                    console.debug('[EDMS] router.refresh() via __edms_refresh__ OK')
                    return
                }
            }
        } catch {
        }

        const snapshot = {
            messages: messagesRef.current,
            threadId,
            isOpen: true,
            savedAt: Date.now(),
        }
        try {
            chrome.storage.local.set({edmsWidgetSnapshot: snapshot}, () => {
                window.location.reload()
            })
        } catch {
            window.location.reload()
        }
    }

    // ── Восстановление чата из snapshot после location.reload() ─────────────
    useEffect(() => {
        chrome.storage.local.get(['edmsWidgetSnapshot'], (r) => {
            const snap = r?.edmsWidgetSnapshot
            if (snap) chrome.storage.local.remove('edmsWidgetSnapshot')
            if (!snap) return
            const age = Date.now() - (snap.savedAt ?? 0)
            if (age > 30_000) return
            if (Array.isArray(snap.messages) && snap.messages.length > 0) {
                setMessages(snap.messages)
            }
            if (snap.threadId) setThreadId(snap.threadId)
            setIsOpen(true)
            setTimeout(() => bottomRef.current?.scrollIntoView({behavior: 'smooth'}), 200)
        })
    }, [])

    const abort = () => {
        if (!requestIdRef.current) return
        chrome.runtime.sendMessage({type: 'abortRequest', payload: {requestId: requestIdRef.current}})
        setLoading(false)
        requestIdRef.current = null
        setMessages(prev => [...prev, {role: 'assistant', content: '_Запрос отменён._', id: newMsgId()}])
    }

    const toggleMic = () => {
        if (!recognition) return
        if (isListening) recognition.stop()
        else {
            try {
                recognition.start()
                setIsListening(true)
            } catch (e) {
                console.error(e)
            }
        }
    }

    const ActionButtons = memo(({msg}: { msg: Message }) => {
        if (msg.action_type !== 'summarize_selection') return null
        return (
            <div className="mt-2 flex flex-wrap gap-2">
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
                            e.stopPropagation()
                            sendWithLabel(btn.id, btn.label)
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl border border-indigo-200 bg-white/60 text-indigo-700 hover:bg-indigo-600 hover:text-white hover:border-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all backdrop-blur-sm"
                    >
                        {btn.icon}{btn.label}
                    </button>
                ))}
            </div>
        )
    })

    const parseDisambigCandidates = (content: string): Array<{ id: string; name: string; dept: string }> => {
        const m = content.match(/<!--CANDIDATES:(.+?)-->/)
        if (!m) return []
        try {
            return JSON.parse(m[1])
        } catch {
            return []
        }
    }

    const cleanDisambigMessage = (content: string): string =>
        content.replace(/\n\n<!--CANDIDATES:.+?-->/, '').trimEnd()

    // Определяем тип кандидата по имени — для иконки и стиля карточки
    const getCandidateType = (name: string): 'employee' | 'docx' | 'xlsx' | 'pdf' | 'file' => {
        const lower = name.toLowerCase()
        if (lower.endsWith('.docx') || lower.endsWith('.doc')) return 'docx'
        if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) return 'xlsx'
        if (lower.endsWith('.pdf')) return 'pdf'
        if (!lower.includes('.')) return 'employee'
        return 'file'
    }

    const CandidateIcon = ({type, className}: { type: string; className?: string }) => {
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
    }

    // Цветовые акценты по типу (иконка + hover)
    const typeAccent: Record<string, string> = {
        employee: 'text-violet-500 group-hover:text-white',
        docx: 'text-blue-500 group-hover:text-white',
        xlsx: 'text-emerald-500 group-hover:text-white',
        pdf: 'text-rose-500 group-hover:text-white',
        file: 'text-slate-400 group-hover:text-white',
    }

    const DisambiguationButtons = memo(({msg}: { msg: Message }) => {
        if (msg.action_type !== 'requires_disambiguation') return null
        const candidates = parseDisambigCandidates(msg.content)
        if (candidates.length === 0) return null

        const isEmployeeList = candidates.every(c => getCandidateType(c.name) === 'employee')
        const isFileList = candidates.every(c => getCandidateType(c.name) !== 'employee')

        return (
            <div className="mt-3">
                {/* Заголовок группы */}
                <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400 mb-2 px-0.5">
                    {isEmployeeList ? '👤 Выберите сотрудника' : isFileList ? '📎 Выберите вложение' : '✦ Выберите вариант'}
                </p>

                {/* Сетка карточек: для сотрудников */}
                <div className={`flex flex-col gap-1.5`}>
                    {candidates.map((c, idx) => {
                        const ctype = getCandidateType(c.name)
                        const accent = typeAccent[ctype] ?? typeAccent.file
                        return (
                            <button
                                key={c.id}
                                type="button"
                                disabled={loading}
                                onClick={e => {
                                    e.stopPropagation()
                                    sendWithLabel(c.id, c.name)
                                }}
                                className="group flex items-center gap-2.5 w-full px-3 py-2.5 text-xs rounded-xl border border-slate-200/80 bg-white/70 text-left text-slate-700 hover:bg-indigo-600 hover:border-indigo-600 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150 backdrop-blur-sm shadow-sm hover:shadow-md"
                            >
                                {/* Номер */}
                                <span
                                    className="flex-shrink-0 w-5 h-5 rounded-full bg-slate-100 group-hover:bg-indigo-500 text-[10px] font-bold text-slate-500 group-hover:text-white flex items-center justify-center transition-colors">
                                    {idx + 1}
                                </span>

                                {/* Иконка типа */}
                                <CandidateIcon type={ctype} className={accent}/>

                                {/* Имя + должность/отдел */}
                                <div className="flex-1 min-w-0">
                                    <p className="font-semibold truncate leading-tight">{c.name}</p>
                                    {c.dept && (
                                        <p className="text-[10px] opacity-60 group-hover:opacity-80 truncate mt-0.5 leading-tight">
                                            {c.dept}
                                        </p>
                                    )}
                                </div>

                                {/* Стрелка */}
                                <svg
                                    className="w-3.5 h-3.5 flex-shrink-0 opacity-30 group-hover:opacity-80 transition-opacity"
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

    if (!isEnabled) return null

    return (
        <div
            className="fixed bottom-5 right-5 z-[2147483647] flex flex-col items-end pointer-events-none select-none font-sans">
            <LiquidGlassFilter/>

            {!isOpen && (
                <>
                    <style>{ANIM_STYLES}</style>
                    <button
                        type="button"
                        onClick={() => setIsOpen(true)}
                        className="pointer-events-auto relative w-16 h-16 rounded-full flex items-center justify-center glass hover:scale-110 active:scale-95 transition-transform group"
                    >
                        <span style={{
                            position: 'absolute', inset: 0,
                            borderRadius: '50%',
                            background: 'rgba(99,102,241,0.18)',
                            animation: 'edms-ripple 3s cubic-bezier(0.4,0,0.2,1) infinite',
                            pointerEvents: 'none',
                        }}/>
                        <MessageSquare size={28}
                                       className="text-indigo-600/90 group-hover:rotate-12 transition-transform z-10"/>
                    </button>
                </>
            )}

            {isOpen && (
                <>
                    <style>{ANIM_STYLES}</style>
                    <div
                        className="pointer-events-auto flex flex-col w-[480px] h-[720px] rounded-[28px] overflow-hidden glass"
                        style={{animation: 'edms-fade-in .3s cubic-bezier(.22,1,.36,1) forwards'}}
                    >
                        <header
                            className="flex items-center justify-between px-4 py-3 border-b border-white/20 shrink-0 bg-white/10">
                            <div className="flex items-center gap-2">
                                <button
                                    type="button"
                                    onClick={() => setIsSidebarOpen(v => !v)}
                                    className={`p-2 rounded-xl transition-all flex flex-col justify-center items-center gap-[5px] w-9 h-9 ${isSidebarOpen ? 'bg-white/30 text-indigo-700' : 'text-slate-500 hover:bg-white/20'}`}
                                >
                                    <span
                                        className={`h-px w-4 bg-current rounded transition-all duration-300 origin-center ${isSidebarOpen ? 'rotate-45 translate-y-[6px]' : ''}`}/>
                                    <span
                                        className={`h-px w-4 bg-current rounded transition-all duration-200 ${isSidebarOpen ? 'opacity-0 scale-x-0' : ''}`}/>
                                    <span
                                        className={`h-px w-4 bg-current rounded transition-all duration-300 origin-center ${isSidebarOpen ? '-rotate-45 -translate-y-[6px]' : ''}`}/>
                                </button>
                                <h3 className="font-bold text-slate-800/80 text-sm tracking-tight">EDMS Assistant</h3>
                            </div>
                            <button
                                type="button"
                                onClick={() => setIsOpen(false)}
                                className="p-2 rounded-xl text-slate-400 hover:text-red-500 hover:bg-red-50/30 transition-colors"
                            >
                                <X size={18}/>
                            </button>
                        </header>

                        <div className="flex-1 flex overflow-hidden">
                            <aside
                                className={`shrink-0 flex flex-col bg-white/10 backdrop-blur-md border-r border-white/20 transition-all duration-300 overflow-hidden ${isSidebarOpen ? 'w-60' : 'w-0'}`}>
                                <div className="p-3 w-60 flex flex-col h-full">
                                    <button
                                        type="button"
                                        onClick={newChat}
                                        className="w-full py-2 px-3 rounded-xl bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-700 active:scale-95 transition-all mb-4"
                                    >
                                        + Новый диалог
                                    </button>
                                    <div className="flex items-center gap-2 px-1 mb-2">
                                        <History size={11} className="text-slate-400"/>
                                        <span
                                            className="text-[9px] uppercase tracking-widest text-slate-400 font-bold">История</span>
                                    </div>
                                    <div className="flex-1 overflow-y-auto scrollbar-thin flex flex-col gap-1.5">
                                        {threads.length === 0 ? (
                                            <p className="text-[11px] text-slate-400 italic text-center py-6 bg-white/10 rounded-xl border border-dashed border-white/30">История
                                                пуста</p>
                                        ) : threads.map(t => (
                                            <div
                                                key={t.id}
                                                className={`group relative flex items-stretch rounded-xl border transition-all ${threadId === t.id ? 'bg-white/40 border-white/50' : 'hover:bg-white/20 border-transparent'}`}
                                            >
                                                <button
                                                    type="button"
                                                    onClick={() => loadHistory(t.id)}
                                                    className="flex-1 text-left p-2.5 text-[11px] min-w-0"
                                                >
                                                    <p className="text-slate-700 font-medium line-clamp-2 leading-relaxed pr-5">{t.preview}</p>
                                                    <span
                                                        className="text-[9px] text-slate-400 mt-0.5 block">{t.date}</span>
                                                </button>
                                                {/* Кнопка удаления — видна только при hover */}
                                                <button
                                                    type="button"
                                                    onClick={(e) => deleteThread(t.id, e)}
                                                    title="Удалить диалог"
                                                    className="absolute top-1.5 right-1.5 p-1 rounded-lg opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 hover:bg-red-50/40 transition-all"
                                                >
                                                    <X size={11}/>
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </aside>

                            <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
                                <div className="flex-1 p-4 overflow-y-auto scrollbar-thin flex flex-col gap-3">
                                    {messages.length === 0 && !loading && (
                                        <div
                                            className="flex-1 flex flex-col items-center justify-center gap-3 opacity-40 select-none">
                                            <MessageSquare size={44} strokeWidth={1} className="text-slate-500"/>
                                            <p className="text-sm text-slate-500 font-medium">Чем я могу помочь?</p>
                                        </div>
                                    )}
                                    {messages.map((msg) => {
                                        const displayContent = msg.action_type === 'requires_disambiguation'
                                            ? cleanDisambigMessage(msg.content)
                                            : msg.content
                                        return (
                                            <div key={msg.id} className="flex flex-col">
                                                <ChatMessage content={displayContent} role={msg.role}
                                                             isError={msg.isError}/>
                                                <ActionButtons msg={msg}/>
                                                <DisambiguationButtons msg={msg}/>
                                            </div>
                                        )
                                    })}
                                    {loading && <TypingDots/>}
                                    <div ref={bottomRef}/>
                                </div>

                                <footer className="px-3 pb-3 pt-2 shrink-0 border-t border-white/20 bg-white/5">
                                    {isListening && <SoundWave/>}
                                    {attachedFile && (
                                        <div
                                            className="flex items-center gap-2 mb-2 px-3 py-1.5 bg-white/50 border border-white/50 rounded-xl w-fit text-[11px]">
                                            <Paperclip size={13} className="text-indigo-500"/>
                                            <span
                                                className="text-slate-700 font-medium truncate max-w-[180px]">{attachedFile.name}</span>
                                            <button type="button" onClick={() => setAttachedFile(null)}
                                                    className="ml-1 text-slate-400 hover:text-red-500 transition-colors">
                                                <X size={13}/>
                                            </button>
                                        </div>
                                    )}

                                    <form
                                        onSubmit={send}
                                        style={{
                                            boxShadow: isFocused
                                                ? '0 0 15px rgba(99, 102, 241, 0.4), 0 0 30px rgba(168, 85, 247, 0.2)'
                                                : 'none'
                                        }}
                                        className={`flex items-end gap-1 rounded-2xl p-1 bg-white/20 border transition-all duration-300 shadow-sm backdrop-blur-md ${
                                            isFocused ? 'border-indigo-400/60 bg-white/40' : 'border-white/30'
                                        }`}
                                    >
                                        <button
                                            type="button"
                                            onClick={() => fileRef.current?.click()}
                                            className="p-2 mb-0.5 rounded-xl text-slate-400 hover:text-indigo-600 hover:bg-white/30 transition-colors"
                                        >
                                            <Paperclip size={18}/>
                                        </button>

                                        <button
                                            type="button"
                                            onClick={toggleMic}
                                            className={`p-2 mb-0.5 rounded-xl transition-all ${isListening ? 'text-red-500 bg-red-50/40 animate-pulse' : 'text-slate-400 hover:text-indigo-600 hover:bg-white/30'}`}
                                        >
                                            {isListening ? <Square size={16} fill="currentColor"/> : <Mic size={18}/>}
                                        </button>

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
                                            placeholder={isListening ? 'Слушаю...' : 'Спросите AI...'}
                                            className="flex-1 bg-transparent border-none outline-none text-sm text-slate-800 placeholder-slate-400 px-1 py-2.5 resize-none max-h-[150px] scrollbar-thin"
                                        />

                                        {loading ? (
                                            <button
                                                type="button"
                                                onClick={abort}
                                                className="p-2 mb-0.5 rounded-xl bg-red-500/80 text-white hover:bg-red-600 active:scale-95 transition-all"
                                            >
                                                <StopCircle size={17}/>
                                            </button>
                                        ) : (
                                            <button
                                                type="submit"
                                                disabled={!input.trim() && !attachedFile}
                                                className="p-2 mb-0.5 rounded-xl bg-indigo-600/80 text-white hover:bg-indigo-700 disabled:opacity-30 active:scale-95 transition-all"
                                            >
                                                <Send size={17}/>
                                            </button>
                                        )}
                                    </form>
                                    <div style={{
                                        fontSize: '10px',
                                        color: 'rgba(30, 41, 59, 0.45)',
                                        textAlign: 'center',
                                        marginTop: '6px',
                                        padding: '0 10px',
                                        lineHeight: '1.2'
                                    }}>
                                        EDMS Assistant - это ИИ. Он может ошибаться, в том числе давать неверную
                                        информацию.
                                    </div>
                                </footer>
                            </main>
                        </div>
                    </div>
                </>
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