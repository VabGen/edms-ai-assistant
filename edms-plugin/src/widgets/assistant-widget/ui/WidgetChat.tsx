import {
  useRef,
  useState,
  useEffect,
  useCallback,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import { Paperclip, X, Mic, Send, Square, MessageSquare } from 'lucide-react'
import { useChatStore } from '@features/chat/model/useChatStore'
import { useStreamChat } from '@features/chat/model/useStreamChat'
import { useSpeechRecognition } from '@features/voice/model/useSpeechRecognition'
import { useApplyPreferences } from '@features/settings/model/useApplyPreferences'
import { sendMessage } from '@shared/api/messaging'
import { getAuthToken } from '@/shared/lib/auth'
import { extractDocIdFromUrl } from '@/shared/lib/url'
import { toast } from '@/shared/lib/toast'
import { ChatMessage } from '@/shared/ui/ChatMessage'
import { cn } from '@shared/lib/cn'
import type { ChatMessage as ChatMessageType } from '@entities/message/model/types'
import type { ResumeValue } from '@entities/interrupt/model/types'

function SoundWave({ active }: { active: boolean }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 2, height: 14 }}>
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
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '6px 4px' }}>
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
  const { messages, loading, threadId, setThreadId, setMessages, saveSnapshot } = useChatStore()
  const { startStream, abort } = useStreamChat()
  const { prefs } = useApplyPreferences()

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

  const { isListening, interimTranscript, isSupported: isSpeechSupported, toggle: toggleMic, stop: stopMic } =
    useSpeechRecognition({
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
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const getOrCreateThreadId = useCallback(async (): Promise<string> => {
    if (threadId) return threadId
    const token = getAuthToken()
    if (!token) throw new Error('Войдите в систему')
    const res = await sendMessage('createNewChat', { user_token: token })
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

    const handle = startStream({
      message: text,
      userToken: token,
      threadId: tid,
      contextUiId: extractDocIdFromUrl(),
      filePath,
      resumeValue: pendingResume,
    })
    streamHandleRef.current = handle
    setPendingResume(null)
  }, [input, attachedFile, loading, pendingResume, stopMic, getOrCreateThreadId, startStream])

  sendRef.current = () => { void send() }
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
    (resume: ResumeValue) => {
      setPendingResume(resume)
      setInput('')
      void send()
    },
    [send],
  )

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file) return
      const dataUrl = await new Promise<string>((resolve) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.readAsDataURL(file)
      })
      setAttachedFile({ path: dataUrl, name: file.name })
      if (fileInputRef.current) fileInputRef.current.value = ''
    },
    [],
  )

  const handleSubmit = useCallback(
    (e: FormEvent) => {
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

  const lastMsg = messages[messages.length - 1]
  const interruptPayload =
    lastMsg?.role === 'assistant' && !loading ? (lastMsg.interrupt ?? null) : null

  return (
    <div className="flex flex-col flex-1 min-w-0 min-h-0">
      <div className={cn('flex-1 overflow-y-auto overflow-x-hidden px-3 py-3 min-h-0 scrollbar-none', messages.length === 0 ? 'flex flex-col' : 'space-y-3')}>
        {messages.length === 0 && (
          <EmptyState
            onAction={(text) => setInput(text)}
            showQuickActions={prefs.documents.showQuickActionHints}
          />
        )}
        {messages.map((msg, idx) => {
          if (loading && idx === messages.length - 1 && msg.role === 'assistant' && msg.content === '') return null
          return (
            <MessageRow
              key={msg.id}
              msg={msg}
              onInterruptReply={handleInterruptReply}
            />
          )
        })}
        {loading && messages[messages.length - 1]?.content === '' && (
          <div className="flex justify-start pl-2">
            <TypingDots />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {attachedFile && (
        <div
          className="mx-3 mb-1 px-3 py-1.5 rounded-xl flex items-center gap-2 text-[11px]"
          style={{ background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.12)' }}
        >
          <Paperclip size={11} style={{ color: '#6366f1', flexShrink: 0 }} />
          <span className="flex-1 text-indigo-700 truncate">{attachedFile.name}</span>
          <button
            type="button"
            onClick={() => setAttachedFile(null)}
            className="text-slate-400 hover:text-slate-600 transition-colors"
          >
            <X size={12} />
          </button>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="shrink-0 px-3 pb-3"
      >
        <div
          className={cn(
            'flex items-center gap-1.5 rounded-full px-3 py-2 transition-all duration-200',
            isFocused
              ? 'shadow-[0_0_0_2px_rgba(99,102,241,0.15),0_2px_8px_rgba(0,0,0,0.05)]'
              : 'shadow-[0_2px_8px_rgba(0,0,0,0.05)]',
          )}
          style={{
            background: 'rgba(255,255,255,0.9)',
            border: `1px solid ${isFocused ? 'rgba(99,102,241,0.25)' : 'rgba(0,0,0,0.07)'}`,
          }}
        >
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            title="Прикрепить файл"
            className="shrink-0 w-7 h-7 flex items-center justify-center rounded-lg text-slate-400 hover:text-indigo-500 hover:bg-indigo-50 transition-all"
          >
            <Paperclip size={14} />
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
            placeholder={isListening ? 'Слушаю…' : attachedFile ? 'Добавьте комментарий...' : 'Спросите AI...'}
            disabled={loading && !isListening}
            className={cn(
              'flex-1 resize-none bg-transparent text-slate-900 placeholder:text-slate-400',
              'focus:outline-none text-[12px] leading-relaxed py-0.5 min-h-[22px] max-h-[150px] overflow-y-hidden text-center',
              isListening && 'text-indigo-700',
            )}
          />

          {isSpeechSupported && (
            <button
              type="button"
              onClick={toggleMic}
              title={isListening ? 'Остановить' : 'Голосовой ввод'}
              className={cn(
                'shrink-0 w-7 h-7 flex items-center justify-center rounded-lg transition-all',
                isListening
                  ? 'text-indigo-500 bg-indigo-50'
                  : 'text-slate-400 hover:text-indigo-500 hover:bg-indigo-50',
              )}
            >
              {isListening ? <SoundWave active /> : <Mic size={14} />}
            </button>
          )}

          {loading ? (
            <button
              type="button"
              onClick={handleStop}
              title="Остановить"
              style={{
                width: 32, height: 32, borderRadius: '50%',
                background: '#ef4444', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'white', flexShrink: 0,
              }}
            >
              <Square size={12} fill="white" strokeWidth={0} />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!displayInput.trim() && !attachedFile}
              title="Отправить"
              style={{
                width: 32, height: 32, borderRadius: '50%',
                background: (displayInput.trim() || attachedFile) ? '#6366f1' : 'rgba(0,0,0,0.06)',
                border: 'none',
                cursor: (displayInput.trim() || attachedFile) ? 'pointer' : 'not-allowed',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: (displayInput.trim() || attachedFile) ? 'white' : '#cbd5e1',
                flexShrink: 0,
                transition: 'background 0.15s',
              }}
            >
              <Send size={13} />
            </button>
          )}
        </div>
      </form>

      <p
        className="shrink-0 text-center px-3 pb-2"
        style={{ fontSize: 10, color: '#cbd5e1', lineHeight: 1.4 }}
      >
        EDMS Assistant — ИИ. Может ошибаться и давать неверную информацию.
      </p>
    </div>
  )
}

interface MessageRowProps {
  msg: ChatMessageType
  onInterruptReply: (resume: ResumeValue) => void
}

function MessageRow({ msg, onInterruptReply }: MessageRowProps) {
  return (
    <div className={cn('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
      <div className={cn('max-w-[85%]', msg.role === 'user' ? 'max-w-[75%]' : 'w-full')}>
        <ChatMessage
          content={msg.content}
          role={msg.role}
          timestamp={msg.timestamp}
          isError={msg.isError === true}
        />
      </div>
    </div>
  )
}

const QUICK_ACTIONS = ['Суммаризация', 'Поиск', 'Тезисы'] as const

function EmptyState({ onAction, showQuickActions }: { onAction: (text: string) => void; showQuickActions: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 gap-4 px-6">
      <div
        style={{
          width: 72,
          height: 72,
          borderRadius: 20,
          background: 'rgba(99,102,241,0.08)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <MessageSquare size={30} strokeWidth={1.5} color="#6366f1" />
      </div>
      <div className="text-center">
        <p style={{ fontSize: 16, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>
          Чем могу помочь?
        </p>
        <p style={{ fontSize: 13, color: '#94a3b8' }}>Задайте вопрос ...</p>
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
