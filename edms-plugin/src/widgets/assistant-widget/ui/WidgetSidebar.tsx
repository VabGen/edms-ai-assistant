import { useEffect, useCallback } from 'react'
import { PenSquare, Settings, Trash2 } from 'lucide-react'
import { storage } from 'wxt/storage'
import { useChatStore, threadsStorage } from '@features/chat/model/useChatStore'
import { sendMessage } from '@shared/api/messaging'
import { getAuthToken } from '@/shared/lib/auth'
import { cn } from '@shared/lib/cn'
import type { Thread } from '@entities/message/model/types'

const MAX_THREADS = 20

interface WidgetSidebarProps {
  onOpenSettings: () => void
}

export function WidgetSidebar({ onOpenSettings }: WidgetSidebarProps) {
  const { threads, threadId, messages, setMessages, setThreadId, setThreads, clearSnapshot } =
    useChatStore()

  useEffect(() => {
    void threadsStorage.getValue().then((saved) => {
      if (saved?.length) setThreads(saved)
    })
  }, [setThreads])

  const handleNewChat = useCallback(async () => {
    if (messages.length > 0 && threadId) {
      const preview = buildPreview(messages.filter((m) => m.role === 'user'))
      const updated = [
        { id: threadId, preview, date: new Date().toLocaleDateString('ru-RU') },
        ...threads.filter((t) => t.id !== threadId),
      ].slice(0, MAX_THREADS)
      setThreads(updated)
      await threadsStorage.setValue(updated)
    }

    const token = getAuthToken()
    if (token) {
      try {
        const res = await sendMessage('createNewChat', { user_token: token })
        setThreadId(res.thread_id)
      } catch {
        setThreadId(null)
      }
    } else {
      setThreadId(null)
    }
    setMessages([])
    await clearSnapshot()
  }, [messages, threadId, threads, setThreads, setThreadId, setMessages, clearSnapshot])

  const handleSelectThread = useCallback(
    async (thread: Thread) => {
      if (thread.id === threadId) return
      if (messages.length > 0 && threadId) {
        const preview = buildPreview(messages.filter((m) => m.role === 'user'))
        const updated = [
          { id: threadId, preview, date: new Date().toLocaleDateString('ru-RU') },
          ...threads.filter((t) => t.id !== threadId),
        ].slice(0, MAX_THREADS)
        setThreads(updated)
        await threadsStorage.setValue(updated)
      }
      setMessages(thread.messages ?? [])
      setThreadId(thread.id)
    },
    [messages, threadId, threads, setThreads, setMessages, setThreadId],
  )

  const handleDeleteThread = useCallback(
    async (e: React.MouseEvent, id: string) => {
      e.stopPropagation()
      const updated = threads.filter((t) => t.id !== id)
      setThreads(updated)
      await threadsStorage.setValue(updated)
      if (id === threadId) {
        setMessages([])
        setThreadId(null)
        await clearSnapshot()
      }
    },
    [threads, threadId, setThreads, setMessages, setThreadId, clearSnapshot],
  )

  return (
    <aside
      className="flex flex-col shrink-0 border-r overflow-hidden"
      style={{
        width: 200,
        borderColor: 'rgba(0,0,0,0.05)',
        background: 'rgba(248,250,252,0.7)',
      }}
    >
      <div className="flex items-center justify-between px-3 py-2.5 shrink-0" style={{ borderBottom: '1px solid rgba(0,0,0,0.05)' }}>
        <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-slate-400">
          История
        </span>
        <button
          type="button"
          onClick={handleNewChat}
          title="Новый диалог"
          className="w-6 h-6 flex items-center justify-center rounded-lg text-indigo-500 hover:bg-indigo-50 transition-colors"
        >
          <PenSquare size={13} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto py-1 min-h-0 scrollbar-none">
        {threads.length === 0 ? (
          <p className="px-3 py-4 text-[10px] text-slate-400 text-center leading-relaxed">
            История пуста.<br />Начните новый диалог.
          </p>
        ) : (
          threads.map((thread) => (
            <ThreadItem
              key={thread.id}
              thread={thread}
              isActive={thread.id === threadId}
              onSelect={handleSelectThread}
              onDelete={handleDeleteThread}
            />
          ))
        )}
      </div>

      <div className="shrink-0 p-2" style={{ borderTop: '1px solid rgba(0,0,0,0.05)' }}>
        <button
          type="button"
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2 px-2.5 py-2 rounded-xl text-slate-500 hover:bg-white/80 hover:text-slate-700 transition-all duration-150"
        >
          <Settings size={13} />
          <span className="text-[11px] font-medium">Настройки</span>
        </button>
      </div>
    </aside>
  )
}

interface ThreadItemProps {
  thread: Thread
  isActive: boolean
  onSelect: (thread: Thread) => void
  onDelete: (e: React.MouseEvent, id: string) => void
}

function ThreadItem({ thread, isActive, onSelect, onDelete }: ThreadItemProps) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onSelect(thread)}
      onKeyDown={(e) => e.key === 'Enter' && onSelect(thread)}
      className={cn(
        'group relative mx-1.5 my-0.5 px-2.5 py-2 rounded-xl cursor-pointer transition-all duration-150',
        isActive
          ? 'bg-white shadow-sm'
          : 'hover:bg-white/60',
      )}
    >
      <p
        className={cn(
          'text-[11px] leading-snug line-clamp-2 pr-5',
          isActive ? 'text-slate-800 font-semibold' : 'text-slate-600',
        )}
      >
        {thread.preview || 'Диалог'}
      </p>
      <span className="text-[9px] text-slate-400 mt-0.5 block">{thread.date}</span>
      <button
        type="button"
        onClick={(e) => onDelete(e, thread.id)}
        title="Удалить"
        className="absolute right-1.5 top-1/2 -translate-y-1/2 w-5 h-5 flex items-center justify-center rounded-lg text-slate-300 hover:text-red-400 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition-all duration-150"
      >
        <Trash2 size={11} />
      </button>
    </div>
  )
}

function buildPreview(userMessages: { content: string }[]): string {
  const meaningful = userMessages.find((m) => m.content.trim().length > 15) ?? userMessages[0]
  return (meaningful?.content ?? 'Диалог').trim().slice(0, 50)
}
