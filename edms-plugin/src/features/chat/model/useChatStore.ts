import { create } from 'zustand'
import { storage } from 'wxt/storage'
import { STORAGE_KEYS } from '@shared/storage/keys'
import type { ChatMessage, Thread } from '@entities/message/model/types'

interface ChatSnapshot {
  messages: ChatMessage[]
  threadId: string | null
  isOpen: boolean
  savedAt: number
}

const snapshotStorage = storage.defineItem<ChatSnapshot>(STORAGE_KEYS.widgetSnapshot, {
  fallback: null as unknown as ChatSnapshot,
})

const threadsStorage = storage.defineItem<Thread[]>(STORAGE_KEYS.widgetThreads, {
  fallback: [],
})

interface ChatState {
  messages: ChatMessage[]
  threadId: string | null
  loading: boolean
  threads: Thread[]
  isSnapshotLoaded: boolean
}

interface ChatActions {
  setMessages: (messages: ChatMessage[]) => void
  appendMessage: (message: ChatMessage) => void
  updateLastMessage: (updater: (msg: ChatMessage) => ChatMessage) => void
  updateMessage: (id: string, updater: (msg: ChatMessage) => ChatMessage) => void
  setThreadId: (id: string | null) => void
  setLoading: (loading: boolean) => void
  setThreads: (threads: Thread[]) => void
  loadSnapshot: () => Promise<void>
  saveSnapshot: (isOpen: boolean) => Promise<void>
  clearSnapshot: () => Promise<void>
}

export const useChatStore = create<ChatState & ChatActions>((set, get) => ({
  messages: [],
  threadId: null,
  loading: false,
  threads: [],
  isSnapshotLoaded: false,

  setMessages: (messages) => set({ messages }),

  appendMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  updateLastMessage: (updater) =>
    set((state) => {
      const msgs = [...state.messages]
      const last = msgs[msgs.length - 1]
      if (!last) return state
      msgs[msgs.length - 1] = updater(last)
      return { messages: msgs }
    }),

  updateMessage: (id, updater) =>
    set((state) => {
      const idx = state.messages.findIndex((m) => m.id === id)
      if (idx < 0) return state
      const target = state.messages[idx]
      if (!target) return state
      const msgs = [...state.messages]
      msgs[idx] = updater(target)
      return { messages: msgs }
    }),

  setThreadId: (threadId) => set({ threadId }),
  setLoading: (loading) => set({ loading }),
  setThreads: (threads) => set({ threads }),

  loadSnapshot: async () => {
    const [snapshot, threads] = await Promise.all([
      snapshotStorage.getValue(),
      threadsStorage.getValue(),
    ])
    const updates: Partial<ChatState> = {
      isSnapshotLoaded: true,
      threads: threads ?? [],
    }
    if (snapshot?.isOpen) {
      updates.messages = snapshot.messages
      updates.threadId = snapshot.threadId
    }
    set(updates)
  },

  saveSnapshot: async (isOpen: boolean) => {
    const { messages, threadId } = get()
    if (messages.length === 0) return
    await snapshotStorage.setValue({
      messages,
      threadId,
      isOpen,
      savedAt: Date.now(),
    })
  },

  clearSnapshot: async () => {
    await snapshotStorage.removeValue()
    set({ messages: [], threadId: null })
  },
}))

export { threadsStorage }
