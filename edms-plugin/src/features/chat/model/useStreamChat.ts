import { useCallback, useRef } from 'react'
import { useChatStore } from './useChatStore'
import { parseSseEvent, parseCompliancePayload } from '@shared/api/sse-schemas'
import { InterruptPayloadSchema } from '@entities/interrupt/model/types'
import type { ChatMessage } from '@entities/message/model/types'
import type { ResumeValue } from '@entities/interrupt/model/types'
import type { SseEvent } from '@shared/api/sse-schemas'

interface StreamOptions {
  message: string
  userToken: string
  threadId: string
  contextUiId: string | null
  filePath: string | null
  resumeValue: ResumeValue | null
}

interface StreamHandle {
  abort: () => void
}

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function buildUserMessage(text: string, filePath: string | null): ChatMessage {
  return {
    id: makeId(),
    role: 'user',
    content: text,
    timestamp: Date.now(),
    attachments: filePath ? [{ path: filePath, name: filePath.split('/').pop() ?? filePath }] : undefined,
  }
}

export function useStreamChat() {
  const portRef = useRef<chrome.runtime.Port | null>(null)
  const { appendMessage, updateLastMessage, setLoading, setThreadId } = useChatStore()

  const startStream = useCallback(
    (options: StreamOptions): StreamHandle => {
      const {
        message,
        userToken,
        threadId,
        contextUiId,
        filePath,
        resumeValue,
      } = options

      if (portRef.current) {
        portRef.current.disconnect()
        portRef.current = null
      }

      const isResume = resumeValue !== null
      const portName = isResume ? 'resumeChat' : 'streamChatMessage'
      const port = chrome.runtime.connect({ name: portName })
      portRef.current = port

      const assistantMsgId = makeId()
      setLoading(true)

      if (!isResume) {
        appendMessage(buildUserMessage(message, filePath))
        appendMessage({
          id: assistantMsgId,
          role: 'assistant',
          content: '',
          timestamp: Date.now(),
        })
      }

      port.postMessage({
        message,
        user_token: userToken,
        thread_id: threadId,
        context_ui_id: contextUiId,
        file_path: filePath,
        resume_value: resumeValue,
      })

      port.onMessage.addListener((raw: unknown) => {
        const msg = raw as { type: string; data?: unknown; error?: string }

        if (msg.type === 'sse_event' && msg.data) {
          const event = parseSseEvent(msg.data)
          if (event) handleSseEvent(event, assistantMsgId)
        }

        if (msg.type === 'sse_error') {
          updateLastMessage((m) => ({
            ...m,
            content: msg.error ?? 'Stream error',
            isError: true,
          }))
          setLoading(false)
          portRef.current = null
        }

        if (msg.type === 'sse_done') {
          setLoading(false)
          portRef.current = null
        }
      })

      port.onDisconnect.addListener(() => {
        setLoading(false)
        portRef.current = null
      })

      return {
        abort: () => {
          port.disconnect()
          portRef.current = null
          setLoading(false)
        },
      }
    },
    [appendMessage, updateLastMessage, setLoading],
  )

  function handleSseEvent(event: SseEvent, assistantMsgId: string): void {
    switch (event.kind) {
      case 'message': {
        updateLastMessage((m) =>
          m.id === assistantMsgId
            ? { ...m, content: m.content + event.data.content }
            : m,
        )
        break
      }

      case 'done': {
        if (event.data.thread_id) {
          setThreadId(event.data.thread_id)
        }
        break
      }

      case 'interrupt': {
        const payload = InterruptPayloadSchema.safeParse(event.data.payload)
        if (!payload.success) break
        updateLastMessage((m) =>
          m.id === assistantMsgId
            ? { ...m, interrupt: payload.data }
            : m,
        )
        break
      }

      case 'ui_component': {
        if (event.data.component_type === 'compliance_result') {
          const compliance = parseCompliancePayload(event.data.payload)
          if (compliance) {
            updateLastMessage((m) =>
              m.id === assistantMsgId ? { ...m, compliance } : m,
            )
          }
        }
        break
      }

      case 'error': {
        updateLastMessage((m) =>
          m.id === assistantMsgId
            ? { ...m, content: event.data.message, isError: true }
            : m,
        )
        break
      }
    }
  }

  const abort = useCallback(() => {
    if (portRef.current) {
      portRef.current.disconnect()
      portRef.current = null
    }
    setLoading(false)
  }, [setLoading])

  return { startStream, abort }
}
