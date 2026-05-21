// edms-plugin/src/features/chat/model/useStreamChat.ts
import {useCallback, useRef} from 'react'
import {useChatStore} from './useChatStore'
import {parseSseEvent, parseCompliancePayload} from '@shared/api/sse-schemas'
import {InterruptPayloadSchema} from '@entities/interrupt/model/types'
import type {ChatMessage, ComplianceData} from '@entities/message/model/types'
import type {ResumeValue} from '@entities/interrupt/model/types'
import type {SseEvent} from '@shared/api/sse-schemas'
import {sendMessage} from '@shared/api/messaging'

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
        attachments: filePath
            ? [{path: filePath, name: filePath.split('/').pop() ?? filePath}]
            : undefined,
    }
}

export function useStreamChat() {
    const portRef = useRef<chrome.runtime.Port | null>(null)
    const {appendMessage, updateLastMessage, setLoading, setThreadId} = useChatStore()

    const startStream = useCallback(
        (options: StreamOptions): StreamHandle => {
            const {message, userToken, threadId, contextUiId, filePath, resumeValue} = options

            if (portRef.current) {
                portRef.current.disconnect()
                portRef.current = null
            }

            const isResume = resumeValue !== null
            const portName = isResume ? 'resumeChat' : 'streamChatMessage'
            const port = chrome.runtime.connect({name: portName})
            portRef.current = port

            const assistantMsgId = makeId()
            setLoading(true)

            if (!isResume) {
                appendMessage(buildUserMessage(message, filePath))
            }

            appendMessage({
                id: assistantMsgId,
                role: 'assistant',
                content: '',
                timestamp: Date.now(),
            })

            port.postMessage({
                message,
                user_token: userToken,
                thread_id: threadId,
                context_ui_id: contextUiId,
                file_path: filePath,
                resume_value: resumeValue,
            })

            let streamFinishedSuccessfully = false

            port.onMessage.addListener((raw: unknown) => {
                const msg = raw as { type: string; data?: unknown; error?: string }
                if (msg.type === 'ping') return;
                if (msg.type === 'sse_event' && msg.data) {
                    const event = parseSseEvent(msg.data)
                    if (event) {
                        handleSseEvent(event, assistantMsgId)
                    } else {
                        console.warn('[useStreamChat] Failed to parse SSE event:', msg.data)
                    }
                }

                if (msg.type === 'sse_error') {
                    streamFinishedSuccessfully = true
                    updateLastMessage((m) => ({
                        ...m,
                        content: msg.error ?? 'Stream error',
                        isError: true,
                    }))
                    setLoading(false)
                    portRef.current = null
                }

                if (msg.type === 'sse_done') {
                    streamFinishedSuccessfully = true
                    setLoading(false)
                    portRef.current = null
                }
            })

            port.onDisconnect.addListener(() => {
                if (!streamFinishedSuccessfully) {
                    updateLastMessage((m) =>
                        m.id === assistantMsgId && m.content === ''
                            ? {...m, content: 'Соединение потеряно', isError: true}
                            : m,
                    )
                }
                setLoading(false)
                portRef.current = null
            })

            return {
                abort: () => {
                    streamFinishedSuccessfully = true
                    port.disconnect()
                    portRef.current = null
                    setLoading(false)
                },
            }
        },
        [appendMessage, updateLastMessage, setLoading, setThreadId],
    )

    function handleSseEvent(event: SseEvent, assistantMsgId: string): void {
        switch (event.kind) {
            case 'message': {
                updateLastMessage((m) =>
                    m.id === assistantMsgId
                        ? {...m, content: m.content + event.data.content}
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
                const parseResult = InterruptPayloadSchema.safeParse(event.data.payload)
                if (parseResult.success) {
                    updateLastMessage((m) =>
                        m.id === assistantMsgId
                            ? {...m, interrupt: parseResult.data}
                            : m,
                    )
                } else {
                    console.warn(
                        '[useStreamChat] Invalid interrupt payload:',
                        parseResult.error.flatten(),
                        'raw:', event.data.payload,
                    )
                    updateLastMessage((m) =>
                        m.id === assistantMsgId
                            ? {...m, interrupt: event.data.payload}
                            : m,
                    )
                }
                break
            }

            case 'ui_component': {
                const data = event.data as Record<string, unknown> & { type: string }

                if (data.type === 'navigate' && typeof data.url === 'string') {
                    void sendMessage('navigateTo', {
                        url: data.url as string,
                        newTab: data.new_tab !== false,
                    })
                }

                if (data.type === 'compliance_result') {
                    const compliance =
                        parseCompliancePayload(data) ?? (data as unknown as ComplianceData | null)
                    if (compliance) {
                        updateLastMessage((m) =>
                            m.id === assistantMsgId
                                ? {
                                    ...m,
                                    compliance: compliance as ComplianceData,
                                    refreshMeta:
                                        (data.refresh_meta as ChatMessage['refreshMeta']) ?? undefined,
                                }
                                : m,
                        )
                    }
                }
                break
            }

            case 'error': {
                updateLastMessage((m) =>
                    m.id === assistantMsgId
                        ? {...m, content: event.data.message, isError: true}
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

    return {startStream, abort}
}