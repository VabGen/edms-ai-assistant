/** chrome.runtime helpers — simple sendMsg + SSE streaming via port */

import type {
    InterruptEvent,
    MessageEvent,
    DoneEvent,
    ErrorEvent,
    ResumeValue,
} from '@entities/interrupt/model/types'

// ── Simple request/response ──────────────────────────────────────────────

export function sendMsg<T = unknown>(
    type: string,
    payload: unknown,
): Promise<T> {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({type, payload}, (res) => {
            if (chrome.runtime.lastError) {
                reject(new Error(chrome.runtime.lastError.message))
                return
            }
            if (res?.success) resolve(res.data as T)
            else reject(new Error(res?.error ?? 'Unknown error'))
        })
    })
}

// ── UI-component event payload ───────────────────────────────────────────

/** Payload carried by ``event: ui_component`` SSE frames. */
export interface UiComponentEvent {
    /** Discriminator: ``"compliance_result"`` | ``"navigate"`` | … */
    type: string
    /** Compliance fields (when type === "compliance_result") */
    overall?: string
    summary?: string
    document_id?: string
    document_category?: string
    fields?: Array<{
        field_key: string
        label: string
        card_value?: string
        file_value?: string | null
        correct_value?: string | null
        status?: string
        update_field?: string
        recommendation?: string | null
    }>
    stats?: { total: number; mismatches: number; ok: number; not_found: number }
    fix_hint?: string
    /** Navigate URL (when type === "navigate") */
    url?: string
    /** Whether to open navigate URL in a new tab (default true) */
    new_tab?: boolean
}

// ── SSE streaming ───────────────────────────────────────────────────────

export type SseEvent =
    | { kind: 'message'; data: MessageEvent }
    | { kind: 'interrupt'; data: InterruptEvent }
    | { kind: 'ui_component'; data: UiComponentEvent }
    | { kind: 'done'; data: DoneEvent }
    | { kind: 'error'; data: ErrorEvent }

export interface StreamHandle {
    /** Async iterator yielding parsed SSE events */
    events: AsyncIterable<SseEvent>

    /** Abort the in-flight stream */
    abort(): void
}

/**
 * Start an SSE stream through the background service-worker.
 *
 * The background script does `fetch` + `ReadableStream` parsing and
 * forwards each SSE event through a `chrome.runtime.Port`.
 */
export function streamChat(
    portName: 'streamChatMessage' | 'resumeChat',
    payload: Record<string, unknown>,
): StreamHandle {
    const port = chrome.runtime.connect({name: portName})
    port.postMessage(payload)

    const queue: SseEvent[] = []
    let done = false
    let error: Error | null = null

    const pendingResolvers: Array<{
        resolve: (value: IteratorResult<SseEvent>) => void
        reject: (reason?: unknown) => void
    }> = []

    function flush() {
        while (queue.length > 0 && pendingResolvers.length > 0) {
            const ev = queue.shift()!
            const {resolve} = pendingResolvers.shift()!
            resolve({value: ev, done: false})
        }
        if ((done || error) && pendingResolvers.length > 0) {
            const {resolve, reject} = pendingResolvers.shift()!
            if (error) reject(error)
            else resolve({value: undefined, done: true} as IteratorResult<SseEvent>)
        }
    }

    port.onMessage.addListener((msg: { type: string; data?: unknown; error?: string }) => {
        if (msg.type === 'sse_event' && msg.data) {
            queue.push(msg.data as SseEvent)
        } else if (msg.type === 'sse_done') {
            done = true
        } else if (msg.type === 'sse_error') {
            error = new Error(msg.error ?? 'Stream error')
        }
        flush()
    })

    port.onDisconnect.addListener(() => {
        if (!done && !error) error = new Error('Port disconnected')
        done = true
        flush()
    })

    const asyncIterator: AsyncIterable<SseEvent> = {
        [Symbol.asyncIterator]() {
            return {
                async next(): Promise<IteratorResult<SseEvent>> {
                    if (queue.length > 0) {
                        return {value: queue.shift()!, done: false}
                    }
                    if (done) return {value: undefined, done: true} as IteratorResult<SseEvent>
                    if (error) throw error
                    return new Promise<IteratorResult<SseEvent>>((resolve, reject) => {
                        pendingResolvers.push({resolve, reject})
                    })
                },
                return(): Promise<IteratorResult<SseEvent>> {
                    done = true
                    return Promise.resolve({value: undefined, done: true} as IteratorResult<SseEvent>)
                },
            }
        },
    }

    return {
        events: asyncIterator,
        abort() {
            port.disconnect()
            done = true
            flush()
        },
    }
}

// ── Convenience: resume chat ─────────────────────────────────────────────

export function resumeChat(
    threadId: string,
    userToken: string,
    resumeValue: ResumeValue,
    interruptId?: string | null,
): StreamHandle {
    return streamChat('resumeChat', {
        thread_id: threadId,
        user_token: userToken,
        resume_value: resumeValue,
        interrupt_id: interruptId ?? undefined,
    })
}