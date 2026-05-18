import {z} from 'zod'
import {
    InterruptEventSchema,
    MessageEventSchema,
    DoneEventSchema,
    ErrorEventSchema,
    type ResumeValue
} from '@entities/interrupt/model/types'
import {ComplianceDataSchema} from '@entities/message/model/types'

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

// ── Zod Schemas for SSE Events ─────────────────────────────────────────

const SseMessageEventSchema = z.object({
    kind: z.literal('message'),
    data: MessageEventSchema,
})

const SseInterruptEventSchema = z.object({
    kind: z.literal('interrupt'),
    data: InterruptEventSchema,
})

const SseUiComponentEventSchema = z.object({
    kind: z.literal('ui_component'),
    data: z.object({
        type: z.string(),
        url: z.string().optional(),
        new_tab: z.boolean().optional(),
        refresh_meta: z.record(z.unknown()).optional(),
    }).passthrough(),
})

const SseDoneEventSchema = z.object({
    kind: z.literal('done'),
    data: DoneEventSchema,
})

const SseErrorEventSchema = z.object({
    kind: z.literal('error'),
    data: ErrorEventSchema,
})

export const SseEventSchema = z.discriminatedUnion('kind', [
    SseMessageEventSchema,
    SseInterruptEventSchema,
    SseUiComponentEventSchema,
    SseDoneEventSchema,
    SseErrorEventSchema,
])

export type SseEvent = z.infer<typeof SseEventSchema>

export function parseSseEvent(raw: unknown): SseEvent | null {
    const result = SseEventSchema.safeParse(raw)
    return result.success ? result.data : null
}

export function parseCompliancePayload(raw: unknown): unknown | null {
    const result = ComplianceDataSchema.safeParse(raw)
    return result.success ? result.data : null
}

// ── SSE streaming ───────────────────────────────────────────────────────

export interface StreamHandle {
    events: AsyncIterable<SseEvent>
    abort(): void
}

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
            const parsed = parseSseEvent(msg.data)
            if (parsed) queue.push(parsed)
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