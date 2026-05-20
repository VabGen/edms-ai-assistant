import {onMessage, SummarizeResponse} from '@shared/api/messaging'
import {normalizeUuid} from '@shared/lib/normalize'
import {parseSseEvent} from '@shared/api/sse-schemas'

const API = import.meta.env.VITE_API_URL as string

const controllers = new Map<string, AbortController>()

function authHeaders(userToken?: string): Record<string, string> {
    const headers: Record<string, string> = {'Content-Type': 'application/json'}
    if (userToken) headers['Authorization'] = `Bearer ${userToken}`
    return headers
}

async function postJson(
    url: string,
    payload: unknown,
    signal?: AbortSignal,
): Promise<unknown> {
    const res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
        signal: signal ?? null,
    })
    const data: unknown = await res.json()
    if (!res.ok) {
        const detail = (data as Record<string, unknown>)['detail']
        throw new Error(typeof detail === 'string' ? detail : `Server error: ${res.status}`)
    }
    return data
}

export default defineBackground({
    type: 'module',
    main() {
        registerSsePort()
        registerMessageHandlers()
    },
})

function registerSsePort(): void {
    chrome.runtime.onConnect.addListener((port) => {
        if (port.name !== 'streamChatMessage' && port.name !== 'resumeChat') return

        const ctrl = new AbortController()
        let started = false

        port.onMessage.addListener(async (payload: unknown) => {
            if (started) return
            started = true

            const p = payload as Record<string, any>
            const msgLower = (p.message || '').trim().toLowerCase()
            const QUICK_FORMAT_MAP: Record<string, 'thesis' | 'extractive' | 'abstractive'> = {
                'тезисы': 'thesis',
                'факты': 'extractive',
                'пересказ': 'abstractive',
                'суммаризация': 'abstractive',
            }
            const quickFormat = QUICK_FORMAT_MAP[msgLower]

            if (port.name === 'streamChatMessage' && quickFormat) {
                try {
                    const format = quickFormat
                    const res = await fetch(`${API}/actions/summarize/stream`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            message: p.message,
                            user_token: p.user_token,
                            context_ui_id: p.context_ui_id,
                            preferred_summary_format: format,
                            file_path: p.file_path,
                        }),
                        signal: ctrl.signal,
                    })

                    if (!res.ok) {
                        const errBody = (await res.json().catch(() => ({}))) as Record<string, unknown>
                        port.postMessage({
                            type: 'sse_error',
                            error: (errBody['detail'] as string) ?? `HTTP ${res.status}`,
                        })
                        port.disconnect()
                        return
                    }

                    const reader = res.body?.getReader()
                    if (!reader) throw new Error('No response body')

                    const decoder = new TextDecoder()
                    let buffer = ''

                    while (true) {
                        const {done, value} = await reader.read()
                        if (done) break

                        buffer += decoder.decode(value, {stream: true})
                        const parts = buffer.split('\n\n')
                        buffer = parts.pop() ?? ''

                        for (const part of parts) {
                            if (!part.trim()) continue
                            let dataStr = ''
                            for (const line of part.split('\n')) {
                                if (line.startsWith('data: ')) {
                                    dataStr = line.slice(6)
                                }
                            }
                            if (!dataStr) continue

                            try {
                                if (dataStr === '[DONE]') {
                                    port.postMessage({type: 'sse_done'})
                                    continue
                                }
                                const payload = JSON.parse(dataStr)
                                if (payload.event === 'delta' && payload.text) {
                                    port.postMessage({
                                        type: 'sse_event',
                                        data: {kind: 'message', data: {role: 'assistant', content: payload.text}}
                                    })
                                } else if (payload.event === 'result') {
                                    // Overwrite content with the final complete text (raw JSON for structural render)
                                    port.postMessage({
                                        type: 'sse_event',
                                        data: {kind: 'message', data: {role: 'assistant', content: payload.response}}
                                    })
                                } else if (payload.event === 'error') {
                                    port.postMessage({type: 'sse_error', error: payload.message})
                                }
                            } catch { /* ignore malformed */ }
                        }
                    }
                } catch (err: any) {
                    port.postMessage({type: 'sse_error', error: err.message || 'Summarization failed'})
                } finally {
                    port.disconnect()
                }
                return
            }

            const url =
                port.name === 'resumeChat' ? `${API}/chat/resume` : `${API}/chat/stream`

            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload),
                    signal: ctrl.signal,
                })

                if (!res.ok) {
                    const errBody = (await res.json().catch(() => ({}))) as Record<string, unknown>
                    const detail = errBody['detail']
                    port.postMessage({
                        type: 'sse_error',
                        error: typeof detail === 'string' ? detail : `HTTP ${res.status}`,
                    })
                    port.disconnect()
                    return
                }

                const reader = res.body?.getReader()
                if (!reader) {
                    port.postMessage({type: 'sse_error', error: 'No response body'})
                    port.disconnect()
                    return
                }

                const decoder = new TextDecoder()
                let buffer = ''

                while (true) {
                    const {done, value} = await reader.read()
                    if (done) break

                    buffer += decoder.decode(value, {stream: true})
                    const parts = buffer.split('\n\n')
                    buffer = parts.pop() ?? ''

                    for (const part of parts) {
                        if (!part.trim()) continue

                        let eventType = 'message'
                        let dataStr = ''

                        for (const line of part.split('\n')) {
                            if (line.startsWith('event: ')) eventType = line.slice(7).trim()
                            else if (line.startsWith('data: ')) dataStr = line.slice(6)
                        }

                        if (!dataStr) continue

                        try {
                            const parsed: unknown = JSON.parse(dataStr)
                            const event = parseSseEvent({kind: eventType, data: parsed})
                            if (event) port.postMessage({type: 'sse_event', data: event})
                        } catch {
                            // skip malformed JSON
                        }
                    }
                }

                port.postMessage({type: 'sse_done'})
            } catch (err: unknown) {
                if (err instanceof Error && err.name !== 'AbortError') {
                    port.postMessage({type: 'sse_error', error: err.message})
                }
            } finally {
                port.disconnect()
            }
        })

        port.onDisconnect.addListener(() => {
            ctrl.abort()
        })
    })
}

function registerMessageHandlers(): void {
    onMessage('abortRequest', ({data}) => {
        controllers.get(data.requestId)?.abort()
        controllers.delete(data.requestId)
    })

    onMessage('summarizeDocument', async ({data}) => {
        const ctrl = new AbortController()
        const reqId = crypto.randomUUID()
        controllers.set(reqId, ctrl)
        try {
            const result = await postJson(`${API}/actions/summarize`, data, ctrl.signal)
            return {success: true as const, data: result as NonNullable<SummarizeResponse['data']>}
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : 'Unknown error'
            return {success: false, error: msg}
        } finally {
            controllers.delete(reqId)
        }
    })

    onMessage('uploadFile', async ({data}) => {
        try {
            const blob = await (await fetch(data.file_data)).blob()
            const form = new FormData()
            form.append('file', blob, data.file_name)
            form.append('user_token', data.user_token)
            if (data.thread_id) form.append('thread_id', data.thread_id)
            if (data.context_ui_id) form.append('context_ui_id', data.context_ui_id)
            const res = await fetch(`${API}/upload-file`, {method: 'POST', body: form})
            const result = (await res.json()) as Record<string, unknown>
            if (!res.ok) throw new Error((result['detail'] as string | undefined) ?? 'Upload error')
            const filePath = result['file_path'] as string | undefined
            return {success: true as const, ...(filePath !== undefined ? {file_path: filePath} : {})}
        } catch (err: unknown) {
            return {success: false, error: err instanceof Error ? err.message : 'Unknown error'}
        }
    })

    onMessage('getChatHistory', async ({data}) => {
        const res = await fetch(`${API}/chat/history/${data.thread_id}`, {
            headers: authHeaders(data.user_token),
        })
        const result = (await res.json()) as Record<string, unknown>
        if (!res.ok) throw new Error((result['detail'] as string | undefined) ?? 'History error')
        return result as { messages: { type: 'human' | 'ai'; content: string }[]; thread_id: string }
    })

    onMessage('createNewChat', async ({data}) => {
        const result = await postJson(`${API}/chat/new`, {user_token: data.user_token})
        return result as { thread_id: string }
    })

    onMessage('navigateTo', async ({data}) => {
        try {
            const normalizedUrl = normalizeUuid(data.url)
            const [tab] = await chrome.tabs.query({active: true, currentWindow: true})
            if (!tab?.id) return {success: false, error: 'No active tab found'}

            let origin = ''
            try {
                if (tab.url) origin = new URL(tab.url).origin
            } catch { /* ignore */
            }

            const targetUrl = normalizedUrl.startsWith('http')
                ? normalizedUrl
                : `${origin}${normalizedUrl}`

            if (data.newTab !== false) {
                await chrome.tabs.create({url: targetUrl, active: true})
            } else {
                try {
                    await chrome.scripting.executeScript({
                        target: {tabId: tab.id},
                        func: (url: string) => {
                            window.location.href = url
                        },
                        args: [targetUrl],
                    })
                } catch {
                    await chrome.tabs.update(tab.id, {url: targetUrl})
                }
            }
            return {success: true}
        } catch (err: unknown) {
            return {success: false, error: err instanceof Error ? err.message : 'Navigation failed'}
        }
    })

    onMessage('deleteCache', async ({data}) => {
        try {
            const parts = [
                `${API}/summarize/cache/${encodeURIComponent(data.file_identifier ?? '')}`,
            ]
            if (data.summary_type) parts.push(encodeURIComponent(data.summary_type))
            const res = await fetch(parts.join('/'), {method: 'DELETE'})
            const result = (await res.json()) as Record<string, unknown>
            if (!res.ok) throw new Error((result['detail'] as string | undefined) ?? `Cache error ${res.status}`)
            return {success: true}
        } catch (err: unknown) {
            return {success: false, error: err instanceof Error ? err.message : 'Unknown error'}
        }
    })

    onMessage('refreshDocument', async ({data}) => {
        const urls = [
            `${API}/document/${data.doc_id}?token=${data.user_token}`,
        ]
        for (const url of urls) {
            try {
                const res = await fetch(url, {headers: authHeaders(data.user_token)})
                if (res.ok) {
                    const result: unknown = await res.json()
                    return {success: true, data: result}
                }
            } catch { /* try next */
            }
        }
        return {success: false, error: 'Document endpoint not found'}
    })

    onMessage('fetchSettingsMeta', async (_msg) => {
        try {
            const res = await fetch(`${API}/api/settings/meta`, {
                headers: {'Content-Type': 'application/json'},
            })
            const data = (await res.json()) as Record<string, unknown>
            if (!res.ok) return {show_technical: false}
            return {show_technical: Boolean(data['show_technical'])}
        } catch {
            return {show_technical: false}
        }
    })

    onMessage('fetchSettings', async ({data}) => {
        const res = await fetch(`${API}/api/settings`, {
            headers: authHeaders(data.user_token),
        })
        const result = (await res.json()) as Record<string, unknown>
        if (!res.ok) throw new Error((result['detail'] as string | undefined) ?? 'Settings fetch error')
        return result
    })

    onMessage('updateSettings', async ({data}) => {
        const res = await fetch(`${API}/api/settings`, {
            method: 'PATCH',
            headers: authHeaders(data.user_token),
            body: JSON.stringify(data.settings),
        })
        const result = (await res.json()) as Record<string, unknown>
        if (!res.ok) throw new Error((result['detail'] as string | undefined) ?? 'Settings update error')
        return result
    })

    onMessage('resetSettings', async ({data}) => {
        const res = await fetch(`${API}/api/settings`, {
            method: 'DELETE',
            headers: authHeaders(data.user_token),
        })
        if (!res.ok) {
            const result = (await res.json().catch(() => ({}))) as Record<string, unknown>
            throw new Error((result['detail'] as string | undefined) ?? 'Settings reset error')
        }
    })

    onMessage('reloadActiveTab', async (_msg) => {
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true})
        if (tab?.id) await chrome.tabs.reload(tab.id)
        return {success: true}
    })
}

// 5