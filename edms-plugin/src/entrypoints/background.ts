export default defineBackground({
    type: 'module',
    main() {
        const API = 'http://localhost:8000'
        const controllers = new Map<string, AbortController>()

        function normalizeUuid(raw: string): string {
            return raw.replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00AD\uFE58\uFE63\uFF0D]/g, '-').trim()
        }

        chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
            const reqId: string = msg.payload?.requestId ?? 'default'

            switch (msg.type) {
                case 'abortRequest': {
                    controllers.get(reqId)?.abort()
                    controllers.delete(reqId)
                    return false
                }

                case 'sendChatMessage':
                    doFetch(`${API}/chat`, msg.payload, reqId, sendResponse)
                    return true

                case 'summarizeDocument':
                    doFetch(`${API}/actions/summarize`, msg.payload, reqId, sendResponse)
                    return true

                case 'uploadFile':
                    doUpload(msg.payload, sendResponse)
                    return true

                case 'getChatHistory':
                    doGetHistory(msg.payload.thread_id, sendResponse)
                    return true

                case 'createNewChat':
                    doFetch(`${API}/chat/new`, {user_token: msg.payload.user_token}, reqId, sendResponse)
                    return true

                case 'autofillAppeal':
                    doFetch(`${API}/appeal/autofill`, {
                        message: msg.payload.message ?? 'Заполни обращение',
                        user_token: msg.payload.user_token,
                        context_ui_id: msg.payload.context_ui_id,
                        file_path: msg.payload.file_path ?? null,
                    }, reqId, sendResponse)
                    return true

                case 'refreshDocumentData':
                    doRefreshDocument(msg.payload, sendResponse)
                    return true

                case 'fetchSettingsMeta':
                    doFetchSettingsMeta(sendResponse)
                    return true

                case 'fetchSettings':
                    doFetchSettings(msg.payload?.user_token, sendResponse)
                    return true

                case 'updateSettings':
                    doPatchSettings(msg.payload?.user_token, msg.payload?.settings, sendResponse)
                    return true

                case 'navigateTo':
                    doNavigateTo(msg.payload, sendResponse)
                    return true

                case 'deleteCache':
                    doDeleteCache(msg.payload, sendResponse)
                    return true

                case 'reloadActiveTab': {
                    chrome.tabs.query({active: true, currentWindow: true}).then(([tab]) => {
                        if (tab?.id) chrome.tabs.reload(tab.id)
                    })
                    sendResponse({success: true})
                    return true
                }

                default:
                    return false
            }
        })

        async function doFetch(
            url: string,
            payload: unknown,
            reqId: string,
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ) {
            const ctrl = new AbortController()
            controllers.set(reqId, ctrl)
            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload),
                    signal: ctrl.signal,
                })
                const data = await res.json()
                if (!res.ok) throw new Error(data.detail ?? `Ошибка сервера: ${res.status}`)
                respond({success: true, data})
            } catch (e: any) {
                if (e.name === 'AbortError') respond({success: false, error: 'Request aborted'})
                else respond({success: false, error: e.message})
            } finally {
                controllers.delete(reqId)
            }
        }

        async function doUpload(
            payload: { fileData: string; fileName: string; user_token: string },
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ) {
            try {
                const blob = await (await fetch(payload.fileData)).blob()
                const form = new FormData()
                form.append('file', blob, payload.fileName)
                form.append('user_token', payload.user_token)
                const res = await fetch(`${API}/upload-file`, {method: 'POST', body: form})
                const data = await res.json()
                if (!res.ok) throw new Error(data.detail ?? 'Ошибка загрузки файла')
                respond({success: true, data})
            } catch (e: any) {
                respond({success: false, error: e.message})
            }
        }

        async function doGetHistory(
            threadId: string,
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ) {
            try {
                const res = await fetch(`${API}/chat/history/${threadId}`)
                const data = await res.json()
                if (!res.ok) throw new Error(data.detail ?? 'Ошибка истории')
                respond({success: true, data})
            } catch (e: any) {
                respond({success: false, error: e.message})
            }
        }

        async function doRefreshDocument(
            payload: { edmsApiUrl: string; documentId: string; user_token: string },
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ) {
            try {
                const urls = [
                    `${API}/document/${payload.documentId}?token=${payload.user_token}`,
                    `${payload.edmsApiUrl}/api/documents/${payload.documentId}`,
                    `${payload.edmsApiUrl}/api/document/${payload.documentId}`,
                ]
                for (const url of urls) {
                    try {
                        const res = await fetch(url, {
                            headers: {
                                'Authorization': `Bearer ${payload.user_token}`,
                                'Content-Type': 'application/json',
                            },
                        })
                        if (res.ok) {
                            const data = await res.json()
                            respond({success: true, data})
                            return
                        }
                    } catch {
                        continue
                    }
                }
                respond({success: false, error: 'Document API endpoint not found'})
            } catch (e: any) {
                respond({success: false, error: e.message})
            }
        }

        /**
         * doNavigateTo — умная навигация.
         *
         * payload.newTab = true (по умолчанию) → открыть в НОВОЙ вкладке
         *   Используется для DocCard клика (документы из поиска)
         *
         * payload.newTab = false → навигация в ТЕКУЩЕЙ вкладке
         *   Используется только для create_document_from_file (после создания документа)
         *
         * Нормализует URL: заменяет типографские тире в UUID на ASCII дефис.
         */
        async function doNavigateTo(
            payload: { url: string; newTab?: boolean },
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ): Promise<void> {
            try {
                const normalizedUrl = normalizeUuid(payload.url)

                const [tab] = await chrome.tabs.query({active: true, currentWindow: true})
                if (!tab?.id) {
                    respond({success: false, error: 'No active tab found'})
                    return
                }

                const tabUrl = tab.url ?? ''
                let origin = ''
                try {
                    origin = tabUrl ? new URL(tabUrl).origin : ''
                } catch { /* ignore */
                }

                const targetUrl = normalizedUrl.startsWith('http')
                    ? normalizedUrl
                    : `${origin}${normalizedUrl}`

                const openInNew = payload.newTab !== false

                if (openInNew) {
                    await chrome.tabs.create({url: targetUrl, active: true})
                    respond({success: true, data: {url: targetUrl, newTab: true}})
                } else {
                    try {
                        await chrome.scripting.executeScript({
                            target: {tabId: tab.id},
                            func: (url: string) => {
                                window.location.href = url
                            },
                            args: [targetUrl],
                        })
                        respond({success: true, data: {url: targetUrl, newTab: false}})
                    } catch {
                        await chrome.tabs.update(tab.id, {url: targetUrl})
                        respond({success: true, data: {url: targetUrl, newTab: false}})
                    }
                }
            } catch (e: any) {
                respond({success: false, error: e.message ?? 'Navigation failed'})
            }
        }

        async function doDeleteCache(
            payload: { file_identifier: string; summary_type?: string },
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ): Promise<void> {
            try {
                const url = payload.summary_type
                    ? `${API}/api/cache/summarization/${encodeURIComponent(payload.file_identifier)}/${encodeURIComponent(payload.summary_type)}`
                    : `${API}/api/cache/summarization/${encodeURIComponent(payload.file_identifier)}`
                const res = await fetch(url, {method: 'DELETE'})
                const data = await res.json()
                if (!res.ok) throw new Error(data.detail ?? `Cache delete error: ${res.status}`)
                respond({success: true, data})
            } catch (e: any) {
                respond({success: false, error: e.message})
            }
        }

        async function doFetchSettingsMeta(
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ): Promise<void> {
            try {
                const res = await fetch(`${API}/api/settings/meta`, {
                    method: 'GET',
                    headers: {'Content-Type': 'application/json'},
                })
                const data = await res.json()
                if (!res.ok) throw new Error(data.detail ?? `Settings meta error: ${res.status}`)
                respond({success: true, data})
            } catch {
                respond({success: true, data: {show_technical: false}})
            }
        }

        async function doFetchSettings(
            userToken: string | undefined,
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ): Promise<void> {
            try {
                const headers: Record<string, string> = {'Content-Type': 'application/json'}
                if (userToken) headers['Authorization'] = `Bearer ${userToken}`
                const res = await fetch(`${API}/api/settings`, {method: 'GET', headers})
                const data = await res.json()
                if (!res.ok) throw new Error(data.detail ?? `Settings fetch error: ${res.status}`)
                respond({success: true, data})
            } catch (e: any) {
                respond({success: false, error: e.message})
            }
        }

        async function doPatchSettings(
            userToken: string | undefined,
            settings: unknown,
            respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
        ): Promise<void> {
            try {
                const headers: Record<string, string> = {'Content-Type': 'application/json'}
                if (userToken) headers['Authorization'] = `Bearer ${userToken}`
                const res = await fetch(`${API}/api/settings`, {
                    method: 'PATCH',
                    headers,
                    body: JSON.stringify(settings ?? {}),
                })
                const data = await res.json()
                if (!res.ok) throw new Error(data.detail ?? `Settings update error: ${res.status}`)
                respond({success: true, data})
            } catch (e: any) {
                respond({success: false, error: e.message})
            }
        }
    },
})