export default defineBackground({
    type: 'module',
    main() {
        const API = 'http://localhost:8000'
        const controllers = new Map<string, AbortController>()

        chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
            const reqId: string = msg.payload?.requestId ?? 'default'

            switch (msg.type) {
                // ── Abort ──────────────────────────────────────────────────────────
                case 'abortRequest': {
                    controllers.get(reqId)?.abort()
                    controllers.delete(reqId)
                    return false
                }

                // ── Chat ───────────────────────────────────────────────────────────
                case 'sendChatMessage':
                    doFetch(`${API}/chat`, msg.payload, reqId, sendResponse)
                    return true

                // ── Summarize attachment ───────────────────────────────────────────
                case 'summarizeDocument':
                    doFetch(`${API}/actions/summarize`, msg.payload, reqId, sendResponse)
                    return true

                // ── Upload file ────────────────────────────────────────────────────
                case 'uploadFile':
                    doUpload(msg.payload, sendResponse)
                    return true

                // ── History ────────────────────────────────────────────────────────
                case 'getChatHistory':
                    doGetHistory(msg.payload.thread_id, sendResponse)
                    return true

                // ── New chat ───────────────────────────────────────────────────────
                case 'createNewChat':
                    doFetch(`${API}/chat/new`, {user_token: msg.payload.user_token}, reqId, sendResponse)
                    return true

                // ── Autofill appeal ────────────────────────────────────────────────
                case 'autofillAppeal':
                    doFetch(`${API}/appeal/autofill`, {
                        message: msg.payload.message ?? 'Заполни обращение',
                        user_token: msg.payload.user_token,
                        context_ui_id: msg.payload.context_ui_id,
                        file_path: msg.payload.file_path ?? null,
                    }, reqId, sendResponse)
                    return true

                // ── Refresh document data ──────────────────────────────────────────
                case 'refreshDocumentData':
                    doRefreshDocument(msg.payload, sendResponse)
                    return true

                // ── Settings: feature flags ────────────────────────────────────────
                case 'fetchSettingsMeta':
                    doFetchSettingsMeta(sendResponse)
                    return true

                // ── Settings: GET current technical settings ───────────────────────
                case 'fetchSettings':
                    doFetchSettings(msg.payload?.user_token, sendResponse)
                    return true

                // ── Settings: PATCH technical settings ────────────────────────────
                case 'updateSettings':
                    doPatchSettings(msg.payload?.user_token, msg.payload?.settings, sendResponse)
                    return true

                default:
                    return false
            }
        })

        // ── Helpers ─────────────────────────────────────────────────────────────

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

        // ── Settings helpers (новые) ─────────────────────────────────────────────

        /**
         * Fetches settings panel feature flags from backend.
         *
         * Reads SETTINGS_PANEL_SHOW_TECHNICAL from server config.
         * On network error falls back to safe default: show_technical = false.
         *
         * @param respond - Chrome message response callback.
         */
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
                respond({
                    success: true,
                    data: {show_technical: false},
                })
            }
        }

        /**
         * Fetches current effective technical settings from backend.
         *
         * Returns merged .env defaults + any in-memory runtime overrides.
         *
         * @param userToken - Optional JWT bearer token.
         * @param respond   - Chrome message response callback.
         */
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

        /**
         * Sends a PATCH request to update runtime technical settings.
         *
         * Backend applies patch in-memory and returns resulting effective settings.
         * Returns 403 if SETTINGS_PANEL_SHOW_TECHNICAL=false on the server.
         *
         * @param userToken - Optional JWT bearer token.
         * @param settings  - UpdateSettingsRequest body (snake_case groups).
         * @param respond   - Chrome message response callback.
         */
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