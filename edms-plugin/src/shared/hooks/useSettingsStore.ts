// edms-plugin/src/shared/hooks/useSettingsStore.ts
import {useState, useEffect, useCallback, useRef} from 'react'
import {sendMsg} from '../lib/messaging'
import {getAuthToken} from '../lib/auth'

export type FontSize = 'small' | 'medium' | 'large'
export type WidgetPosition = 'bottom-right' | 'bottom-left'
export type SummaryFormat = 'ask' | 'abstractive' | 'extractive' | 'thesis'
export type STTLanguage = 'ru-RU' | 'kk-KZ' | 'en-US'
export type AutoSendPauseMs = 800 | 1400 | 2000 | 3000

export interface AppearancePrefs {
    fontSize: FontSize
    widgetPosition: WidgetPosition
    glassOpacity: number
}

export interface VoicePrefs {
    handsFreeEnabled: boolean
    autoSendPauseMs: AutoSendPauseMs
    sttLanguage: STTLanguage
}

export interface DocumentPrefs {
    defaultSummaryFormat: SummaryFormat
    autoAnalyzeOnOpen: boolean
    showQuickActionHints: boolean
}

export interface UserPreferences {
    appearance: AppearancePrefs
    voice: VoicePrefs
    documents: DocumentPrefs
}

export interface LLMSettings {
    generativeUrl: string
    generativeModel: string
    embeddingUrl: string
    embeddingModel: string
    temperature: number
    maxTokens: number
    timeout: number
    maxRetries: number
}

export interface AgentSettings {
    maxIterations: number
    maxContextMessages: number
    timeout: number
    maxRetries: number
    enableTracing: boolean
    logLevel: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
}

export interface RAGSettings {
    chunkSize: number
    chunkOverlap: number
    batchSize: number
    embeddingBatchSize: number
}

export interface EDMSSettings {
    baseUrl: string
    timeout: number
    apiVersion: string
}

export interface TechSettings {
    llm: LLMSettings
    agent: AgentSettings
    rag: RAGSettings
    edms: EDMSSettings
}

export interface AllSettings {
    user: UserPreferences
    tech: TechSettings
}

export type SettingsTab = 'appearance' | 'voice' | 'documents' | 'llm' | 'agent' | 'rag' | 'edms'
export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error'

export const DEFAULT_USER_PREFS: UserPreferences = {
    appearance: {fontSize: 'medium', widgetPosition: 'bottom-right', glassOpacity: 0.15},
    voice: {handsFreeEnabled: false, autoSendPauseMs: 1400, sttLanguage: 'ru-RU'},
    documents: {defaultSummaryFormat: 'ask', autoAnalyzeOnOpen: false, showQuickActionHints: true},
}
export const DEFAULT_TECH: TechSettings = {
    llm: {
        generativeUrl: import.meta.env.VITE_LLM_GENERATIVE_URL,
        generativeModel: import.meta.env.VITE_LLM_GENERATIVE_MODEL,
        embeddingUrl: import.meta.env.VITE_LLM_EMBEDDING_URL,
        embeddingModel: import.meta.env.VITE_LLM_EMBEDDING_MODEL,
        temperature: Number(import.meta.env.VITE_LLM_TEMPERATURE),
        maxTokens: Number(import.meta.env.VITE_LLM_MAX_TOKENS),
        timeout: Number(import.meta.env.VITE_LLM_TIMEOUT),
        maxRetries: Number(import.meta.env.VITE_LLM_MAX_RETRIES),
    },
    agent: {
        maxIterations: Number(import.meta.env.VITE_AGENT_MAX_ITERATIONS),
        maxContextMessages: Number(import.meta.env.VITE_AGENT_MAX_CONTEXT_MESSAGES),
        timeout: Number(import.meta.env.VITE_AGENT_TIMEOUT),
        maxRetries: Number(import.meta.env.VITE_AGENT_MAX_RETRIES),
        enableTracing: import.meta.env.VITE_AGENT_ENABLE_TRACING === 'true',
        logLevel: import.meta.env.VITE_AGENT_LOG_LEVEL as AgentSettings['logLevel'],
    },
    rag: {
        chunkSize: Number(import.meta.env.VITE_RAG_CHUNK_SIZE),
        chunkOverlap: Number(import.meta.env.VITE_RAG_CHUNK_OVERLAP),
        batchSize: Number(import.meta.env.VITE_RAG_BATCH_SIZE),
        embeddingBatchSize: Number(import.meta.env.VITE_RAG_EMBEDDING_BATCH_SIZE),
    },
    edms: {
        baseUrl: import.meta.env.VITE_EDMS_BASE_URL,
        timeout: Number(import.meta.env.VITE_EDMS_TIMEOUT),
        apiVersion: import.meta.env.VITE_EDMS_API_VERSION,
    },
}

const USER_KEY = 'edmsUserPreferences'
const TECH_KEY = 'edmsTechSettingsCache'

function techFromBackend(d: Record<string, any>): TechSettings {
    const l = d.llm ?? {};
    const a = d.agent ?? {}
    const r = d.rag ?? {};
    const e = d.edms ?? {}
    const D = DEFAULT_TECH
    return {
        llm: {
            generativeUrl: l.generative_url ?? D.llm.generativeUrl,
            generativeModel: l.generative_model ?? D.llm.generativeModel,
            embeddingUrl: l.embedding_url ?? D.llm.embeddingUrl,
            embeddingModel: l.embedding_model ?? D.llm.embeddingModel,
            temperature: l.temperature ?? D.llm.temperature,
            maxTokens: l.max_tokens ?? D.llm.maxTokens,
            timeout: l.timeout ?? D.llm.timeout,
            maxRetries: l.max_retries ?? D.llm.maxRetries
        },
        agent: {
            maxIterations: a.max_iterations ?? D.agent.maxIterations,
            maxContextMessages: a.max_context_messages ?? D.agent.maxContextMessages,
            timeout: a.timeout ?? D.agent.timeout,
            maxRetries: a.max_retries ?? D.agent.maxRetries,
            enableTracing: a.enable_tracing ?? D.agent.enableTracing,
            logLevel: a.log_level ?? D.agent.logLevel
        },
        rag: {
            chunkSize: r.chunk_size ?? D.rag.chunkSize,
            chunkOverlap: r.chunk_overlap ?? D.rag.chunkOverlap,
            batchSize: r.batch_size ?? D.rag.batchSize,
            embeddingBatchSize: r.embedding_batch_size ?? D.rag.embeddingBatchSize
        },
        edms: {
            baseUrl: e.base_url ?? D.edms.baseUrl,
            timeout: e.timeout ?? D.edms.timeout,
            apiVersion: e.api_version ?? D.edms.apiVersion
        },
    }
}

function techToBackend(s: TechSettings): Record<string, any> {
    return {
        llm: {
            generative_url: s.llm.generativeUrl,
            generative_model: s.llm.generativeModel,
            embedding_url: s.llm.embeddingUrl,
            embedding_model: s.llm.embeddingModel,
            temperature: s.llm.temperature,
            max_tokens: s.llm.maxTokens,
            timeout: s.llm.timeout,
            max_retries: s.llm.maxRetries
        },
        agent: {
            max_iterations: s.agent.maxIterations,
            max_context_messages: s.agent.maxContextMessages,
            timeout: s.agent.timeout,
            max_retries: s.agent.maxRetries,
            enable_tracing: s.agent.enableTracing,
            log_level: s.agent.logLevel
        },
        rag: {
            chunk_size: s.rag.chunkSize,
            chunk_overlap: s.rag.chunkOverlap,
            batch_size: s.rag.batchSize,
            embedding_batch_size: s.rag.embeddingBatchSize
        },
        edms: {base_url: s.edms.baseUrl, timeout: s.edms.timeout, api_version: s.edms.apiVersion},
    }
}

async function chromeGet<T>(key: string): Promise<T | null> {
    return new Promise<T | null>((resolve) => {
        try {
            chrome.storage.local.get([key], (r) => resolve((r?.[key] as T) ?? null))
        } catch {
            resolve(null)
        }
    })
}

function chromeSet(key: string, value: unknown): void {
    try {
        chrome.storage.local.set({[key]: value})
    } catch {
    }
}

async function loadUserPrefs(): Promise<UserPreferences> {
    const s = await chromeGet<Partial<UserPreferences>>(USER_KEY)
    if (!s) return DEFAULT_USER_PREFS
    return {
        appearance: {...DEFAULT_USER_PREFS.appearance, ...(s.appearance ?? {})},
        voice: {...DEFAULT_USER_PREFS.voice, ...(s.voice ?? {})},
        documents: {...DEFAULT_USER_PREFS.documents, ...(s.documents ?? {})},
    }
}

async function loadTechCache(): Promise<TechSettings | null> {
    const s = await chromeGet<Partial<TechSettings>>(TECH_KEY)
    if (!s) return null
    return {
        llm: {...DEFAULT_TECH.llm, ...(s.llm ?? {})},
        agent: {...DEFAULT_TECH.agent, ...(s.agent ?? {})},
        rag: {...DEFAULT_TECH.rag, ...(s.rag ?? {})},
        edms: {...DEFAULT_TECH.edms, ...(s.edms ?? {})},
    }
}

export interface UseSettingsStoreReturn {
    draft: AllSettings
    savedUser: UserPreferences
    savedTech: TechSettings
    isDirty: boolean
    saveStatus: SaveStatus
    showTechnical: boolean
    isLoading: boolean
    isTechOffline: boolean
    updateUser: <K extends keyof UserPreferences>(group: K, patch: Partial<UserPreferences[K]>) => void
    updateTech: <K extends keyof TechSettings>(group: K, patch: Partial<TechSettings[K]>) => void
    saveAll: () => Promise<void>
    resetAll: () => void
    resetToDefaults: () => Promise<void>
    discardDraft: () => void
}

export function useSettingsStore(): UseSettingsStoreReturn {
    const [savedUser, setSavedUser] = useState<UserPreferences>(DEFAULT_USER_PREFS)
    const [savedTech, setSavedTech] = useState<TechSettings>(DEFAULT_TECH)
    const [draft, setDraft] = useState<AllSettings>({user: DEFAULT_USER_PREFS, tech: DEFAULT_TECH})
    const [isLoading, setIsLoading] = useState(true)
    const [isTechOffline, setTechOffline] = useState(false)
    const [showTechnical, setShowTechnical] = useState(false)
    const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle')
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const scheduleReset = useCallback((ms: number) => {
        if (timerRef.current) clearTimeout(timerRef.current)
        timerRef.current = setTimeout(() => setSaveStatus('idle'), ms)
    }, [])

    useEffect(() => {
        let alive = true
        ;(async () => {
            const [userPrefs, techCache] = await Promise.all([loadUserPrefs(), loadTechCache()])
            if (!alive) return
            setSavedUser(userPrefs)
            const techInit = techCache ?? DEFAULT_TECH
            setSavedTech(techInit)
            setDraft({user: userPrefs, tech: techInit})

            const [metaRes, techRes] = await Promise.allSettled([
                sendMsg<{ show_technical: boolean }>('fetchSettingsMeta', {}),
                sendMsg<Record<string, any>>('fetchSettings', {user_token: getAuthToken() ?? ''}),
            ])
            if (!alive) return

            if (metaRes.status === 'fulfilled') setShowTechnical(metaRes.value?.show_technical ?? false)

            if (techRes.status === 'fulfilled' && techRes.value) {
                const mapped = techFromBackend(techRes.value)
                setSavedTech(mapped)
                setDraft(prev => ({...prev, tech: mapped}))
                chromeSet(TECH_KEY, mapped)
                setTechOffline(false)
            } else {
                setTechOffline(true)
            }
            setIsLoading(false)
        })()
        return () => {
            alive = false
        }
    }, [])

    useEffect(() => () => {
        if (timerRef.current) clearTimeout(timerRef.current)
    }, [])

    const updateUser = useCallback(
        <K extends keyof UserPreferences>(group: K, patch: Partial<UserPreferences[K]>) => {
            setDraft(prev => ({...prev, user: {...prev.user, [group]: {...prev.user[group], ...patch}}}))
        }, [],
    )
    const updateTech = useCallback(
        <K extends keyof TechSettings>(group: K, patch: Partial<TechSettings[K]>) => {
            setDraft(prev => ({...prev, tech: {...prev.tech, [group]: {...prev.tech[group], ...patch}}}))
        }, [],
    )

    const saveAll = useCallback(async () => {
        setSaveStatus('saving')
        try {
            chromeSet(USER_KEY, draft.user)
            setSavedUser(draft.user)
            if (showTechnical) {
                const updated = await sendMsg<Record<string, any>>('updateSettings', {
                    user_token: getAuthToken() ?? '',
                    settings: techToBackend(draft.tech)
                })
                const mapped = techFromBackend(updated)
                setSavedTech(mapped)
                setDraft(prev => ({...prev, tech: mapped}))
                chromeSet(TECH_KEY, mapped)
            } else {
                setSavedTech(draft.tech)
            }
            setSaveStatus('saved')
            scheduleReset(2500)
        } catch {
            setSaveStatus('error')
            scheduleReset(3000)
        }
    }, [draft, showTechnical, scheduleReset])

    const resetAll = useCallback(() => setDraft({user: DEFAULT_USER_PREFS, tech: DEFAULT_TECH}), [])

    const resetToDefaults = useCallback(async () => {
        setSaveStatus('saving')
        try {
            await sendMsg<void>('resetSettings', {user_token: getAuthToken() ?? ''})
            const updated = await sendMsg<Record<string, any>>('fetchSettings', {user_token: getAuthToken() ?? ''})
            if (updated) {
                const mapped = techFromBackend(updated)
                setSavedTech(mapped)
                setDraft(prev => ({...prev, user: DEFAULT_USER_PREFS, tech: mapped}))
                chromeSet(USER_KEY, DEFAULT_USER_PREFS)
                chromeSet(TECH_KEY, mapped)
            } else {
                setDraft({user: DEFAULT_USER_PREFS, tech: DEFAULT_TECH})
            }
            setSaveStatus('saved')
            scheduleReset(2500)
        } catch {
            setSaveStatus('error')
            scheduleReset(3000)
        }
    }, [scheduleReset])

    const discardDraft = useCallback(() => setDraft({user: savedUser, tech: savedTech}), [savedUser, savedTech])

    const isDirty =
        JSON.stringify(draft.user) !== JSON.stringify(savedUser) ||
        JSON.stringify(draft.tech) !== JSON.stringify(savedTech)

    return {
        draft,
        savedUser,
        savedTech,
        isDirty,
        saveStatus,
        showTechnical,
        isLoading,
        isTechOffline,
        updateUser,
        updateTech,
        saveAll,
        resetAll,
        resetToDefaults,
        discardDraft
    }
}