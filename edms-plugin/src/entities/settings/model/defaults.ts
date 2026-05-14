import type { UserPreferences, TechSettings } from './types'

export const DEFAULT_USER_PREFS = {
  appearance: {
    fontSize: 'medium',
    widgetPosition: 'bottom-right',
    glassOpacity: 0.15,
  },
  voice: {
    handsFreeEnabled: false,
    autoSendPauseMs: 1400,
    sttLanguage: 'ru-RU',
  },
  documents: {
    defaultSummaryFormat: 'ask',
    autoAnalyzeOnOpen: false,
    showQuickActionHints: true,
  },
} as const satisfies UserPreferences

export const DEFAULT_TECH = {
  llm: {
    generativeUrl: import.meta.env.VITE_LLM_GENERATIVE_URL as string,
    generativeModel: import.meta.env.VITE_LLM_GENERATIVE_MODEL as string,
    embeddingUrl: import.meta.env.VITE_LLM_EMBEDDING_URL as string,
    embeddingModel: import.meta.env.VITE_LLM_EMBEDDING_MODEL as string,
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
    logLevel: (import.meta.env.VITE_AGENT_LOG_LEVEL ?? 'INFO') as AgentLogLevel,
  },
  rag: {
    chunkSize: Number(import.meta.env.VITE_RAG_CHUNK_SIZE),
    chunkOverlap: Number(import.meta.env.VITE_RAG_CHUNK_OVERLAP),
    batchSize: Number(import.meta.env.VITE_RAG_BATCH_SIZE),
    embeddingBatchSize: Number(import.meta.env.VITE_RAG_EMBEDDING_BATCH_SIZE),
  },
  edms: {
    baseUrl: import.meta.env.VITE_EDMS_BASE_URL as string,
    timeout: Number(import.meta.env.VITE_EDMS_TIMEOUT),
    apiVersion: (import.meta.env.VITE_EDMS_API_VERSION ?? 'v1') as string,
  },
} satisfies TechSettings

type AgentLogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
