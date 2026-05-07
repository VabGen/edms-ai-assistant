/// <reference types="wxt/browser" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_EDMS_FRONTEND_URL: string
  // LLM
  readonly VITE_LLM_GENERATIVE_URL: string
  readonly VITE_LLM_GENERATIVE_MODEL: string
  readonly VITE_LLM_EMBEDDING_URL: string
  readonly VITE_LLM_EMBEDDING_MODEL: string
  readonly VITE_LLM_TEMPERATURE: string
  readonly VITE_LLM_MAX_TOKENS: string
  readonly VITE_LLM_TIMEOUT: string
  readonly VITE_LLM_MAX_RETRIES: string
  // Agent
  readonly VITE_AGENT_MAX_ITERATIONS: string
  readonly VITE_AGENT_MAX_CONTEXT_MESSAGES: string
  readonly VITE_AGENT_TIMEOUT: string
  readonly VITE_AGENT_MAX_RETRIES: string
  readonly VITE_AGENT_ENABLE_TRACING: string
  readonly VITE_AGENT_LOG_LEVEL: string
  // RAG
  readonly VITE_RAG_CHUNK_SIZE: string
  readonly VITE_RAG_CHUNK_OVERLAP: string
  readonly VITE_RAG_BATCH_SIZE: string
  readonly VITE_RAG_EMBEDDING_BATCH_SIZE: string
  // EDMS backend
  readonly VITE_EDMS_BASE_URL: string
  readonly VITE_EDMS_TIMEOUT: string
  readonly VITE_EDMS_API_VERSION: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare interface WxtContentScriptContext {
  abort(reason?: string): void
  onInvalidated: (cb: () => void) => void
  isValid: boolean
}

declare interface BackgroundDefinition {
  type?: 'module'
  persistent?: boolean
  main(): void
}

declare interface ContentScriptDefinition {
  matches: string[]
  runAt?: 'document_start' | 'document_end' | 'document_idle'
  allFrames?: boolean
  cssInjectionMode?: 'manifest' | 'manual' | 'ui'
  main(ctx: WxtContentScriptContext): void | Promise<void>
}

declare function defineBackground(def: BackgroundDefinition): BackgroundDefinition
declare function defineContentScript(def: ContentScriptDefinition): ContentScriptDefinition

declare function createShadowRootUi<T>(
  ctx: WxtContentScriptContext,
  options: {
    name: string
    position: 'inline' | 'overlay' | 'modal'
    anchor?: string | Element | (() => Element | null)
    append?: 'first' | 'last' | 'replace' | ((anchor: Element, ui: Element) => void)
    onMount(container: HTMLElement): T
    onRemove?(mounted: T): void
  },
): Promise<{ mount(): void; remove(): void }>
