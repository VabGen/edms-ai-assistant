import { useCallback, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { storage } from 'wxt/storage'
import { sendMessage } from '@shared/api/messaging'
import { STORAGE_KEYS } from '@shared/storage/keys'
import {
  DEFAULT_USER_PREFS,
  DEFAULT_TECH,
} from '@entities/settings/model/defaults'
import type {
  UserPreferences,
  TechSettings,
  AllSettings,
  SaveStatus,
} from '@entities/settings/model/types'

const userPrefsStorage = storage.defineItem<UserPreferences>(
  STORAGE_KEYS.userPreferences,
  { fallback: DEFAULT_USER_PREFS },
)

const techCacheStorage = storage.defineItem<TechSettings>(
  STORAGE_KEYS.techSettingsCache,
  { fallback: DEFAULT_TECH },
)

function techFromBackend(d: Record<string, unknown>): TechSettings {
  const l = (d['llm'] ?? {}) as Record<string, unknown>
  const a = (d['agent'] ?? {}) as Record<string, unknown>
  const r = (d['rag'] ?? {}) as Record<string, unknown>
  const e = (d['edms'] ?? {}) as Record<string, unknown>
  const D = DEFAULT_TECH
  return {
    llm: {
      generativeUrl: (l['generative_url'] as string | undefined) ?? D.llm.generativeUrl,
      generativeModel: (l['generative_model'] as string | undefined) ?? D.llm.generativeModel,
      embeddingUrl: (l['embedding_url'] as string | undefined) ?? D.llm.embeddingUrl,
      embeddingModel: (l['embedding_model'] as string | undefined) ?? D.llm.embeddingModel,
      temperature: (l['temperature'] as number | undefined) ?? D.llm.temperature,
      maxTokens: (l['max_tokens'] as number | undefined) ?? D.llm.maxTokens,
      timeout: (l['timeout'] as number | undefined) ?? D.llm.timeout,
      maxRetries: (l['max_retries'] as number | undefined) ?? D.llm.maxRetries,
    },
    agent: {
      maxIterations: (a['max_iterations'] as number | undefined) ?? D.agent.maxIterations,
      maxContextMessages: (a['max_context_messages'] as number | undefined) ?? D.agent.maxContextMessages,
      timeout: (a['timeout'] as number | undefined) ?? D.agent.timeout,
      maxRetries: (a['max_retries'] as number | undefined) ?? D.agent.maxRetries,
      enableTracing: (a['enable_tracing'] as boolean | undefined) ?? D.agent.enableTracing,
      logLevel: (a['log_level'] as TechSettings['agent']['logLevel'] | undefined) ?? D.agent.logLevel,
    },
    rag: {
      chunkSize: (r['chunk_size'] as number | undefined) ?? D.rag.chunkSize,
      chunkOverlap: (r['chunk_overlap'] as number | undefined) ?? D.rag.chunkOverlap,
      batchSize: (r['batch_size'] as number | undefined) ?? D.rag.batchSize,
      embeddingBatchSize: (r['embedding_batch_size'] as number | undefined) ?? D.rag.embeddingBatchSize,
    },
    edms: {
      baseUrl: (e['base_url'] as string | undefined) ?? D.edms.baseUrl,
      timeout: (e['timeout'] as number | undefined) ?? D.edms.timeout,
      apiVersion: (e['api_version'] as string | undefined) ?? D.edms.apiVersion,
    },
  }
}

export function techToBackend(s: TechSettings): Record<string, unknown> {
  return {
    llm: {
      generative_url: s.llm.generativeUrl,
      generative_model: s.llm.generativeModel,
      embedding_url: s.llm.embeddingUrl,
      embedding_model: s.llm.embeddingModel,
      temperature: s.llm.temperature,
      max_tokens: s.llm.maxTokens,
      timeout: s.llm.timeout,
      max_retries: s.llm.maxRetries,
    },
    agent: {
      max_iterations: s.agent.maxIterations,
      max_context_messages: s.agent.maxContextMessages,
      timeout: s.agent.timeout,
      max_retries: s.agent.maxRetries,
      enable_tracing: s.agent.enableTracing,
      log_level: s.agent.logLevel,
    },
    rag: {
      chunk_size: s.rag.chunkSize,
      chunk_overlap: s.rag.chunkOverlap,
      batch_size: s.rag.batchSize,
      embedding_batch_size: s.rag.embeddingBatchSize,
    },
    edms: {
      base_url: s.edms.baseUrl,
      timeout: s.edms.timeout,
      api_version: s.edms.apiVersion,
    },
  }
}

export function useSettingsMetaQuery() {
  return useQuery({
    queryKey: ['settings-meta'] as const,
    queryFn: () => sendMessage('fetchSettingsMeta', undefined),
    staleTime: 10 * 60 * 1000,
    retry: 1,
  })
}

export function useSettingsQuery(userToken: string) {
  return useQuery({
    queryKey: ['settings', userToken] as const,
    queryFn: async () => {
      const [userPrefs, techCache, backendData] = await Promise.all([
        userPrefsStorage.getValue(),
        techCacheStorage.getValue(),
        sendMessage('fetchSettings', { user_token: userToken }).catch(() => null),
      ])
      const tech = backendData ? techFromBackend(backendData) : (techCache ?? DEFAULT_TECH)
      if (backendData) {
        await techCacheStorage.setValue(tech)
      }
      return {
        user: userPrefs ?? DEFAULT_USER_PREFS,
        tech,
        isTechOffline: backendData === null,
      }
    },
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })
}

export function useUpdateSettingsMutation(userToken: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (settings: AllSettings) => {
      await userPrefsStorage.setValue(settings.user)
      const backendResult = await sendMessage('updateSettings', {
        user_token: userToken,
        settings: techToBackend(settings.tech),
      })
      const mapped = techFromBackend(backendResult)
      await techCacheStorage.setValue(mapped)
      return { ...settings, tech: mapped }
    },
    onSuccess: (data) => {
      qc.setQueryData(['settings', userToken], (prev: typeof data | undefined) =>
        prev ? { ...prev, ...data } : data,
      )
    },
  })
}

export function useSaveUserPrefsMutation(userToken: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (prefs: UserPreferences) => {
      await userPrefsStorage.setValue(prefs)
      return prefs
    },
    onSuccess: (prefs) => {
      qc.setQueryData(
        ['settings', userToken],
        (prev: { user: UserPreferences; tech: TechSettings; isTechOffline: boolean } | undefined) =>
          prev ? { ...prev, user: prefs } : undefined,
      )
    },
  })
}

export function useResetSettingsMutation(userToken: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => {
      await sendMessage('resetSettings', { user_token: userToken })
      const backendData = await sendMessage('fetchSettings', { user_token: userToken })
      const tech = techFromBackend(backendData)
      await Promise.all([
        userPrefsStorage.setValue(DEFAULT_USER_PREFS),
        techCacheStorage.setValue(tech),
      ])
      return { user: DEFAULT_USER_PREFS, tech, isTechOffline: false }
    },
    onSuccess: (data) => {
      qc.setQueryData(['settings', userToken], data)
    },
  })
}

export interface UseDraftSettingsReturn {
  draft: AllSettings
  isDirty: boolean
  saveStatus: SaveStatus
  updateUser: <K extends keyof UserPreferences>(group: K, patch: Partial<UserPreferences[K]>) => void
  updateTech: <K extends keyof TechSettings>(group: K, patch: Partial<TechSettings[K]>) => void
  commit: () => void
  discard: () => void
  setSaveStatus: (s: SaveStatus) => void
}

export function useDraftSettings(saved: AllSettings): UseDraftSettingsReturn {
  const draftRef = useRef<AllSettings>(saved)
  const savedRef = useRef<AllSettings>(saved)
  const saveStatusRef = useRef<SaveStatus>('idle')

  const updateUser = useCallback(
    <K extends keyof UserPreferences>(group: K, patch: Partial<UserPreferences[K]>) => {
      draftRef.current = {
        ...draftRef.current,
        user: {
          ...draftRef.current.user,
          [group]: { ...draftRef.current.user[group], ...patch },
        },
      }
    },
    [],
  )

  const updateTech = useCallback(
    <K extends keyof TechSettings>(group: K, patch: Partial<TechSettings[K]>) => {
      draftRef.current = {
        ...draftRef.current,
        tech: {
          ...draftRef.current.tech,
          [group]: { ...draftRef.current.tech[group], ...patch },
        },
      }
    },
    [],
  )

  const commit = useCallback(() => {
    savedRef.current = draftRef.current
  }, [])

  const discard = useCallback(() => {
    draftRef.current = savedRef.current
  }, [])

  const setSaveStatus = useCallback((s: SaveStatus) => {
    saveStatusRef.current = s
  }, [])

  const isDirty =
    JSON.stringify(draftRef.current.user) !== JSON.stringify(savedRef.current.user) ||
    JSON.stringify(draftRef.current.tech) !== JSON.stringify(savedRef.current.tech)

  return {
    draft: draftRef.current,
    isDirty,
    saveStatus: saveStatusRef.current,
    updateUser,
    updateTech,
    commit,
    discard,
    setSaveStatus,
  }
}
