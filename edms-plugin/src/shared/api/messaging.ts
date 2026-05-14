import { defineExtensionMessaging, ProtocolWithReturn } from '@webext-core/messaging'
import type { ResumeValue } from '@entities/interrupt/model/types'
import type { TechSettings } from '@entities/settings/model/types'

export interface ApiResponse {
  success: boolean
  data?: unknown
  error?: string
}

export interface HistoryMessage {
  type: 'human' | 'ai'
  content: string
}

export interface HistoryResponse {
  messages: HistoryMessage[]
  thread_id: string
}

export interface UploadPayload {
  user_token: string
  thread_id: string
  file_data: string
  file_name: string
  context_ui_id: string | null
}

export interface UploadResponse {
  success: boolean
  file_path?: string
  error?: string
}

export interface SummarizePayload {
  message: string
  user_token: string
  context_ui_id: string | null
  file_path: string | null
  preferred_summary_format?: string
}

export interface SummarizeResponse {
  success: boolean
  data?: {
    response?: string
    metadata?: {
      cache_file_identifier?: string | null
      cache_summary_type?: string | null
      cache_context_ui_id?: string | null
    }
  }
  error?: string
}

export interface StreamStartPayload {
  message: string
  user_token: string
  thread_id: string
  context_ui_id: string | null
  file_path: string | null
  resume_value: ResumeValue | null
  requestId: string
}

export interface StreamAbortPayload {
  requestId: string
}

export interface GetHistoryPayload {
  user_token: string
  thread_id: string
}

export interface NavigatePayload {
  url: string
  newTab?: boolean
}

export interface DeleteCachePayload {
  user_token: string
  thread_id: string
  file_identifier: string | null
  summary_type: string | null
  context_id: string | null
  file_path: string | null
}

export interface RefreshDocumentPayload {
  user_token: string
  doc_id: string
}

export interface UpdateSettingsPayload {
  user_token: string
  settings: Record<string, unknown>
}

export type BackendSettingsResponse = Record<string, unknown>

export const { sendMessage, onMessage } = defineExtensionMessaging<{
  summarizeDocument: ProtocolWithReturn<SummarizePayload, SummarizeResponse>
  uploadFile: ProtocolWithReturn<UploadPayload, UploadResponse>
  getChatHistory: ProtocolWithReturn<GetHistoryPayload, HistoryResponse>
  createNewChat: ProtocolWithReturn<{ user_token: string }, { thread_id: string }>
  navigateTo: ProtocolWithReturn<NavigatePayload, ApiResponse>
  deleteCache: ProtocolWithReturn<DeleteCachePayload, ApiResponse>
  refreshDocument: ProtocolWithReturn<RefreshDocumentPayload, ApiResponse>
  fetchSettingsMeta: ProtocolWithReturn<undefined, { show_technical: boolean }>
  fetchSettings: ProtocolWithReturn<{ user_token: string }, BackendSettingsResponse>
  updateSettings: ProtocolWithReturn<UpdateSettingsPayload, BackendSettingsResponse>
  resetSettings: ProtocolWithReturn<{ user_token: string }, void>
  reloadActiveTab: ProtocolWithReturn<undefined, { success: boolean }>
  abortRequest: ProtocolWithReturn<StreamAbortPayload, void>
}>()
