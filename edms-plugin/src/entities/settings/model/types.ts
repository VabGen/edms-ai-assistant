import { z } from 'zod'

export const FontSizeSchema = z.enum(['small', 'medium', 'large'])
export type FontSize = z.infer<typeof FontSizeSchema>

export const WidgetPositionSchema = z.enum(['bottom-right', 'bottom-left'])
export type WidgetPosition = z.infer<typeof WidgetPositionSchema>

export const SummaryFormatSchema = z.enum(['ask', 'abstractive', 'extractive', 'thesis'])
export type SummaryFormat = z.infer<typeof SummaryFormatSchema>

export const STTLanguageSchema = z.enum(['ru-RU', 'kk-KZ', 'en-US'])
export type STTLanguage = z.infer<typeof STTLanguageSchema>

export const AutoSendPauseMsSchema = z.union([
  z.literal(800),
  z.literal(1400),
  z.literal(2000),
  z.literal(3000),
])
export type AutoSendPauseMs = z.infer<typeof AutoSendPauseMsSchema>

export const AppearancePrefsSchema = z.object({
  fontSize: FontSizeSchema,
  widgetPosition: WidgetPositionSchema,
  glassOpacity: z.number().min(0).max(0.5),
})
export type AppearancePrefs = z.infer<typeof AppearancePrefsSchema>

export const VoicePrefsSchema = z.object({
  handsFreeEnabled: z.boolean(),
  autoSendPauseMs: AutoSendPauseMsSchema,
  sttLanguage: STTLanguageSchema,
})
export type VoicePrefs = z.infer<typeof VoicePrefsSchema>

export const DocumentPrefsSchema = z.object({
  defaultSummaryFormat: SummaryFormatSchema,
  autoAnalyzeOnOpen: z.boolean(),
  showQuickActionHints: z.boolean(),
})
export type DocumentPrefs = z.infer<typeof DocumentPrefsSchema>

export const UserPreferencesSchema = z.object({
  appearance: AppearancePrefsSchema,
  voice: VoicePrefsSchema,
  documents: DocumentPrefsSchema,
})
export type UserPreferences = z.infer<typeof UserPreferencesSchema>

export const LLMSettingsSchema = z.object({
  generativeUrl: z.string().url(),
  generativeModel: z.string().min(1),
  embeddingUrl: z.string().url(),
  embeddingModel: z.string().min(1),
  temperature: z.number().min(0).max(2),
  maxTokens: z.number().int().positive(),
  timeout: z.number().int().positive(),
  maxRetries: z.number().int().min(0),
})
export type LLMSettings = z.infer<typeof LLMSettingsSchema>

export const AgentSettingsSchema = z.object({
  maxIterations: z.number().int().positive(),
  maxContextMessages: z.number().int().positive(),
  timeout: z.number().int().positive(),
  maxRetries: z.number().int().min(0),
  enableTracing: z.boolean(),
  logLevel: z.enum(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
})
export type AgentSettings = z.infer<typeof AgentSettingsSchema>

export const RAGSettingsSchema = z.object({
  chunkSize: z.number().int().positive(),
  chunkOverlap: z.number().int().min(0),
  batchSize: z.number().int().positive(),
  embeddingBatchSize: z.number().int().positive(),
})
export type RAGSettings = z.infer<typeof RAGSettingsSchema>

export const EDMSSettingsSchema = z.object({
  baseUrl: z.string().url(),
  timeout: z.number().int().positive(),
  apiVersion: z.string().min(1),
})
export type EDMSSettings = z.infer<typeof EDMSSettingsSchema>

export const TechSettingsSchema = z.object({
  llm: LLMSettingsSchema,
  agent: AgentSettingsSchema,
  rag: RAGSettingsSchema,
  edms: EDMSSettingsSchema,
})
export type TechSettings = z.infer<typeof TechSettingsSchema>

export const AllSettingsSchema = z.object({
  user: UserPreferencesSchema,
  tech: TechSettingsSchema,
})
export type AllSettings = z.infer<typeof AllSettingsSchema>

export type SettingsTab = 'appearance' | 'voice' | 'documents' | 'llm' | 'agent' | 'rag' | 'edms'
export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error'
