import { z } from 'zod'

export const MessageRoleSchema = z.enum(['user', 'assistant'])
export type MessageRole = z.infer<typeof MessageRoleSchema>

export const AttachedFileSchema = z.object({
  path: z.string(),
  name: z.string(),
})
export type AttachedFile = z.infer<typeof AttachedFileSchema>

export const ComplianceFieldSchema = z.object({
  field_key: z.string(),
  label: z.string(),
  status: z.enum(['ok', 'mismatch', 'missing', 'warning']),
  card_value: z.string().nullable().optional(),
  correct_value: z.string().nullable().optional(),
  comment: z.string().nullable().optional(),
})
export type ComplianceField = z.infer<typeof ComplianceFieldSchema>

export const ComplianceDataSchema = z.object({
  overall: z.enum(['ok', 'has_mismatches', 'cannot_verify']),
  fields: z.array(ComplianceFieldSchema),
  summary: z.string().optional(),
})
export type ComplianceData = z.infer<typeof ComplianceDataSchema>

export const RefreshMetaSchema = z.object({
  cache_file_identifier: z.string().nullable().optional(),
  cache_summary_type: z.string().nullable().optional(),
  cache_context_id: z.string().nullable().optional(),
  cache_file_path: z.string().nullable().optional(),
  doc_url: z.string().nullable().optional(),
})
export type RefreshMeta = z.infer<typeof RefreshMetaSchema>

export const ChatMessageSchema = z.object({
  id: z.string(),
  role: MessageRoleSchema,
  content: z.string(),
  timestamp: z.number(),
  isError: z.boolean().optional(),
  attachments: z.array(AttachedFileSchema).optional(),
  interrupt: z.unknown().optional(),
  compliance: ComplianceDataSchema.optional(),
  refreshMeta: RefreshMetaSchema.optional(),
})
export type ChatMessage = z.infer<typeof ChatMessageSchema>

export const ThreadSchema = z.object({
  id: z.string(),
  preview: z.string(),
  date: z.string(),
  messages: z.array(ChatMessageSchema).optional(),
})
export type Thread = z.infer<typeof ThreadSchema>
