import { z } from 'zod'
import { InterruptEventSchema } from '@entities/interrupt/model/types'
import { ComplianceDataSchema } from '@entities/message/model/types'

const SseMessageEventSchema = z.object({
  kind: z.literal('message'),
  data: z.object({
    role: z.literal('assistant'),
    content: z.string(),
  }),
})

const SseInterruptEventSchema = z.object({
  kind: z.literal('interrupt'),
  data: InterruptEventSchema,
})

const SseUiComponentEventSchema = z.object({
  kind: z.literal('ui_component'),
  data: z.object({
    component_type: z.string(),
    payload: z.unknown(),
  }),
})

const SseDoneEventSchema = z.object({
  kind: z.literal('done'),
  data: z.object({
    thread_id: z.string(),
    paused: z.boolean(),
  }),
})

const SseErrorEventSchema = z.object({
  kind: z.literal('error'),
  data: z.object({
    code: z.string(),
    message: z.string(),
    thread_id: z.string().optional(),
  }),
})

export const SseEventSchema = z.discriminatedUnion('kind', [
  SseMessageEventSchema,
  SseInterruptEventSchema,
  SseUiComponentEventSchema,
  SseDoneEventSchema,
  SseErrorEventSchema,
])
export type SseEvent = z.infer<typeof SseEventSchema>
export type SseMessageEvent = z.infer<typeof SseMessageEventSchema>
export type SseInterruptEvent = z.infer<typeof SseInterruptEventSchema>
export type SseUiComponentEvent = z.infer<typeof SseUiComponentEventSchema>
export type SseDoneEvent = z.infer<typeof SseDoneEventSchema>
export type SseErrorEvent = z.infer<typeof SseErrorEventSchema>

export const ComplianceUiPayloadSchema = z.object({
  overall: z.enum(['ok', 'has_mismatches', 'cannot_verify']),
  fields: z.array(
    z.object({
      field_key: z.string(),
      label: z.string(),
      status: z.enum(['ok', 'mismatch', 'missing', 'warning']),
      card_value: z.string().nullable().optional(),
      correct_value: z.string().nullable().optional(),
      comment: z.string().nullable().optional(),
    }),
  ),
  summary: z.string().optional(),
})
export type ComplianceUiPayload = z.infer<typeof ComplianceUiPayloadSchema>

export function parseSseEvent(raw: unknown): SseEvent | null {
  const result = SseEventSchema.safeParse(raw)
  return result.success ? result.data : null
}

export function parseCompliancePayload(raw: unknown): ComplianceUiPayload | null {
  const result = ComplianceDataSchema.safeParse(raw)
  return result.success ? result.data : null
}
