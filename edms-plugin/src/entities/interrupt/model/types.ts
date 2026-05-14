import { z } from 'zod'

export const INTERRUPT_SCHEMA_VERSION = 1 as const

const InterruptOptionSchema = z.object({
  id: z.string(),
  label: z.string(),
  description: z.string().nullable().optional(),
  metadata: z.record(z.unknown()).nullable().optional(),
})
export type InterruptOption = z.infer<typeof InterruptOptionSchema>

const InterruptCardItemSchema = z.object({
  id: z.string(),
  label: z.string(),
  description: z.string().nullable().optional(),
})
export type InterruptCardItem = z.infer<typeof InterruptCardItemSchema>

const DisambiguationInterruptSchema = z.object({
  schema_version: z.literal(INTERRUPT_SCHEMA_VERSION),
  kind: z.literal('disambiguation'),
  entity_type: z.string(),
  prompt: z.string(),
  options: z.array(InterruptOptionSchema),
  multiple: z.boolean(),
  search_term: z.string().nullable().optional(),
})

const ConfirmationInterruptSchema = z.object({
  schema_version: z.literal(INTERRUPT_SCHEMA_VERSION),
  kind: z.literal('confirmation'),
  prompt: z.string(),
  danger: z.boolean(),
  confirm_label: z.string(),
  cancel_label: z.string(),
})

const TextInputInterruptSchema = z.object({
  schema_version: z.literal(INTERRUPT_SCHEMA_VERSION),
  kind: z.literal('text_input'),
  prompt: z.string(),
  placeholder: z.string().nullable().optional(),
  secret: z.boolean(),
  validator_regex: z.string().nullable().optional(),
})

const SelectInterruptSchema = z.object({
  schema_version: z.literal(INTERRUPT_SCHEMA_VERSION),
  kind: z.literal('select'),
  prompt: z.string(),
  options: z.array(InterruptOptionSchema),
  default: z.string().nullable().optional(),
})

const CardSelectInterruptSchema = z.object({
  schema_version: z.literal(INTERRUPT_SCHEMA_VERSION).optional(),
  kind: z.literal('card_select'),
  prompt: z.string(),
  cards: z.array(InterruptCardItemSchema),
  multiple: z.boolean(),
  layout: z.enum(['list', 'grid']).optional(),
})

const GenericSelectInterruptSchema = z.object({
  schema_version: z.literal(INTERRUPT_SCHEMA_VERSION).optional(),
  kind: z.string(),
  prompt: z.string(),
  options: z.array(InterruptOptionSchema),
  multiple: z.boolean().optional(),
  entity_type: z.string().optional(),
})

export const InterruptPayloadSchema = z.discriminatedUnion('kind', [
  DisambiguationInterruptSchema,
  ConfirmationInterruptSchema,
  TextInputInterruptSchema,
  SelectInterruptSchema,
  CardSelectInterruptSchema,
])

export type DisambiguationInterrupt = z.infer<typeof DisambiguationInterruptSchema>
export type ConfirmationInterrupt = z.infer<typeof ConfirmationInterruptSchema>
export type TextInputInterrupt = z.infer<typeof TextInputInterruptSchema>
export type SelectInterrupt = z.infer<typeof SelectInterruptSchema>
export type CardSelectInterrupt = z.infer<typeof CardSelectInterruptSchema>
export type GenericSelectInterrupt = z.infer<typeof GenericSelectInterruptSchema>
export type InterruptPayload = z.infer<typeof InterruptPayloadSchema>

export const DisambiguationResumeSchema = z.object({
  kind: z.literal('disambiguation'),
  selected_ids: z.tuple([z.string()]).rest(z.string()),
})

export const ConfirmationResumeSchema = z.object({
  kind: z.literal('confirmation'),
  confirmed: z.boolean(),
})

export const TextInputResumeSchema = z.object({
  kind: z.literal('text_input'),
  value: z.string(),
})

export const SelectResumeSchema = z.object({
  kind: z.literal('select'),
  selected_id: z.string(),
})

export const CardSelectResumeSchema = z.object({
  kind: z.literal('card_select'),
  selected_ids: z.tuple([z.string()]).rest(z.string()),
})

export const AbortResumeSchema = z.object({
  kind: z.literal('__abort__'),
  reason: z.string().nullable().optional(),
})

export const ResumeValueSchema = z.discriminatedUnion('kind', [
  DisambiguationResumeSchema,
  ConfirmationResumeSchema,
  TextInputResumeSchema,
  SelectResumeSchema,
  CardSelectResumeSchema,
  AbortResumeSchema,
])
export type ResumeValue = z.infer<typeof ResumeValueSchema>

export const InterruptEventSchema = z.object({
  interrupt_id: z.string().nullable(),
  thread_id: z.string(),
  payload: InterruptPayloadSchema,
})
export type InterruptEvent = z.infer<typeof InterruptEventSchema>
