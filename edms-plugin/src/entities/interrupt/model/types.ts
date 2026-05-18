import {z} from 'zod'

export const INTERRUPT_SCHEMA_VERSION = 1 as const

const InterruptOptionSchema = z.object({
    id: z.string(),
    label: z.string(),
    description: z.string().nullable().optional(),
    metadata: z.record(z.unknown()).nullable().optional(),
})
export type InterruptOption = z.infer<typeof InterruptOptionSchema>

const InterruptCardSchema = z.object({
    id: z.string(),
    label: z.string(),
    description: z.string().nullable().optional(),
    image_url: z.string().nullable().optional(),
    badges: z.array(z.string()).default([]),
    primary_attrs: z.record(z.string()).default({}),
    metadata: z.record(z.unknown()).nullable().optional(),
})

const FileRefSchema = z.object({
    file_id: z.string(),
    file_name: z.string(),
    mime_type: z.string().nullable().optional(),
    size_bytes: z.number().nullable().optional(),
})
// Removed unused `export type FileRef`

const DisambiguationInterruptSchema = z.object({
    schema_version: z.literal(INTERRUPT_SCHEMA_VERSION).optional(),
    kind: z.literal('disambiguation'),
    entity_type: z.string().optional(),
    prompt: z.string(),
    options: z.array(InterruptOptionSchema),
    multiple: z.boolean().optional().default(false),
    search_term: z.string().nullable().optional(),
})

const ConfirmationInterruptSchema = z.object({
    schema_version: z.literal(INTERRUPT_SCHEMA_VERSION).optional(),
    kind: z.literal('confirmation'),
    prompt: z.string(),
    danger: z.boolean().optional().default(false),
    confirm_label: z.string().optional().default('Подтвердить'),
    cancel_label: z.string().optional().default('Отмена'),
})

const TextInputInterruptSchema = z.object({
    schema_version: z.literal(INTERRUPT_SCHEMA_VERSION).optional(),
    kind: z.literal('text_input'),
    prompt: z.string(),
    placeholder: z.string().nullable().optional(),
    secret: z.boolean().optional().default(false),
    validator_regex: z.string().nullable().optional(),
})

const SelectInterruptSchema = z.object({
    schema_version: z.literal(INTERRUPT_SCHEMA_VERSION).optional(),
    kind: z.literal('select'),
    prompt: z.string(),
    options: z.array(InterruptOptionSchema),
    default: z.string().nullable().optional(),
})

const CardSelectInterruptSchema = z.object({
    schema_version: z.literal(INTERRUPT_SCHEMA_VERSION).optional(),
    kind: z.literal('card_select'),
    prompt: z.string(),
    cards: z.array(InterruptCardSchema),
    multiple: z.boolean().optional().default(false),
    layout: z.enum(['list', 'grid']).optional(),
})

const FilePickerInterruptSchema = z.object({
    schema_version: z.literal(INTERRUPT_SCHEMA_VERSION).optional(),
    kind: z.literal('file_picker'),
    prompt: z.string(),
    accept_mime: z.array(z.string()).nullable().optional(),
    max_size_bytes: z.number().nullable().optional(),
    multiple: z.boolean().optional().default(false),
})

export const InterruptPayloadSchema = z.discriminatedUnion('kind', [
    DisambiguationInterruptSchema,
    ConfirmationInterruptSchema,
    TextInputInterruptSchema,
    SelectInterruptSchema,
    CardSelectInterruptSchema,
    FilePickerInterruptSchema,
])
export type InterruptPayload = z.infer<typeof InterruptPayloadSchema>

// ── Inbound resume values (frontend → tool) ───────────────────────────────

const DisambiguationResumeSchema = z.object({
    kind: z.literal('disambiguation'),
    selected_ids: z.array(z.string()).min(1),
})

const ConfirmationResumeSchema = z.object({
    kind: z.literal('confirmation'),
    confirmed: z.boolean(),
})

const TextInputResumeSchema = z.object({
    kind: z.literal('text_input'),
    value: z.string(),
})

const SelectResumeSchema = z.object({
    kind: z.literal('select'),
    selected_id: z.string(),
})

const CardSelectResumeSchema = z.object({
    kind: z.literal('card_select'),
    selected_ids: z.array(z.string()).min(1),
})

const FilePickerResumeSchema = z.object({
    kind: z.literal('file_picker'),
    file_refs: z.array(FileRefSchema).min(1),
})

const AbortResumeSchema = z.object({
    kind: z.literal('__abort__'),
    reason: z.string().nullable().optional(),
})

export const ResumeValueSchema = z.discriminatedUnion('kind', [
    DisambiguationResumeSchema,
    ConfirmationResumeSchema,
    TextInputResumeSchema,
    SelectResumeSchema,
    CardSelectResumeSchema,
    FilePickerResumeSchema,
    AbortResumeSchema,
])
export type ResumeValue = z.infer<typeof ResumeValueSchema>

export const MessageEventSchema = z.object({
    role: z.literal('assistant'),
    content: z.string(),
})
export type MessageEvent = z.infer<typeof MessageEventSchema>

export const DoneEventSchema = z.object({
    thread_id: z.string(),
    paused: z.boolean(),
})
export type DoneEvent = z.infer<typeof DoneEventSchema>

export const ErrorEventSchema = z.object({
    code: z.string(),
    message: z.string(),
    thread_id: z.string().optional(),
})
export type ErrorEvent = z.infer<typeof ErrorEventSchema>

export const InterruptEventSchema = z.object({
    interrupt_id: z.string().nullable(),
    thread_id: z.string(),
    payload: InterruptPayloadSchema,
})
export type InterruptEvent = z.infer<typeof InterruptEventSchema>