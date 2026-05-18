import {z} from 'zod'

const MessageRoleSchema = z.enum(['user', 'assistant'])

const AttachedFileSchema = z.object({
    path: z.string(),
    name: z.string(),
})

export const ComplianceFieldSchema = z.object({
    field_key: z.string(),
    label: z.string(),
    status: z.enum(['ok', 'mismatch', 'not_found', 'missing', 'warning']),
    card_value: z.string().nullable().optional(),
    file_value: z.string().nullable().optional(),
    correct_value: z.string().nullable().optional(),
    update_field: z.string().optional(),
    recommendation: z.string().nullable().optional(),
    comment: z.string().nullable().optional(),
})
export type ComplianceField = z.infer<typeof ComplianceFieldSchema>

export const ComplianceDataSchema = z.object({
    overall: z.enum(['ok', 'has_mismatches', 'cannot_verify']),
    fields: z.array(ComplianceFieldSchema),
    summary: z.string().optional(),
    document_id: z.string().nullable().optional(),
    stats: z.object({
        total: z.number().optional(),
        ok: z.number().optional(),
        mismatches: z.number().optional(),
        not_found: z.number().optional(),
    }).optional(),
    fix_hint: z.string().nullable().optional(),
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

export const ThesisPlanSchema = z.object({
    main_argument: z.string(),
    sections: z.array(z.object({
        title: z.string(),
        thesis: z.string(),
        points: z.array(z.object({
            claim: z.string(),
            evidence: z.string().nullable().optional(),
            sub_points: z.array(z.string()).optional(),
        })),
    })),
    conclusion: z.string(),
})
export type ThesisPlanData = z.infer<typeof ThesisPlanSchema>

export const AbstractiveSchema = z.object({
    summary: z.string(),
    key_themes: z.array(z.string()),
})
export type AbstractiveData = z.infer<typeof AbstractiveSchema>

export const ExtractiveSchema = z.object({
    facts: z.array(z.object({
        category: z.string(),
        label: z.string(),
        value: z.string(),
    })),
    document_summary: z.string(),
})
export type ExtractiveData = z.infer<typeof ExtractiveSchema>

export const ActionItemsSchema = z.object({
    action_items: z.array(z.object({
        task: z.string(),
        owner: z.string().nullable().optional(),
        deadline: z.string().nullable().optional(),
        priority: z.enum(['high', 'medium', 'low']),
        source_fragment: z.string().optional(),
        confidence: z.number().optional(),
    })),
    document_context: z.string(),
})
export type ActionItemsData = z.infer<typeof ActionItemsSchema>

export const ExecutiveSummarySchema = z.object({
    headline: z.string(),
    bullets: z.array(z.string()),
    recommendation: z.string().nullable().optional(),
})
export type ExecutiveSummaryData = z.infer<typeof ExecutiveSummarySchema>

export const DetailedNotesSchema = z.object({
    document_type: z.string(),
    sections: z.array(z.object({
        title: z.string(),
        content: z.string(),
        subsections: z.array(z.string()).optional(),
    })),
    key_entities: z.array(z.string()),
    date_range: z.string().nullable().optional(),
})
export type DetailedNotesData = z.infer<typeof DetailedNotesSchema>

export const MultilingualSchema = z.object({
    detected_language: z.string(),
    summary_language: z.string(),
    summary: z.string(),
    translation_notes: z.string().nullable().optional(),
})
export type MultilingualData = z.infer<typeof MultilingualSchema>

export type StructuredOutput =
    | { type: 'compliance'; data: ComplianceData }
    | { type: 'thesis'; data: ThesisPlanData }
    | { type: 'abstractive'; data: AbstractiveData }
    | { type: 'extractive'; data: ExtractiveData }
    | { type: 'action_items'; data: ActionItemsData }
    | { type: 'executive'; data: ExecutiveSummaryData }
    | { type: 'detailed_notes'; data: DetailedNotesData }
    | { type: 'multilingual'; data: MultilingualData }

export const ChatMessageSchema = z.object({
    id: z.string(),
    role: MessageRoleSchema,
    content: z.string(),
    timestamp: z.number(),
    isError: z.boolean().optional(),
    attachments: z.array(AttachedFileSchema).optional(),
    interrupt: z.unknown().nullable().optional(),
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