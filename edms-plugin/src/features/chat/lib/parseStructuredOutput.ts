import type { StructuredOutput } from '@/entities/message/model/types'

export function parseStructuredOutput(content: string): StructuredOutput | null {
    if (typeof content !== 'string') return null

    let parsed: any = null

    try {
        parsed = JSON.parse(content)
    } catch {
        const m = content.match(/```json\s*([\s\S]*?)\s*```/)
            ?? content.match(/```\s*([\s\S]*?)\s*```/)
        if (m) {
            try {
                parsed = JSON.parse(m[1] ?? '')
            } catch {
            }
        }
    }

    if (!parsed || typeof parsed !== 'object') return null

    // Hide internal tool calls
    if (parsed.tool_use || parsed.tool_calls || parsed.action === 'call_tool') return null

    if ('overall' in parsed && 'fields' in parsed)
        return {type: 'compliance', data: parsed}
    if ('main_argument' in parsed && 'sections' in parsed)
        return {type: 'thesis', data: parsed}
    if ('key_themes' in parsed && 'summary' in parsed && !('headline' in parsed))
        return {type: 'abstractive', data: parsed}
    if ('facts' in parsed && 'document_summary' in parsed)
        return {type: 'extractive', data: parsed}
    if ('action_items' in parsed)
        return {type: 'action_items', data: parsed}
    if ('headline' in parsed && 'bullets' in parsed)
        return {type: 'executive', data: parsed}
    if ('document_type' in parsed && 'sections' in parsed && !('main_argument' in parsed))
        return {type: 'detailed_notes', data: parsed}
    if ('detected_language' in parsed && 'summary_language' in parsed)
        return {type: 'multilingual', data: parsed}

    return null
}
