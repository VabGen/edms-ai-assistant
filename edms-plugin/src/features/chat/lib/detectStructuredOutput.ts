import type { StructuredOutput } from '@/entities/message/model/types'

/**
 * Detects if the message content contains structured output (JSON).
 * If the message is purely JSON (or JSON wrapped in markdown blocks) that matches
 * a known domain entity, it returns the structured data and a 'pure' flag.
 *
 * Technical tool calls (tool_use, action: call_tool, etc.) are always suppressed.
 */
export function detectStructuredOutput(content: string): {
    output: StructuredOutput | null,
    shouldHideText: boolean
} {
    if (typeof content !== 'string' || !content.trim()) {
        return { output: null, shouldHideText: false }
    }

    let parsed: any = null
    let isPure = false

    const trimmed = content.trim()

    // Attempt to parse as raw JSON first
    try {
        parsed = JSON.parse(trimmed)
        isPure = true
    } catch {
        // Look for JSON blocks
        const m = trimmed.match(/^```json\s*([\s\S]*?)\s*```$/i)
            ?? trimmed.match(/^```\s*([\s\S]*?)\s*```$/i)

        if (m) {
            try {
                parsed = JSON.parse(m[1] ?? '')
                isPure = true
            } catch {
                // Not valid JSON
            }
        } else {
            // Partial match (JSON block inside other text)
            const partialMatch = content.match(/```json\s*([\s\S]*?)\s*```/i)
                ?? content.match(/```\s*([\s\S]*?)\s*```/i)
            if (partialMatch) {
                try {
                    parsed = JSON.parse(partialMatch[1] ?? '')
                    isPure = false
                } catch {
                    // Not valid JSON
                }
            }
        }
    }

    if (!parsed || typeof parsed !== 'object') {
        return { output: null, shouldHideText: false }
    }

    // Always hide internal technical tool calls/results if they leaked into content
    const isTechnical = !!(parsed.tool_use || parsed.tool_calls || parsed.action === 'call_tool' || parsed.tool_name)
    if (isTechnical) {
        return { output: null, shouldHideText: isPure }
    }

    let output: StructuredOutput | null = null

    if ('overall' in parsed && 'fields' in parsed)
        output = {type: 'compliance', data: parsed}
    else if ('main_argument' in parsed && 'sections' in parsed)
        output = {type: 'thesis', data: parsed}
    else if ('key_themes' in parsed && 'summary' in parsed && !('headline' in parsed))
        output = {type: 'abstractive', data: parsed}
    else if ('facts' in parsed && 'document_summary' in parsed)
        output = {type: 'extractive', data: parsed}
    else if ('action_items' in parsed)
        output = {type: 'action_items', data: parsed}
    else if ('headline' in parsed && 'bullets' in parsed)
        output = {type: 'executive', data: parsed}
    else if ('document_type' in parsed && 'sections' in parsed && !('main_argument' in parsed))
        output = {type: 'detailed_notes', data: parsed}
    else if ('detected_language' in parsed && 'summary_language' in parsed)
        output = {type: 'multilingual', data: parsed}

    return {
        output,
        shouldHideText: isPure && (output !== null || isTechnical)
    }
}
