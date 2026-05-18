const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

/** Извлекает UUID документа из текущего URL */
export function extractDocIdFromUrl(): string | null {
    try {
        return window.location.pathname.split('/').find(p => UUID_RE.test(p)) ?? null
    } catch {
        return null
    }
}

export function normalizeUuid(raw: string): string {
    return raw
        .replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00AD\uFE58\uFE63\uFF0D]/g, '-')
        .trim()
}

export function isValidUuid(raw: string): boolean {
    const normalized = normalizeUuid(raw)
    return UUID_RE.test(normalized)
}
