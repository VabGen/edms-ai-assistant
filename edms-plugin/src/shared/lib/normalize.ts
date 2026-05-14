const UNICODE_DASHES =
  /[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00AD\uFE58\uFE63\uFF0D]/g

export function normalizeUuid(raw: string): string {
  return raw.replace(UNICODE_DASHES, '-').trim()
}
