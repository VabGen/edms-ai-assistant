import {createContext, useContext} from 'react'
import {remarkLazyList} from '../plugins/remarkLazyList'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import dayjs from 'dayjs'
import {XCircle, AlertTriangle, CheckCircle, FileText} from 'lucide-react'

interface Props {
    content: string
    role: 'user' | 'assistant'
    timestamp: number
    isError?: boolean
    onAttachmentClick?: (fileName: string) => void
    onDocumentClick?: (documentId: string) => void
}

const AttachmentClickContext = createContext<((fileName: string) => void) | null>(null)
const DocumentClickContext = createContext<((documentId: string) => void) | null>(null)

interface ComplianceCheckData {
    summary: string
    overall: 'ok' | 'has_mismatches' | 'cannot_verify'
    fields: Array<{
        field_key: string
        label: string
        card_value: string
        file_value: string | null
        status: 'ok' | 'mismatch' | 'not_found'
        recommendation: string | null
    }>
}

function ComplianceCheckResult({data}: { data: ComplianceCheckData }) {
    const isError = data.overall === 'has_mismatches'
    const isWarning = data.overall === 'cannot_verify'

    const statusColor = isError ? '#ef4444' : (isWarning ? '#f59e0b' : '#10b981')
    const statusIcon = isError ? <XCircle size={18} color={statusColor}/> : (isWarning ?
        <AlertTriangle size={18} color={statusColor}/> : <CheckCircle size={18} color={statusColor}/>)
    const statusText = isError ? 'Найдены расхождения' : (isWarning ? 'Требуется проверка' : 'Проверка пройдена успешно')

    return (
        <div style={{
            background: '#ffffff',
            borderRadius: 12,
            border: '1px solid rgba(0,0,0,0.06)',
            overflow: 'hidden',
            fontSize: 13,
        }}>
            <div style={{
                padding: '12px 16px',
                background: isError ? 'rgba(239, 68, 68, 0.04)' : (isWarning ? 'rgba(245, 158, 11, 0.04)' : 'rgba(16, 185, 129, 0.04)'),
                borderBottom: '1px solid rgba(0,0,0,0.05)',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
            }}>
                <FileText size={18} style={{color: '#64748b'}}/>
                <div style={{flex: 1}}>
                    <div style={{
                        fontWeight: 700,
                        color: '#0f172a',
                        fontSize: 14,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8
                    }}>
                        {statusIcon}
                        {statusText}
                    </div>
                    <div style={{color: '#64748b', fontSize: 12, marginTop: 2}}>
                        {data.summary}
                    </div>
                </div>
            </div>

            <div style={{padding: '0 0 8px 0'}}>
                {data.fields.map((field, idx) => {
                    const isFieldError = field.status === 'mismatch'
                    const isFieldOk = field.status === 'ok'

                    return (
                        <div key={idx} style={{
                            padding: '10px 16px',
                            borderBottom: idx < data.fields.length - 1 ? '1px solid rgba(0,0,0,0.04)' : 'none',
                            background: idx % 2 === 0 ? 'transparent' : 'rgba(248,250,252,0.5)',
                        }}>
                            <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 4}}>
                                <span style={{fontWeight: 600, color: '#334155'}}>{field.label}</span>
                                <span style={{
                                    fontSize: 11,
                                    fontWeight: 600,
                                    padding: '2px 8px',
                                    borderRadius: 10,
                                    background: isFieldError ? 'rgba(239, 68, 68, 0.1)' : (isFieldOk ? 'rgba(16, 185, 129, 0.1)' : 'rgba(148, 163, 184, 0.1)'),
                                    color: isFieldError ? '#b91c1c' : (isFieldOk ? '#047857' : '#64748b'),
                                    textTransform: 'uppercase',
                                }}>
                                    {isFieldError ? 'Ошибка' : (isFieldOk ? 'OK' : 'Не найдено')}
                                </span>
                            </div>

                            <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 12}}>
                                <div>
                                    <div style={{fontSize: 10, color: '#94a3b8', marginBottom: 2}}>В карточке</div>
                                    <div style={{color: '#1e293b', wordBreak: 'break-word'}}>{field.card_value}</div>
                                </div>
                                <div>
                                    <div style={{fontSize: 10, color: '#94a3b8', marginBottom: 2}}>В файле</div>
                                    <div style={{
                                        color: field.file_value ? '#1e293b' : '#cbd5e1',
                                        wordBreak: 'break-word'
                                    }}>
                                        {field.file_value || '—'}
                                    </div>
                                </div>
                            </div>

                            {field.recommendation && (
                                <div style={{
                                    marginTop: 6,
                                    padding: '6px 10px',
                                    background: '#fffbeb',
                                    border: '1px solid #fcd34d',
                                    borderRadius: 6,
                                    fontSize: 11,
                                    color: '#92400e',
                                    display: 'flex',
                                    gap: 6,
                                    alignItems: 'flex-start'
                                }}>
                                    <span style={{fontWeight: 700}}>💡 Рекомендация:</span>
                                    <span>{field.recommendation}</span>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            <div style={{
                padding: '8px 16px',
                fontSize: 11,
                color: '#94a3b8',
                borderTop: '1px solid rgba(0,0,0,0.05)',
                background: '#f8fafc'
            }}>
                Проверено AI. Результат добавлен в краткое содержание документа.
            </div>
        </div>
    )
}

export function isErrorMessage(content: string): boolean {
    return content.startsWith('__error__:')
}

export function extractErrorText(content: string): string {
    return content
        .replace(/^__error__:\s*/, '')
        .replace(/^(Error:\s*|Ошибка:\s*)/i, '')
}

function humanizeError(raw: string): string {
    const lower = raw.toLowerCase()
    if (lower.includes('failed to fetch') || lower.includes('networkerror') || lower.includes('network'))
        return 'Нет соединения с сервером. Проверьте подключение к интернету и доступность сервиса.'
    if (lower.includes('timeout') || lower.includes('timed out'))
        return 'Сервер не ответил вовремя. Возможно, он перегружен — попробуйте чуть позже.'
    if (lower.includes('401') || lower.includes('unauthorized') || lower.includes('unauthenticated'))
        return 'Ошибка авторизации. Возможно, сессия истекла — обновите страницу или войдите снова.'
    if (lower.includes('403') || lower.includes('forbidden'))
        return 'Доступ запрещён. У вас нет прав на выполнение этого действия.'
    if (lower.includes('404') || lower.includes('not found'))
        return 'Запрашиваемый ресурс не найден. Возможно, он был удалён или перемещён.'
    if (lower.includes('500') || lower.includes('internal server'))
        return 'Внутренняя ошибка сервера. Мы уже работаем над её устранением.'
    if (lower.includes('503') || lower.includes('service unavailable'))
        return 'Сервис временно недоступен. Попробуйте повторить запрос через несколько минут.'
    if (lower.includes('aborted') || lower.includes('abort'))
        return 'Запрос был отменён.'
    return raw || 'Произошла неизвестная ошибка.'
}

function sanitizeHtmlToMarkdown(raw: string): string {
    return raw
        .replace(/<pre[^>]*>\s*<code[^>]*>([\s\S]*?)<\/code>\s*<\/pre>/gi, (_, code) =>
            '\n```\n' + code.trim() + '\n```\n'
        )
        .replace(/<br\s*\/?>/gi, '  \n')
        .replace(/<strong[^>]*>([\s\S]*?)<\/strong>/gi, '**$1**')
        .replace(/<b[^>]*>([\s\S]*?)<\/b>/gi, '**$1**')
        .replace(/<em[^>]*>([\s\S]*?)<\/em>/gi, '*$1*')
        .replace(/<i[^>]*>([\s\S]*?)<\/i>/gi, '*$1*')
        .replace(/<u[^>]*>([\s\S]*?)<\/u>/gi, '$1')
        .replace(/<(?:s|del|strike)[^>]*>([\s\S]*?)<\/(?:s|del|strike)>/gi, '~~$1~~')
        .replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, '`$1`')
        .replace(/<h1[^>]*>([\s\S]*?)<\/h1>/gi, '\n# $1\n')
        .replace(/<h2[^>]*>([\s\S]*?)<\/h2>/gi, '\n## $1\n')
        .replace(/<h3[^>]*>([\s\S]*?)<\/h3>/gi, '\n### $1\n')
        .replace(/<h4[^>]*>([\s\S]*?)<\/h4>/gi, '\n#### $1\n')
        .replace(/<h5[^>]*>([\s\S]*?)<\/h5>/gi, '\n##### $1\n')
        .replace(/<h6[^>]*>([\s\S]*?)<\/h6>/gi, '\n###### $1\n')
        .replace(/<p[^>]*>([\s\S]*?)<\/p>/gi, '\n$1\n')
        .replace(/<\/?(?:ul|ol)[^>]*>/gi, '\n')
        .replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, '- $1\n')
        .replace(/<hr\s*\/?>/gi, '\n---\n')
        .replace(/<a[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/gi, '[$2]($1)')
        .replace(/<a[^>]*href='([^']*)'[^>]*>([\s\S]*?)<\/a>/gi, '[$2]($1)')
        .replace(/<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*\/?>/gi, '![$2]($1)')
        .replace(/<img[^>]*src="([^"]*)"[^>]*\/?>/gi, '![]($1)')
        .replace(/<blockquote[^>]*>([\s\S]*?)<\/blockquote>/gi, (_, inner) => {
            const lines = inner.trim().split('\n')
            return '\n' + lines.map(l => '> ' + l.trim()).join('\n') + '\n'
        })
        .replace(/<th[^>]*>([\s\S]*?)<\/th>/gi, '| $1 ')
        .replace(/<\/tr>/gi, '|\n')
        .replace(/<tr[^>]*>/gi, '|')
        .replace(/<td[^>]*>([\s\S]*?)<\/td>/gi, '| $1 ')
        .replace(/<\/?(?:table|thead|tbody|tfoot|caption|colgroup|col)[^>]*>/gi, '\n')
        .replace(/<div[^>]*>/gi, '\n')
        .replace(/<\/div>/gi, '\n')
        .replace(/<\/?span[^>]*>/gi, '')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&nbsp;/g, ' ')
        .replace(/<[^>]+>/g, '')
        .replace(/\n{3,}/g, '\n\n')
        .trim()
}

function normalizeUuid(raw: string): string {
    return raw
        .replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00AD\uFE58\uFE63\uFF0D]/g, '-')
        .trim()
}

function isValidUuid(raw: string): boolean {
    const normalized = normalizeUuid(raw)
    return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(normalized)
}

interface TableData {
    headers: string[]
    rows: string[][]
}

function extractTableData(children: React.ReactNode): TableData {
    const headers: string[] = []
    const rows: string[][] = []
    const childArr = Array.isArray(children) ? children : [children]
    for (const child of childArr) {
        if (!child || typeof child !== 'object') continue
        const c = child as any
        if (c.type === 'thead' || c.props?.node?.tagName === 'thead') {
            const thCells = c.props?.children
            const thRow = Array.isArray(thCells) ? thCells[0] : thCells
            const ths = thRow?.props?.children
            const thArr = Array.isArray(ths) ? ths : [ths]
            for (const th of thArr) {
                const text = extractText(th)
                if (text) headers.push(text)
            }
        }
        if (c.type === 'tbody' || c.props?.node?.tagName === 'tbody') {
            const trs = c.props?.children
            const trArr = Array.isArray(trs) ? trs : [trs]
            for (const tr of trArr) {
                if (!tr) continue
                const tds = tr?.props?.children
                const tdArr = Array.isArray(tds) ? tds : [tds]
                const row: string[] = []
                for (const td of tdArr) row.push(extractText(td) || '')
                if (row.some(Boolean)) rows.push(row)
            }
        }
    }
    return {headers, rows}
}

function extractText(node: any): string {
    if (!node) return ''
    if (typeof node === 'string') return node
    if (typeof node === 'number') return String(node)
    if (Array.isArray(node)) return node.map(extractText).join('')
    if (node?.props?.children) return extractText(node.props.children)
    return ''
}

function isAttachmentTable(headers: string[]): boolean {
    const h = headers.join(' ').toLowerCase()
    return /файл|вложени|название файл/.test(h) || /размер|size/.test(h)
}

function isKeyValueTable(headers: string[]): boolean {
    if (headers.length !== 2) return false
    const h0 = headers[0].toLowerCase()
    const h1 = headers[1].toLowerCase()
    const kvKeys = ['параметр', 'поле', 'ключ', 'field', 'key', 'свойство', 'атрибут']
    const kvVals = ['информация', 'значение', 'value', 'данные', 'data', 'содержание']
    return kvKeys.some(k => h0.includes(k)) || kvVals.some(k => h1.includes(k))
}

function KeyValueList({rows}: { rows: string[][] }) {
    return (
        <div style={{display: 'flex', flexDirection: 'column', gap: 2, margin: '8px 0'}}>
            {rows.map(([key, value], i) => (
                <div key={i} style={{
                    display: 'flex', gap: 12, padding: '6px 10px', borderRadius: 10,
                    background: i % 2 === 0 ? 'rgba(248,250,252,0.70)' : 'transparent',
                    alignItems: 'flex-start',
                }}>
                    <span style={{
                        fontSize: 11, fontWeight: 600, color: '#64748b',
                        minWidth: 120, flexShrink: 0, lineHeight: 1.6,
                    }}>
                        {key}
                    </span>
                    <span style={{fontSize: 12, color: '#1e293b', lineHeight: 1.6, wordBreak: 'break-word'}}>
                        {value || '—'}
                    </span>
                </div>
            ))}
        </div>
    )
}

const CATEGORY_COLORS: Record<string, { bg: string; text: string; label: string }> = {
    'INCOMING': {bg: 'rgba(59,130,246,0.08)', text: '#1d4ed8', label: 'Входящий'},
    'OUTGOING': {bg: 'rgba(16,185,129,0.08)', text: '#065f46', label: 'Исходящий'},
    'INTERN': {bg: 'rgba(139,92,246,0.08)', text: '#5b21b6', label: 'Внутренний'},
    'APPEAL': {bg: 'rgba(245,158,11,0.08)', text: '#92400e', label: 'Обращение'},
    'CONTRACT': {bg: 'rgba(239,68,68,0.08)', text: '#991b1b', label: 'Договор'},
    'MEETING': {bg: 'rgba(99,102,241,0.08)', text: '#3730a3', label: 'Совещание'},
}

function getCategoryStyle(raw: string) {
    const upper = raw.toUpperCase().replace(/[()]/g, '').trim()
    for (const [key, val] of Object.entries(CATEGORY_COLORS)) {
        if (upper.includes(key)) return val
    }
    return {bg: 'rgba(100,116,139,0.08)', text: '#334155', label: raw}
}

function IconOpenNew() {
    return (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"
             style={{flexShrink: 0}}>
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
            <polyline points="15 3 21 3 21 9"/>
            <line x1="10" y1="14" x2="21" y2="3"/>
        </svg>
    )
}

function DocCard({headers, row, index}: { headers: string[]; row: string[]; index: number }) {
    const onDocumentClick = useContext(DocumentClickContext)

    const pairs = headers.map((h, i) => ({key: h.trim(), value: (row[i] || '—').trim()}))

    const num = pairs.find(p => /^[№#]$/.test(p.key))?.value
    const regNum = pairs.find(p => /рег.*номер|reg.*num|^номер$/i.test(p.key))?.value
    const date = pairs.find(p => /^дата$|^date$|рег.*дата|reg.*date/i.test(p.key))?.value
    const category = pairs.find(p => /категор|category|тип|type/i.test(p.key))?.value
    const summary = pairs.find(p => /содержан|summary|краткое|описан/i.test(p.key))?.value
    const author = pairs.find(p => /автор|author/i.test(p.key))?.value
    const status = pairs.find(p => /статус|status/i.test(p.key))?.value
    const address = pairs.find(p => /адрес|address/i.test(p.key))?.value

    const rawId = pairs.find(p => /^id$/i.test(p.key))?.value ?? ''
    const docId = rawId ? normalizeUuid(rawId) : ''
    const isClickable = Boolean(onDocumentClick && docId && isValidUuid(docId))

    const _skipKeys = /^[№#]$|^id$|рег.*номер|reg.*num|^номер$|^дата$|^date$|рег.*дата|reg.*date|категор|category|тип|type|содержан|summary|краткое|описан|автор|author|статус|status|адрес|address/i
    const extraPairs = pairs.filter(p => !_skipKeys.test(p.key) && p.value && p.value !== '—')

    const catStyle = category ? getCategoryStyle(category) : null

    return (
        <div
            onClick={isClickable ? () => onDocumentClick!(docId) : undefined}
            title={isClickable ? 'Открыть документ в новой вкладке' : undefined}
            style={{
                background: '#ffffff',
                border: '1px solid rgba(0,0,0,0.05)',
                borderRadius: 14,
                padding: '12px 14px',
                marginBottom: 6,
                boxShadow: '0 1px 3px rgba(0,0,0,0.03)',
                cursor: isClickable ? 'pointer' : 'default',
                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                display: 'flex',
                flexDirection: 'column',
                gap: 4
            }}
            onMouseEnter={isClickable ? e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.background = '#fafbff'
                el.style.borderColor = 'rgba(99,102,241,0.18)'
                el.style.boxShadow = '0 2px 12px rgba(99,102,241,0.08)'
                el.style.transform = 'translateY(-1px)'
            } : undefined}
            onMouseLeave={isClickable ? e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.background = '#ffffff'
                el.style.borderColor = 'rgba(0,0,0,0.05)'
                el.style.boxShadow = '0 1px 3px rgba(0,0,0,0.03)'
                el.style.transform = 'translateY(0)'
            } : undefined}
        >
            <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
            }}>
                <span style={{
                    width: 24, height: 24, borderRadius: 8,
                    background: isClickable ? 'rgba(99,102,241,0.08)' : 'rgba(148,163,184,0.08)',
                    color: isClickable ? '#6366f1' : '#94a3b8',
                    fontSize: 10, fontWeight: 700,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                }}>{num ?? index + 1}</span>

                {regNum && regNum !== '—'
                    ? <span style={{fontSize: 12, fontWeight: 700, color: '#0f172a', flex: 1}}>{regNum}</span>
                    : <span style={{fontSize: 12, color: '#94a3b8', flex: 1, fontStyle: 'italic'}}>Без номера</span>
                }

                {date && date !== '—' && (
                    <span style={{fontSize: 10, color: '#94a3b8', flexShrink: 0}}>{date}</span>
                )}

                {isClickable && (
                    <span style={{
                        flexShrink: 0, color: '#6366f1', opacity: 0.5,
                        display: 'flex', alignItems: 'center',
                    }}>
                        <IconOpenNew/>
                    </span>
                )}
            </div>

            {summary && summary !== '—' && (
                <p style={{
                    fontSize: 12, color: '#475569', lineHeight: 1.5, margin: 0,
                    display: '-webkit-box', WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical', overflow: 'hidden',
                }}>{summary}</p>
            )}

            <div style={{display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'flex-start'}}>
                {catStyle && (
                    <span style={{
                        fontSize: 10, fontWeight: 600,
                        padding: '2px 8px', borderRadius: 20,
                        background: catStyle.bg, color: catStyle.text,
                    }}>
                        {catStyle.label}
                    </span>
                )}
                {author && author !== '—' && (
                    <span style={{
                        fontSize: 10, color: '#64748b',
                        padding: '2px 8px', borderRadius: 20,
                        background: 'rgba(100,116,139,0.06)',
                    }}>👤 {author}</span>
                )}
                {status && status !== '—' && (
                    <span style={{
                        fontSize: 10, color: '#64748b',
                        padding: '2px 8px', borderRadius: 20,
                        background: 'rgba(100,116,139,0.06)',
                    }}>{status}</span>
                )}

                {address && address !== '—' && (
                    <div style={{
                        width: '100%',
                        marginTop: 2,
                        fontSize: 11,
                        color: '#64748b',
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 6,
                        lineHeight: 1.4,
                        padding: '2px 0'
                    }}>
                        <span style={{flexShrink: 0, marginTop: '1px'}}>📍</span>
                        <span style={{wordBreak: 'break-word'}}>{address}</span>
                    </div>
                )}

                {extraPairs.map(({key, value}) => {
                    const isContact = key.toLowerCase().includes('контакт') || key.toLowerCase().includes('contact');

                    if (isContact && value) {
                        const parts = value.split(/\s{2,}|\n/).filter(part => part.trim().length > 0);

                        return (
                            <div key={key} className="flex flex-wrap gap-2 items-center w-full mt-1">
                                {parts.map((part, i) => (
                                    <span
                                        key={i}
                                        style={{
                                            fontSize: 10,
                                            color: '#475569',
                                            padding: '3px 8px',
                                            borderRadius: 6,
                                            background: '#f1f5f9',
                                            whiteSpace: 'nowrap',
                                            border: '1px solid rgba(0,0,0,0.03)',
                                            fontFamily: 'ui-monospace, monospace',
                                            lineHeight: 1.4
                                        }}
                                    >
                                        {part.trim()}
                                    </span>
                                ))}
                            </div>
                        );
                    }

                    return (
                        <span key={key} style={{
                            fontSize: 10, color: '#64748b',
                            padding: '2px 8px', borderRadius: 20,
                            background: 'rgba(100,116,139,0.06)',
                        }}>{key}: {value}</span>
                    );
                })}
            </div>
        </div>
    )
}

function AttachmentCard({headers, row, index}: { headers: string[]; row: string[]; index: number }) {
    const onAttachmentClick = useContext(AttachmentClickContext)
    const pairs = headers.map((h, i) => ({key: h, value: row[i] || ''}))

    const fileName = pairs.find(p => /файл|название|name/i.test(p.key))?.value
        || pairs.find(p => p.value && /\.(docx?|pdf|xlsx?|txt|rtf)/i.test(p.value))?.value || ''
    const fileSize = pairs.find(p => /размер|size/i.test(p.key))?.value || ''
    const fileDate = pairs.find(p => /дата|date/i.test(p.key))?.value || ''

    const ext = (fileName.match(/\.([a-z]+)$/i)?.[1] || '').toLowerCase()
    const icon = ext === 'pdf' ? '📄' : (ext === 'doc' || ext === 'docx') ? '📝'
        : (ext === 'xls' || ext === 'xlsx') ? '📊' : '📎'

    return (
        <div
            onClick={() => fileName && onAttachmentClick?.(fileName)}
            style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '10px 14px', marginBottom: 5, borderRadius: 12,
                background: 'rgba(248,250,252,0.80)',
                border: '1px solid rgba(0,0,0,0.04)',
                cursor: onAttachmentClick ? 'pointer' : 'default',
                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
            onMouseEnter={e => {
                if (!onAttachmentClick) return
                const el = e.currentTarget as HTMLDivElement
                el.style.background = 'rgba(99,102,241,0.05)'
                el.style.borderColor = 'rgba(99,102,241,0.15)'
                el.style.transform = 'translateX(2px)'
            }}
            onMouseLeave={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.background = 'rgba(248,250,252,0.80)'
                el.style.borderColor = 'rgba(0,0,0,0.04)'
                el.style.transform = 'translateX(0)'
            }}
        >
            <span style={{fontSize: 20, flexShrink: 0, lineHeight: 1}}>{icon}</span>
            <div style={{flex: 1, minWidth: 0}}>
                <p style={{
                    margin: 0, fontSize: 12, fontWeight: 600, color: '#1e293b',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>{fileName || `Вложение ${index + 1}`}</p>
                <p style={{margin: 0, fontSize: 10, color: '#94a3b8', marginTop: 2}}>
                    {[fileDate, fileSize].filter(Boolean).join('  ·  ')}
                </p>
            </div>
            {onAttachmentClick && <span style={{color: '#94a3b8', fontSize: 14, flexShrink: 0}}>›</span>}
        </div>
    )
}

function SmartTable({children}: { children: React.ReactNode }) {
    const {headers, rows} = extractTableData(children)

    if (!headers.length || !rows.length) {
        return (
            <div style={{
                overflowX: 'auto', margin: '8px 0', borderRadius: 12,
                border: '1px solid rgba(0,0,0,0.05)',
            }}>
                <table style={{width: '100%', fontSize: 12, borderCollapse: 'collapse'}}>{children}</table>
            </div>
        )
    }

    if (isKeyValueTable(headers)) return <KeyValueList rows={rows}/>

    if (isAttachmentTable(headers)) {
        return (
            <div style={{margin: '6px 0'}}>
                {rows.map((row, i) => <AttachmentCard key={i} headers={headers} row={row} index={i}/>)}
            </div>
        )
    }

    return (
        <div style={{margin: '6px 0'}}>
            {rows.map((row, i) => <DocCard key={i} headers={headers} row={row} index={i}/>)}
        </div>
    )
}

export function ChatMessage({content, role, timestamp, isError, onAttachmentClick, onDocumentClick}: Props) {
    const isUser = role === 'user'
    const isErr = isError ?? false
    const isRawError = typeof content === 'string' && content.startsWith('__error__:')

    const display = isRawError ? content.replace(/^__error__:\s*/, '') : content
    const timeLabel = dayjs(timestamp).format('HH:mm')
    const sanitized = sanitizeHtmlToMarkdown(display)

    let complianceData: ComplianceCheckData | null = null

    if (!isUser && !isRawError) {
        try {
            let jsonStr = content

            try {
                const parsed = JSON.parse(content)
                if (parsed && typeof parsed === 'object' && 'overall' in parsed && 'fields' in parsed) {
                    complianceData = parsed as ComplianceCheckData
                }
            } catch (e) {
                const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/) || content.match(/```\s*([\s\S]*?)\s*```/)
                if (jsonMatch) {
                    jsonStr = jsonMatch[1]
                    const parsed = JSON.parse(jsonStr)
                    if (parsed && typeof parsed === 'object' && 'overall' in parsed && 'fields' in parsed) {
                        complianceData = parsed as ComplianceCheckData
                    }
                }
            }
        } catch (e) {
        }
    }

    if (isRawError) {
        return (
            <div className="flex w-full justify-start">
                {/* ... error rendering ... */}
            </div>
        )
    }

    return (
        <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
             style={{animation: 'edms-fade-in .25s ease-out'}}>
            <div
                className="max-w-full px-4 py-3 leading-relaxed edms-chat-text"
                style={isUser ? {
                    background: 'linear-gradient(135deg, #6366f1 0%, #7c3aed 100%)',
                    color: '#ffffff',
                    borderRadius: '22px 22px 6px 22px',
                    boxShadow: '0 2px 16px rgba(99,102,241,0.25)',
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word',
                    minWidth: 200,
                } : {
                    background: '#ffffff',
                    color: '#0f172a',
                    borderRadius: '22px 22px 22px 6px',
                    boxShadow: '0 1px 4px rgba(0,0,0,0.04), 0 0 0 1px rgba(0,0,0,0.03)',
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word',
                    minWidth: 300,
                }}
            >
                <AttachmentClickContext.Provider value={onAttachmentClick ?? null}>
                    <DocumentClickContext.Provider value={onDocumentClick ?? null}>

                        {complianceData ? (
                            <ComplianceCheckResult data={complianceData}/>
                        ) : (
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm, remarkLazyList]}
                                components={{
                                    table: ({children}) => <SmartTable>{children}</SmartTable>,
                                    thead: ({children}) => <thead>{children}</thead>,
                                    tbody: ({children}) => <tbody>{children}</tbody>,
                                    tr: ({children}) => <tr>{children}</tr>,
                                    th: ({children}) => <th style={{
                                        padding: '7px 12px',
                                        borderBottom: '1px solid rgba(0,0,0,0.05)',
                                        fontWeight: 600,
                                        background: 'rgba(248,250,252,0.60)',
                                        textAlign: 'left',
                                        fontSize: 11,
                                    }}>{children}</th>,
                                    td: ({children}) => <td style={{
                                        padding: '7px 12px',
                                        borderBottom: '1px solid rgba(0,0,0,0.03)',
                                        fontSize: 12,
                                    }}>{children}</td>,
                                    code({inline, className, children, ...props}: any) {
                                        if (!inline) {
                                            return (
                                                <pre style={{
                                                    overflow: 'auto',
                                                    padding: '12px 14px',
                                                    margin: '8px 0',
                                                    borderRadius: 12,
                                                    background: '#0f172a',
                                                    border: '1px solid rgba(255,255,255,0.04)',
                                                }}>
                                                <code className={className} style={{
                                                    fontFamily: 'ui-monospace, monospace',
                                                    fontSize: 12,
                                                    color: '#a5b4fc',
                                                }} {...props}>{children}</code>
                                            </pre>
                                            )
                                        }
                                        return <code style={{
                                            fontFamily: 'ui-monospace, monospace',
                                            fontSize: '0.88em',
                                            padding: '1px 6px',
                                            borderRadius: 5,
                                            background: isUser ? 'rgba(255,255,255,0.18)' : 'rgba(99,102,241,0.07)',
                                            color: isUser ? '#fff' : '#4338ca',
                                        }} {...props}>{children}</code>
                                    },
                                    p: ({children}) => {
                                        const rawText = Array.isArray(children)
                                            ? children.map(c => typeof c === 'string' ? c : (c?.props?.children ?? '')).join('')
                                            : typeof children === 'string' ? children : ''
                                        const fileMatch = rawText.match(/Файл:\s*([^\s][^\t\n]+?)(?:\s{2,}|\s*Размер:)/)
                                        if (fileMatch && !isUser && onAttachmentClick) {
                                            const fileName = fileMatch[1].trim()
                                            return (
                                                <button type="button" onClick={() => onAttachmentClick(fileName)}
                                                        style={{
                                                            display: 'flex', alignItems: 'center', gap: 8,
                                                            width: '100%', padding: '9px 12px', marginBottom: 4,
                                                            borderRadius: 12, cursor: 'pointer', textAlign: 'left',
                                                            background: 'rgba(248,250,252,0.80)',
                                                            border: '1px solid rgba(0,0,0,0.04)',
                                                            color: '#334155', fontSize: 12, fontWeight: 500,
                                                            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                                                        }}
                                                        onMouseEnter={e => {
                                                            const el = e.currentTarget as HTMLButtonElement
                                                            el.style.background = 'rgba(99,102,241,0.05)'
                                                            el.style.borderColor = 'rgba(99,102,241,0.15)'
                                                            el.style.transform = 'translateX(2px)'
                                                        }}
                                                        onMouseLeave={e => {
                                                            const el = e.currentTarget as HTMLButtonElement
                                                            el.style.background = 'rgba(248,250,252,0.80)'
                                                            el.style.borderColor = 'rgba(0,0,0,0.04)'
                                                            el.style.transform = 'translateX(0)'
                                                        }}
                                                >
                                                    <span style={{fontSize: 15, flexShrink: 0}}>📎</span>
                                                    <span style={{
                                                        flex: 1,
                                                        minWidth: 0,
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                        whiteSpace: 'nowrap',
                                                    }}>{fileName}</span>
                                                    <span style={{
                                                        fontSize: 10, color: '#94a3b8', flexShrink: 0, fontWeight: 400,
                                                    }}>{rawText.match(/Размер:\s*([^\n]+)/)?.[1]?.trim() ?? ''}</span>
                                                </button>
                                            )
                                        }
                                        return <p style={{
                                            marginBottom: 6, lineHeight: 1.7,
                                            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                                        }}>{children}</p>
                                    },
                                    ul: ({children}) => <ul style={{paddingLeft: 20, marginBottom: 6}}>{children}</ul>,
                                    ol: ({children}) => <ol style={{paddingLeft: 20, marginBottom: 6}}>{children}</ol>,
                                    li: ({children}) => <li style={{marginBottom: 2}}>{children}</li>,
                                    h2: ({children}) => {
                                        const text = typeof children === 'string' ? children : Array.isArray(children) ? children.map(c => typeof c === 'string' ? c : '').join('') : ''
                                        const clean = text.replace(/^\d+\s+/, '').replace(/^[①②③④⑤⑥⑦⑧⑨⑩]\s*/, '').trim()
                                        return <p style={{
                                            fontWeight: 700, fontSize: 13.5, marginTop: 12, marginBottom: 5,
                                            color: isUser ? '#fff' : '#0f172a',
                                            paddingBottom: 4,
                                            borderBottom: isUser ? '1px solid rgba(255,255,255,0.15)' : '1px solid rgba(0,0,0,0.05)',
                                        }}>{clean || children}</p>
                                    },
                                    h3: ({children}) => {
                                        const text = typeof children === 'string' ? children : Array.isArray(children) ? children.map(c => typeof c === 'string' ? c : '').join('') : ''
                                        const clean = text.replace(/^\d+\s+/, '').replace(/^[①②③④⑤⑥⑦⑧⑨⑩]\s*/, '').trim()
                                        return <p style={{
                                            fontWeight: 600, fontSize: 12.5, marginTop: 10, marginBottom: 3,
                                            color: isUser ? 'rgba(255,255,255,0.85)' : '#334155',
                                        }}>{clean || children}</p>
                                    },
                                    strong: ({children}) => <strong
                                        style={{
                                            fontWeight: 650,
                                            color: isUser ? '#fff' : '#0f172a'
                                        }}>{children}</strong>,
                                    a: ({children, href}) => <a href={href} target="_blank" rel="noopener noreferrer"
                                                                style={{
                                                                    textDecoration: 'underline',
                                                                    fontWeight: 500,
                                                                    color: isUser ? 'rgba(255,255,255,0.85)' : '#6366f1',
                                                                }}>{children}</a>,
                                    blockquote: ({children}) => (
                                        <blockquote style={{
                                            borderLeft: `3px solid ${isUser ? 'rgba(255,255,255,0.30)' : '#c7d2fe'}`,
                                            paddingLeft: 12, marginLeft: 0,
                                            fontStyle: 'italic',
                                            color: isUser ? 'rgba(255,255,255,0.75)' : '#64748b',
                                        }}>
                                            {children}
                                        </blockquote>
                                    ),
                                }}
                            >
                                {sanitized}
                            </ReactMarkdown>
                        )}
                    </DocumentClickContext.Provider>
                </AttachmentClickContext.Provider>

                <div style={{
                    marginTop: 5,
                    opacity: 0.45,
                    textAlign: isUser ? 'right' : 'left',
                    color: isUser ? 'rgba(255,255,255,0.75)' : '#94a3b8',
                    fontSize: 10,
                }}>
                    {timeLabel}
                </div>
            </div>
        </div>
    )
}