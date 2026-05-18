import {
    useState,
    useCallback,
} from 'react'
import {remarkLazyList} from '../plugins/remarkLazyList'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import dayjs from 'dayjs'
import {
    Copy,
    Check,
    AlertCircle,
    ExternalLink,
    User,
    FileText,
} from 'lucide-react'
import { DocCard } from './cards/DocCard'
import { AttachmentCard } from './cards/AttachmentCard'
import { AttachmentClickContext, DocumentClickContext } from './ChatContext'
import { parseStructuredOutput } from '@/features/chat/lib/parseStructuredOutput'
import { ComplianceCheckResult } from '@/features/chat/ui/structured/ComplianceCheckResult'
import { ThesisPlanResult } from '@/features/chat/ui/structured/ThesisPlanResult'
import { AbstractiveResult } from '@/features/chat/ui/structured/AbstractiveResult'
import { ExtractiveResult } from '@/features/chat/ui/structured/ExtractiveResult'
import { ActionItemsResult } from '@/features/chat/ui/structured/ActionItemsResult'
import { ExecutiveSummaryResult } from '@/features/chat/ui/structured/ExecutiveSummaryResult'
import { DetailedNotesResult } from '@/features/chat/ui/structured/DetailedNotesResult'
import { MultilingualResult } from '@/features/chat/ui/structured/MultilingualResult'
import type { StructuredOutput } from '@/entities/message/model/types'


interface Props {
    content: string
    role: 'user' | 'assistant'
    timestamp: number
    isError?: boolean
    onAttachmentClick?: (fileName: string) => void
    onDocumentClick?: (documentId: string) => void
}


function CopyButton({text}: { text: string }) {
    const [copied, setCopied] = useState(false)
    const handleCopy = useCallback(() => {
        navigator.clipboard.writeText(text).then(() => {
            setCopied(true)
            setTimeout(() => setCopied(false), 1500)
        })
    }, [text])

    return (
        <button
            onClick={handleCopy}
            title={copied ? 'Скопировано!' : 'Копировать код'}
            style={{
                position: 'absolute',
                top: 8,
                right: 8,
                background: copied ? 'rgba(16,185,129,0.15)' : 'rgba(255,255,255,0.08)',
                border: 'none',
                borderRadius: 6,
                padding: '4px 7px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                color: copied ? '#6ee7b7' : '#94a3b8',
                fontSize: 11,
                fontWeight: 500,
                transition: 'all 0.15s',
            }}
        >
            {copied ? <Check size={12}/> : <Copy size={12}/>}
            {copied ? 'OK' : ''}
        </button>
    )
}

function StructuredOutputRenderer({output}: { output: StructuredOutput }) {
    switch (output.type) {
        case 'compliance':
            return <ComplianceCheckResult data={output.data}/>
        case 'thesis':
            return <ThesisPlanResult data={output.data}/>
        case 'abstractive':
            return <AbstractiveResult data={output.data}/>
        case 'extractive':
            return <ExtractiveResult data={output.data}/>
        case 'action_items':
            return <ActionItemsResult data={output.data}/>
        case 'executive':
            return <ExecutiveSummaryResult data={output.data}/>
        case 'detailed_notes':
            return <DetailedNotesResult data={output.data}/>
        case 'multilingual':
            return <MultilingualResult data={output.data}/>
    }
}

// ─── Error utilities ─────────────────────────────────────────────────────────

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

// ─── HTML → Markdown sanitiser ───────────────────────────────────────────────

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
            return '\n' + lines.map((l: string) => '> ' + l.trim()).join('\n') + '\n'
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

// ─── UUID helpers ────────────────────────────────────────────────────────────

function normalizeUuid(raw: string): string {
    return raw
        .replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00AD\uFE58\uFE63\uFF0D]/g, '-')
        .trim()
}

function isValidUuid(raw: string): boolean {
    const normalized = normalizeUuid(raw)
    return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(normalized)
}

// ─── Table extraction ────────────────────────────────────────────────────────

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
    const h0 = (headers[0] ?? '').toLowerCase()
    const h1 = (headers[1] ?? '').toLowerCase()
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
                    <span style={{
                        fontSize: 12, color: '#1e293b', lineHeight: 1.6, wordBreak: 'break-word',
                    }}>
                        {value || '—'}
                    </span>
                </div>
            ))}
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

    const h = headers.join(' ').toLowerCase()
    const isDocList = /(id|uuid|идентификатор|doc.*id|document.*id)/.test(h) || /рег.*номер|рег\.номер/.test(h) || (/дата/.test(h) && /категор/.test(h))

    if (isDocList) {
        return (
            <div style={{margin: '6px 0'}}>
                {rows.map((row, i) => <DocCard key={i} headers={headers} row={row} index={i}/>)}
            </div>
        )
    }

    return (
        <div style={{
            overflowX: 'auto', margin: '8px 0', borderRadius: 12,
            border: '1px solid rgba(0,0,0,0.05)', background: '#ffffff',
        }}>
            <table style={{width: '100%', fontSize: 12, borderCollapse: 'collapse'}}>{children}</table>
        </div>
    )
}

// ─── Error message component ─────────────────────────────────────────────────

function ErrorMessage({raw, timeLabel}: { raw: string; timeLabel: string }) {
    const friendly = humanizeError(raw)
    return (
        <div style={{
            background: 'linear-gradient(135deg, rgba(239,68,68,0.04), rgba(244,63,94,0.04))',
            border: '1px solid rgba(239,68,68,0.12)',
            borderRadius: '22px 22px 22px 6px',
            padding: '14px 18px',
            maxWidth: '85%',
            animation: 'edms-slide-up .25s ease-out',
        }}>
            <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                marginBottom: 6,
            }}>
                <AlertCircle size={16} style={{color: '#ef4444', flexShrink: 0}}/>
                <span style={{fontSize: 13, fontWeight: 600, color: '#991b1b'}}>
                    Ошибка
                </span>
            </div>
            <p style={{
                margin: 0, fontSize: 12, color: '#7f1d1d', lineHeight: 1.6,
            }}>
                {friendly}
            </p>
            <div style={{
                marginTop: 6, opacity: 0.45, textAlign: 'left',
                color: '#94a3b8', fontSize: 10,
            }}>
                {timeLabel}
            </div>
        </div>
    )
}

// ─── Main ChatMessage ────────────────────────────────────────────────────────

export function ChatMessage({
                                content,
                                role,
                                timestamp,
                                onAttachmentClick,
                                onDocumentClick,
                            }: Props) {
    const isUser = role === 'user'
    const isRawError = typeof content === 'string' && content.startsWith('__error__:')

    const timeLabel = dayjs(timestamp).format('HH:mm')

    // ── Error path ──
    if (isRawError) {
        const rawText = content.replace(/^__error__:\s*/, '')
        return (
            <div className="flex w-full justify-start">
                <ErrorMessage raw={rawText} timeLabel={timeLabel}/>
            </div>
        )
    }

    // ── Structured output detection ──
    const structured = !isUser ? parseStructuredOutput(content) : null

    // ── Sanitize HTML → markdown ──
    const sanitized = sanitizeHtmlToMarkdown(content)

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
                } : {
                    background: '#ffffff',
                    color: '#0f172a',
                    borderRadius: '22px 22px 22px 6px',
                    boxShadow: '0 1px 4px rgba(0,0,0,0.04), 0 0 0 1px rgba(0,0,0,0.03)',
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word',
                }}
            >
                <AttachmentClickContext.Provider value={onAttachmentClick ?? null}>
                    <DocumentClickContext.Provider value={onDocumentClick ?? null}>

                        {structured ? (
                            <StructuredOutputRenderer output={structured}/>
                        ) : (
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm, remarkLazyList]}
                                components={{
                                    table: ({children}) => <SmartTable>{children}</SmartTable>,
                                    thead: ({children}) => <thead>{children}</thead>,
                                    tbody: ({children}) => <tbody>{children}</tbody>,
                                    tr: ({children}) => <tr>{children}</tr>,
                                    th: ({children}) => <th style={{
                                        padding: '8px 14px',
                                        borderBottom: '2px solid rgba(99,102,241,0.08)',
                                        fontWeight: 600,
                                        background: 'rgba(99,102,241,0.03)',
                                        textAlign: 'left',
                                        fontSize: 11,
                                        color: '#475569',
                                    }}>{children}</th>,
                                    td: ({children}) => <td style={{
                                        padding: '8px 14px',
                                        borderBottom: '1px solid rgba(0,0,0,0.03)',
                                        fontSize: 12,
                                        color: '#334155',
                                    }}>{children}</td>,
                                    code({inline, className, children, ...props}: any) {
                                        const codeText = String(children).replace(/\n$/, '')
                                        if (!inline) {
                                            const lang = className?.replace(/^language-/, '') || ''
                                            return (
                                                <div style={{position: 'relative', margin: '10px 0'}}>
                                                    {lang && (
                                                        <div style={{
                                                            position: 'absolute',
                                                            top: 8,
                                                            left: 12,
                                                            fontSize: 9,
                                                            fontWeight: 600,
                                                            color: '#64748b',
                                                            textTransform: 'uppercase',
                                                            letterSpacing: 0.5,
                                                            zIndex: 1,
                                                        }}>
                                                            {lang}
                                                        </div>
                                                    )}
                                                    <pre style={{
                                                        overflow: 'auto',
                                                        padding: lang ? '28px 14px 14px' : '14px',
                                                        borderRadius: 12,
                                                        background: '#0f172a',
                                                        border: '1px solid rgba(255,255,255,0.04)',
                                                        fontSize: 12,
                                                        lineHeight: 1.6,
                                                    }}>
                                                        <code className={className} style={{
                                                            fontFamily: 'ui-monospace, monospace',
                                                            fontSize: 12,
                                                            color: '#a5b4fc',
                                                        }} {...props}>{children}</code>
                                                    </pre>
                                                    <CopyButton text={codeText}/>
                                                </div>
                                            )
                                        }
                                        return <code style={{
                                            fontFamily: 'ui-monospace, monospace',
                                            fontSize: '0.88em',
                                            padding: '2px 7px',
                                            borderRadius: 5,
                                            background: isUser ? 'rgba(255,255,255,0.18)' : 'rgba(99,102,241,0.07)',
                                            color: isUser ? '#fff' : '#4338ca',
                                        }} {...props}>{children}</code>
                                    },
                                    p: ({children}) => {
                                        const rawText = Array.isArray(children)
                                            ? children.map(c => typeof c === 'string' ? c : (c?.props?.children ?? '')).join('')
                                            : typeof children === 'string' ? children : ''
                                        const fileMatch = rawText.match(/Файл:\s*(\S[^\t\n]+?)(?:\s{2,}|\s*Размер:)/)
                                        if (fileMatch?.[1] && !isUser && onAttachmentClick) {
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
                                                        flex: 1, minWidth: 0,
                                                        overflow: 'hidden', textOverflow: 'ellipsis',
                                                        whiteSpace: 'nowrap',
                                                    }}>{fileName}</span>
                                                    <span style={{
                                                        fontSize: 10, color: '#94a3b8',
                                                        flexShrink: 0, fontWeight: 400,
                                                    }}>
                                                        {rawText.match(/Размер:\s*([^\n]+)/)?.[1]?.trim() ?? ''}
                                                    </span>
                                                </button>
                                            )
                                        }
                                        return <p style={{
                                            marginBottom: 8, lineHeight: 1.75,
                                            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                                        }}>{children}</p>
                                    },
                                    ul: ({children, depth}: any) => (
                                        <ul style={{
                                            paddingLeft: depth > 0 ? 16 : 20,
                                            marginBottom: 8,
                                            listStyle: 'none',
                                        }}>{children}</ul>
                                    ),
                                    ol: ({children}: any) => (
                                        <ol style={{
                                            paddingLeft: 20, marginBottom: 8,
                                            counterReset: 'edms-ol',
                                            listStyle: 'none',
                                        }}>{children}</ol>
                                    ),
                                    li: ({children, ordered, index, depth}: any) => (
                                        <li style={{
                                            marginBottom: 3, position: 'relative',
                                            paddingLeft: 4,
                                            lineHeight: 1.65,
                                        }}>
                                            <span style={{
                                                position: 'absolute',
                                                left: depth > 0 ? -16 : -14,
                                                color: isUser ? 'rgba(255,255,255,0.6)' : '#6366f1',
                                                fontSize: 10,
                                                fontWeight: 700,
                                            }}>
                                                {ordered ? `${(index ?? 0) + 1}.` : '•'}
                                            </span>
                                            {children}
                                        </li>
                                    ),
                                    h1: ({children}) => {
                                        const text = typeof children === 'string'
                                            ? children
                                            : Array.isArray(children)
                                                ? children.map(c => typeof c === 'string' ? c : '').join('')
                                                : ''
                                        return <div style={{
                                            fontWeight: 700, fontSize: 15, marginTop: 14, marginBottom: 6,
                                            color: isUser ? '#fff' : '#0f172a',
                                            paddingBottom: 6,
                                            borderBottom: isUser
                                                ? '1px solid rgba(255,255,255,0.15)'
                                                : '2px solid rgba(99,102,241,0.12)',
                                            lineHeight: 1.4,
                                        }}>{text || children}</div>
                                    },
                                    h2: ({children}) => {
                                        const text = typeof children === 'string'
                                            ? children
                                            : Array.isArray(children)
                                                ? children.map(c => typeof c === 'string' ? c : '').join('')
                                                : ''
                                        if (text.includes('Открой любой документ')) return null
                                        return <div style={{
                                            fontWeight: 700, fontSize: 14, marginTop: 14, marginBottom: 6,
                                            color: isUser ? '#fff' : '#0f172a',
                                            paddingBottom: 5,
                                            borderBottom: isUser
                                                ? '1px solid rgba(255,255,255,0.12)'
                                                : '1px solid rgba(99,102,241,0.1)',
                                            lineHeight: 1.4,
                                        }}>{text || children}</div>
                                    },
                                    h3: ({children}) => {
                                        const text = typeof children === 'string'
                                            ? children
                                            : Array.isArray(children)
                                                ? children.map(c => typeof c === 'string' ? c : '').join('')
                                                : ''
                                        return <div style={{
                                            fontWeight: 600, fontSize: 13, marginTop: 10, marginBottom: 4,
                                            color: isUser ? 'rgba(255,255,255,0.9)' : '#334155',
                                            lineHeight: 1.4,
                                        }}>{text || children}</div>
                                    },
                                    h4: ({children}) => {
                                        const text = typeof children === 'string'
                                            ? children
                                            : Array.isArray(children)
                                                ? children.map(c => typeof c === 'string' ? c : '').join('')
                                                : ''
                                        return <div style={{
                                            fontWeight: 600, fontSize: 12, marginTop: 8, marginBottom: 3,
                                            color: isUser ? 'rgba(255,255,255,0.85)' : '#475569',
                                        }}>{text || children}</div>
                                    },
                                    strong: ({children}) => <strong style={{
                                        fontWeight: 650,
                                        color: isUser ? '#fff' : '#0f172a',
                                    }}>{children}</strong>,
                                    em: ({children}) => <em style={{
                                        fontStyle: 'italic',
                                        color: isUser ? 'rgba(255,255,255,0.85)' : '#64748b',
                                    }}>{children}</em>,
                                    a: ({children, href}) => <a href={href} target="_blank" rel="noopener noreferrer"
                                                                style={{
                                                                    textDecoration: 'underline',
                                                                    fontWeight: 500,
                                                                    color: isUser ? 'rgba(255,255,255,0.85)' : '#6366f1',
                                                                    transition: 'opacity 0.15s',
                                                                }}>{children}</a>,
                                    blockquote: ({children}) => (
                                        <blockquote style={{
                                            borderLeft: `3px solid ${isUser ? 'rgba(255,255,255,0.30)' : '#c7d2fe'}`,
                                            paddingLeft: 14, marginLeft: 0, marginRight: 0,
                                            fontStyle: 'italic',
                                            color: isUser ? 'rgba(255,255,255,0.75)' : '#64748b',
                                            margin: '8px 0',
                                        }}>
                                            {children}
                                        </blockquote>
                                    ),
                                    hr: () => (
                                        <hr style={{
                                            border: 'none',
                                            height: 1,
                                            background: isUser
                                                ? 'rgba(255,255,255,0.15)'
                                                : 'rgba(0,0,0,0.06)',
                                            margin: '12px 0',
                                        }}/>
                                    ),
                                    img: ({src, alt}) => (
                                        <img src={src} alt={alt || ''}
                                             style={{
                                                 maxWidth: '100%',
                                                 borderRadius: 10,
                                                 margin: '8px 0',
                                                 border: '1px solid rgba(0,0,0,0.05)',
                                             }}
                                        />
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