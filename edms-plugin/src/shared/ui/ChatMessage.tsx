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
    Paperclip,
    Clock,
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
import { cn } from '@shared/lib/cn'


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
            className={cn(
                "absolute top-2 right-2 p-1.5 rounded-lg border transition-all duration-200",
                copied
                  ? "bg-emerald-50 border-emerald-200 text-emerald-600"
                  : "bg-white/80 border-zinc-200 text-zinc-400 hover:text-zinc-600 hover:border-zinc-300 shadow-sm"
            )}
        >
            {copied ? <Check size={14}/> : <Copy size={14}/>}
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
        <div className="flex flex-col gap-1 my-3 bg-zinc-50 dark:bg-zinc-800/30 rounded-xl p-2 border border-zinc-100 dark:border-zinc-800">
            {rows.map(([key, value], i) => (
                <div key={i} className="flex gap-4 p-2.5 rounded-lg hover:bg-white dark:hover:bg-zinc-800 transition-colors shadow-none hover:shadow-sm">
                    <span className="text-[11px] font-bold text-zinc-400 dark:text-zinc-500 min-w-[120px] flex-shrink-0 uppercase tracking-wider mt-0.5">
                        {key}
                    </span>
                    <span className="text-[13px] text-zinc-800 dark:text-zinc-200 font-medium break-words leading-relaxed">
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
            <div className="overflow-x-auto my-3 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm">
                <table className="w-full text-[13px] border-collapse bg-white dark:bg-zinc-900">{children}</table>
            </div>
        )
    }

    if (isKeyValueTable(headers)) return <KeyValueList rows={rows}/>

    if (isAttachmentTable(headers)) {
        return (
            <div className="my-2 space-y-2">
                {rows.map((row, i) => <AttachmentCard key={i} headers={headers} row={row} index={i}/>)}
            </div>
        )
    }

    const h = headers.join(' ').toLowerCase()
    const isDocList = /(id|uuid|идентификатор|doc.*id|document.*id)/.test(h) || /рег.*номер|рег\.номер/.test(h) || (/дата/.test(h) && /категор/.test(h))

    if (isDocList) {
        return (
            <div className="my-2 space-y-3">
                {rows.map((row, i) => <DocCard key={i} headers={headers} row={row} index={i}/>)}
            </div>
        )
    }

    return (
        <div className="overflow-x-auto my-3 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm">
            <table className="w-full text-[13px] border-collapse bg-white dark:bg-zinc-900">{children}</table>
        </div>
    )
}

// ─── Error message component ─────────────────────────────────────────────────

function ErrorMessage({raw, timeLabel}: { raw: string; timeLabel: string }) {
    const friendly = humanizeError(raw)
    return (
        <div className="bg-rose-50 dark:bg-rose-900/10 border border-rose-100 dark:border-rose-900/20 rounded-xl p-4 max-w-[90%] animate-edms-slide-up">
            <div className="flex items-center gap-2 mb-2">
                <AlertCircle size={16} className="text-rose-500" />
                <span className="text-[13px] font-bold text-rose-800 dark:text-rose-400 uppercase tracking-tight">Ошибка</span>
            </div>
            <p className="text-[14px] text-rose-700 dark:text-rose-300 leading-relaxed m-0">
                {friendly}
            </p>
            <div className="mt-2.5 text-[11px] text-rose-400 dark:text-rose-500/60 font-medium flex items-center gap-1.5">
                <Clock size={11} />
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
            <div className="flex w-full justify-start px-2">
                <ErrorMessage raw={rawText} timeLabel={timeLabel}/>
            </div>
        )
    }

    // ── Structured output detection ──
    const structured = !isUser ? parseStructuredOutput(content) : null

    // ── Sanitize HTML → markdown ──
    const sanitized = sanitizeHtmlToMarkdown(content)

    return (
        <div className={cn("flex w-full mb-1", isUser ? "justify-end" : "justify-start")}>
            <div
                className={cn(
                    "max-w-[92%] px-4 py-3 edms-chat-text transition-all duration-300 animate-edms-fade-in",
                    isUser
                        ? "bg-blue-600 dark:bg-blue-600 text-white rounded-2xl rounded-tr-sm shadow-sm"
                        : "bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 border border-zinc-200 dark:border-zinc-800 rounded-2xl rounded-tl-sm shadow-sm"
                )}
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
                                    thead: ({children}) => <thead className="bg-zinc-50 dark:bg-zinc-800/50">{children}</thead>,
                                    tbody: ({children}) => <tbody>{children}</tbody>,
                                    tr: ({children}) => <tr className="border-b border-zinc-100 dark:border-zinc-800 last:border-0">{children}</tr>,
                                    th: ({children}) => <th className="px-4 py-2.5 text-left text-[11px] font-bold text-zinc-500 uppercase tracking-wider">{children}</th>,
                                    td: ({children}) => <td className="px-4 py-3 text-[13px] text-zinc-700 dark:text-zinc-300 leading-relaxed">{children}</td>,
                                    code({inline, className, children, ...props}: any) {
                                        const codeText = String(children).replace(/\n$/, '')
                                        if (!inline) {
                                            const lang = className?.replace(/^language-/, '') || ''
                                            return (
                                                <div className="relative my-4 rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-700 shadow-sm">
                                                    {lang && (
                                                        <div className="bg-zinc-50 dark:bg-zinc-800 px-4 py-1.5 text-[10px] font-bold text-zinc-500 border-b border-zinc-200 dark:border-zinc-700 uppercase tracking-widest">
                                                            {lang}
                                                        </div>
                                                    )}
                                                    <pre className="p-4 bg-zinc-900 dark:bg-black overflow-x-auto">
                                                        <code className={cn("font-mono text-[13px] leading-relaxed text-zinc-200", className)} {...props}>{children}</code>
                                                    </pre>
                                                    <CopyButton text={codeText}/>
                                                </div>
                                            )
                                        }
                                        return <code className={cn(
                                            "font-mono text-[0.9em] px-1.5 py-0.5 rounded-md",
                                            isUser ? "bg-white/20 text-white" : "bg-zinc-100 dark:bg-zinc-800 text-blue-600 dark:text-blue-400"
                                        )} {...props}>{children}</code>
                                    },
                                    p: ({children}) => {
                                        const rawText = Array.isArray(children)
                                            ? children.map(c => typeof c === 'string' ? c : (c?.props?.children ?? '')).join('')
                                            : typeof children === 'string' ? children : ''
                                        const fileMatch = rawText.match(/Файл:\s*(\S[^\t\n]+?)(?:\s{2,}|\s*Размер:)/)
                                        if (fileMatch?.[1] && !isUser && onAttachmentClick) {
                                            const fileName = fileMatch[1].trim()
                                            const fileSize = rawText.match(/Размер:\s*([^\n]+)/)?.[1]?.trim() ?? ''
                                            return (
                                                <div
                                                  className="flex items-center gap-3 p-3 my-2 bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 rounded-xl hover:shadow-md hover:border-blue-200 dark:hover:border-blue-900/50 transition-all cursor-pointer group"
                                                  onClick={() => onAttachmentClick(fileName)}
                                                >
                                                    <div className="p-2 bg-white dark:bg-zinc-800 rounded-lg shadow-sm group-hover:text-blue-500">
                                                      <Paperclip size={16} />
                                                    </div>
                                                    <div className="flex-1 min-w-0">
                                                        <div className="text-[13px] font-semibold text-zinc-900 dark:text-zinc-100 truncate">{fileName}</div>
                                                        {fileSize && <div className="text-[11px] text-zinc-500 mt-0.5">{fileSize}</div>}
                                                    </div>
                                                    <ExternalLink size={14} className="text-zinc-400 group-hover:text-blue-500" />
                                                </div>
                                            )
                                        }
                                        return <p className="mb-2 last:mb-0 leading-relaxed text-[15px]">{children}</p>
                                    },
                                    ul: ({children, depth}: any) => (
                                        <ul className={cn("mb-3 space-y-1.5 list-none", depth > 0 ? "pl-4" : "pl-1")}>{children}</ul>
                                    ),
                                    ol: ({children}: any) => (
                                        <ol className="mb-3 space-y-1.5 list-none pl-1 counter-reset-edms-ol">{children}</ol>
                                    ),
                                    li: ({children, ordered, index}: any) => (
                                        <li className="relative pl-7 text-[15px] leading-relaxed">
                                            <span className="absolute left-0 top-1.5 flex items-center justify-center w-4 h-4 text-[10px] font-bold text-blue-500 dark:text-blue-400">
                                                {ordered ? `${(index ?? 0) + 1}.` : '•'}
                                            </span>
                                            {children}
                                        </li>
                                    ),
                                    h1: ({children}) => <h1 className="text-xl font-bold mt-6 mb-3 text-zinc-900 dark:text-zinc-100 tracking-tight">{children}</h1>,
                                    h2: ({children}) => <h2 className="text-lg font-bold mt-5 mb-2.5 text-zinc-900 dark:text-zinc-100 tracking-tight">{children}</h2>,
                                    h3: ({children}) => <h3 className="text-base font-bold mt-4 mb-2 text-zinc-900 dark:text-zinc-100 tracking-tight">{children}</h3>,
                                    h4: ({children}) => <h4 className="text-sm font-bold mt-3 mb-1.5 text-zinc-900 dark:text-zinc-100 tracking-tight uppercase tracking-wider">{children}</h4>,
                                    strong: ({children}) => <strong className="font-bold text-zinc-900 dark:text-zinc-100">{children}</strong>,
                                    em: ({children}) => <em className="italic text-zinc-700 dark:text-zinc-300">{children}</em>,
                                    a: ({children, href}) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 font-medium underline underline-offset-4 decoration-blue-500/30 hover:decoration-blue-500 transition-all">{children}</a>,
                                    blockquote: ({children}) => (
                                        <blockquote className="border-l-4 border-zinc-200 dark:border-zinc-800 pl-4 my-4 italic text-zinc-600 dark:text-zinc-400 leading-relaxed">
                                            {children}
                                        </blockquote>
                                    ),
                                    hr: () => <hr className="my-6 border-zinc-100 dark:border-zinc-800"/>,
                                    img: ({src, alt}) => (
                                        <img src={src} alt={alt || ''} className="max-w-full rounded-xl my-4 border border-zinc-100 dark:border-zinc-800 shadow-md"/>
                                    ),
                                }}
                            >
                                {sanitized}
                            </ReactMarkdown>
                        )}
                    </DocumentClickContext.Provider>
                </AttachmentClickContext.Provider>

                <div className={cn(
                    "mt-2 text-[10px] font-medium flex items-center gap-1 opacity-50",
                    isUser ? "justify-end text-white" : "justify-start text-zinc-500"
                )}>
                    <Clock size={10} />
                    {timeLabel}
                </div>
            </div>
        </div>
    )
}
