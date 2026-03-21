import {createContext, useContext} from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import dayjs from 'dayjs'

interface Props {
    content: string
    role: 'user' | 'assistant'
    timestamp: number
    isError?: boolean
    onAttachmentClick?: (fileName: string) => void
}


// ─── Attachment click context ─────────────────────────────────────────────────
const AttachmentClickContext = createContext<((fileName: string) => void) | null>(null)

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

// ─── Table data extraction helpers ────────────────────────────────────────────

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

// ─── Detect key-value table ────────────────────────────────────────────────────

function isAttachmentTable(headers: string[]): boolean {
    const h = headers.join(' ').toLowerCase()
    const hasFile = /файл|вложени|название файл/.test(h)
    const hasSize = /размер|size/.test(h)
    return hasFile || hasSize
}

function isKeyValueTable(headers: string[]): boolean {
    if (headers.length !== 2) return false
    const h0 = headers[0].toLowerCase()
    const h1 = headers[1].toLowerCase()
    const kvKeys = ['параметр', 'поле', 'ключ', 'field', 'key', 'свойство', 'атрибут']
    const kvVals = ['информация', 'значение', 'value', 'данные', 'data', 'содержание']
    return kvKeys.some(k => h0.includes(k)) || kvVals.some(k => h1.includes(k))
}

// ─── Key-value list renderer ───────────────────────────────────────────────────

function KeyValueList({rows}: { rows: string[][] }) {
    return (
        <div style={{display: 'flex', flexDirection: 'column', gap: 4, margin: '6px 0'}}>
            {rows.map(([key, value], i) => (
                <div key={i} style={{
                    display: 'flex',
                    gap: 8,
                    padding: '5px 8px',
                    borderRadius: 8,
                    background: i % 2 === 0 ? 'rgba(241,245,249,0.60)' : 'transparent',
                    alignItems: 'flex-start',
                }}>
                    <span style={{
                        fontSize: 11,
                        fontWeight: 600,
                        color: '#475569',
                        minWidth: 120,
                        flexShrink: 0,
                        lineHeight: 1.5,
                    }}>
                        {key}
                    </span>
                    <span style={{
                        fontSize: 12,
                        color: '#0f172a',
                        lineHeight: 1.5,
                        wordBreak: 'break-word',
                    }}>
                        {value || '—'}
                    </span>
                </div>
            ))}
        </div>
    )
}

// ─── Category badge styles ─────────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, { bg: string; text: string; label: string }> = {
    'INCOMING': {bg: 'rgba(59,130,246,0.10)', text: '#1d4ed8', label: 'Входящий'},
    'OUTGOING': {bg: 'rgba(16,185,129,0.10)', text: '#065f46', label: 'Исходящий'},
    'INTERN': {bg: 'rgba(139,92,246,0.10)', text: '#5b21b6', label: 'Внутренний'},
    'APPEAL': {bg: 'rgba(245,158,11,0.10)', text: '#92400e', label: 'Обращение'},
    'CONTRACT': {bg: 'rgba(239,68,68,0.10)', text: '#991b1b', label: 'Договор'},
    'MEETING': {bg: 'rgba(99,102,241,0.10)', text: '#3730a3', label: 'Совещание'},
}

function getCategoryStyle(raw: string) {
    const upper = raw.toUpperCase().replace(/[()]/g, '').trim()
    for (const [key, val] of Object.entries(CATEGORY_COLORS)) {
        if (upper.includes(key)) return val
    }
    return {bg: 'rgba(100,116,139,0.10)', text: '#334155', label: raw}
}

// ─── DocCard — карточка для поиска документов ─────────────────────────────────

function DocCard({headers, row, index}: { headers: string[]; row: string[]; index: number }) {
    const pairs = headers.map((h, i) => ({key: h, value: row[i] || '—'}))
    const num = pairs.find(p => p.key === '№' || p.key === '#')?.value
    const regNum = pairs.find(p => /рег|номер|number/i.test(p.key))?.value
    const date = pairs.find(p => /дата|date/i.test(p.key))?.value
    const category = pairs.find(p => /категор|category|тип|type/i.test(p.key))?.value
    const summary = pairs.find(p => /содержан|summary|краткое|описан/i.test(p.key))?.value
    const author = pairs.find(p => /автор|author/i.test(p.key))?.value
    const status = pairs.find(p => /статус|status/i.test(p.key))?.value
    const catStyle = category ? getCategoryStyle(category) : null

    return (
        <div style={{
            background: '#fff',
            border: '1px solid rgba(226,232,240,0.80)',
            borderRadius: 12,
            padding: '10px 12px',
            marginBottom: 6,
            boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
        }}>
            <div style={{display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6}}>
                <span style={{
                    width: 22, height: 22, borderRadius: 6,
                    background: 'rgba(99,102,241,0.10)',
                    color: '#6366f1', fontSize: 10, fontWeight: 700,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                }}>{num ?? index + 1}</span>
                {regNum && <span style={{fontSize: 12, fontWeight: 700, color: '#0f172a', flex: 1}}>{regNum}</span>}
                {date && <span style={{fontSize: 10, color: '#94a3b8', flexShrink: 0}}>{date}</span>}
            </div>
            {summary && (
                <p style={{
                    fontSize: 11.5, color: '#334155', lineHeight: 1.4, margin: '0 0 6px 0',
                    display: '-webkit-box', WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical', overflow: 'hidden',
                }}>{summary}</p>
            )}
            <div style={{display: 'flex', flexWrap: 'wrap', gap: 4}}>
                {catStyle && (
                    <span style={{
                        fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 20,
                        background: catStyle.bg, color: catStyle.text,
                    }}>{catStyle.label}</span>
                )}
                {author && (
                    <span style={{
                        fontSize: 10,
                        color: '#64748b',
                        padding: '2px 7px',
                        borderRadius: 20,
                        background: 'rgba(100,116,139,0.08)'
                    }}>
                        👤 {author}
                    </span>
                )}
                {status && (
                    <span style={{
                        fontSize: 10,
                        color: '#64748b',
                        padding: '2px 7px',
                        borderRadius: 20,
                        background: 'rgba(100,116,139,0.08)'
                    }}>
                        {status}
                    </span>
                )}
                {pairs
                    .filter(p => !['№', '#'].includes(p.key)
                        && !/рег|номер|number|дата|date|категор|category|тип|type|содержан|summary|краткое|описан|автор|author|статус|status/i.test(p.key)
                        && p.value && p.value !== '—')
                    .map(({key, value}) => (
                        <span key={key} style={{
                            fontSize: 10,
                            color: '#64748b',
                            padding: '2px 7px',
                            borderRadius: 20,
                            background: 'rgba(100,116,139,0.08)'
                        }}>
                            {key}: {value}
                        </span>
                    ))}
            </div>
        </div>
    )
}


// ─── AttachmentCard — кликабельная карточка вложения ─────────────────────────
function AttachmentCard({headers, row, index}: { headers: string[]; row: string[]; index: number }) {
    const onAttachmentClick = useContext(AttachmentClickContext)
    const pairs = headers.map((h, i) => ({key: h, value: row[i] || ''}))

    // Извлекаем имя файла и размер из любого столбца
    const fileName = pairs.find(p => /файл|название|name/i.test(p.key))?.value
        || pairs.find(p => p.value && /\.(docx?|pdf|xlsx?|txt|rtf)/i.test(p.value))?.value
        || ''
    const fileSize = pairs.find(p => /размер|size/i.test(p.key))?.value || ''
    const fileDate = pairs.find(p => /дата|date/i.test(p.key))?.value || ''

    const ext = (fileName.match(/\.([a-z]+)$/i)?.[1] || '').toLowerCase()
    const icon = ext === 'pdf' ? '📄' : ext === 'doc' || ext === 'docx' ? '📝'
        : ext === 'xls' || ext === 'xlsx' ? '📊' : '📎'

    const handleClick = () => {
        if (fileName && onAttachmentClick) onAttachmentClick(fileName)
    }

    return (
        <div
            onClick={handleClick}
            style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '9px 12px', marginBottom: 5, borderRadius: 10,
                background: 'rgba(241,245,249,0.90)',
                border: '1px solid rgba(203,213,225,0.70)',
                cursor: onAttachmentClick ? 'pointer' : 'default',
                transition: 'all 0.15s',
                userSelect: 'none',
            }}
            onMouseEnter={e => {
                if (!onAttachmentClick) return
                const el = e.currentTarget as HTMLDivElement
                el.style.background = 'rgba(99,102,241,0.08)'
                el.style.borderColor = 'rgba(99,102,241,0.35)'
            }}
            onMouseLeave={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.background = 'rgba(241,245,249,0.90)'
                el.style.borderColor = 'rgba(203,213,225,0.70)'
            }}
        >
            {/* Иконка файла */}
            <span style={{fontSize: 20, flexShrink: 0, lineHeight: 1}}>{icon}</span>

            {/* Имя файла + размер */}
            <div style={{flex: 1, minWidth: 0}}>
                <p style={{
                    margin: 0, fontSize: 12, fontWeight: 600, color: '#1e293b',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                    {fileName || `Вложение ${index + 1}`}
                </p>
                <p style={{margin: 0, fontSize: 10, color: '#94a3b8', marginTop: 1}}>
                    {[fileDate, fileSize].filter(Boolean).join('  ·  ')}
                </p>
            </div>

            {/* Стрелка "открыть" */}
            {onAttachmentClick && (
                <span style={{color: '#94a3b8', fontSize: 12, flexShrink: 0}}>›</span>
            )}
        </div>
    )
}

// ─── SmartTable — главный роутер таблиц ──────────────────────────────────────

function SmartTable({children}: { children: React.ReactNode }) {
    const {headers, rows} = extractTableData(children)

    if (!headers.length || !rows.length) {
        return (
            <div style={{
                overflowX: 'auto',
                margin: '8px 0',
                borderRadius: 8,
                border: '1px solid rgba(226,232,240,0.70)'
            }}>
                <table style={{width: '100%', fontSize: 12, borderCollapse: 'collapse'}}>{children}</table>
            </div>
        )
    }

    // Роутинг: key-value таблица → KeyValueList
    if (isKeyValueTable(headers)) {
        return <KeyValueList rows={rows}/>
    }

    // Роутинг: таблица вложений → AttachmentCard (кликабельные)
    if (isAttachmentTable(headers)) {
        return (
            <div style={{margin: '6px 0'}}>
                {rows.map((row, i) => (
                    <AttachmentCard key={i} headers={headers} row={row} index={i}/>
                ))}
            </div>
        )
    }

    // Роутинг: таблица документов → DocCards
    return (
        <div style={{margin: '6px 0'}}>
            {rows.map((row, i) => (
                <DocCard key={i} headers={headers} row={row} index={i}/>
            ))}
        </div>
    )
}

// ─── ChatMessage ──────────────────────────────────────────────────────────────

export function ChatMessage({content, role, timestamp, isError, onAttachmentClick}: Props) {
    const isUser = role === 'user'
    const isErr = isError ?? isErrorMessage(content)
    const display = isErr ? extractErrorText(content) : content
    const timeLabel = dayjs(timestamp).format('HH:mm')

    if (isErr) {
        return (
            <div className="flex w-full justify-start" style={{animation: 'edms-fade-in .25s ease-out'}}>
                <div style={{
                    maxWidth: '88%', padding: '12px 14px',
                    borderRadius: '18px 18px 18px 4px',
                    background: 'rgba(254,226,226,0.92)',
                    border: '1px solid rgba(252,165,165,0.50)',
                    boxShadow: '0 1px 6px rgba(220,38,38,0.08)',
                }}>
                    <div style={{display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6}}>
                        <span style={{
                            width: 22, height: 22, borderRadius: 6,
                            background: 'rgba(220,38,38,0.12)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: 13, flexShrink: 0,
                        }}>⚠️</span>
                        <span style={{fontSize: 12, fontWeight: 700, color: '#b91c1c', letterSpacing: '-0.01em'}}>
                            Не удалось выполнить запрос
                        </span>
                    </div>
                    <p style={{fontSize: 12, color: '#991b1b', lineHeight: 1.5, margin: 0, opacity: 0.9}}>
                        {humanizeError(display)}
                    </p>
                    <p style={{fontSize: 11, color: '#b91c1c', opacity: 0.6, marginTop: 6, marginBottom: 0}}>
                        Попробуйте повторить запрос или обновите страницу.
                    </p>
                    <div style={{fontSize: 10, opacity: 0.45, marginTop: 6, color: '#7f1d1d'}}>{timeLabel}</div>
                </div>
            </div>
        )
    }

    return (
        <div
            className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
            style={{animation: 'edms-fade-in .25s ease-out'}}
        >
            <div
                className="max-w-full px-4 py-3 leading-relaxed edms-chat-text"
                style={isUser ? {
                    background: '#6366f1',
                    color: '#ffffff',
                    borderRadius: '20px 20px 4px 20px',
                    boxShadow: '0 2px 12px rgba(99,102,241,0.28)',
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word',
                } : {
                    background: '#ffffff',
                    color: '#0f172a',
                    borderRadius: '20px 20px 20px 4px',
                    boxShadow: '0 1px 6px rgba(0,0,0,0.08)',
                    border: '1px solid rgba(226,232,240,0.80)',
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word',
                }}
            >
                <AttachmentClickContext.Provider value={onAttachmentClick ?? null}>
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            // Таблица → SmartTable роутер
                            table: ({children}) => <SmartTable>{children}</SmartTable>,
                            thead: ({children}) => <thead>{children}</thead>,
                            tbody: ({children}) => <tbody>{children}</tbody>,
                            tr: ({children}) => <tr>{children}</tr>,
                            th: ({children}) => <th style={{
                                padding: '6px 10px',
                                borderBottom: '1px solid rgba(226,232,240,0.70)',
                                fontWeight: 600,
                                background: 'rgba(241,245,249,0.80)',
                                textAlign: 'left', fontSize: 12,
                            }}>{children}</th>,
                            td: ({children}) => <td style={{
                                padding: '6px 10px',
                                borderBottom: '1px solid rgba(226,232,240,0.40)',
                                fontSize: 12,
                            }}>{children}</td>,
                            code({inline, className, children, ...props}: any) {
                                if (!inline) {
                                    return (
                                        <pre style={{
                                            overflow: 'auto', padding: '10px 12px', margin: '8px 0',
                                            borderRadius: 10, background: '#1e293b',
                                            border: '1px solid rgba(255,255,255,0.06)',
                                        }}>
                                        <code className={className} style={{
                                            fontFamily: 'monospace', fontSize: 12, color: '#a5b4fc',
                                        }} {...props}>{children}</code>
                                    </pre>
                                    )
                                }
                                return (
                                    <code style={{
                                        fontFamily: 'monospace', fontSize: 12,
                                        padding: '1px 5px', borderRadius: 4,
                                        background: isUser ? 'rgba(255,255,255,0.20)' : 'rgba(99,102,241,0.09)',
                                        color: isUser ? '#fff' : '#4338ca',
                                    }} {...props}>{children}</code>
                                )
                            },
                            p: ({children}) => {
                                const rawText = Array.isArray(children)
                                    ? children.map(c => typeof c === 'string' ? c : (c?.props?.children ?? '')).join('')
                                    : typeof children === 'string' ? children : ''
                                const fileMatch = rawText.match(/Файл:\s*([^\s][^\t\n]+?)(?:\s{2,}|\s*Размер:)/)
                                if (fileMatch && !isUser && onAttachmentClick) {
                                    const fileName = fileMatch[1].trim()
                                    return (
                                        <button
                                            type="button"
                                            onClick={() => onAttachmentClick(fileName)}
                                            style={{
                                                display: 'flex', alignItems: 'center', gap: 6,
                                                width: '100%', padding: '7px 10px', marginBottom: 4,
                                                borderRadius: 10, cursor: 'pointer', textAlign: 'left',
                                                background: 'rgba(241,245,249,0.90)',
                                                border: '1px solid rgba(203,213,225,0.70)',
                                                color: '#334155', fontSize: 12, fontWeight: 500,
                                                transition: 'all 0.15s',
                                            }}
                                            onMouseEnter={e => {
                                                const el = e.currentTarget as HTMLButtonElement
                                                el.style.background = 'rgba(99,102,241,0.08)'
                                                el.style.borderColor = 'rgba(99,102,241,0.30)'
                                                el.style.color = '#4338ca'
                                            }}
                                            onMouseLeave={e => {
                                                const el = e.currentTarget as HTMLButtonElement
                                                el.style.background = 'rgba(241,245,249,0.90)'
                                                el.style.borderColor = 'rgba(203,213,225,0.70)'
                                                el.style.color = '#334155'
                                            }}
                                        >
                                            <span style={{fontSize: 14, flexShrink: 0}}>📎</span>
                                            <span style={{
                                                flex: 1,
                                                minWidth: 0,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap'
                                            }}>
                                        {fileName}
                                    </span>
                                            <span style={{
                                                fontSize: 10,
                                                color: '#94a3b8',
                                                flexShrink: 0,
                                                fontWeight: 400
                                            }}>
                                        {rawText.match(/Размер:\s*([^\n]+)/)?.[1]?.trim() ?? ''}
                                    </span>
                                        </button>
                                    )
                                }
                                return <p style={{
                                    marginBottom: 6, lineHeight: 1.65,
                                    whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                                }}>{children}</p>
                            },
                            ul: ({children}) => <ul style={{paddingLeft: 20, marginBottom: 6}}>{children}</ul>,
                            ol: ({children}) => <ol style={{paddingLeft: 20, marginBottom: 6}}>{children}</ol>,
                            li: ({children}) => <li style={{marginBottom: 2}}>{children}</li>,

                            h2: ({children}) => {
                                const text = typeof children === 'string' ? children
                                    : Array.isArray(children) ? children.map(c => typeof c === 'string' ? c : '').join('') : ''
                                const clean = text.replace(/^\d+\s+/, '').replace(/^[①②③④⑤⑥⑦⑧⑨⑩]\s*/, '').trim()
                                return (
                                    <p style={{
                                        fontWeight: 700, fontSize: 13, marginTop: 10, marginBottom: 4,
                                        color: isUser ? '#fff' : '#1e293b',
                                        borderBottom: isUser ? '1px solid rgba(255,255,255,0.2)' : '1px solid rgba(226,232,240,0.70)',
                                        paddingBottom: 3,
                                    }}>{clean || children}</p>
                                )
                            },
                            h3: ({children}) => {
                                const text = typeof children === 'string' ? children
                                    : Array.isArray(children) ? children.map(c => typeof c === 'string' ? c : '').join('') : ''
                                const clean = text.replace(/^\d+\s+/, '').replace(/^[①②③④⑤⑥⑦⑧⑨⑩]\s*/, '').trim()
                                return (
                                    <p style={{
                                        fontWeight: 600, fontSize: 12, marginTop: 8, marginBottom: 2,
                                        color: isUser ? 'rgba(255,255,255,0.85)' : '#334155',
                                    }}>{clean || children}</p>
                                )
                            },
                            strong: ({children}) => <strong style={{
                                fontWeight: 600, color: isUser ? '#fff' : '#0f172a',
                            }}>{children}</strong>,
                            a: ({children, href}) => (
                                <a href={href} target="_blank" rel="noopener noreferrer" style={{
                                    textDecoration: 'underline', fontWeight: 500,
                                    color: isUser ? 'rgba(255,255,255,0.85)' : '#6366f1',
                                }}>
                                    {children}
                                </a>
                            ),
                            blockquote: ({children}) => (
                                <blockquote style={{
                                    borderLeft: `3px solid ${isUser ? 'rgba(255,255,255,0.40)' : '#c7d2fe'}`,
                                    paddingLeft: 10, marginLeft: 0, fontStyle: 'italic',
                                    color: isUser ? 'rgba(255,255,255,0.80)' : '#475569',
                                }}>
                                    {children}
                                </blockquote>
                            ),
                        }}
                    >
                        {display}
                    </ReactMarkdown>
                </AttachmentClickContext.Provider>

                <div style={{
                    marginTop: 4, opacity: 0.55,
                    textAlign: isUser ? 'right' : 'left',
                    color: isUser ? 'rgba(255,255,255,0.80)' : '#94a3b8',
                    fontSize: 10,
                }}>
                    {timeLabel}
                </div>
            </div>
        </div>
    )
}

// 5