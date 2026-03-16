import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import dayjs from 'dayjs'

interface Props {
    content: string
    role: 'user' | 'assistant'
    timestamp: number
    isError?: boolean
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

export function ChatMessage({content, role, timestamp, isError}: Props) {
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
                            width: 22,
                            height: 22,
                            borderRadius: 6,
                            background: 'rgba(220,38,38,0.12)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: 13,
                            flexShrink: 0
                        }}>⚠️</span>
                        <span style={{fontSize: 12, fontWeight: 700, color: '#b91c1c', letterSpacing: '-0.01em'}}>Не удалось выполнить запрос</span>
                    </div>
                    <p style={{
                        fontSize: 12,
                        color: '#991b1b',
                        lineHeight: 1.5,
                        margin: 0,
                        opacity: 0.9
                    }}>{humanizeError(display)}</p>
                    <p style={{fontSize: 11, color: '#b91c1c', opacity: 0.6, marginTop: 6, marginBottom: 0}}>Попробуйте
                        повторить запрос или обновите страницу.</p>
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
                className="max-w-[88%] px-4 py-3 leading-relaxed edms-chat-text"
                style={isUser ? {
                    background: '#6366f1',
                    color: '#ffffff',
                    borderRadius: '20px 20px 4px 20px',
                    boxShadow: '0 2px 12px rgba(99,102,241,0.28)',
                } : {
                    background: '#ffffff',
                    color: '#0f172a',
                    borderRadius: '20px 20px 20px 4px',
                    boxShadow: '0 1px 6px rgba(0,0,0,0.08)',
                    border: '1px solid rgba(226,232,240,0.80)',
                }}
            >
                <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                        code({inline, className, children, ...props}: any) {
                            if (!inline) {
                                return (
                                    <pre style={{
                                        overflow: 'auto',
                                        padding: '10px 12px',
                                        margin: '8px 0',
                                        borderRadius: 10,
                                        background: '#1e293b',
                                        border: '1px solid rgba(255,255,255,0.06)'
                                    }}>
                                        <code className={className} style={{
                                            fontFamily: 'monospace',
                                            fontSize: 12,
                                            color: '#a5b4fc'
                                        }} {...props}>{children}</code>
                                    </pre>
                                )
                            }
                            return (
                                <code style={{
                                    fontFamily: 'monospace',
                                    fontSize: 12,
                                    padding: '1px 5px',
                                    borderRadius: 4,
                                    background: isUser ? 'rgba(255,255,255,0.20)' : 'rgba(99,102,241,0.09)',
                                    color: isUser ? '#fff' : '#4338ca'
                                }} {...props}>{children}</code>
                            )
                        },
                        p: ({children}) => <p style={{
                            marginBottom: 6,
                            lineHeight: 1.65,
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word'
                        }}>{children}</p>,
                        ul: ({children}) => <ul style={{paddingLeft: 20, marginBottom: 6}}>{children}</ul>,
                        ol: ({children}) => <ol style={{paddingLeft: 20, marginBottom: 6}}>{children}</ol>,
                        li: ({children}) => <li style={{marginBottom: 2}}>{children}</li>,
                        strong: ({children}) => <strong
                            style={{fontWeight: 600, color: isUser ? '#fff' : '#0f172a'}}>{children}</strong>,
                        a: ({children, href}) => (
                            <a href={href} target="_blank" rel="noopener noreferrer"
                               style={{
                                   textDecoration: 'underline',
                                   fontWeight: 500,
                                   color: isUser ? 'rgba(255,255,255,0.85)' : '#6366f1'
                               }}>
                                {children}
                            </a>
                        ),
                        blockquote: ({children}) => (
                            <blockquote style={{
                                borderLeft: `3px solid ${isUser ? 'rgba(255,255,255,0.40)' : '#c7d2fe'}`,
                                paddingLeft: 10,
                                marginLeft: 0,
                                fontStyle: 'italic',
                                color: isUser ? 'rgba(255,255,255,0.80)' : '#475569'
                            }}>
                                {children}
                            </blockquote>
                        ),
                        table: ({children}) => (
                            <div style={{
                                overflowX: 'auto',
                                margin: '8px 0',
                                borderRadius: 8,
                                border: '1px solid rgba(226,232,240,0.70)'
                            }}>
                                <table
                                    style={{width: '100%', fontSize: 12, borderCollapse: 'collapse'}}>{children}</table>
                            </div>
                        ),
                        th: ({children}) => <th style={{
                            padding: '6px 10px',
                            borderBottom: '1px solid rgba(226,232,240,0.70)',
                            fontWeight: 600,
                            background: 'rgba(241,245,249,0.80)',
                            textAlign: 'left'
                        }}>{children}</th>,
                        td: ({children}) => <td style={{
                            padding: '6px 10px',
                            borderBottom: '1px solid rgba(226,232,240,0.40)'
                        }}>{children}</td>,
                    }}
                >
                    {display}
                </ReactMarkdown>

                <div style={{
                    marginTop: 4,
                    opacity: 0.55,
                    textAlign: isUser ? 'right' : 'left',
                    color: isUser ? 'rgba(255,255,255,0.80)' : '#94a3b8',
                    fontSize: 10
                }}>
                    {timeLabel}
                </div>
            </div>
        </div>
    )
}