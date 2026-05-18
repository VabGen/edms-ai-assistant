import {
    createContext,
    useContext,
    useState,
    useCallback,
    type ReactNode
} from 'react'
import {remarkLazyList} from '../plugins/remarkLazyList'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import dayjs from 'dayjs'
import {
    XCircle,
    AlertTriangle,
    CheckCircle,
    FileText,
    Copy,
    Check,
    AlertCircle,
    Sparkles,
    ListChecks,
    BookOpen,
    Target,
    Globe,
    FileSearch,
    Brain,
    ChevronDown,
    ChevronRight,
    Clock,
    User,
    Flame,
    Zap,
    ArrowRight,
    Quote,
    ExternalLink,
} from 'lucide-react'


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

interface ThesisPlanData {
    main_argument: string
    sections: Array<{
        title: string
        thesis: string
        points: Array<{
            claim: string
            evidence?: string | null
            sub_points?: string[]
        }>
    }>
    conclusion: string
}

interface AbstractiveData {
    summary: string
    key_themes: string[]
}

interface ExtractiveData {
    facts: Array<{
        category: string
        label: string
        value: string
    }>
    document_summary: string
}

interface ActionItemsData {
    action_items: Array<{
        task: string
        owner?: string | null
        deadline?: string | null
        priority: 'high' | 'medium' | 'low'
        source_fragment?: string
        confidence?: number
    }>
    document_context: string
}

interface ExecutiveSummaryData {
    headline: string
    bullets: string[]
    recommendation?: string | null
}

interface DetailedNotesData {
    document_type: string
    sections: Array<{
        title: string
        content: string
        subsections?: string[]
    }>
    key_entities: string[]
    date_range?: string | null
}

interface MultilingualData {
    detected_language: string
    summary_language: string
    summary: string
    translation_notes?: string | null
}

type StructuredOutput =
    | { type: 'compliance'; data: ComplianceCheckData }
    | { type: 'thesis'; data: ThesisPlanData }
    | { type: 'abstractive'; data: AbstractiveData }
    | { type: 'extractive'; data: ExtractiveData }
    | { type: 'action_items'; data: ActionItemsData }
    | { type: 'executive'; data: ExecutiveSummaryData }
    | { type: 'detailed_notes'; data: DetailedNotesData }
    | { type: 'multilingual'; data: MultilingualData }


function detectStructuredOutput(content: string): StructuredOutput | null {
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

    if ('overall' in parsed && 'fields' in parsed)
        return {type: 'compliance', data: parsed as ComplianceCheckData}
    if (parsed.tool_use || parsed.tool_calls || parsed.action === 'call_tool') return null
    if ('main_argument' in parsed && 'sections' in parsed)
        return {type: 'thesis', data: parsed as ThesisPlanData}
    if ('key_themes' in parsed && 'summary' in parsed && !('headline' in parsed))
        return {type: 'abstractive', data: parsed as AbstractiveData}
    if ('facts' in parsed && 'document_summary' in parsed)
        return {type: 'extractive', data: parsed as ExtractiveData}
    if ('action_items' in parsed)
        return {type: 'action_items', data: parsed as ActionItemsData}
    if ('headline' in parsed && 'bullets' in parsed)
        return {type: 'executive', data: parsed as ExecutiveSummaryData}
    if ('document_type' in parsed && 'sections' in parsed && !('main_argument' in parsed))
        return {type: 'detailed_notes', data: parsed as DetailedNotesData}
    if ('detected_language' in parsed && 'summary_language' in parsed)
        return {type: 'multilingual', data: parsed as MultilingualData}

    return null
}


const CARD: React.CSSProperties = {
    background: '#ffffff',
    borderRadius: 14,
    border: '1px solid rgba(0,0,0,0.06)',
    overflow: 'hidden',
    fontSize: 13,
    animation: 'edms-slide-up .3s ease-out',
}

const CARD_HEADER: React.CSSProperties = {
    padding: '14px 16px',
    borderBottom: '1px solid rgba(0,0,0,0.05)',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
}

const BADGE_BASE: React.CSSProperties = {
    fontSize: 10,
    fontWeight: 600,
    padding: '3px 9px',
    borderRadius: 20,
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    whiteSpace: 'nowrap',
}

const THEME_PALETTE: Record<string, { bg: string; text: string; border: string }> = {
    default: {bg: 'rgba(100,116,139,0.07)', text: '#475569', border: 'rgba(100,116,139,0.12)'},
    blue: {bg: 'rgba(59,130,246,0.07)', text: '#1d4ed8', border: 'rgba(59,130,246,0.12)'},
    indigo: {bg: 'rgba(99,102,241,0.07)', text: '#4338ca', border: 'rgba(99,102,241,0.12)'},
    violet: {bg: 'rgba(139,92,246,0.07)', text: '#5b21b6', border: 'rgba(139,92,246,0.12)'},
    green: {bg: 'rgba(16,185,129,0.07)', text: '#065f46', border: 'rgba(16,185,129,0.12)'},
    amber: {bg: 'rgba(245,158,11,0.07)', text: '#92400e', border: 'rgba(245,158,11,0.12)'},
    red: {bg: 'rgba(239,68,68,0.07)', text: '#991b1b', border: 'rgba(239,68,68,0.12)'},
    rose: {bg: 'rgba(244,63,94,0.07)', text: '#9f1239', border: 'rgba(244,63,94,0.12)'},
    cyan: {bg: 'rgba(6,182,212,0.07)', text: '#155e75', border: 'rgba(6,182,212,0.12)'},
}


const PRIORITY_CFG: Record<string, { bg: string; text: string; icon: ReactNode; label: string }> = {
    high: {
        bg: 'rgba(239,68,68,0.08)',
        text: '#b91c1c',
        icon: <Flame size={10}/>,
        label: 'Высокий',
    },
    medium: {
        bg: 'rgba(245,158,11,0.08)',
        text: '#92400e',
        icon: <Zap size={10}/>,
        label: 'Средний',
    },
    low: {
        bg: 'rgba(100,116,139,0.08)',
        text: '#475569',
        icon: <Clock size={10}/>,
        label: 'Низкий',
    },
}

const STRICT_SVG_PROPS = {
    width: 12,
    height: 12,
    viewBox: '0 0 16 16',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.5,
    strokeLinecap: 'square' as const,
    strokeLinejoin: 'miter' as const,
};

const FACT_CATEGORY: Record<string, { bg: string; text: string; icon: React.ReactNode; border: string }> = {
    'ДАТА': {
        bg: 'rgba(59,130,246,0.08)',
        text: '#1d4ed8',
        border: 'rgba(59,130,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 4h12v10H2V4zM2 7h12M5 4V2M11 4V2"/>
        </svg>,
    },
    'ПЕРСОНА': {
        bg: 'rgba(139,92,246,0.08)',
        text: '#5b21b6',
        border: 'rgba(139,92,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 6a2 2 0 100-4 2 2 0 000 4zM3 14v-2a5 5 0 0110 0v2"/>
        </svg>,
    },
    'ОРГАНИЗАЦИЯ': {
        bg: 'rgba(16,185,129,0.08)',
        text: '#065f46',
        border: 'rgba(16,185,129,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 14V2h5v4h7v8H7M7 6v8"/>
        </svg>,
    },
    'СУММА': {
        bg: 'rgba(245,158,11,0.08)',
        text: '#92400e',
        border: 'rgba(245,158,11,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 4h12v8H2V4zM5 4v8M8 7h3M8 10h3"/>
        </svg>,
    },
    'ТРЕБОВАНИЕ': {
        bg: 'rgba(239,68,68,0.08)',
        text: '#991b1b',
        border: 'rgba(239,68,68,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 1l6 3v5c0 3.5-6 6-6 6s-6-2.5-6-6V4l6-3zM8 5v3M8 11.5v0.5"/>
        </svg>,
    },
    'СРОК': {
        bg: 'rgba(244,63,94,0.08)',
        text: '#9f1239',
        border: 'rgba(244,63,94,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M4 2h8v3L8 8l4 3v3H4v-3l4-3-4-3V2z"/>
        </svg>,
    },
    'АДРЕС': {
        bg: 'rgba(6,182,212,0.08)',
        text: '#155e75',
        border: 'rgba(6,182,212,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 14s-5-4-5-8a5 5 0 0110 0c0 4-5 8-5 8z"/>
        </svg>,
    },
    'ТЕЛЕФОН': {
        bg: 'rgba(139,92,246,0.08)',
        text: '#5b21b6',
        border: 'rgba(139,92,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M4 1h8v14H4V1zM7 13h2"/>
        </svg>, // Строгий смартфон
    },
    'ЭЛ. ПОЧТА': {
        bg: 'rgba(59,130,246,0.08)',
        text: '#1d4ed8',
        border: 'rgba(59,130,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 3h12v10H2V3zM2 3l6 6 6-6"/>
        </svg>, // Конверт
    },
    'ДОКУМЕНТ': {
        bg: 'rgba(99,102,241,0.08)',
        text: '#4338ca',
        border: 'rgba(99,102,241,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M3 2h7l3 3v9H3V2zM10 2v3h3M5 7h6M5 9h6M5 11h4"/>
        </svg>,
    },
    'ПОДРАЗДЕЛЕНИЕ': {
        bg: 'rgba(20,184,166,0.08)',
        text: '#115e59',
        border: 'rgba(20,184,166,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 2v4M4 6h8M4 6v3M12 6v3M4 9h2M12 9h2"/>
        </svg>,
    },
    'ДОЛЖНОСТЬ': {
        bg: 'rgba(249,115,22,0.08)',
        text: '#9a3412',
        border: 'rgba(249,115,22,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 2l2.5 5H5.5L8 2zM6 7v5M10 7v5M5 12h6"/>
        </svg>,
    },
    'НОМЕР': {
        bg: 'rgba(100,116,139,0.08)',
        text: '#334155',
        border: 'rgba(100,116,139,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M4 3h8M7 2v12M4 13h8"/>
        </svg>, // Исправлен невалидный 'T' на 'M'
    },
    'СТАТУС': {
        bg: 'rgba(192,38,211,0.08)',
        text: '#86198f',
        border: 'rgba(192,38,211,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 2h12v12H2V2zM4 8l3 3 5-5"/>
        </svg>,
    },
    'ЗАКОН': {
        bg: 'rgba(14,165,233,0.08)',
        text: '#0369a1',
        border: 'rgba(14,165,233,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 4h12M4 4l2 8M12 4l-2 8M6 12h4M8 2v2"/>
        </svg>,
    },
    'ССЫЛКА': {
        bg: 'rgba(59,130,246,0.08)',
        text: '#1d4ed8',
        border: 'rgba(59,130,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M6 6L3 9l4 4 3-3M10 10l3-3-4-4-3 3"/>
        </svg>,
    },
    'КОНТАКТ': {
        bg: 'rgba(139,92,246,0.08)',
        text: '#5b21b6',
        border: 'rgba(139,92,246,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M8 6a2 2 0 100-4 2 2 0 000 4zM3 14v-2a5 5 0 0110 0v2"/>
        </svg>, // Дубль персоны, так как контакт = лицо
    },
    'РИСК': {
        bg: 'rgba(239,68,68,0.08)',
        text: '#991b1b',
        border: 'rgba(239,68,68,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M2 14L8 2l6 12H2zM8 6v3M8 11.5v0.5"/>
        </svg>,
    },
    'ДЕЙСТВИЕ': {
        bg: 'rgba(245,158,11,0.08)',
        text: '#92400e',
        border: 'rgba(245,158,11,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M4 2l9 6-9 6V2z"/>
        </svg>,
    },
    'ПРОЧЕЕ': {
        bg: 'rgba(100,116,139,0.08)',
        text: '#475569',
        border: 'rgba(100,116,139,0.12)',
        icon: <svg {...STRICT_SVG_PROPS}>
            <path d="M3 4h10M3 8h10M3 12h10"/>
        </svg>,
    },
};

function factCat(key: string) {
    const upper = key.toUpperCase()
        .replace(/EMAIL/g, 'ЭЛ. ПОЧТА')
        .replace(/E-?MAIL/g, 'ЭЛ. ПОЧТА')
        .replace(/ПОЧТА/g, 'ЭЛ. ПОЧТА')
        .replace(/PHONE/g, 'ТЕЛЕФОН')
        .replace(/TEL/g, 'ТЕЛЕФОН')
        .replace(/ТЕЛ\./g, 'ТЕЛЕФОН')
        .replace(/MOBILE/g, 'ТЕЛЕФОН')
        .replace(/ADDRESS/g, 'АДРЕС')
        .replace(/NUMBER/g, 'НОМЕР')
        .replace(/NUM/g, 'НОМЕР');

    return (FACT_CATEGORY[upper] ?? FACT_CATEGORY['ПРОЧЕЕ'])!
}

const LANG_NAMES: Record<string, string> = {
    ru: 'Русский', en: 'English', be: 'Белорусская',
    de: 'Deutsch', fr: 'Français', es: 'Español',
    zh: '中文', ja: '日本語', ko: '한국어',
}

function langName(code: string) {
    return LANG_NAMES[code] ?? code.toUpperCase()
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

function CollapsibleSection({
                                title,
                                children,
                                defaultOpen = true,
                                icon,
                                right,
                            }: {
    title: string
    children: ReactNode
    defaultOpen?: boolean
    icon?: ReactNode
    right?: ReactNode
}) {
    const [open, setOpen] = useState(defaultOpen)
    return (
        <div style={{marginBottom: 4}}>
            <button
                onClick={() => setOpen(v => !v)}
                style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    padding: '8px 16px',
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    textAlign: 'left',
                    color: '#0f172a',
                    fontSize: 13,
                    fontWeight: 600,
                    transition: 'background 0.15s',
                }}
                onMouseEnter={e => {
                    (e.currentTarget as HTMLElement).style.background = 'rgba(99,102,241,0.03)'
                }}
                onMouseLeave={e => {
                    (e.currentTarget as HTMLElement).style.background = 'transparent'
                }}
            >
                {open ? <ChevronDown size={14} style={{flexShrink: 0, color: '#94a3b8'}}/>
                    : <ChevronRight size={14} style={{flexShrink: 0, color: '#94a3b8'}}/>}
                {icon}
                <span style={{flex: 1}}>{title}</span>
                {right}
            </button>
            {open && <div style={{paddingLeft: 16, paddingRight: 16}}>{children}</div>}
        </div>
    )
}

function ComplianceCheckResult({data}: { data: ComplianceCheckData }) {
    const isError = data.overall === 'has_mismatches'
    const isWarning = data.overall === 'cannot_verify'

    const statusColor = isError ? '#ef4444' : (isWarning ? '#f59e0b' : '#10b981')
    const statusIcon = isError
        ? <XCircle size={18} color={statusColor}/>
        : (isWarning ? <AlertTriangle size={18} color={statusColor}/> : <CheckCircle size={18} color={statusColor}/>)
    const statusText = isError
        ? 'Найдены расхождения'
        : (isWarning ? 'Требуется проверка' : 'Проверка пройдена успешно')

    const okCount = data.fields.filter(f => f.status === 'ok').length
    const errCount = data.fields.filter(f => f.status === 'mismatch').length
    const naCount = data.fields.filter(f => f.status === 'not_found').length

    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: isError
                    ? 'rgba(239,68,68,0.04)'
                    : (isWarning ? 'rgba(245,158,11,0.04)' : 'rgba(16,185,129,0.04)'),
            }}>
                <FileText size={18} style={{color: '#64748b'}}/>
                <div style={{flex: 1}}>
                    <div style={{
                        fontWeight: 700, color: '#0f172a', fontSize: 14,
                        display: 'flex', alignItems: 'center', gap: 8,
                    }}>
                        {statusIcon}
                        {statusText}
                    </div>
                    <div style={{color: '#64748b', fontSize: 12, marginTop: 2}}>
                        {data.summary}
                    </div>
                </div>
            </div>

            {/* Stats bar */}
            <div style={{
                display: 'flex', gap: 16, padding: '10px 16px',
                borderBottom: '1px solid rgba(0,0,0,0.04)',
                background: '#fafbfc',
            }}>
                {okCount > 0 && (
                    <span style={{fontSize: 11, color: '#059669', fontWeight: 600}}>
                        ✓ {okCount} совпадают
                    </span>
                )}
                {errCount > 0 && (
                    <span style={{fontSize: 11, color: '#dc2626', fontWeight: 600}}>
                        ✗ {errCount} расхождений
                    </span>
                )}
                {naCount > 0 && (
                    <span style={{fontSize: 11, color: '#94a3b8', fontWeight: 500}}>
                        ? {naCount} не найдено
                    </span>
                )}
            </div>

            <div style={{padding: '0 0 8px 0'}}>
                {data.fields.map((field, idx) => {
                    const isFieldError = field.status === 'mismatch'
                    const isFieldOk = field.status === 'ok'

                    return (
                        <div key={idx} style={{
                            padding: '10px 16px',
                            borderBottom: idx < data.fields.length - 1
                                ? '1px solid rgba(0,0,0,0.04)' : 'none',
                            background: idx % 2 === 0 ? 'transparent' : 'rgba(248,250,252,0.5)',
                        }}>
                            <div style={{
                                display: 'flex', justifyContent: 'space-between',
                                alignItems: 'center', marginBottom: 4,
                            }}>
                                <span style={{fontWeight: 600, color: '#334155'}}>{field.label}</span>
                                <span style={{
                                    ...BADGE_BASE,
                                    background: isFieldError
                                        ? 'rgba(239,68,68,0.1)'
                                        : (isFieldOk ? 'rgba(16,185,129,0.1)' : 'rgba(148,163,184,0.1)'),
                                    color: isFieldError
                                        ? '#b91c1c'
                                        : (isFieldOk ? '#047857' : '#64748b'),
                                    textTransform: 'uppercase',
                                }}>
                                    {isFieldError ? 'Ошибка' : (isFieldOk ? 'OK' : 'Не найдено')}
                                </span>
                            </div>

                            <div style={{
                                display: 'grid', gridTemplateColumns: '1fr 1fr',
                                gap: 12, fontSize: 12,
                            }}>
                                <div>
                                    <div style={{fontSize: 10, color: '#94a3b8', marginBottom: 2}}>В карточке</div>
                                    <div style={{color: '#1e293b', wordBreak: 'break-word'}}>{field.card_value}</div>
                                </div>
                                <div>
                                    <div style={{fontSize: 10, color: '#94a3b8', marginBottom: 2}}>В файле</div>
                                    <div style={{
                                        color: field.file_value ? '#1e293b' : '#cbd5e1',
                                        wordBreak: 'break-word',
                                    }}>
                                        {field.file_value || '—'}
                                    </div>
                                </div>
                            </div>

                            {field.recommendation && (
                                <div style={{
                                    marginTop: 6, padding: '6px 10px',
                                    background: '#fffbeb', border: '1px solid #fcd34d',
                                    borderRadius: 6, fontSize: 11, color: '#92400e',
                                    display: 'flex', gap: 6, alignItems: 'flex-start',
                                }}>
                                    <span style={{fontWeight: 700}}>💡</span>
                                    <span>{field.recommendation}</span>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            <div style={{
                padding: '8px 16px', fontSize: 11, color: '#94a3b8',
                borderTop: '1px solid rgba(0,0,0,0.05)', background: '#f8fafc',
            }}>
                Проверено AI. Результат добавлен в краткое содержание документа.
            </div>
        </div>
    )
}

function ThesisPlanResult({data}: { data: ThesisPlanData }) {
    return (
        <div style={CARD}>
            {/* Header — main argument */}
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(99,102,241,0.04), rgba(139,92,246,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(99,102,241,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <Target size={16} style={{color: '#6366f1'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{
                        fontSize: 10, fontWeight: 600, color: '#6366f1',
                        textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2
                    }}>
                        Главный тезис
                    </div>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a', lineHeight: 1.45}}>
                        {data.main_argument}
                    </div>
                </div>
            </div>

            {/* Sections */}
            <div style={{padding: '4px 0'}}>
                {data.sections.map((section, si) => (
                    <CollapsibleSection
                        key={si}
                        title={section.title || `Раздел ${si + 1}`}
                        icon={<span style={{
                            ...BADGE_BASE,
                            background: 'rgba(99,102,241,0.08)',
                            color: '#4338ca',
                            borderRadius: 6,
                            fontSize: 10,
                        }}>{si + 1}</span>}
                        right={
                            <span style={{fontSize: 10, color: '#94a3b8'}}>
                                {section.points.length} тезисов
                            </span>
                        }
                    >
                        {/* Section thesis */}
                        <div style={{
                            padding: '8px 12px', marginBottom: 8,
                            background: 'rgba(99,102,241,0.03)',
                            borderLeft: '3px solid #c7d2fe',
                            borderRadius: '0 8px 8px 0',
                            fontSize: 12, color: '#334155', lineHeight: 1.6,
                        }}>
                            {section.thesis}
                        </div>

                        {/* Points */}
                        {section.points.map((point, pi) => (
                            <div key={pi} style={{
                                padding: '8px 12px', marginBottom: 6,
                                background: idx_bg(pi),
                                borderRadius: 10,
                                border: '1px solid rgba(0,0,0,0.03)',
                            }}>
                                <div style={{
                                    display: 'flex', alignItems: 'flex-start', gap: 8,
                                }}>
                                    <span style={{
                                        minWidth: 20, height: 20, borderRadius: 6,
                                        background: 'rgba(99,102,241,0.08)',
                                        color: '#4338ca', fontSize: 10, fontWeight: 700,
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        flexShrink: 0, marginTop: 1,
                                    }}>
                                        {pi + 1}
                                    </span>
                                    <div style={{flex: 1}}>
                                        <div style={{
                                            fontSize: 12, fontWeight: 600, color: '#1e293b',
                                            lineHeight: 1.5,
                                        }}>
                                            {point.claim}
                                        </div>

                                        {point.evidence && (
                                            <div style={{
                                                marginTop: 4, padding: '5px 10px',
                                                background: '#f8fafc', borderRadius: 6,
                                                border: '1px solid rgba(0,0,0,0.04)',
                                                fontSize: 11, color: '#64748b',
                                                fontStyle: 'italic', lineHeight: 1.5,
                                                display: 'flex', gap: 5, alignItems: 'flex-start',
                                            }}>
                                                <Quote size={11} style={{
                                                    flexShrink: 0, marginTop: 2,
                                                    color: '#94a3b8',
                                                }}/>
                                                {point.evidence}
                                            </div>
                                        )}

                                        {point.sub_points && point.sub_points.length > 0 && (
                                            <div style={{
                                                marginTop: 4, paddingLeft: 8,
                                                borderLeft: '2px solid rgba(99,102,241,0.15)',
                                            }}>
                                                {point.sub_points.map((sp, spi) => (
                                                    <div key={spi} style={{
                                                        fontSize: 11, color: '#64748b',
                                                        lineHeight: 1.5, padding: '1px 0',
                                                    }}>
                                                        → {sp}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </CollapsibleSection>
                ))}
            </div>

            {/* Conclusion */}
            {data.conclusion && (
                <div style={{
                    margin: '0 16px 12px', padding: '10px 14px',
                    background: 'linear-gradient(135deg, rgba(16,185,129,0.04), rgba(6,182,212,0.04))',
                    border: '1px solid rgba(16,185,129,0.1)',
                    borderRadius: 10,
                    fontSize: 12, color: '#065f46', lineHeight: 1.6,
                    display: 'flex', gap: 8, alignItems: 'flex-start',
                }}>
                    <ArrowRight size={14} style={{flexShrink: 0, marginTop: 2, color: '#10b981'}}/>
                    <div>
                        <div style={{
                            fontSize: 10, fontWeight: 600, textTransform: 'uppercase',
                            letterSpacing: 0.5, color: '#059669', marginBottom: 2,
                        }}>
                            Вывод
                        </div>
                        {data.conclusion}
                    </div>
                </div>
            )}

            <CardFooter/>
        </div>
    )
}

/** Alternating row background helper */
function idx_bg(i: number) {
    return i % 2 === 0 ? 'rgba(248,250,252,0.6)' : 'transparent'
}

function AbstractiveResult({data}: { data: AbstractiveData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(16,185,129,0.04), rgba(6,182,212,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(16,185,129,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <BookOpen size={16} style={{color: '#10b981'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Краткое изложение
                    </div>
                    <div style={{fontSize: 11, color: '#64748b', marginTop: 1}}>
                        Абстрактивная суммаризация
                    </div>
                </div>
            </div>

            {/* Key themes */}
            {data.key_themes.length > 0 && (
                <div style={{
                    display: 'flex', flexWrap: 'wrap', gap: 6,
                    padding: '10px 16px',
                    borderBottom: '1px solid rgba(0,0,0,0.04)',
                    background: '#fafbfc',
                }}>
                    {data.key_themes.map((theme, i) => {
                        const colors = Object.values(THEME_PALETTE)
                        const c = colors[(i + 1) % colors.length] ?? colors[0]!
                        return (
                            <span key={i} style={{
                                ...BADGE_BASE,
                                background: c.bg, color: c.text,
                                border: `1px solid ${c.border}`,
                            }}>
                                <Sparkles size={9}/>
                                {theme}
                            </span>
                        )
                    })}
                </div>
            )}

            {/* Summary text */}
            <div style={{
                padding: '14px 16px',
                fontSize: 13, color: '#334155', lineHeight: 1.75,
            }}>
                {data.summary.split(/\n\n+/).map((para, i) => (
                    <p key={i} style={{
                        margin: 0, marginBottom: i === data.summary.split(/\n\n+/).length - 1 ? 0 : 10,
                    }}>
                        {para}
                    </p>
                ))}
            </div>

            <CardFooter/>
        </div>
    )
}

function ExtractiveResult({data}: { data: ExtractiveData }) {
    const categories = [...new Set(data.facts.map(f => f.category.toUpperCase()))]

    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(59,130,246,0.04), rgba(99,102,241,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(59,130,246,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <FileSearch size={16} style={{color: '#3b82f6'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Извлечённые факты
                    </div>
                    {data.document_summary && (
                        <div style={{fontSize: 11, color: '#64748b', marginTop: 1}}>
                            {data.document_summary}
                        </div>
                    )}
                </div>
                <span style={{
                    ...BADGE_BASE,
                    background: 'rgba(59,130,246,0.08)', color: '#1d4ed8',
                }}>
                    {data.facts.length} фактов
                </span>
            </div>

            {/* Category tabs */}
            {categories.length > 1 && (
                <div style={{
                    display: 'flex', flexWrap: 'wrap', gap: 4,
                    padding: '8px 16px',
                    borderBottom: '1px solid rgba(0,0,0,0.04)',
                    background: '#fafbfc',
                }}>
                    {categories.map(cat => {
                        const cfg = factCat(cat)
                        const count = data.facts.filter(f => f.category.toUpperCase() === cat).length
                        return (
                            <span key={cat} style={{
                                ...BADGE_BASE,
                                background: cfg.bg, color: cfg.text,
                            }}>
                                {cfg.icon} {cat} ({count})
                            </span>
                        )
                    })}
                </div>
            )}

            {/* Facts grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
                gap: 8,
                padding: '12px 16px',
            }}>
                {data.facts.map((fact, i) => {
                    const cfg = factCat(fact.category)
                    return (
                        <div key={i} style={{
                            padding: '10px 12px',
                            borderRadius: 10,
                            background: cfg.bg,
                            border: `1px solid ${cfg.border}`,
                        }}>
                            <div style={{
                                display: 'flex', justifyContent: 'space-between',
                                alignItems: 'center', marginBottom: 4,
                            }}>
                                <span style={{
                                    fontSize: 11,
                                    fontWeight: 600,
                                    color: cfg.text,
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: 4
                                }}>
                                    {cfg.icon} {fact.label}
                                </span>
                                <span style={{
                                    fontSize: 9, fontWeight: 600, textTransform: 'uppercase',
                                    color: cfg.text, opacity: 0.7,
                                    display: 'inline-flex', alignItems: 'center', gap: 3
                                }}>
                                   {cfg.icon} {fact.category}
                                </span>
                            </div>
                            <div style={{
                                fontSize: 12, color: '#1e293b',
                                wordBreak: 'break-word', lineHeight: 1.5,
                            }}>
                                {fact.value}
                            </div>
                        </div>
                    )
                })}
            </div>

            <CardFooter/>
        </div>
    )
}

function ActionItemsResult({data}: { data: ActionItemsData }) {
    const sorted = [...data.action_items].sort((a, b) => {
        const order = {high: 0, medium: 1, low: 2}
        return (order[a.priority] ?? 1) - (order[b.priority] ?? 1)
    })

    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(245,158,11,0.04), rgba(239,68,68,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(245,158,11,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <ListChecks size={16} style={{color: '#f59e0b'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Задачи и действия
                    </div>
                    {data.document_context && (
                        <div style={{fontSize: 11, color: '#64748b', marginTop: 1}}>
                            {data.document_context}
                        </div>
                    )}
                </div>
                <span style={{
                    ...BADGE_BASE,
                    background: 'rgba(245,158,11,0.08)', color: '#92400e',
                }}>
                    {data.action_items.length} задач
                </span>
            </div>

            <div style={{padding: '8px 0'}}>
                {sorted.map((item, i) => {
                    const cfg = (PRIORITY_CFG[item.priority] ?? PRIORITY_CFG['medium'])!
                    return (
                        <div key={i} style={{
                            padding: '10px 16px',
                            borderBottom: i < sorted.length - 1
                                ? '1px solid rgba(0,0,0,0.04)' : 'none',
                            background: i % 2 === 0 ? 'transparent' : 'rgba(248,250,252,0.5)',
                        }}>
                            <div style={{
                                display: 'flex', alignItems: 'flex-start', gap: 10,
                            }}>
                                {/* Priority indicator */}
                                <div style={{
                                    minWidth: 6, height: 6, borderRadius: '50%',
                                    background: cfg.text,
                                    marginTop: 6, flexShrink: 0,
                                }}/>

                                <div style={{flex: 1}}>
                                    <div style={{
                                        fontSize: 12, fontWeight: 600, color: '#1e293b',
                                        lineHeight: 1.5,
                                    }}>
                                        {item.task}
                                    </div>

                                    <div style={{
                                        display: 'flex', flexWrap: 'wrap', gap: 6,
                                        marginTop: 5,
                                    }}>
                                        <span style={{
                                            ...BADGE_BASE,
                                            background: cfg.bg, color: cfg.text,
                                        }}>
                                            {cfg.icon} {cfg.label}
                                        </span>

                                        {item.owner && (
                                            <span style={{
                                                ...BADGE_BASE,
                                                background: 'rgba(139,92,246,0.06)',
                                                color: '#5b21b6',
                                            }}>
                                                <User size={9}/> {item.owner}
                                            </span>
                                        )}

                                        {item.deadline && (
                                            <span style={{
                                                ...BADGE_BASE,
                                                background: 'rgba(59,130,246,0.06)',
                                                color: '#1d4ed8',
                                            }}>
                                                <Clock size={9}/> {formatDate(item.deadline)}
                                            </span>
                                        )}
                                    </div>

                                    {item.source_fragment && (
                                        <div style={{
                                            marginTop: 5, padding: '4px 10px',
                                            background: '#f8fafc', borderRadius: 6,
                                            border: '1px solid rgba(0,0,0,0.04)',
                                            fontSize: 11, color: '#64748b',
                                            fontStyle: 'italic', lineHeight: 1.5,
                                            display: 'flex', gap: 5,
                                        }}>
                                            <Quote size={10} style={{
                                                flexShrink: 0, marginTop: 2, color: '#94a3b8',
                                            }}/>
                                            {item.source_fragment}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>

            <CardFooter/>
        </div>
    )
}

function formatDate(raw: string): string {
    try {
        const d = new Date(raw)
        if (isNaN(d.getTime())) return raw
        return d.toLocaleDateString('ru-RU', {day: 'numeric', month: 'short', year: 'numeric'})
    } catch {
        return raw
    }
}

function ExecutiveSummaryResult({data}: { data: ExecutiveSummaryData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(139,92,246,0.04), rgba(99,102,241,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(139,92,246,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <Sparkles size={16} style={{color: '#8b5cf6'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{
                        fontSize: 10, fontWeight: 600, color: '#8b5cf6',
                        textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2,
                    }}>
                        Резюме
                    </div>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a', lineHeight: 1.45}}>
                        {data.headline}
                    </div>
                </div>
            </div>

            {data.bullets.length > 0 && (
                <div style={{padding: '12px 16px'}}>
                    {data.bullets.map((bullet, i) => (
                        <div key={i} style={{
                            display: 'flex', alignItems: 'flex-start', gap: 10,
                            padding: '7px 0',
                            borderBottom: i < data.bullets.length - 1
                                ? '1px solid rgba(0,0,0,0.04)' : 'none',
                        }}>
                            <span style={{
                                minWidth: 22, height: 22, borderRadius: 7,
                                background: `rgba(139,92,246,${0.06 + i * 0.02})`,
                                color: '#5b21b6', fontSize: 10, fontWeight: 700,
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                flexShrink: 0, marginTop: 1,
                            }}>
                                {i + 1}
                            </span>
                            <span style={{fontSize: 12, color: '#334155', lineHeight: 1.6}}>
                                {bullet}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {data.recommendation && (
                <div style={{
                    margin: '0 16px 12px', padding: '10px 14px',
                    background: '#fffbeb', border: '1px solid #fcd34d',
                    borderRadius: 10, fontSize: 12, color: '#92400e',
                    display: 'flex', gap: 8, alignItems: 'flex-start', lineHeight: 1.5,
                }}>
                    <span style={{fontSize: 14, flexShrink: 0}}>💡</span>
                    <div>
                        <div style={{fontWeight: 600, marginBottom: 2}}>Рекомендация</div>
                        {data.recommendation}
                    </div>
                </div>
            )}

            <CardFooter/>
        </div>
    )
}

function DetailedNotesResult({data}: { data: DetailedNotesData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(6,182,212,0.04), rgba(59,130,246,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(6,182,212,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <Brain size={16} style={{color: '#06b6d4'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Подробные заметки
                    </div>
                    <div style={{
                        display: 'flex', gap: 8, marginTop: 3, flexWrap: 'wrap',
                    }}>
                        <span style={{
                            ...BADGE_BASE,
                            background: 'rgba(6,182,212,0.08)', color: '#155e75',
                        }}>
                            {data.document_type}
                        </span>
                        {data.date_range && (
                            <span style={{
                                ...BADGE_BASE,
                                background: 'rgba(59,130,246,0.06)', color: '#1d4ed8',
                            }}>
                                <Clock size={9}/> {data.date_range}
                            </span>
                        )}
                    </div>
                </div>
            </div>

            <div style={{padding: '4px 0'}}>
                {data.sections.map((section, i) => (
                    <CollapsibleSection
                        key={i}
                        title={section.title}
                        defaultOpen={i === 0}
                        icon={
                            <span style={{
                                ...BADGE_BASE, borderRadius: 6, fontSize: 10,
                                background: 'rgba(6,182,212,0.08)', color: '#155e75',
                            }}>
                                {i + 1}
                            </span>
                        }
                    >
                        <div style={{
                            fontSize: 12, color: '#334155',
                            lineHeight: 1.7, marginBottom: 6,
                        }}>
                            {section.content}
                        </div>

                        {section.subsections && section.subsections.length > 0 && (
                            <div style={{
                                paddingLeft: 12,
                                borderLeft: '2px solid rgba(6,182,212,0.2)',
                                marginTop: 4,
                            }}>
                                {section.subsections.map((sub, j) => (
                                    <div key={j} style={{
                                        fontSize: 11, color: '#64748b',
                                        lineHeight: 1.5, padding: '2px 0',
                                    }}>
                                        → {sub}
                                    </div>
                                ))}
                            </div>
                        )}
                    </CollapsibleSection>
                ))}
            </div>

            {/* Key entities */}
            {data.key_entities.length > 0 && (
                <div style={{
                    padding: '10px 16px',
                    borderTop: '1px solid rgba(0,0,0,0.04)',
                    background: '#fafbfc',
                }}>
                    <div style={{
                        fontSize: 10, fontWeight: 600, color: '#94a3b8',
                        textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6
                    }}>
                        Ключевые сущности
                    </div>
                    <div style={{display: 'flex', flexWrap: 'wrap', gap: 4}}>
                        {data.key_entities.map((entity, i) => (
                            <span key={i} style={{
                                ...BADGE_BASE,
                                background: 'rgba(100,116,139,0.06)',
                                color: '#475569',
                                fontFamily: 'ui-monospace, monospace',
                                fontSize: 10,
                            }}>
                                {entity}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            <CardFooter/>
        </div>
    )
}

function MultilingualResult({data}: { data: MultilingualData }) {
    return (
        <div style={CARD}>
            <div style={{
                ...CARD_HEADER,
                background: 'linear-gradient(135deg, rgba(244,63,94,0.04), rgba(139,92,246,0.04))',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 10,
                    background: 'rgba(244,63,94,0.1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <Globe size={16} style={{color: '#f43f5e'}}/>
                </div>
                <div style={{flex: 1}}>
                    <div style={{fontSize: 14, fontWeight: 700, color: '#0f172a'}}>
                        Многоязычная суммаризация
                    </div>
                    <div style={{
                        display: 'flex', gap: 6, marginTop: 3,
                    }}>
                        <span style={{
                            ...BADGE_BASE,
                            background: 'rgba(244,63,94,0.08)', color: '#9f1239',
                        }}>
                            {langName(data.detected_language)}
                        </span>
                        <span style={{
                            fontSize: 10, color: '#94a3b8', display: 'flex',
                            alignItems: 'center'
                        }}>
                            →
                        </span>
                        <span style={{
                            ...BADGE_BASE,
                            background: 'rgba(139,92,246,0.08)', color: '#5b21b6',
                        }}>
                            {langName(data.summary_language)}
                        </span>
                    </div>
                </div>
            </div>

            <div style={{
                padding: '14px 16px',
                fontSize: 13, color: '#334155', lineHeight: 1.75,
            }}>
                {data.summary.split(/\n\n+/).map((para, i) => (
                    <p key={i} style={{
                        margin: 0,
                        marginBottom: i === data.summary.split(/\n\n+/).length - 1 ? 0 : 10,
                    }}>
                        {para}
                    </p>
                ))}
            </div>

            {data.translation_notes && (
                <div style={{
                    margin: '0 16px 12px', padding: '10px 14px',
                    background: '#f8fafc', border: '1px solid rgba(0,0,0,0.06)',
                    borderRadius: 10, fontSize: 11, color: '#64748b',
                    display: 'flex', gap: 8, alignItems: 'flex-start', lineHeight: 1.5,
                }}>
                    <AlertCircle size={14} style={{flexShrink: 0, marginTop: 1, color: '#94a3b8'}}/>
                    <div>
                        <div style={{fontWeight: 600, marginBottom: 2, color: '#475569'}}>
                            Примечания к переводу
                        </div>
                        {data.translation_notes}
                    </div>
                </div>
            )}

            <CardFooter/>
        </div>
    )
}

function CardFooter() {
    return (
        <div style={{
            padding: '8px 16px', fontSize: 10, color: '#94a3b8',
            borderTop: '1px solid rgba(0,0,0,0.04)', background: '#fafbfc',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
            <span>Сгенерировано AI</span>
            <span style={{display: 'flex', alignItems: 'center', gap: 3}}>
                <Sparkles size={9}/> EDMS Assistant
            </span>
        </div>
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

const CATEGORY_COLORS: Record<string, { bg: string; text: string; label: string }> = {
    'INCOMING': {bg: 'rgba(59,130,246,0.08)', text: '#1d4ed8', label: 'Входящий'},
    'OUTGOING': {bg: 'rgba(16,185,129,0.08)', text: '#065f46', label: 'Исходящий'},
    'INTERN': {bg: 'rgba(139,92,246,0.08)', text: '#5b21b6', label: 'Внутренний'},
    'APPEAL': {bg: 'rgba(245,158,11,0.08)', text: '#92400e', label: 'Обращение'},
    'CONTRACT': {bg: 'rgba(239,68,68,0.08)', text: '#991b1b', label: 'Договор'},
    'MEETING': {bg: 'rgba(99,102,241,0.08)', text: '#3730a3', label: 'Совещание'},
    'ORDER': {bg: 'rgba(244,63,94,0.08)', text: '#9f1239', label: 'Приказ'},
    'CITIZEN': {bg: 'rgba(245,158,11,0.08)', text: '#92400e', label: 'Гражданин'},
}

function getCategoryStyle(raw: string) {
    const upper = raw.toUpperCase().replace(/[()]/g, '').trim()
    for (const [key, val] of Object.entries(CATEGORY_COLORS)) {
        if (upper.includes(key)) return val
    }
    return {bg: 'rgba(100,116,139,0.08)', text: '#334155', label: raw}
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
                gap: 4,
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
            <div style={{display: 'flex', alignItems: 'center', gap: 8}}>
                <span style={{
                    width: 24, height: 24, borderRadius: '50%',
                    background: isClickable ? 'rgba(99,102,241,0.08)' : 'rgba(148,163,184,0.08)',
                    color: isClickable ? '#6366f1' : '#94a3b8',
                    fontSize: 10, fontWeight: 700,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                }}>{num ?? index + 1}</span>

                {regNum && regNum !== '—'
                    ? <span style={{fontSize: 12, fontWeight: 700, color: '#0f172a', flex: 1}}>{regNum}</span>
                    : <span style={{fontSize: 12, color: '#94a3b8', flex: 1}}>—</span>
                }

                {date && date !== '—' && (
                    <span style={{fontSize: 10, color: '#94a3b8', flexShrink: 0}}>{date}</span>
                )}

                {isClickable && (
                    <span style={{
                        flexShrink: 0, color: '#6366f1', opacity: 0.5,
                        display: 'flex', alignItems: 'center',
                    }}>
                        <ExternalLink size={12}/>
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

            <div style={{display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center'}}>
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
                        display: 'flex', alignItems: 'center', gap: 4,
                    }}>
                        <User size={10} style={{opacity: 0.7}}/>
                        {author}
                    </span>
                )}
                {status && status !== '—' && (
                    <span style={{
                        fontSize: 9, color: '#64748b', fontWeight: 600,
                        padding: '2px 8px', borderRadius: 4,
                        background: 'rgba(100,116,139,0.08)',
                        textTransform: 'uppercase',
                    }}>{status}</span>
                )}

                {address && address !== '—' && (
                    <div style={{
                        width: '100%', marginTop: 2, fontSize: 11, color: '#64748b',
                        display: 'flex', alignItems: 'flex-start', gap: 6,
                        lineHeight: 1.4, padding: '2px 0',
                    }}>
                        <span style={{flexShrink: 0, marginTop: '1px'}}>📍</span>
                        <span style={{wordBreak: 'break-word'}}>{address}</span>
                    </div>
                )}

                {extraPairs.map(({key, value}) => {
                    const isContact = key.toLowerCase().includes('контакт') || key.toLowerCase().includes('contact')
                    if (isContact && value) {
                        const parts = value.split(/\s{2,}|\n/).filter(part => part.trim().length > 0)
                        return (
                            <div key={key} className="flex flex-wrap gap-2 items-center w-full mt-1">
                                {parts.map((part, i) => (
                                    <span key={i} style={{
                                        fontSize: 10, color: '#475569',
                                        padding: '3px 8px', borderRadius: 6,
                                        background: '#f1f5f9', whiteSpace: 'nowrap',
                                        border: '1px solid rgba(0,0,0,0.03)',
                                        fontFamily: 'ui-monospace, monospace', lineHeight: 1.4,
                                    }}>
                                        {part.trim()}
                                    </span>
                                ))}
                            </div>
                        )
                    }
                    return (
                        <span key={key} style={{
                            fontSize: 10, color: '#64748b',
                            padding: '2px 8px', borderRadius: 20,
                            background: 'rgba(100,116,139,0.06)',
                        }}>{key}: {value}</span>
                    )
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

    return (
        <div
            onClick={() => fileName && onAttachmentClick?.(fileName)}
            style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '12px 16px', marginBottom: 6, borderRadius: 16,
                background: '#ffffff',
                border: '1px solid rgba(0,0,0,0.06)',
                cursor: onAttachmentClick ? 'pointer' : 'default',
                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                boxShadow: '0 1px 3px rgba(0,0,0,0.03)',
            }}
            onMouseEnter={e => {
                if (!onAttachmentClick) return
                const el = e.currentTarget as HTMLDivElement
                el.style.background = '#fafbff'
                el.style.borderColor = 'rgba(99,102,241,0.18)'
                el.style.transform = 'translateY(-1px)'
                el.style.boxShadow = '0 2px 12px rgba(99,102,241,0.08)'
            }}
            onMouseLeave={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.background = '#ffffff'
                el.style.borderColor = 'rgba(0,0,0,0.06)'
                el.style.transform = 'translateY(0)'
                el.style.boxShadow = '0 1px 3px rgba(0,0,0,0.03)'
            }}
        >
            <div style={{
                width: 32, height: 32, borderRadius: '50%',
                background: 'rgba(99,102,241,0.06)',
                color: '#6366f1',
                fontSize: 12, fontWeight: 700,
                display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
            }}>{index + 1}</div>

            <FileText size={18} style={{color: '#6366f1', opacity: 0.7, flexShrink: 0}}/>

            <div style={{flex: 1, minWidth: 0}}>
                <div style={{
                    fontSize: 13, fontWeight: 600, color: '#0f172a',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>{fileName || `Вложение ${index + 1}`}</div>
                <div style={{fontSize: 11, color: '#64748b', marginTop: 1, display: 'flex', gap: 8}}>
                    {fileSize && <span>{fileSize}</span>}
                    {fileDate && <span>{fileDate}</span>}
                </div>
            </div>
            {onAttachmentClick && <ChevronRight size={16} style={{color: '#cbd5e1', flexShrink: 0}}/>}
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
    const isDocList = /id/.test(h) || /рег.*номер|рег\.номер/.test(h) || (/дата/.test(h) && /категор/.test(h))

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
    const structured = !isUser ? detectStructuredOutput(content) : null

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