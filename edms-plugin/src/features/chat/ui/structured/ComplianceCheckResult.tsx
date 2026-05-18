import { FileText, XCircle, AlertTriangle, CheckCircle } from 'lucide-react'
import type { ComplianceData, ComplianceField } from '@/entities/message/model/types'
import { CARD, CARD_HEADER, BADGE_BASE } from './common'

export function ComplianceCheckResult({data}: { data: ComplianceData }) {
    const isError = data.overall === 'has_mismatches'
    const isWarning = data.overall === 'cannot_verify'

    const statusColor = isError ? '#ef4444' : (isWarning ? '#f59e0b' : '#10b981')
    const statusIcon = isError
        ? <XCircle size={18} color={statusColor}/>
        : (isWarning ? <AlertTriangle size={18} color={statusColor}/> : <CheckCircle size={18} color={statusColor}/>)
    const statusText = isError
        ? 'Найдены расхождения'
        : (isWarning ? 'Требуется проверка' : 'Проверка пройдена успешно')

    const okCount = data.fields.filter((f: ComplianceField) => f.status === 'ok').length
    const errCount = data.fields.filter((f: ComplianceField) => f.status === 'mismatch').length
    const naCount = data.fields.filter((f: ComplianceField) => f.status === 'not_found').length

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
                {data.fields.map((field: ComplianceField, idx: number) => {
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
