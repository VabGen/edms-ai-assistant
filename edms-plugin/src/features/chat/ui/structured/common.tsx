import React, { useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight, Sparkles } from 'lucide-react'

export const CARD: React.CSSProperties = {
    background: '#ffffff',
    borderRadius: 14,
    border: '1px solid rgba(0,0,0,0.06)',
    overflow: 'hidden',
    fontSize: 13,
}

export const CARD_HEADER: React.CSSProperties = {
    padding: '14px 16px',
    borderBottom: '1px solid rgba(0,0,0,0.05)',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
}

export const BADGE_BASE: React.CSSProperties = {
    fontSize: 10,
    fontWeight: 600,
    padding: '3px 9px',
    borderRadius: 20,
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    whiteSpace: 'nowrap',
}

export const THEME_PALETTE: Record<string, { bg: string; text: string; border: string }> = {
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

export function CardFooter() {
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

export function CollapsibleSection({
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
