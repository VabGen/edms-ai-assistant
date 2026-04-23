/**
 * @file Popup.tsx
 * @description Chrome Extension popup — toggle EDMS Assistant.
 */

import {useState, useEffect, useRef, type KeyboardEvent as ReactKeyboardEvent} from 'react'

export function Popup() {
    const [enabled, setEnabled] = useState(true)
    const [mounted, setMounted] = useState(false)
    const [glow, setGlow] = useState(false)
    const ref = useRef<HTMLDivElement>(null)

    useEffect(() => {
        chrome.storage.local.get(['assistantEnabled'], r => {
            if (r.assistantEnabled !== undefined) setEnabled(r.assistantEnabled as boolean)
        })
        requestAnimationFrame(() => requestAnimationFrame(() => setMounted(true)))
    }, [])

    const toggle = () => {
        const next = !enabled
        setGlow(true)
        setEnabled(next)
        setTimeout(() => setGlow(false), 600)
        chrome.storage.local.set({assistantEnabled: next})
    }

    const onKey = (e: ReactKeyboardEvent) => {
        if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault()
            toggle()
        }
    }

    return (
        <div style={{
            padding: 10,
            fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif',
            WebkitFontSmoothing: 'antialiased',
        }}>
            <div ref={ref} style={{
                position: 'relative',
                borderRadius: 18,
                overflow: 'hidden',
                background: 'rgba(255,255,255,0.22)',
                backdropFilter: 'blur(52px) saturate(200%)',
                WebkitBackdropFilter: 'blur(52px) saturate(200%)',
                border: '1px solid rgba(255,255,255,0.50)',
                boxShadow: '0 8px 36px rgba(0,0,0,0.07), 0 1px 0 rgba(255,255,255,0.65) inset',
                transform: mounted ? 'translateY(0) scale(1)' : 'translateY(8px) scale(0.97)',
                opacity: mounted ? 1 : 0,
                transition: 'transform 0.45s cubic-bezier(0.22,1,0.36,1), opacity 0.3s ease',
            }}>
                {/* specular line */}
                <div style={{
                    height: 1,
                    background: 'linear-gradient(90deg, transparent 8%, rgba(255,255,255,0.75) 35%, rgba(255,255,255,0.75) 65%, transparent 92%)',
                }}/>

                {/* glow flash on toggle */}
                {glow && (
                    <div style={{
                        position: 'absolute',
                        inset: -2,
                        borderRadius: 20,
                        background: enabled
                            ? 'radial-gradient(circle at 85% 45%, rgba(99,102,241,0.18) 0%, transparent 60%)'
                            : 'none',
                        animation: 'popup-flash 0.6s ease-out forwards',
                        pointerEvents: 'none',
                        zIndex: 0,
                    }}/>
                )}

                <div style={{padding: '16px 18px 14px', position: 'relative', zIndex: 1}}>

                    {/* header row */}
                    <div style={{display: 'flex', alignItems: 'center', gap: 11}}>

                        {/* icon */}
                        <div style={{
                            position: 'relative',
                            width: 38,
                            height: 38,
                            borderRadius: 11,
                            background: enabled
                                ? 'linear-gradient(145deg, #6366f1, #8b5cf6)'
                                : 'linear-gradient(145deg, rgba(148,163,184,0.14), rgba(100,116,139,0.09))',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                            boxShadow: enabled ? '0 3px 12px rgba(99,102,241,0.35)' : 'none',
                            transition: 'all 0.4s cubic-bezier(0.34,1.56,0.64,1)',
                        }}>
                            {enabled && (
                                <div style={{
                                    position: 'absolute',
                                    inset: -5,
                                    borderRadius: 16,
                                    background: 'linear-gradient(135deg, rgba(99,102,241,0.22), rgba(139,92,246,0.10))',
                                    filter: 'blur(7px)',
                                }}/>
                            )}
                            <svg
                                width="17"
                                height="17"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke={enabled ? '#fff' : '#94a3b8'}
                                strokeWidth="2.2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                style={{position: 'relative', zIndex: 1, transition: 'stroke 0.3s'}}
                            >
                                <path
                                    d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
                            </svg>
                        </div>

                        {/* label */}
                        <div style={{flex: 1, minWidth: 0}}>
                            <div style={{
                                fontSize: 13.5,
                                fontWeight: 700,
                                letterSpacing: -0.3,
                                color: enabled ? '#0f172a' : '#64748b',
                                lineHeight: 1.2,
                                transition: 'color 0.35s',
                            }}>
                                EDMS Assistant
                            </div>
                            <div style={{display: 'flex', alignItems: 'center', gap: 5, marginTop: 3}}>
                                <div style={{
                                    width: 5.5,
                                    height: 5.5,
                                    borderRadius: '50%',
                                    background: enabled ? '#22c55e' : '#cbd5e1',
                                    boxShadow: enabled ? '0 0 5px rgba(34,197,94,0.50)' : 'none',
                                    transition: 'all 0.35s',
                                }}/>
                                <span style={{
                                    fontSize: 11,
                                    fontWeight: 550,
                                    letterSpacing: -0.1,
                                    color: enabled ? '#6366f1' : '#94a3b8',
                                    transition: 'color 0.35s',
                                }}>
                                    {enabled ? 'Активен' : 'Выключен'}
                                </span>
                            </div>
                        </div>

                        {/* toggle */}
                        <div
                            onClick={toggle}
                            onKeyDown={onKey}
                            tabIndex={0}
                            role="switch"
                            aria-checked={enabled}
                            aria-label="Включить EDMS Assistant"
                            style={{
                                position: 'relative',
                                width: 48,
                                height: 28,
                                borderRadius: 14,
                                cursor: 'pointer',
                                flexShrink: 0,
                                outline: 'none',
                                background: enabled
                                    ? 'linear-gradient(135deg, #6366f1, #818cf8)'
                                    : 'rgba(120,120,128,0.14)',
                                boxShadow: enabled
                                    ? '0 2px 8px rgba(99,102,241,0.38), 0 1px 0 rgba(255,255,255,0.22) inset'
                                    : '0 1px 3px rgba(0,0,0,0.07) inset',
                                transition: 'background 0.3s, box-shadow 0.3s',
                            }}
                        >
                            {enabled && (
                                <div style={{
                                    position: 'absolute',
                                    inset: 1,
                                    borderRadius: 13,
                                    background: 'linear-gradient(180deg, rgba(255,255,255,0.16) 0%, transparent 45%)',
                                    pointerEvents: 'none',
                                }}/>
                            )}
                            <div style={{
                                position: 'absolute',
                                top: 2.5,
                                left: enabled ? 23.5 : 2.5,
                                width: 23,
                                height: 23,
                                borderRadius: '50%',
                                background: '#fff',
                                boxShadow: enabled
                                    ? '0 2px 8px rgba(99,102,241,0.28), 0 1px 0 rgba(255,255,255,0.85) inset'
                                    : '0 2px 5px rgba(0,0,0,0.16), 0 1px 0 rgba(255,255,255,0.75) inset',
                                transition: 'left 0.3s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.3s',
                            }}/>
                        </div>
                    </div>

                    {/* divider */}
                    <div style={{
                        height: 1,
                        margin: '13px 0 11px',
                        background: enabled
                            ? 'linear-gradient(90deg, transparent, rgba(99,102,241,0.10) 25%, rgba(99,102,241,0.10) 75%, transparent)'
                            : 'rgba(0,0,0,0.04)',
                        transition: 'background 0.4s',
                    }}/>

                    {/* description */}
                    <p style={{
                        fontSize: 11,
                        color: enabled ? 'rgba(15,23,42,0.40)' : 'rgba(60,60,67,0.32)',
                        lineHeight: 1.55,
                        margin: 0,
                        letterSpacing: -0.05,
                        transition: 'color 0.35s',
                    }}>
                        {enabled
                            ? 'Ассистент на странице. Нажмите на иконку для взаимодействия.'
                            : 'Ассистент скрыт. Включите для доступа к AI-помощнику.'}
                    </p>
                </div>

                {/* footer */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '7px 18px 8px',
                    borderTop: '1px solid rgba(0,0,0,0.025)',
                }}>
                    <div style={{display: 'flex', alignItems: 'center', gap: 4}}>
                        <div style={{
                            width: 4,
                            height: 4,
                            borderRadius: '50%',
                            background: enabled ? 'rgba(99,102,241,0.40)' : 'rgba(148,163,184,0.22)',
                            transition: 'background 0.4s',
                        }}/>
                        <span style={{
                            fontSize: 8.5,
                            fontWeight: 700,
                            letterSpacing: 0.5,
                            textTransform: 'uppercase',
                            color: enabled ? 'rgba(99,102,241,0.45)' : 'rgba(148,163,184,0.35)',
                            transition: 'color 0.4s',
                        }}>EDMS</span>
                    </div>
                    <span style={{fontSize: 9, color: 'rgba(148,163,184,0.25)', fontWeight: 500}}>
                        v1.0.0
                    </span>
                </div>
            </div>
        </div>
    )
}