/**
 * @file hooks/useSpeechRecognition.ts
 * @description Production-ready Web Speech API hook for Russian STT
 *              with optional hands-free auto-send mode.
 *
 * ── Возможности ────────────────────────────────────────────────────────────
 *
 * 1. continuous: true + interimResults: true
 *    Живой предварительный текст в textarea; финальные сегменты накапливаются
 *    через onFinalResult-callback.
 *
 * 2. Авторестарт при неожиданном onend
 *    Chrome Web Speech API самопроизвольно останавливается (~60 сек,
 *    потеря соединения). Хук перезапускает экземпляр если wantListening=true.
 *
 * 3. Таймер тишины (silenceMs, default 3000 мс)
 *    Если за silenceMs не пришло ни одного onresult — микрофон останавливается.
 *    Предотвращает «висящий микрофон».
 *
 * 4. Hands-free автоотправка (autoSendMs, default 1400 мс)
 *    После последнего isFinal-сегмента запускается таймер autoSendMs.
 *    Если за это время не пришёл новый результат — вызывается onAutoSend().
 *    Любой новый interim/final сбрасывает таймер — пользователь продолжает
 *    говорить без потери слов.
 *    Включается опцией autoSend: true; по умолчанию выключен.
 *
 * 5. recognition в useRef, не в useState
 *    Исключает race-condition при быстрых start/stop.
 *
 * 6. Корректный teardown
 *    useEffect cleanup сбрасывает ВСЕ таймеры и вызывает abort().
 *
 * ── Инварианты таймеров ────────────────────────────────────────────────────
 *
 *   autoSendMs < silenceMs   (всегда)
 *   ┌──────────────────────────────────────────── время ──►
 *   │  последний isFinal
 *   │       │
 *   │       ├──── autoSendMs ────► onAutoSend() + stopMic()
 *   │       │
 *   │       └──────────── silenceMs ────► stop() (резерв, если autoSend=false)
 *
 * @example Обычный режим
 * ```tsx
 * const { isListening, interimTranscript, toggle } = useSpeechRecognition({
 *   onFinalResult: (text) => setInput(prev => prev ? `${prev} ${text}` : text),
 * })
 * ```
 *
 * @example Hands-free режим
 * ```tsx
 * const { isListening, interimTranscript, toggle } = useSpeechRecognition({
 *   autoSend:      true,
 *   autoSendMs:    1400,
 *   onFinalResult: (text) => setInput(prev => prev ? `${prev} ${text}` : text),
 *   onAutoSend:    () => sendFormRef.current?.requestSubmit(),
 * })
 * ```
 */

import {useState, useRef, useEffect, useCallback} from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SpeechRecognitionOptions {
    /**
     * BCP-47 language tag.
     * @default 'ru-RU'
     */
    lang?: string

    /**
     * Milliseconds of total silence (no onresult at all) before mic auto-stops.
     * Acts as a safety net — should be larger than autoSendMs.
     * @default 3000
     */
    silenceMs?: number

    /**
     * Enable hands-free mode: auto-send after autoSendMs of post-final silence.
     * @default false
     */
    autoSend?: boolean

    /**
     * Milliseconds after the last isFinal segment before onAutoSend fires.
     * Must be less than silenceMs, otherwise the mic stops before the send.
     * @default 1400
     */
    autoSendMs?: number

    /**
     * Called with the delta text of each final result segment (isFinal === true).
     * Append the delta to your input state here.
     * NOT called for interim results.
     */
    onFinalResult: (deltaText: string) => void

    /**
     * Called in hands-free mode when autoSendMs of silence follows the last
     * final segment.  Trigger your form submit here.
     * Only called when autoSend: true.
     */
    onAutoSend?: () => void

    /**
     * Called when mic fully stops for any reason (user click, silence, error).
     */
    onStop?: () => void
}

export interface SpeechRecognitionState {
    /** True while the microphone is actively capturing. */
    isListening: boolean

    /** Current unconfirmed (interim) transcript segment — for live UI preview. */
    interimTranscript: string

    /**
     * True if the browser supports Web Speech API.
     * Hide the mic button when false.
     */
    isSupported: boolean

    /**
     * Whether hands-free auto-send is currently active.
     * Use to show a visual indicator ("отправлю автоматически...").
     */
    autoSendPending: boolean

    /** Toggle mic on/off. */
    toggle: () => void

    /** Unconditionally stop mic and cancel any pending auto-send. */
    stop: () => void
}

// ---------------------------------------------------------------------------
// Internal timer helper — typed wrapper around setTimeout / clearTimeout
// ---------------------------------------------------------------------------

type TimerRef = ReturnType<typeof setTimeout> | null

function clearTimer(ref: { current: TimerRef }): void {
    if (ref.current !== null) {
        clearTimeout(ref.current)
        ref.current = null
    }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSpeechRecognition({
                                         lang = 'ru-RU',
                                         silenceMs = 3000,
                                         autoSend = false,
                                         autoSendMs = 1400,
                                         onFinalResult,
                                         onAutoSend,
                                         onStop,
                                     }: SpeechRecognitionOptions): SpeechRecognitionState {

    const [isListening, setIsListening] = useState(false)
    const [interimTranscript, setInterimTranscript] = useState('')
    const [autoSendPending, setAutoSendPending] = useState(false)

    // ── Internal refs (never trigger re-renders) ────────────────────────────
    const recognitionRef = useRef<SpeechRecognition | null>(null)

    /**
     * Intent flag.
     * true  = пользователь хочет продолжать слушать.
     * false = пользователь нажал стоп / сработал таймер тишины.
     * Используется в onend для различения "пользователь остановил" vs
     * "браузер сам остановился" → авторестарт только в последнем случае.
     */
    const wantListeningRef = useRef(false)

    /** Silence watchdog: no onresult at all for silenceMs → stop mic. */
    const silenceTimerRef = useRef<TimerRef>(null)

    /**
     * Auto-send countdown: started after each isFinal segment.
     * Fires onAutoSend after autoSendMs of post-final silence.
     * Reset on any new result (interim or final) — user is still talking.
     */
    const autoSendTimerRef = useRef<TimerRef>(null)

    /** Stable callback refs — no stale closures in SR event handlers. */
    const onFinalRef = useRef(onFinalResult)
    const onAutoSendRef = useRef(onAutoSend)
    const onStopRef = useRef(onStop)
    const autoSendRef = useRef(autoSend)
    const autoSendMsRef = useRef(autoSendMs)
    const silenceMsRef = useRef(silenceMs)

    useEffect(() => {
        onFinalRef.current = onFinalResult
    }, [onFinalResult])
    useEffect(() => {
        onAutoSendRef.current = onAutoSend
    }, [onAutoSend])
    useEffect(() => {
        onStopRef.current = onStop
    }, [onStop])
    useEffect(() => {
        autoSendRef.current = autoSend
    }, [autoSend])
    useEffect(() => {
        autoSendMsRef.current = autoSendMs
    }, [autoSendMs])
    useEffect(() => {
        silenceMsRef.current = silenceMs
    }, [silenceMs])

    // ── Browser support ─────────────────────────────────────────────────────
    const isSupported =
        typeof window !== 'undefined' &&
        !!(window.SpeechRecognition || window.webkitSpeechRecognition)

    // ── Timer helpers ────────────────────────────────────────────────────────

    /** Сброс таймера тишины (вызывается при любом onresult). */
    const resetSilenceTimer = useCallback(() => {
        clearTimer(silenceTimerRef)
        silenceTimerRef.current = setTimeout(() => {
            if (wantListeningRef.current) {
                wantListeningRef.current = false
                recognitionRef.current?.stop()
            }
        }, silenceMsRef.current)
    }, [])

    /**
     * Сброс auto-send таймера (вызывается при любом onresult — interim или final).
     * Запускаем новый countdown только если только что был isFinal.
     */
    const scheduleAutoSend = useCallback(() => {
        clearTimer(autoSendTimerRef)
        if (!autoSendRef.current) return

        setAutoSendPending(true)
        autoSendTimerRef.current = setTimeout(() => {
            setAutoSendPending(false)
            wantListeningRef.current = false
            recognitionRef.current?.stop()
            onAutoSendRef.current?.()
        }, autoSendMsRef.current)
    }, [])

    /** Отмена auto-send countdown (новый interim/final пока таймер тикал). */
    const cancelAutoSend = useCallback(() => {
        clearTimer(autoSendTimerRef)
        setAutoSendPending(false)
    }, [])

    // ── Build SpeechRecognition instance ─────────────────────────────────────

    const buildInstance = useCallback((): SpeechRecognition | null => {
        if (!isSupported) return null

        const SR = window.SpeechRecognition ?? window.webkitSpeechRecognition!
        const inst = new SR()

        inst.lang = lang
        inst.continuous = true
        inst.interimResults = true

        // ── onresult ────────────────────────────────────────────────────────────
        inst.onresult = (ev: SpeechRecognitionEvent) => {
            resetSilenceTimer()

            let interim = ''
            let finalDelta = ''

            for (let i = ev.resultIndex; i < ev.results.length; i++) {
                const result = ev.results[i]
                const text = result[0].transcript

                if (result.isFinal) {
                    finalDelta += (finalDelta ? ' ' : '') + text.trim()
                } else {
                    interim += text
                }
            }

            if (interim) {
                cancelAutoSend()
                setInterimTranscript(interim)
            }

            if (finalDelta) {
                setInterimTranscript('')
                onFinalRef.current(finalDelta)
                scheduleAutoSend()
            }
        }

        // ── onspeechend ─────────────────────────────────────────────────────────
        inst.onspeechend = () => {
            resetSilenceTimer()
        }

        // ── onerror ─────────────────────────────────────────────────────────────
        inst.onerror = (ev: SpeechRecognitionErrorEvent) => {
            switch (ev.error) {
                case 'not-allowed':
                case 'service-not-allowed':
                    wantListeningRef.current = false
                    cancelAutoSend()
                    break
                case 'no-speech':
                    break
                case 'aborted':
                    break
                default:
                    break
            }
        }

        // ── onend ───────────────────────────────────────────────────────────────
        inst.onend = () => {
            clearTimer(silenceTimerRef)
            setInterimTranscript('')

            if (wantListeningRef.current) {
                setTimeout(() => {
                    if (wantListeningRef.current && recognitionRef.current) {
                        try {
                            recognitionRef.current.start()
                        } catch {

                        }
                    }
                }, 150)
            } else {
                setIsListening(false)
                onStopRef.current?.()
            }
        }

        return inst
    }, [
        lang,
        isSupported,
        resetSilenceTimer,
        scheduleAutoSend,
        cancelAutoSend,
    ])

    // ── Lifecycle ────────────────────────────────────────────────────────────

    useEffect(() => {
        if (!isSupported) return
        recognitionRef.current = buildInstance()

        return () => {
            wantListeningRef.current = false
            clearTimer(silenceTimerRef)
            clearTimer(autoSendTimerRef)
            recognitionRef.current?.abort()
            recognitionRef.current = null
        }
    }, [isSupported, buildInstance])

    // ── Public API ───────────────────────────────────────────────────────────

    /** Остановить микрофон и отменить любой pending auto-send. */
    const stop = useCallback(() => {
        wantListeningRef.current = false
        clearTimer(silenceTimerRef)
        cancelAutoSend()
        recognitionRef.current?.stop()
        // setIsListening(false) вызовется через onend
    }, [cancelAutoSend])

    /** Переключить микрофон. */
    const toggle = useCallback(() => {
        if (!recognitionRef.current) return

        if (wantListeningRef.current) {
            stop()
        } else {
            wantListeningRef.current = true
            setIsListening(true)
            setInterimTranscript('')
            setAutoSendPending(false)
            resetSilenceTimer()
            try {
                recognitionRef.current.start()
            } catch {

            }
        }
    }, [stop, resetSilenceTimer])

    return {
        isListening,
        interimTranscript,
        isSupported,
        autoSendPending,
        toggle,
        stop,
    }
}