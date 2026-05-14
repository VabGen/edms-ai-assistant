import { useState, useRef, useEffect, useCallback } from 'react'

export interface SpeechRecognitionOptions {
  lang?: string
  silenceMs?: number
  autoSend?: boolean
  autoSendMs?: number
  onFinalResult: (deltaText: string) => void
  onAutoSend?: () => void
  onStop?: () => void
}

export interface SpeechRecognitionState {
  isListening: boolean
  interimTranscript: string
  isSupported: boolean
  autoSendPending: boolean
  toggle: () => void
  stop: () => void
}

type TimerRef = ReturnType<typeof setTimeout> | null

function clearTimer(ref: { current: TimerRef }): void {
  if (ref.current !== null) {
    clearTimeout(ref.current)
    ref.current = null
  }
}

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

  const recognitionRef = useRef<SpeechRecognition | null>(null)
  const wantListeningRef = useRef(false)
  const silenceTimerRef = useRef<TimerRef>(null)
  const autoSendTimerRef = useRef<TimerRef>(null)

  const onFinalRef = useRef(onFinalResult)
  const onAutoSendRef = useRef(onAutoSend)
  const onStopRef = useRef(onStop)
  const autoSendRef = useRef(autoSend)
  const autoSendMsRef = useRef(autoSendMs)
  const silenceMsRef = useRef(silenceMs)

  useEffect(() => { onFinalRef.current = onFinalResult }, [onFinalResult])
  useEffect(() => { onAutoSendRef.current = onAutoSend }, [onAutoSend])
  useEffect(() => { onStopRef.current = onStop }, [onStop])
  useEffect(() => { autoSendRef.current = autoSend }, [autoSend])
  useEffect(() => { autoSendMsRef.current = autoSendMs }, [autoSendMs])
  useEffect(() => { silenceMsRef.current = silenceMs }, [silenceMs])

  const isSupported =
    typeof window !== 'undefined' &&
    !!(window.SpeechRecognition ?? window.webkitSpeechRecognition)

  const cancelAutoSend = useCallback(() => {
    clearTimer(autoSendTimerRef)
    setAutoSendPending(false)
  }, [])

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

  const resetSilenceTimer = useCallback(() => {
    clearTimer(silenceTimerRef)
    silenceTimerRef.current = setTimeout(() => {
      if (wantListeningRef.current) {
        wantListeningRef.current = false
        recognitionRef.current?.stop()
      }
    }, silenceMsRef.current)
  }, [])

  const buildInstance = useCallback((): SpeechRecognition | null => {
    if (!isSupported) return null
    const SR = window.SpeechRecognition ?? window.webkitSpeechRecognition!
    const inst = new SR()
    inst.lang = lang
    inst.continuous = true
    inst.interimResults = true

    inst.onresult = (ev: SpeechRecognitionEvent) => {
      resetSilenceTimer()
      let interim = ''
      let finalDelta = ''
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const result = ev.results[i]
        if (!result) continue
        const text = result[0]?.transcript ?? ''
        if (result.isFinal) finalDelta += (finalDelta ? ' ' : '') + text.trim()
        else interim += text
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

    inst.onspeechend = () => resetSilenceTimer()

    inst.onerror = (ev: SpeechRecognitionErrorEvent) => {
      if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
        wantListeningRef.current = false
        cancelAutoSend()
      }
    }

    inst.onend = () => {
      clearTimer(silenceTimerRef)
      setInterimTranscript('')
      if (wantListeningRef.current) {
        setTimeout(() => {
          if (wantListeningRef.current && recognitionRef.current) {
            try { recognitionRef.current.start() } catch { /* ignore */ }
          }
        }, 150)
      } else {
        setIsListening(false)
        onStopRef.current?.()
      }
    }

    return inst
  }, [lang, isSupported, resetSilenceTimer, scheduleAutoSend, cancelAutoSend])

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

  const stop = useCallback(() => {
    wantListeningRef.current = false
    clearTimer(silenceTimerRef)
    cancelAutoSend()
    recognitionRef.current?.stop()
  }, [cancelAutoSend])

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
      try { recognitionRef.current.start() } catch { /* ignore */ }
    }
  }, [stop, resetSilenceTimer])

  return { isListening, interimTranscript, isSupported, autoSendPending, toggle, stop }
}
