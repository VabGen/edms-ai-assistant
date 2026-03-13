/**
 * @file speech.d.ts
 * @description Ambient type declarations for the Web Speech API.
 *
 * The Web Speech API is not yet included in TypeScript's built-in lib.dom.d.ts
 * at a stable level.  Chrome ships `window.webkitSpeechRecognition` (prefixed)
 * and modern browsers expose `window.SpeechRecognition` (unprefixed).
 *
 * Place this file anywhere TypeScript can pick it up (e.g. `src/types/`) and
 * ensure the directory is covered by `tsconfig.json` → `include`.
 */

// ---------------------------------------------------------------------------
// SpeechRecognition — minimal surface used by AssistantWidget
// ---------------------------------------------------------------------------

interface SpeechRecognitionEventMap {
    audioend: Event
    audiostart: Event
    end: Event
    error: SpeechRecognitionErrorEvent
    nomatch: SpeechRecognitionEvent
    result: SpeechRecognitionEvent
    soundend: Event
    soundstart: Event
    speechend: Event
    speechstart: Event
    start: Event
}

interface SpeechRecognition extends EventTarget {
    continuous: boolean
    grammars: SpeechGrammarList
    interimResults: boolean
    lang: string
    maxAlternatives: number
    onaudioend: ((this: SpeechRecognition, ev: Event) => unknown) | null
    onaudiostart: ((this: SpeechRecognition, ev: Event) => unknown) | null
    onend: ((this: SpeechRecognition, ev: Event) => unknown) | null
    onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => unknown) | null
    onnomatch: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => unknown) | null
    onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => unknown) | null
    onsoundend: ((this: SpeechRecognition, ev: Event) => unknown) | null
    onsoundstart: ((this: SpeechRecognition, ev: Event) => unknown) | null
    onspeechend: ((this: SpeechRecognition, ev: Event) => unknown) | null
    onspeechstart: ((this: SpeechRecognition, ev: Event) => unknown) | null
    onstart: ((this: SpeechRecognition, ev: Event) => unknown) | null

    abort(): void

    start(): void

    stop(): void

    addEventListener<K extends keyof SpeechRecognitionEventMap>(
        type: K,
        listener: (this: SpeechRecognition, ev: SpeechRecognitionEventMap[K]) => unknown,
        options?: boolean | AddEventListenerOptions,
    ): void

    addEventListener(
        type: string,
        listener: EventListenerOrEventListenerObject,
        options?: boolean | AddEventListenerOptions,
    ): void

    removeEventListener<K extends keyof SpeechRecognitionEventMap>(
        type: K,
        listener: (this: SpeechRecognition, ev: SpeechRecognitionEventMap[K]) => unknown,
        options?: boolean | EventListenerOptions,
    ): void

    removeEventListener(
        type: string,
        listener: EventListenerOrEventListenerObject,
        options?: boolean | EventListenerOptions,
    ): void
}

declare var SpeechRecognition: {
    prototype: SpeechRecognition
    new(): SpeechRecognition
}

// ---------------------------------------------------------------------------
// SpeechRecognitionEvent
// ---------------------------------------------------------------------------

interface SpeechRecognitionEvent extends Event {
    readonly resultIndex: number
    readonly results: SpeechRecognitionResultList
}

interface SpeechRecognitionResultList {
    readonly length: number

    item(index: number): SpeechRecognitionResult

    [index: number]: SpeechRecognitionResult
}

interface SpeechRecognitionResult {
    readonly isFinal: boolean
    readonly length: number

    item(index: number): SpeechRecognitionAlternative

    [index: number]: SpeechRecognitionAlternative
}

interface SpeechRecognitionAlternative {
    readonly confidence: number
    readonly transcript: string
}

// ---------------------------------------------------------------------------
// SpeechRecognitionErrorEvent
// ---------------------------------------------------------------------------

type SpeechRecognitionErrorCode =
    | 'aborted'
    | 'audio-capture'
    | 'bad-grammar'
    | 'language-not-supported'
    | 'network'
    | 'no-speech'
    | 'not-allowed'
    | 'service-not-allowed'

interface SpeechRecognitionErrorEvent extends Event {
    readonly error: SpeechRecognitionErrorCode
    readonly message: string
}

// ---------------------------------------------------------------------------
// SpeechGrammar / SpeechGrammarList (required by SpeechRecognition.grammars)
// ---------------------------------------------------------------------------

interface SpeechGrammar {
    src: string
    weight: number
}

declare var SpeechGrammar: {
    prototype: SpeechGrammar
    new(): SpeechGrammar
}

interface SpeechGrammarList {
    readonly length: number

    addFromString(string: string, weight?: number): void

    addFromURI(src: string, weight?: number): void

    item(index: number): SpeechGrammar

    [index: number]: SpeechGrammar
}

declare var SpeechGrammarList: {
    prototype: SpeechGrammarList
    new(): SpeechGrammarList
}

// ---------------------------------------------------------------------------
// Window augmentation — unprefixed + webkit-prefixed
// ---------------------------------------------------------------------------

interface Window {
    SpeechRecognition: typeof SpeechRecognition | undefined
    webkitSpeechRecognition: typeof SpeechRecognition | undefined
    SpeechGrammarList: typeof SpeechGrammarList | undefined
    webkitSpeechGrammarList: typeof SpeechGrammarList | undefined
}