import {useState, useEffect, CSSProperties} from 'react'
import type {UserPreferences, FontSize, WidgetPosition} from './useSettingsStore'
import {DEFAULT_USER_PREFS} from './useSettingsStore'

const USER_KEY = 'edmsUserPreferences'

const FONT_SIZE_MAP: Record<FontSize, string> = {
    small: '12px',
    medium: '14px',
    large: '16px',
}

function resolvePositionClass(pos: WidgetPosition): string {
    if (pos === 'bottom-left') {
        return 'fixed bottom-5 left-5 z-[2147483647] flex flex-col items-start pointer-events-none select-none font-sans'
    }
    return 'fixed bottom-5 right-5 z-[2147483647] flex flex-col items-end pointer-events-none select-none font-sans'
}

function resolveStyle(prefs: UserPreferences): CSSProperties {
    const {glassOpacity, fontSize} = prefs.appearance
    const alpha = Math.round((0.20 + glassOpacity * 1.20) * 100) / 100

    return {
        ['--glass-bg' as string]: `rgba(255, 255, 255, ${alpha})`,
        ['--glass-border' as string]: 'rgba(255, 255, 255, 0.50)',
        ['--glass-shadow' as string]: 'rgba(31, 38, 135, 0.12)',
        ['--edms-font-size' as string]: FONT_SIZE_MAP[fontSize],
    }
}

function loadFromStorage(saved: any): UserPreferences {
    if (!saved) return DEFAULT_USER_PREFS
    return {
        appearance: {...DEFAULT_USER_PREFS.appearance, ...(saved.appearance ?? {})},
        voice: {...DEFAULT_USER_PREFS.voice, ...(saved.voice ?? {})},
        documents: {...DEFAULT_USER_PREFS.documents, ...(saved.documents ?? {})},
    }
}

export interface UseApplyPreferencesReturn {
    rootClassName: string
    rootStyle: CSSProperties
    dataTheme: string | undefined
    prefs: UserPreferences
}

export function useApplyPreferences(): UseApplyPreferencesReturn {
    const [prefs, setPrefs] = useState<UserPreferences>(DEFAULT_USER_PREFS)

    useEffect(() => {
        try {
            chrome.storage.local.get([USER_KEY], (r) => {
                setPrefs(loadFromStorage(r?.[USER_KEY]))
            })
        } catch {
        }
    }, [])

    useEffect(() => {
        function onChanged(changes: Record<string, chrome.storage.StorageChange>, area: string) {
            if (area !== 'local' || !(USER_KEY in changes)) return
            setPrefs(loadFromStorage(changes[USER_KEY].newValue))
        }

        try {
            chrome.storage.onChanged.addListener(onChanged)
            return () => chrome.storage.onChanged.removeListener(onChanged)
        } catch {
            return () => {
            }
        }
    }, [])

    return {
        rootClassName: resolvePositionClass(prefs.appearance.widgetPosition),
        rootStyle: resolveStyle(prefs),
        dataTheme: undefined,
        prefs,
    }
}