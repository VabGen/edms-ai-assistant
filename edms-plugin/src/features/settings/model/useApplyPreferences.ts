import { useState, useEffect, type CSSProperties } from 'react'
import { storage } from 'wxt/storage'
import { STORAGE_KEYS } from '@shared/storage/keys'
import { DEFAULT_USER_PREFS } from '@entities/settings/model/defaults'
import type { UserPreferences, FontSize, WidgetPosition } from '@entities/settings/model/types'

const userPrefsStorage = storage.defineItem<UserPreferences>(
  STORAGE_KEYS.userPreferences,
  { fallback: DEFAULT_USER_PREFS },
)

const FONT_SIZE_MAP = {
  small: '12px',
  medium: '14px',
  large: '16px',
} as const satisfies Record<FontSize, string>

function resolvePositionClass(pos: WidgetPosition): string {
  return pos === 'bottom-left'
    ? 'fixed bottom-5 left-5 z-[2147483647] flex flex-col items-start pointer-events-none select-none font-sans'
    : 'fixed bottom-5 right-5 z-[2147483647] flex flex-col items-end pointer-events-none select-none font-sans'
}

function resolveStyle(prefs: UserPreferences): CSSProperties {
  const { glassOpacity, fontSize } = prefs.appearance
  const alpha = Math.min(1, Math.round((0.82 + glassOpacity * 0.36) * 100) / 100)
  return {
    ['--glass-bg' as string]: `rgba(255,255,255,${alpha})`,
    ['--glass-border' as string]: 'rgba(255,255,255,0.50)',
    ['--glass-shadow' as string]: 'rgba(31,38,135,0.12)',
    ['--edms-font-size' as string]: FONT_SIZE_MAP[fontSize],
  }
}

export interface UseApplyPreferencesReturn {
  rootClassName: string
  rootStyle: CSSProperties
  prefs: UserPreferences
}

export function useApplyPreferences(): UseApplyPreferencesReturn {
  const [prefs, setPrefs] = useState<UserPreferences>(DEFAULT_USER_PREFS)

  useEffect(() => {
    let cancelled = false
    void userPrefsStorage.getValue().then((val) => {
      if (!cancelled) setPrefs(val ?? DEFAULT_USER_PREFS)
    })
    const unwatch = userPrefsStorage.watch((val) => {
      setPrefs(val ?? DEFAULT_USER_PREFS)
    })
    return () => {
      cancelled = true
      unwatch()
    }
  }, [])

  return {
    rootClassName: resolvePositionClass(prefs.appearance.widgetPosition),
    rootStyle: resolveStyle(prefs),
    prefs,
  }
}
