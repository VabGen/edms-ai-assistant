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
  // Base opacity is 0.85, setting adds up to 0.15 (total 1.0) or reduces it.
  // Actually, let's make the setting 0.0 to 1.0 for more control in the future,
  // but for now we follow the schema which is 0 to 0.5.
  // We'll map the schema's 0-0.5 to a nice glass range.
  const alpha = 1.0 - glassOpacity;

  return {
    ['--glass-opacity' as string]: alpha.toString(),
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
