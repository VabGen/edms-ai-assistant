import { useState, useCallback } from 'react'
import { storage } from 'wxt/storage'
import { STORAGE_KEYS } from '@shared/storage/keys'

const enabledStorage = storage.defineItem<boolean>(STORAGE_KEYS.assistantEnabled, {
  fallback: true,
})

interface WidgetSize {
  width: number
  height: number
}

const DEFAULT_SIZE: WidgetSize = { width: 500, height: 700 }
const MIN_SIZE: WidgetSize = { width: 320, height: 400 }
const MAX_SIZE: WidgetSize = { width: 900, height: 900 }

export interface UseWidgetStateReturn {
  isOpen: boolean
  isSidebarOpen: boolean
  isSettingsOpen: boolean
  widgetSize: WidgetSize
  setIsOpen: (v: boolean) => void
  toggleOpen: () => void
  setIsSidebarOpen: (v: boolean) => void
  setIsSettingsOpen: (v: boolean) => void
  setWidgetSize: (size: WidgetSize) => void
  clampSize: (w: number, h: number) => WidgetSize
}

export function useWidgetState(): UseWidgetStateReturn {
  const [isOpen, setIsOpen] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [widgetSize, setWidgetSize] = useState<WidgetSize>(DEFAULT_SIZE)

  const toggleOpen = useCallback(() => setIsOpen((v) => !v), [])

  const clampSize = useCallback((w: number, h: number): WidgetSize => ({
    width: Math.min(MAX_SIZE.width, Math.max(MIN_SIZE.width, w)),
    height: Math.min(MAX_SIZE.height, Math.max(MIN_SIZE.height, h)),
  }), [])

  return {
    isOpen,
    isSidebarOpen,
    isSettingsOpen,
    widgetSize,
    setIsOpen,
    toggleOpen,
    setIsSidebarOpen,
    setIsSettingsOpen,
    setWidgetSize,
    clampSize,
  }
}

export { enabledStorage }
