import { useEffect, useState } from 'react'
import { storage } from 'wxt/storage'
import { QueryProvider } from '@app/providers/QueryProvider'
import { useWidgetState, enabledStorage } from '../model/useWidgetState'
import { useApplyPreferences } from '@features/settings/model/useApplyPreferences'
import { useChatStore } from '@features/chat/model/useChatStore'
import { WidgetFab } from './WidgetFab'
import { WidgetPanel } from './WidgetPanel'

function AssistantWidgetInner() {
  const [isEnabled, setIsEnabled] = useState(true)
  const { isOpen, setIsOpen, toggleOpen } = useWidgetState()
  const { rootClassName, rootStyle } = useApplyPreferences()
  const { loadSnapshot, isSnapshotLoaded } = useChatStore()

  useEffect(() => {
    let cancelled = false
    void enabledStorage.getValue().then((val) => {
      if (!cancelled) setIsEnabled(val ?? true)
    })
    const unwatch = enabledStorage.watch((val) => {
      setIsEnabled(val ?? true)
    })
    return () => {
      cancelled = true
      unwatch()
    }
  }, [])

  useEffect(() => {
    void loadSnapshot()
  }, [loadSnapshot])

  if (!isEnabled || !isSnapshotLoaded) return null

  return (
    <div className={rootClassName} style={rootStyle}>
      {isOpen ? (
        <WidgetPanel onClose={() => setIsOpen(false)} />
      ) : (
        <WidgetFab onClick={toggleOpen} />
      )}
    </div>
  )
}

export function AssistantWidget() {
  return (
    <QueryProvider>
      <AssistantWidgetInner />
    </QueryProvider>
  )
}
