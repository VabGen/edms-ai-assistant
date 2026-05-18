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
  const { loadSnapshot, isSnapshotLoaded, appendMessage } = useChatStore()

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

  useEffect(() => {
    const onMessage = (ev: MessageEvent): void => {
      const data = ev.data as Record<string, unknown> | null
      if (!data || data.type !== 'REFRESH_CHAT_HISTORY') return
      const msgs = (data.messages as Array<{ type: string; content: string }> | undefined) ?? []
      const now = Date.now()
      msgs.forEach((m, i) => {
        appendMessage({
          id: `bridge_${now}_${i}`,
          role: m.type === 'human' ? 'user' : 'assistant',
          content: m.content ?? '',
          timestamp: now + i,
          refreshMeta: {
            cache_file_identifier: (data.cache_file_identifier as string | null) ?? undefined,
            cache_summary_type: (data.cache_summary_type as string | null) ?? undefined,
            cache_context_id: (data.cache_context_id as string | null) ?? undefined,
            cache_file_path: (data.cache_file_path as string | null) ?? undefined,
          },
        })
      })
      if (msgs.length > 0) setIsOpen(true)
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [appendMessage, setIsOpen])

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
