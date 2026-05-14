import { useRef, useCallback, useEffect, type CSSProperties } from 'react'
import { X } from 'lucide-react'
import { useChatStore } from '@features/chat/model/useChatStore'
import { useWidgetState } from '../model/useWidgetState'
import { WidgetSidebar } from './WidgetSidebar'
import { WidgetChat } from './WidgetChat'
import { SettingsPanelWrapper } from './SettingsPanelWrapper'

interface WidgetPanelProps {
  onClose: () => void
}

interface ResizeStart {
  x: number
  y: number
  width: number
  height: number
}

export function WidgetPanel({ onClose }: WidgetPanelProps) {
  const {
    isSidebarOpen,
    isSettingsOpen,
    widgetSize,
    setIsSidebarOpen,
    setIsSettingsOpen,
    setWidgetSize,
    clampSize,
  } = useWidgetState()
  const { saveSnapshot } = useChatStore()

  const isResizingRef = useRef(false)
  const resizeStartRef = useRef<ResizeStart | null>(null)

  useEffect(() => {
    function onMouseMove(e: MouseEvent) {
      if (!isResizingRef.current || !resizeStartRef.current) return
      const start = resizeStartRef.current
      setWidgetSize(clampSize(start.width + (start.x - e.clientX), start.height + (start.y - e.clientY)))
    }
    function onMouseUp() {
      isResizingRef.current = false
      resizeStartRef.current = null
    }
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [setWidgetSize, clampSize])

  const handleClose = useCallback(() => {
    void saveSnapshot(false)
    onClose()
  }, [saveSnapshot, onClose])

  const handleToggleSidebar = useCallback(() => {
    if (isSidebarOpen) {
      setIsSidebarOpen(false)
      setIsSettingsOpen(false)
    } else {
      setIsSidebarOpen(true)
    }
  }, [isSidebarOpen, setIsSidebarOpen, setIsSettingsOpen])

  return (
    <div
      className="pointer-events-auto relative flex flex-col overflow-hidden"
      style={{
        width: widgetSize.width,
        height: widgetSize.height,
        background: 'var(--glass-bg, white)',
        backdropFilter: 'blur(20px) saturate(1.5)',
        WebkitBackdropFilter: 'blur(20px) saturate(1.5)',
        borderRadius: 16,
        boxShadow: '0 8px 40px rgba(0,0,0,0.14), 0 2px 8px rgba(0,0,0,0.06)',
        border: '1px solid rgba(255,255,255,0.60)',
        fontSize: 'var(--edms-font-size, 14px)',
      }}
    >
      <div
        onMouseDown={(e) => {
          e.preventDefault()
          isResizingRef.current = true
          resizeStartRef.current = { x: e.clientX, y: e.clientY, width: widgetSize.width, height: widgetSize.height }
        }}
        style={{ position: 'absolute', top: 0, left: 0, width: 20, height: 20, cursor: 'nw-resize', zIndex: 20 }}
      />

      <WidgetHeader
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={handleToggleSidebar}
        onClose={handleClose}
      />

      <div className="flex flex-1 min-h-0">
        {isSidebarOpen && (
          isSettingsOpen
            ? <SettingsPanelWrapper onClose={() => setIsSettingsOpen(false)} />
            : <WidgetSidebar onOpenSettings={() => setIsSettingsOpen(true)} />
        )}
        <WidgetChat />
      </div>
    </div>
  )
}

interface WidgetHeaderProps {
  isSidebarOpen: boolean
  onToggleSidebar: () => void
  onClose: () => void
}

function AnimatedBurger({ isOpen }: { isOpen: boolean }) {
  const bar: CSSProperties = {
    display: 'block',
    width: 16,
    height: 1.5,
    borderRadius: 1,
    background: isOpen ? '#6366f1' : '#94a3b8',
    transition: 'transform 0.22s ease, opacity 0.15s ease, background 0.15s',
    transformOrigin: 'center',
  }
  return (
    <span style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center', width: 16, pointerEvents: 'none' }}>
      <span style={{ ...bar, transform: isOpen ? 'translateY(5.5px) rotate(45deg)' : 'none' }} />
      <span style={{ ...bar, opacity: isOpen ? 0 : 1 }} />
      <span style={{ ...bar, transform: isOpen ? 'translateY(-5.5px) rotate(-45deg)' : 'none' }} />
    </span>
  )
}

function WidgetHeader({ isSidebarOpen, onToggleSidebar, onClose }: WidgetHeaderProps) {
  return (
    <header
      className="flex items-center shrink-0 px-3"
      style={{ height: 52, borderBottom: '1px solid rgba(0,0,0,0.06)' }}
    >
      <button
        type="button"
        onClick={onToggleSidebar}
        title={isSidebarOpen ? 'Скрыть' : 'История'}
        style={{
          width: 32, height: 32,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: 8, border: 'none',
          background: isSidebarOpen ? 'rgba(99,102,241,0.08)' : 'transparent',
          cursor: 'pointer', flexShrink: 0,
        }}
      >
        <AnimatedBurger isOpen={isSidebarOpen} />
      </button>

      <div
        className="flex items-center gap-1.5 flex-1 justify-center"
        style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}
      >
        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#22c55e', flexShrink: 0 }} />
        EDMS Assistant
      </div>

      <button
        type="button"
        onClick={onClose}
        title="Закрыть"
        style={{
          width: 32, height: 32,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: 8, border: 'none', background: 'transparent',
          color: '#94a3b8', cursor: 'pointer', flexShrink: 0,
          transition: 'background 0.15s, color 0.15s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(0,0,0,0.05)'; e.currentTarget.style.color = '#475569' }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#94a3b8' }}
      >
        <X size={15} />
      </button>
    </header>
  )
}
