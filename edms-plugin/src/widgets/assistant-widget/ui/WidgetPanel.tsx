import { useRef, useCallback, useEffect, type CSSProperties } from 'react'
import { X, PanelLeftClose, PanelLeftOpen, LayoutPanelLeft } from 'lucide-react'
import { useChatStore } from '@features/chat/model/useChatStore'
import { useWidgetState } from '../model/useWidgetState'
import { WidgetSidebar } from './WidgetSidebar'
import { WidgetChat } from './WidgetChat'
import { SettingsPanelWrapper } from './SettingsPanelWrapper'
import { cn } from '@shared/lib/cn'

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
      className="pointer-events-auto relative flex flex-col overflow-hidden bg-white dark:bg-zinc-900 border border-white/60 dark:border-zinc-800 shadow-2xl animate-edms-fade-in"
      style={{
        width: widgetSize.width,
        height: widgetSize.height,
        backdropFilter: 'blur(20px) saturate(1.5)',
        WebkitBackdropFilter: 'blur(20px) saturate(1.5)',
        borderRadius: 24,
      }}
    >
      <div
        onMouseDown={(e) => {
          e.preventDefault()
          isResizingRef.current = true
          resizeStartRef.current = { x: e.clientX, y: e.clientY, width: widgetSize.width, height: widgetSize.height }
        }}
        className="absolute top-0 left-0 w-6 h-6 cursor-nw-resize z-50 rounded-tl-3xl hover:bg-blue-500/5 transition-colors"
      />

      <WidgetHeader
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={handleToggleSidebar}
        onClose={handleClose}
      />

      <div className="flex flex-1 min-h-0">
        {isSidebarOpen && (
          <div className="w-72 shrink-0 border-r border-zinc-100 dark:border-zinc-800 animate-slide-in-left">
            {isSettingsOpen
              ? <SettingsPanelWrapper onClose={() => setIsSettingsOpen(false)} />
              : <WidgetSidebar onOpenSettings={() => setIsSettingsOpen(true)} />
            }
          </div>
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

function WidgetHeader({ isSidebarOpen, onToggleSidebar, onClose }: WidgetHeaderProps) {
  return (
    <header
      className="flex items-center justify-between shrink-0 px-4 h-14 border-b border-zinc-100 dark:border-zinc-800 bg-zinc-50/30 dark:bg-zinc-800/20"
    >
      <button
        type="button"
        onClick={onToggleSidebar}
        title={isSidebarOpen ? 'Скрыть' : 'Меню'}
        className={cn(
            "p-2 rounded-xl transition-all duration-200",
            isSidebarOpen ? "bg-white dark:bg-zinc-800 text-blue-600 shadow-sm border border-zinc-200 dark:border-zinc-700" : "text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 dark:hover:bg-zinc-800"
        )}
      >
        {isSidebarOpen ? <PanelLeftClose size={18} /> : <LayoutPanelLeft size={18} />}
      </button>

      <div
        className="flex items-center gap-2 font-bold text-zinc-900 dark:text-zinc-100 tracking-tight"
      >
        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
        <span className="text-[14px]">EDMS Assistant</span>
      </div>

      <button
        type="button"
        onClick={onClose}
        title="Закрыть"
        className="p-2 rounded-xl text-zinc-400 hover:text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-950/30 transition-all border border-transparent hover:border-rose-100 dark:hover:border-rose-900/50"
      >
        <X size={18} />
      </button>
    </header>
  )
}
