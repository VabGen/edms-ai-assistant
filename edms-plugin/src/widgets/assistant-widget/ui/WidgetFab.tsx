import { MessageSquare } from 'lucide-react'

interface WidgetFabProps {
  onClick: () => void
}

export function WidgetFab({ onClick }: WidgetFabProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      title="EDMS Assistant"
      className="pointer-events-auto group"
      style={{
        width: 60,
        height: 60,
        borderRadius: 20,
        background: 'rgb(var(--edms-bg))',
        boxShadow: 'var(--edms-shadow-lg)',
        border: '1px solid rgba(var(--zinc-200), 0.5)',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1)',
      }}
    >
      <div className="absolute inset-0 rounded-[20px] bg-indigo-500/0 group-hover:bg-indigo-500/5 transition-colors duration-300" />
      <MessageSquare size={24} strokeWidth={2} className="text-indigo-600 group-hover:scale-110 transition-transform duration-300" />
    </button>
  )
}
