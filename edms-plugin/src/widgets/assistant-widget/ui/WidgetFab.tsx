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
      className="pointer-events-auto"
      style={{
        width: 52,
        height: 52,
        borderRadius: 16,
        background: 'rgba(255,255,255,0.96)',
        boxShadow: '0 4px 20px rgba(0,0,0,0.12), 0 1px 4px rgba(0,0,0,0.06)',
        border: 'none',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'transform 0.15s, box-shadow 0.15s',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.06)' }}
      onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)' }}
    >
      <MessageSquare size={22} strokeWidth={1.8} color="#6366f1" />
    </button>
  )
}
