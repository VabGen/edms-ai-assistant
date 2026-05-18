import { SettingsPanel } from '@/shared/ui/SettingsPanel'

interface SettingsPanelWrapperProps {
  onClose: () => void
}

export function SettingsPanelWrapper({ onClose }: SettingsPanelWrapperProps) {
  return (
    <div
      className="flex flex-col shrink-0 overflow-hidden"
      style={{
        width: 240,
        borderRight: '1px solid rgba(0,0,0,0.06)',
        background: 'rgba(248,250,252,0.8)',
      }}
    >
      <div className="flex-1 overflow-y-auto p-3 min-h-0 scrollbar-none">
        <SettingsPanel onClose={onClose} />
      </div>
    </div>
  )
}
