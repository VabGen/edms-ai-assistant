import { SettingsPanel } from '@/shared/ui/SettingsPanel'

interface SettingsPanelWrapperProps {
  onClose: () => void
}

export function SettingsPanelWrapper({ onClose }: SettingsPanelWrapperProps) {
  return (
    <div
      className="flex flex-col w-full h-full overflow-hidden"
      style={{
        background: '#f8fafc',
      }}
    >
      <div className="flex-1 overflow-y-auto p-3 min-h-0 scrollbar-none">
        <SettingsPanel onClose={onClose} />
      </div>
    </div>
  )
}
