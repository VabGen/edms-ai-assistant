export const STORAGE_KEYS = {
  userPreferences: 'local:edmsUserPreferences',
  techSettingsCache: 'local:edmsTechSettingsCache',
  widgetSnapshot: 'local:edmsWidgetSnapshot',
  widgetThreads: 'local:edmsWidgetThreads',
  assistantEnabled: 'local:assistantEnabled',
  userContext: 'local:userContext',
} as const satisfies Record<string, `local:${string}` | `sync:${string}` | `session:${string}`>

export type StorageKey = (typeof STORAGE_KEYS)[keyof typeof STORAGE_KEYS]
