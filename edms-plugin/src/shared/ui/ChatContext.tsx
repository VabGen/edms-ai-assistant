import { createContext } from 'react'

export const AttachmentClickContext = createContext<((fileName: string) => void) | null>(null)
export const DocumentClickContext = createContext<((documentId: string) => void) | null>(null)
