import './globals.css'
import type {Metadata} from 'next'

export const metadata: Metadata = {
    title: 'EDMS AI Popup',
    description: 'AI Assistant for EDMS',
}

export default function RootLayout({
                                       children,
                                   }: {
    children: React.ReactNode
}) {
    return (
        <html lang="ru">
        <body className="m-0 p-0 overflow-hidden bg-transparent">
        {children}
        </body>
        </html>
    )
}