import { useContext } from 'react'
import { FileText, Download, FileCode, FileSpreadsheet, FileDigit, FileType, File } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, IconBox } from '../primitives'
import { AttachmentClickContext } from '../ChatContext'

const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase() || ''
    if (['doc', 'docx', 'txt', 'rtf'].includes(ext)) return FileText
    if (['xls', 'xlsx', 'csv'].includes(ext)) return FileSpreadsheet
    if (['pdf'].includes(ext)) return FileType
    if (['zip', 'rar', '7z'].includes(ext)) return FileDigit
    if (['html', 'css', 'js', 'json'].includes(ext)) return FileCode
    return File
}

interface AttachmentCardProps {
    headers: string[]
    row: string[]
    index: number
}

export function AttachmentCard({ headers, row, index }: AttachmentCardProps) {
    const onAttachmentClick = useContext(AttachmentClickContext)
    const pairs = headers.map((h, i) => ({ key: h, value: row[i] || '' }))

    const fileName = pairs.find(p => /файл|название|name/i.test(p.key))?.value
        || pairs.find(p => p.value && /\.(docx?|pdf|xlsx?|txt|rtf|zip|rar|csv)/i.test(p.value))?.value || ''
    const fileSize = pairs.find(p => /размер|size/i.test(p.key))?.value || ''
    const fileDate = pairs.find(p => /дата|date/i.test(p.key))?.value || ''

    const Icon = getFileIcon(fileName)

    const cardContent = (
        <CardHeader className="flex-row items-center gap-3 p-3 space-y-0">
        <IconBox
            icon={Icon}
            variant="zinc"
            size="sm"
        />
        <div className="flex-1 min-w-0">
            <CardTitle className="text-sm truncate group-hover:text-indigo-600 transition-colors">
                {fileName || `Вложение ${index + 1}`}
            </CardTitle>
            {(fileSize || fileDate) && (
                <CardDescription className="text-[11px] mt-0.5 flex items-center gap-2">
                    {fileSize && <span>{fileSize}</span>}
                    {fileSize && fileDate && <span className="w-1 h-1 rounded-full bg-zinc-300" />}
                    {fileDate && <span>{fileDate}</span>}
                </CardDescription>
            )}
        </div>
        <Download size={14} className="text-zinc-400 group-hover:text-indigo-500 transition-colors shrink-0" />
    </CardHeader>
    )

    return (
        <a
            href="#"
            className="block mb-2 no-underline"
            onClick={(e) => {
                e.preventDefault()
                if (fileName && onAttachmentClick) {
                    onAttachmentClick(fileName)
                }
            }}
        >
            <Card
                isClickable={!!onAttachmentClick}
                className="group hover:border-indigo-200 transition-all"
            >
                {cardContent}
            </Card>
        </a>
    )
}
