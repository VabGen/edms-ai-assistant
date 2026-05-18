import { useContext } from 'react'
import { FileText, ChevronRight } from 'lucide-react'
import { BaseCard } from '../primitives/BaseCard'
import { AttachmentClickContext } from '../ChatContext'

interface AttachmentCardProps {
  headers: string[]
  row: string[]
  index: number
}

export function AttachmentCard({ headers, row, index }: AttachmentCardProps) {
    const onAttachmentClick = useContext(AttachmentClickContext)
    const pairs = headers.map((h, i) => ({key: h, value: row[i] || ''}))

    const fileName = pairs.find(p => /файл|название|name/i.test(p.key))?.value
        || pairs.find(p => p.value && /\.(docx?|pdf|xlsx?|txt|rtf)/i.test(p.value))?.value || ''
    const fileSize = pairs.find(p => /размер|size/i.test(p.key))?.value || ''
    const fileDate = pairs.find(p => /дата|date/i.test(p.key))?.value || ''

    return (
        <BaseCard
            onClick={() => fileName && onAttachmentClick?.(fileName)}
            isClickable={!!onAttachmentClick}
        >
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-indigo-50/50 flex items-center justify-center shrink-0 text-indigo-500 text-[12px] font-bold">
                  {index + 1}
                </div>

                <FileText size={18} className="text-indigo-500 opacity-70 shrink-0"/>

                <div className="flex-1 min-w-0">
                    <div className="text-[13px] font-semibold text-slate-900 truncate">
                      {fileName || `Вложение ${index + 1}`}
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5 flex gap-2">
                        {fileSize && <span>{fileSize}</span>}
                        {fileDate && <span>{fileDate}</span>}
                    </div>
                </div>
                {onAttachmentClick && <ChevronRight size={16} className="text-slate-300 shrink-0"/>}
            </div>
        </BaseCard>
    )
}
