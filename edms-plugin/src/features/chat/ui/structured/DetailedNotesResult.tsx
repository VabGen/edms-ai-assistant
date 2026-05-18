import { Brain, Clock, Hash } from 'lucide-react'
import type { DetailedNotesData } from '@/entities/message/model/types'
import { CardFooter, CollapsibleSection } from './common'
import { Card, CardHeader, CardTitle, IconBox } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'

export function DetailedNotesResult({data}: { data: DetailedNotesData }) {
    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 dark:border-zinc-800">
            <CardHeader className="flex-row items-center gap-4 p-4 space-y-0 bg-gradient-to-br from-cyan-50/50 to-blue-50/50 dark:from-cyan-900/10 dark:to-blue-900/10 border-b border-zinc-100 dark:border-zinc-800">
                <IconBox
                    icon={Brain}
                    variant="primary"
                    size="md"
                />
                <div className="flex-1 min-w-0">
                    <CardTitle className="text-base font-bold leading-snug">
                        Подробные заметки
                    </CardTitle>
                    <div className="flex flex-wrap gap-2 mt-1.5">
                        <div className="px-1.5 py-0.5 rounded-md bg-cyan-100 dark:bg-cyan-900/40 text-cyan-700 dark:text-cyan-400 text-[10px] font-bold uppercase tracking-wider border border-cyan-200 dark:border-cyan-800">
                            {data.document_type}
                        </div>
                        {data.date_range && (
                            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-blue-50 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 text-[10px] font-bold border border-blue-100 dark:border-blue-800">
                                <Clock size={10}/> {data.date_range}
                            </div>
                        )}
                    </div>
                </div>
            </CardHeader>

            <div className="px-1 py-1">
                {data.sections.map((section, i) => (
                    <CollapsibleSection
                        key={i}
                        title={section.title}
                        defaultOpen={i === 0}
                    >
                        <div className="text-[14px] text-zinc-700 dark:text-zinc-300 font-medium leading-relaxed mb-4">
                            {section.content}
                        </div>

                        {section.subsections && section.subsections.length > 0 && (
                            <div className="space-y-2 border-l-2 border-cyan-100 dark:border-cyan-900/50 pl-4 py-1">
                                {section.subsections.map((sub, j) => (
                                    <div key={j} className="flex items-start gap-2 text-[12px] text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">
                                        <div className="w-1 h-1 rounded-full bg-cyan-400 mt-2 shrink-0" />
                                        {sub}
                                    </div>
                                ))}
                            </div>
                        )}
                    </CollapsibleSection>
                ))}
            </div>

            {data.key_entities.length > 0 && (
                <div className="p-4 bg-zinc-50/50 dark:bg-zinc-800/30 border-t border-zinc-100 dark:border-zinc-800">
                    <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                        <Hash size={10} /> Ключевые сущности
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {data.key_entities.map((entity, i) => (
                            <div key={i} className="px-2 py-1 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 font-mono text-[11px] font-bold shadow-sm">
                                {entity}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <CardFooter/>
        </Card>
    )
}
