import { Globe, Info, ArrowRight } from 'lucide-react'
import type { MultilingualData } from '@/entities/message/model/types'
import { CardFooter } from './common'
import { Card, CardHeader, CardTitle, IconBox } from '@shared/ui/primitives'

const LANG_NAMES: Record<string, string> = {
    ru: 'Русский', en: 'English', be: 'Белорусская',
    de: 'Deutsch', fr: 'Français', es: 'Español',
    zh: '中文', ja: '日本語', ko: '한국어',
}

function langName(code: string) {
    return LANG_NAMES[code] ?? code.toUpperCase()
}

export function MultilingualResult({data}: { data: MultilingualData }) {
    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 dark:border-zinc-800">
            <CardHeader className="flex-row items-center gap-4 p-4 space-y-0 bg-gradient-to-br from-rose-50/50 to-violet-50/50 dark:from-rose-900/10 dark:to-violet-900/10 border-b border-zinc-100 dark:border-zinc-800">
                <IconBox
                    icon={Globe}
                    variant="error"
                    size="md"
                />
                <div className="flex-1 min-w-0">
                    <CardTitle className="text-base font-bold leading-snug">
                        Многоязычная суммаризация
                    </CardTitle>
                    <div className="flex items-center gap-2 mt-1.5">
                        <div className="px-1.5 py-0.5 rounded-md bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-400 text-[10px] font-bold uppercase tracking-wider border border-rose-200 dark:border-rose-800">
                            {langName(data.detected_language)}
                        </div>
                        <ArrowRight size={10} className="text-zinc-400" />
                        <div className="px-1.5 py-0.5 rounded-md bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-400 text-[10px] font-bold uppercase tracking-wider border border-violet-200 dark:border-violet-800">
                            {langName(data.summary_language)}
                        </div>
                    </div>
                </div>
            </CardHeader>

            <div className="p-4 sm:p-5 text-[15px] text-zinc-700 dark:text-zinc-300 font-medium leading-relaxed">
                {data.summary.split(/\n\n+/).map((para, i) => (
                    <p key={i} className="mb-4 last:mb-0">
                        {para}
                    </p>
                ))}
            </div>

            {data.translation_notes && (
                <div className="mx-4 mb-4 p-4 bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-100 dark:border-zinc-800 rounded-xl flex gap-3 items-start">
                    <Info size={16} className="shrink-0 mt-0.5 text-zinc-400" />
                    <div>
                        <div className="text-[11px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-tight mb-1">Примечания к переводу</div>
                        <div className="text-[13px] text-zinc-600 dark:text-zinc-400 leading-relaxed font-medium italic">
                            {data.translation_notes}
                        </div>
                    </div>
                </div>
            )}

            <CardFooter/>
        </Card>
    )
}
