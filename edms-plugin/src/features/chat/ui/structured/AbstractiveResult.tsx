import { BookOpen, Sparkles } from 'lucide-react'
import type { AbstractiveData } from '@/entities/message/model/types'
import { CardFooter, THEME_PALETTE } from './common'
import { Card, CardHeader, CardTitle, IconBox } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'

export function AbstractiveResult({data}: { data: AbstractiveData }) {
    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 dark:border-zinc-800">
            <CardHeader className="flex-row items-center gap-4 p-4 space-y-0 bg-gradient-to-br from-emerald-50/50 to-cyan-50/50 dark:from-emerald-900/10 dark:to-cyan-900/10 border-b border-zinc-100 dark:border-zinc-800">
                <IconBox
                    icon={BookOpen}
                    variant="success"
                    size="md"
                />
                <div className="flex-1 min-w-0">
                    <CardTitle className="text-base font-bold leading-snug">
                        Краткое изложение
                    </CardTitle>
                    <div className="text-[11px] font-medium text-emerald-600/70 dark:text-emerald-400/70 mt-1">
                        Абстрактивная суммаризация
                    </div>
                </div>
            </CardHeader>

            {data.key_themes.length > 0 && (
                <div className="flex flex-wrap gap-2 p-4 bg-zinc-50/50 dark:bg-zinc-800/30 border-b border-zinc-100 dark:border-zinc-800">
                    {data.key_themes.map((theme, i) => {
                        const palettes = Object.values(THEME_PALETTE)
                        const p = palettes[(i + 1) % palettes.length] ?? palettes[0]!
                        return (
                            <div key={i} className={cn(
                                "flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[10px] font-bold uppercase tracking-tight",
                                p.bg, p.border, p.text
                            )}>
                                <Sparkles size={10} className="opacity-70" />
                                {theme}
                            </div>
                        )
                    })}
                </div>
            )}

            <div className="p-4 sm:p-5 text-[15px] text-zinc-700 dark:text-zinc-300 font-medium leading-relaxed">
                {data.summary.split(/\n\n+/).map((para, i) => (
                    <p key={i} className="mb-4 last:mb-0">
                        {para}
                    </p>
                ))}
            </div>

            <CardFooter/>
        </Card>
    )
}
