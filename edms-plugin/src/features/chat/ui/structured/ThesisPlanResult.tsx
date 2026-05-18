import { Target, Quote, ArrowRight } from 'lucide-react'
import type { ThesisPlanData } from '@/entities/message/model/types'
import { CardFooter, CollapsibleSection } from './common'
import { Card, CardHeader, CardTitle, IconBox } from '@shared/ui/primitives'

export function ThesisPlanResult({data}: { data: ThesisPlanData }) {
    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 dark:border-zinc-800">
            <CardHeader className="flex-row items-center gap-4 p-4 space-y-0 bg-gradient-to-br from-indigo-50/50 to-violet-50/50 dark:from-indigo-900/10 dark:to-violet-900/10 border-b border-zinc-100 dark:border-zinc-800">
                <IconBox
                    icon={Target}
                    variant="primary"
                    size="md"
                />
                <div className="flex-1 min-w-0">
                    <div className="text-[10px] font-bold text-indigo-500 uppercase tracking-widest mb-1">
                        Главный тезис
                    </div>
                    <CardTitle className="text-base font-bold leading-snug">
                        {data.main_argument}
                    </CardTitle>
                </div>
            </CardHeader>

            <div className="px-1 py-1">
                {data.sections.map((section, si) => (
                    <CollapsibleSection
                        key={si}
                        title={section.title || `Раздел ${si + 1}`}
                        right={
                            <span className="text-[10px] font-bold text-zinc-400 bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded-md uppercase tracking-tighter">
                                {section.points.length} тезисов
                            </span>
                        }
                    >
                        <div className="p-3.5 mb-4 bg-indigo-50/30 dark:bg-indigo-900/10 border-l-2 border-indigo-200 dark:border-indigo-800 rounded-r-xl text-[13px] text-zinc-700 dark:text-zinc-300 leading-relaxed font-medium italic">
                            {section.thesis}
                        </div>

                        <div className="space-y-3">
                            {section.points.map((point, pi) => (
                                <div key={pi} className="p-3.5 bg-zinc-50/50 dark:bg-zinc-800/20 rounded-xl border border-zinc-100 dark:border-zinc-800/50 hover:bg-white dark:hover:bg-zinc-800 transition-all hover:shadow-sm">
                                    <div className="flex items-start gap-3">
                                        <div className="w-5 h-5 rounded-md bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400 text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5 border border-indigo-200 dark:border-indigo-800/50">
                                            {pi + 1}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="text-[13px] font-bold text-zinc-900 dark:text-zinc-100 leading-relaxed">
                                                {point.claim}
                                            </div>

                                            {point.evidence && (
                                                <div className="mt-2.5 p-3 bg-white dark:bg-zinc-900 rounded-lg border border-zinc-100 dark:border-zinc-800 text-[12px] text-zinc-500 dark:text-zinc-400 italic leading-relaxed flex gap-2.5 items-start">
                                                    <Quote size={12} className="shrink-0 mt-1 text-zinc-300 dark:text-zinc-600" />
                                                    {point.evidence}
                                                </div>
                                            )}

                                            {point.sub_points && point.sub_points.length > 0 && (
                                                <div className="mt-2.5 space-y-1 pl-1">
                                                    {point.sub_points.map((sp, spi) => (
                                                        <div key={spi} className="flex items-start gap-2 text-[12px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
                                                            <div className="w-1 h-1 rounded-full bg-indigo-300 dark:bg-indigo-700 mt-2 shrink-0" />
                                                            {sp}
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CollapsibleSection>
                ))}
            </div>

            {data.conclusion && (
                <div className="mx-4 mb-4 p-4 bg-gradient-to-br from-emerald-50/50 to-cyan-50/50 dark:from-emerald-900/10 dark:to-cyan-900/10 border border-emerald-100/50 dark:border-emerald-900/30 rounded-xl flex gap-3 items-start">
                    <ArrowRight size={16} className="shrink-0 mt-0.5 text-emerald-500" />
                    <div>
                        <div className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-widest mb-1">
                            Вывод
                        </div>
                        <div className="text-[13px] text-zinc-700 dark:text-zinc-300 leading-relaxed font-medium">
                            {data.conclusion}
                        </div>
                    </div>
                </div>
            )}

            <CardFooter/>
        </Card>
    )
}
