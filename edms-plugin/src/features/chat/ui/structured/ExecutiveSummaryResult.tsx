import { Sparkles, Lightbulb, Info } from 'lucide-react'
import type { ExecutiveSummaryData } from '@/entities/message/model/types'
import { CardFooter } from './common'
import { Card, CardHeader, CardTitle, IconBox } from '@shared/ui/primitives'

export function ExecutiveSummaryResult({data}: { data: ExecutiveSummaryData }) {
    return (
        <Card className="p-0 overflow-hidden shadow-sm border-zinc-200/60 ">
            <CardHeader className="flex-row items-center gap-4 p-4 space-y-0 bg-gradient-to-br from-violet-50/50 to-indigo-50/50   border-b border-zinc-100 ">
                <IconBox
                    icon={Sparkles}
                    variant="primary"
                    size="md"
                />
                <div className="flex-1 min-w-0">
                    <div className="text-[10px] font-bold text-violet-500 uppercase tracking-widest mb-1">
                        Резюме
                    </div>
                    <CardTitle className="text-base font-bold leading-snug">
                        {data.headline}
                    </CardTitle>
                </div>
            </CardHeader>

            {data.bullets.length > 0 && (
                <div className="p-4 space-y-3">
                    {data.bullets.map((bullet, i) => (
                        <div key={i} className="flex items-start gap-3 group">
                            <div className="w-5 h-5 rounded-md bg-violet-50  text-violet-600  text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5 border border-violet-100 ">
                                {i + 1}
                            </div>
                            <span className="text-[14px] text-zinc-700  leading-relaxed">
                                {bullet}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {data.recommendation && (
                <div className="mx-4 mb-4 p-4 bg-amber-50/50  border border-amber-100  rounded-xl flex gap-3 items-start shadow-sm">
                    <IconBox icon={Lightbulb} variant="warning" size="sm" className="bg-white  shadow-sm mt-0.5" />
                    <div>
                        <div className="text-[11px] font-bold text-amber-700  uppercase tracking-tight mb-1">Рекомендация</div>
                        <div className="text-[13px] text-zinc-800  leading-relaxed font-medium">
                            {data.recommendation}
                        </div>
                    </div>
                </div>
            )}

            <CardFooter/>
        </Card>
    )
}
