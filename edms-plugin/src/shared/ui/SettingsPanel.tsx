import {memo, useState, useRef, useEffect} from 'react'
import {
    ArrowLeft, Palette, Mic, FileText,
    Cpu, Bot, Database, Globe,
    Save, RotateCcw, Check, AlertCircle, Loader2, WifiOff, ChevronDown, Sparkles
} from 'lucide-react'
import {
    useSettingsStore,
    type SettingsTab,
    type UserPreferences,
    type TechSettings,
    type SaveStatus,
    type FontSize,
    type WidgetPosition,
    type SummaryFormat,
    type STTLanguage,
    type AutoSendPauseMs,
} from '../hooks/useSettingsStore'
import { Card, CardHeader, CardTitle, IconBox, Button, Toggle, Slider, Field, TextInput, Segmented } from './primitives'
import { cn } from '@shared/lib/cn'

function SelectField<T extends string>({value, onChange, options}: {
    value: T
    onChange: (v: T) => void
    options: readonly {value: T; label: string}[]
}) {
    const [open, setOpen] = useState(false)
    const ref = useRef<HTMLDivElement>(null)
    const selectedLabel = options.find(o => o.value === value)?.label ?? value

    useEffect(() => {
        if (!open) return
        const handler = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
        }
        document.addEventListener('mousedown', handler)
        return () => document.removeEventListener('mousedown', handler)
    }, [open])

    return (
        <div ref={ref} className="relative">
            <button
                type="button"
                onClick={() => setOpen(p => !p)}
                className={cn(
                    "w-full px-3 py-2.5 rounded-xl text-[13px] font-medium flex items-center justify-between transition-all duration-200 bg-white dark:bg-zinc-900 border",
                    open ? "border-blue-500 ring-2 ring-blue-500/10 shadow-sm" : "border-zinc-200 dark:border-zinc-800 shadow-none"
                )}
            >
                <span className="text-zinc-700 dark:text-zinc-200">{selectedLabel}</span>
                <ChevronDown
                    size={14}
                    className={cn(
                        "text-zinc-400 transition-transform duration-200",
                        open && "rotate-180"
                    )}
                />
            </button>
            {open && (
                <div className="absolute left-0 right-0 z-[100] mt-1.5 p-1 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl shadow-lg animate-edms-fade-in">
                    {options.map(o => (
                        <button
                            key={o.value}
                            type="button"
                            onClick={() => { onChange(o.value); setOpen(false) }}
                            className={cn(
                                "w-full px-3 py-2 rounded-lg text-left text-[13px] transition-colors",
                                o.value === value
                                  ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 font-bold"
                                  : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800"
                            )}
                        >
                            {o.label}
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}

function ToggleRow({label, hint, value, onChange}: {
    label: string;
    hint?: string;
    value: boolean;
    onChange: (v: boolean) => void
}) {
    return (
        <div className="flex items-center justify-between gap-4 py-1">
            <div className="flex-1 min-w-0">
                <p className="text-[13px] font-bold text-zinc-700 dark:text-zinc-200 leading-tight mb-0.5">{label}</p>
                {hint && <p className="text-[11px] text-zinc-400 dark:text-zinc-500 leading-snug">{hint}</p>}
            </div>
            <Toggle checked={value} onChange={onChange} />
        </div>
    )
}

function AppearanceTab({s, on}: {
    s: UserPreferences['appearance'];
    on: (p: Partial<UserPreferences['appearance']>) => void
}) {
    return (
        <div className="space-y-5">
            <Field label="Размер текста" hint="Масштаб интерфейса">
                <Segmented<FontSize>
                  value={s.fontSize}
                  onChange={(v) => on({fontSize: v})}
                  options={[
                    {value: 'small', label: 'Мелкий'},
                    {value: 'medium', label: 'Средний'},
                    {value: 'large', label: 'Крупный'}
                  ]}
                />
            </Field>
            <Field label="Положение виджета">
                <Segmented<WidgetPosition>
                  value={s.widgetPosition}
                  onChange={(v) => on({widgetPosition: v})}
                  options={[
                    {value: 'bottom-right', label: 'Справа'},
                    {value: 'bottom-left', label: 'Слева'}
                  ]}
                />
            </Field>
            <Slider
              label="Прозрачность"
              hint="Эффект матового стекла"
              value={s.glassOpacity}
              onChange={(v) => on({glassOpacity: v})}
              min={0} max={0.5} step={0.05}
              format={(v) => `${Math.round(v * 100)}%`}
            />
        </div>
    )
}

function VoiceTab({s, on}: { s: UserPreferences['voice']; on: (p: Partial<UserPreferences['voice']>) => void }) {
    return (
        <div className="space-y-5">
            <ToggleRow label="Hands-Free" hint="Автоотправка при паузе"
                       value={s.handsFreeEnabled} onChange={(v) => on({handsFreeEnabled: v})}
            />
            <Field label="Пауза">
                <Segmented<AutoSendPauseMs>
                  value={s.autoSendPauseMs}
                  onChange={(v) => on({autoSendPauseMs: Number(v) as AutoSendPauseMs})}
                  options={[
                    {value: 800, label: '0.8с'},
                    {value: 1400, label: '1.4с'},
                    {value: 2000, label: '2с'},
                    {value: 3000, label: '3с'}
                  ]}
                />
            </Field>
            <Field label="Язык">
                <Segmented<STTLanguage>
                  value={s.sttLanguage}
                  onChange={(v) => on({sttLanguage: v})}
                  options={[
                    {value: 'ru-RU', label: 'RUS'},
                    {value: 'kk-KZ', label: 'KAZ'},
                    {value: 'en-US', label: 'ENG'}
                  ]}
                />
            </Field>
            <Card className="bg-blue-50/30 border-blue-100 dark:bg-blue-900/10 dark:border-blue-900/30 p-3 shadow-none">
                <div className="flex gap-2.5 items-start">
                    <IconBox icon={Mic} variant="primary" size="sm" />
                    <p className="text-[11px] text-blue-700 dark:text-blue-400 font-medium leading-relaxed">
                        Голосовой ввод работает в Chrome/Edge. <br/>Требуется доступ к микрофону.
                    </p>
                </div>
            </Card>
        </div>
    )
}

function DocumentsTab({s, on}: {
    s: UserPreferences['documents'];
    on: (p: Partial<UserPreferences['documents']>) => void
}) {
    return (
        <div className="space-y-5">
            <Field label="Суммаризация" hint="Формат по умолчанию">
                <SelectField<SummaryFormat>
                    value={s.defaultSummaryFormat}
                    onChange={(v) => on({defaultSummaryFormat: v})}
                    options={[
                        {value: 'ask', label: 'Спрашивать всегда'},
                        {value: 'abstractive', label: 'Пересказ'},
                        {value: 'extractive', label: 'Факты'},
                        {value: 'thesis', label: 'Тезисы'},
                    ]}
                />
            </Field>
            <div className="space-y-3 pt-2">
                <ToggleRow label="Автоанализ" hint="Сразу при открытии документа"
                           value={s.autoAnalyzeOnOpen} onChange={(v) => on({autoAnalyzeOnOpen: v})}
                />
                <ToggleRow label="Подсказки" hint="Быстрые кнопки в чате"
                           value={s.showQuickActionHints} onChange={(v) => on({showQuickActionHints: v})}
                />
            </div>
        </div>
    )
}

function TechField({label, value, onChange, type = 'text'}: any) {
    return (
        <div className="space-y-2">
            <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest pl-1">{label}</label>
            <TextInput value={value} onChange={onChange} type={type} className="font-mono text-[12px]" />
        </div>
    )
}

function LLMTab({s, on}: { s: TechSettings['llm']; on: (p: Partial<TechSettings['llm']>) => void }) {
    return (
        <div className="space-y-4">
            <TechField label="Generative URL" value={s.generativeUrl} onChange={(v: string) => on({generativeUrl: v})} />
            <TechField label="Generative Model" value={s.generativeModel} onChange={(v: string) => on({generativeModel: v})} />
            <TechField label="Embedding URL" value={s.embeddingUrl} onChange={(v: string) => on({embeddingUrl: v})} />
            <TechField label="Embedding Model" value={s.embeddingModel} onChange={(v: string) => on({embeddingModel: v})} />
            <div className="grid grid-cols-2 gap-4">
                <TechField label="Timeout" value={s.timeout} onChange={(v: string) => on({timeout: Number(v)})} type="number" />
                <TechField label="Retries" value={s.maxRetries} onChange={(v: string) => on({maxRetries: Number(v)})} type="number" />
            </div>
        </div>
    )
}

type TabDef = { id: SettingsTab; label: string; icon: any }

const USER_TABS: TabDef[] = [
    {id: 'appearance', label: 'Вид', icon: Palette},
    {id: 'voice', label: 'Голос', icon: Mic},
    {id: 'documents', label: 'Доки', icon: FileText},
]
const TECH_TABS: TabDef[] = [
    {id: 'llm', label: 'LLM', icon: Cpu},
    {id: 'agent', label: 'Agent', icon: Bot},
    {id: 'rag', label: 'RAG', icon: Database},
    {id: 'edms', label: 'EDMS', icon: Globe},
]

function TabBar({tabs, active, onChange}: { tabs: TabDef[]; active: SettingsTab; onChange: (t: SettingsTab) => void }) {
    return (
        <div className="flex p-1 bg-zinc-100 dark:bg-zinc-800 rounded-xl mb-6">
            {tabs.map((t) => {
                const isActive = t.id === active
                const Icon = t.icon
                return (
                    <button
                        key={t.id}
                        type="button"
                        onClick={() => onChange(t.id)}
                        className={cn(
                            "flex-1 flex flex-col items-center gap-1.5 py-2 rounded-lg transition-all",
                            isActive
                              ? "bg-white dark:bg-zinc-900 text-blue-600 dark:text-blue-400 shadow-sm"
                              : "text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
                        )}
                    >
                        <Icon size={14} className={isActive ? "text-blue-500" : "opacity-60"} />
                        <span className="text-[9px] font-bold uppercase tracking-wider">{t.label}</span>
                    </button>
                )
            })}
        </div>
    )
}

function SaveButton({status, isDirty, onSave}: { status: SaveStatus; isDirty: boolean; onSave: () => void }) {
    const isLoading = status === 'saving'
    const isSaved = status === 'saved'
    const isError = status === 'error'

    return (
        <Button
            onClick={onSave}
            disabled={isLoading || (!isDirty && !isSaved && !isError)}
            className={cn(
                "min-w-[120px] transition-all",
                isSaved && "bg-emerald-600 hover:bg-emerald-700",
                isError && "bg-rose-600 hover:bg-rose-700",
                isDirty && !isLoading && "shadow-lg shadow-blue-200 dark:shadow-none"
            )}
        >
            {isLoading ? <Loader2 size={16} className="animate-spin mr-2" /> :
             isSaved ? <Check size={16} className="mr-2" /> :
             isError ? <AlertCircle size={16} className="mr-2" /> :
             <Save size={16} className="mr-2" />}

            {isLoading ? 'Загрузка...' : isSaved ? 'Сохранено' : isError ? 'Ошибка' : 'Сохранить'}
        </Button>
    )
}

interface SettingsPanelProps {
    onClose: () => void
}

export const SettingsPanel = memo(function SettingsPanel({onClose}: SettingsPanelProps) {
    const {
        draft,
        isDirty,
        saveStatus,
        showTechnical,
        isLoading,
        isTechOffline,
        updateUser,
        updateTech,
        saveAll,
        resetAll,
        resetToDefaults,
        discardDraft
    } = useSettingsStore()
    const [activeTab, setActiveTab] = useState<SettingsTab>('appearance')
    const isUserTab = USER_TABS.some(t => t.id === activeTab)

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center flex-1 gap-4 py-20 animate-pulse">
                <IconBox icon={Loader2} variant="primary" size="lg" className="animate-spin" />
                <p className="text-[12px] font-bold text-zinc-400 uppercase tracking-widest">Инициализация...</p>
            </div>
        )
    }

    return (
        <div className="w-full flex flex-col h-full bg-white rounded-2xl overflow-hidden animate-edms-fade-in">
            {/* Header */}
            <div className="px-6 py-5 border-b border-zinc-100 flex items-center justify-between bg-zinc-50/50">
                <div className="flex items-center gap-4">
                    <button
                        type="button"
                        onClick={() => { discardDraft(); onClose() }}
                        className="p-2.5 -ml-2 rounded-2xl hover:bg-white text-zinc-400 hover:text-zinc-900 transition-all border border-transparent hover:border-zinc-200"
                    >
                        <ArrowLeft size={20}/>
                    </button>
                    <div>
                        <h2 className="text-[17px] font-bold text-zinc-900 leading-none">Настройки</h2>
                        <p className="text-[11px] font-bold text-zinc-400 uppercase tracking-widest mt-1.5">Персонализация</p>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    {isDirty && (
                        <div className="flex items-center gap-2 pr-3 border-r border-zinc-100">
                             <button
                                type="button"
                                onClick={showTechnical ? resetToDefaults : resetAll}
                                className="p-2.5 rounded-2xl text-zinc-400 hover:text-amber-600 hover:bg-amber-50 transition-all"
                                title="Сбросить"
                            >
                                <RotateCcw size={18}/>
                            </button>
                        </div>
                    )}
                    <IconBox icon={Sparkles} variant="primary" size="sm" />
                </div>
            </div>

            {/* Offline Alert */}
            {isTechOffline && showTechnical && (
                <div className="mx-6 mt-6 p-4 rounded-[20px] bg-amber-50 border border-amber-100 flex items-start gap-3 animate-edms-slide-up">
                    <WifiOff size={18} className="text-amber-500 shrink-0 mt-0.5" />
                    <p className="text-[12px] font-medium text-amber-800 leading-relaxed">
                        Связь с сервером потеряна. Технические настройки загружены из локального кэша.
                    </p>
                </div>
            )}

            {/* Content */}
            <div className="flex-1 overflow-y-auto px-6 py-8 scrollbar-thin">
                <TabBar
                  tabs={USER_TABS}
                  active={isUserTab ? activeTab : 'appearance'}
                  onChange={setActiveTab}
                />

                <div className="animate-edms-fade-in transition-all">
                    {activeTab === 'appearance' &&
                        <AppearanceTab s={draft.user.appearance} on={(p) => updateUser('appearance', p)}/>}
                    {activeTab === 'voice' && <VoiceTab s={draft.user.voice} on={(p) => updateUser('voice', p)}/>}
                    {activeTab === 'documents' &&
                        <DocumentsTab s={draft.user.documents} on={(p) => updateUser('documents', p)}/>}
                </div>

                {showTechnical && (
                    <div className="mt-12 space-y-8">
                        <div className="flex items-center gap-5">
                            <div className="flex-1 h-px bg-zinc-100" />
                            <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-[0.3em]">Системные</span>
                            <div className="flex-1 h-px bg-zinc-100" />
                        </div>

                        <TabBar
                            tabs={TECH_TABS}
                            active={USER_TABS.some(t => t.id === activeTab) ? 'llm' : activeTab}
                            onChange={setActiveTab}
                        />

                        <div className="animate-edms-fade-in">
                            {activeTab === 'llm' && <LLMTab s={draft.tech.llm} on={(p: any) => updateTech('llm', p)}/>}
                        </div>
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="p-6 border-t border-zinc-100 flex items-center justify-between bg-zinc-50/30">
                {isDirty && saveStatus === 'idle' ? (
                    <Button variant="ghost" onClick={discardDraft} className="text-zinc-500 hover:text-zinc-900 rounded-2xl px-6">
                        Отменить
                    </Button>
                ) : <div/>}
                <SaveButton status={saveStatus} isDirty={isDirty} onSave={saveAll}/>
            </div>
        </div>
    )
})
