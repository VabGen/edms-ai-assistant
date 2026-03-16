import {memo, useState} from 'react'
import {
    ArrowLeft, Palette, Mic, FileText,
    Cpu, Bot, Database, Globe,
    Save, RotateCcw, Check, AlertCircle, Loader2, WifiOff,
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

function Field({label, hint, children}: { label: string; hint?: string; children: React.ReactNode }) {
    return (
        <div className="flex flex-col gap-1.5">
            <span className="text-[9px] font-bold uppercase tracking-[0.12em]" style={{color: '#64748b'}}>{label}</span>
            {children}
            {hint && <p className="text-[9px] leading-relaxed" style={{color: '#64748b'}}>{hint}</p>}
        </div>
    )
}

function Segmented<T extends string | number>({value, onChange, options}: {
    value: T; onChange: (v: T) => void; options: { value: T; label: string }[]
}) {
    return (
        <div className="flex gap-0.5 rounded-xl p-[3px]" style={{background: 'rgba(0,0,0,0.06)'}}>
            {options.map((o) => {
                const active = String(value) === String(o.value)
                return (
                    <button key={String(o.value)} type="button" onClick={() => onChange(o.value)}
                            className="flex-1 py-1.5 rounded-lg text-[10px] font-semibold transition-all duration-150 select-none"
                            style={{
                                background: active ? 'white' : 'transparent',
                                color: active ? '#4338ca' : '#475569',
                                boxShadow: active ? '0 1px 3px rgba(0,0,0,0.12)' : 'none',
                            }}>
                        {o.label}
                    </button>
                )
            })}
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
        <div className="flex items-center justify-between gap-3">
            <div className="flex-1 min-w-0">
                <p className="text-[11px] font-medium leading-snug" style={{color: '#1e293b'}}>{label}</p>
                {hint && <p className="text-[9px] mt-0.5 leading-snug" style={{color: '#64748b'}}>{hint}</p>}
            </div>
            <button type="button" role="switch" aria-checked={value} onClick={() => onChange(!value)}
                    className="shrink-0 focus:outline-none">
                <div className="relative w-8 h-[18px] rounded-full transition-colors duration-200"
                     style={{background: value ? '#6366f1' : 'rgba(148,163,184,0.6)'}}>
                    <div
                        className="absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white shadow transition-all duration-200"
                        style={{left: value ? '18px' : '2px'}}/>
                </div>
            </button>
        </div>
    )
}

function SliderField({label, hint, value, onChange, min, max, step, fmt}: {
    label: string; hint?: string; value: number; onChange: (v: number) => void
    min: number; max: number; step: number; fmt?: (v: number) => string
}) {
    const pct = ((value - min) / (max - min)) * 100
    return (
        <Field label={label} hint={hint}>
            <div className="flex items-center gap-2">
                <input type="range" min={min} max={max} step={step} value={value}
                       onChange={(e) => onChange(Number(e.target.value))}
                       className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
                       style={{background: `linear-gradient(to right, rgba(99,102,241,0.75) ${pct}%, rgba(0,0,0,0.12) ${pct}%)`}}
                />
                <span className="min-w-[40px] text-center text-[10px] font-mono px-1.5 py-0.5 rounded-md tabular-nums"
                      style={{color: '#4338ca', background: 'rgba(99,102,241,0.08)'}}>
                    {fmt ? fmt(value) : value}
                </span>
            </div>
        </Field>
    )
}

function TextInput({value, onChange, type = 'text', min, max, mono, placeholder}: {
    value: string | number; onChange: (v: string) => void
    type?: 'text' | 'number' | 'url'; min?: number; max?: number; mono?: boolean; placeholder?: string
}) {
    return (
        <input type={type} value={value} min={min} max={max} placeholder={placeholder}
               onChange={(e) => onChange(e.target.value)}
               className={`w-full px-2.5 py-1.5 rounded-lg text-[11px] focus:outline-none focus:ring-1 focus:ring-indigo-400 transition-all ${mono ? 'font-mono' : ''}`}
               style={{
                   background: 'rgba(255,255,255,0.70)',
                   border: '1px solid rgba(203,213,225,0.80)',
                   color: '#0f172a'
               }}
        />
    )
}

function AppearanceTab({s, on}: {
    s: UserPreferences['appearance'];
    on: (p: Partial<UserPreferences['appearance']>) => void
}) {
    return (
        <div className="flex flex-col gap-4">
            <Field label="Размер текста">
                <Segmented<FontSize> value={s.fontSize} onChange={(v) => on({fontSize: v})}
                                     options={[{value: 'small', label: 'S'}, {
                                         value: 'medium',
                                         label: 'M'
                                     }, {value: 'large', label: 'L'}]}
                />
            </Field>
            <Field label="Положение виджета">
                <Segmented<WidgetPosition> value={s.widgetPosition} onChange={(v) => on({widgetPosition: v})}
                                           options={[{value: 'bottom-right', label: '↘ Справа'}, {
                                               value: 'bottom-left',
                                               label: '↙ Слева'
                                           }]}
                />
            </Field>
            <SliderField label="Прозрачность фона" hint="Интенсивность стеклянного эффекта"
                         value={s.glassOpacity} onChange={(v) => on({glassOpacity: v})}
                         min={0} max={0.5} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`}
            />
        </div>
    )
}

function VoiceTab({s, on}: { s: UserPreferences['voice']; on: (p: Partial<UserPreferences['voice']>) => void }) {
    return (
        <div className="flex flex-col gap-4">
            <ToggleRow label="Режим Hands-Free" hint="Автоотправка после паузы в речи"
                       value={s.handsFreeEnabled} onChange={(v) => on({handsFreeEnabled: v})}
            />
            <Field label="Пауза перед отправкой">
                <Segmented<AutoSendPauseMs> value={s.autoSendPauseMs}
                                            onChange={(v) => on({autoSendPauseMs: Number(v) as AutoSendPauseMs})}
                                            options={[{value: 800, label: '0.8с'}, {
                                                value: 1400,
                                                label: '1.4с'
                                            }, {value: 2000, label: '2с'}, {value: 3000, label: '3с'}]}
                />
            </Field>
            <Field label="Язык распознавания речи">
                <Segmented<STTLanguage> value={s.sttLanguage} onChange={(v) => on({sttLanguage: v})}
                                        options={[{value: 'ru-RU', label: 'Рус'}, {
                                            value: 'kk-KZ',
                                            label: 'Каз'
                                        }, {value: 'en-US', label: 'Eng'}]}
                />
            </Field>
            <div className="px-2.5 py-2 rounded-lg"
                 style={{background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.15)'}}>
                <p className="text-[9px] leading-relaxed" style={{color: '#4338ca'}}>
                    Голосовой ввод работает в Chrome/Edge. Требует разрешения на микрофон.
                </p>
            </div>
        </div>
    )
}

function DocumentsTab({s, on}: {
    s: UserPreferences['documents'];
    on: (p: Partial<UserPreferences['documents']>) => void
}) {
    return (
        <div className="flex flex-col gap-4">
            <Field label="Формат суммаризации" hint="Убирает шаг выбора при каждом запросе">
                <select value={s.defaultSummaryFormat}
                        onChange={(e) => on({defaultSummaryFormat: e.target.value as SummaryFormat})}
                        className="w-full px-2.5 py-1.5 rounded-lg text-[11px] focus:outline-none focus:ring-1 focus:ring-indigo-400 cursor-pointer"
                        style={{
                            background: 'rgba(255,255,255,0.70)',
                            border: '1px solid rgba(203,213,225,0.80)',
                            color: '#0f172a'
                        }}>
                    <option value="ask">Спрашивать каждый раз</option>
                    <option value="abstractive">Пересказ (abstractive)</option>
                    <option value="extractive">Факты (extractive)</option>
                    <option value="thesis">Тезисы (thesis)</option>
                </select>
            </Field>
            <div className="h-px" style={{background: 'rgba(0,0,0,0.06)'}}/>
            <ToggleRow label="Автоанализ при открытии" hint="Анализировать документ сразу при открытии"
                       value={s.autoAnalyzeOnOpen} onChange={(v) => on({autoAnalyzeOnOpen: v})}
            />
            <ToggleRow label="Подсказки быстрых действий" hint="Кнопки Суммаризация / Поиск / Тезисы"
                       value={s.showQuickActionHints} onChange={(v) => on({showQuickActionHints: v})}
            />
        </div>
    )
}

function LLMTab({s, on}: { s: TechSettings['llm']; on: (p: Partial<TechSettings['llm']>) => void }) {
    return (
        <div className="flex flex-col gap-3">
            <Field label="Generative URL"><TextInput value={s.generativeUrl} onChange={(v) => on({generativeUrl: v})}
                                                     type="url" mono/></Field>
            <Field label="Generative Model"><TextInput value={s.generativeModel}
                                                       onChange={(v) => on({generativeModel: v})} mono/></Field>
            <Field label="Embedding URL"><TextInput value={s.embeddingUrl} onChange={(v) => on({embeddingUrl: v})}
                                                    type="url" mono/></Field>
            <Field label="Embedding Model"><TextInput value={s.embeddingModel} onChange={(v) => on({embeddingModel: v})}
                                                      mono/></Field>
            <div className="h-px" style={{background: 'rgba(0,0,0,0.06)'}}/>
            <SliderField label="Temperature" value={s.temperature} onChange={(v) => on({temperature: v})} min={0}
                         max={2} step={0.1} fmt={(v) => v.toFixed(1)}/>
            <SliderField label="Max Tokens" value={s.maxTokens} onChange={(v) => on({maxTokens: v})} min={256}
                         max={8192} step={256} fmt={(v) => v.toLocaleString()}/>
            <Field label="Timeout (сек)"><TextInput value={s.timeout} onChange={(v) => on({timeout: Number(v)})}
                                                    type="number" min={10} max={600}/></Field>
            <Field label="Max Retries"><TextInput value={s.maxRetries} onChange={(v) => on({maxRetries: Number(v)})}
                                                  type="number" min={0} max={10}/></Field>
        </div>
    )
}

function AgentTab({s, on}: { s: TechSettings['agent']; on: (p: Partial<TechSettings['agent']>) => void }) {
    return (
        <div className="flex flex-col gap-3">
            <SliderField label="Max Iterations" value={s.maxIterations} onChange={(v) => on({maxIterations: v})} min={1}
                         max={50} step={1}/>
            <SliderField label="Context Messages" value={s.maxContextMessages}
                         onChange={(v) => on({maxContextMessages: v})} min={5} max={100} step={5}/>
            <Field label="Timeout (сек)"><TextInput value={s.timeout} onChange={(v) => on({timeout: Number(v)})}
                                                    type="number" min={10} max={600}/></Field>
            <Field label="Max Retries"><TextInput value={s.maxRetries} onChange={(v) => on({maxRetries: Number(v)})}
                                                  type="number" min={0} max={10}/></Field>
            <div className="h-px" style={{background: 'rgba(0,0,0,0.06)'}}/>
            <Field label="Log Level">
                <select value={s.logLevel}
                        onChange={(e) => on({logLevel: e.target.value as TechSettings['agent']['logLevel']})}
                        className="w-full px-2.5 py-1.5 rounded-lg text-[11px] focus:outline-none focus:ring-1 focus:ring-indigo-400 cursor-pointer"
                        style={{
                            background: 'rgba(255,255,255,0.70)',
                            border: '1px solid rgba(203,213,225,0.80)',
                            color: '#0f172a'
                        }}>
                    {(['DEBUG', 'INFO', 'WARNING', 'ERROR'] as const).map(v => <option key={v} value={v}>{v}</option>)}
                </select>
            </Field>
            <ToggleRow label="Трассировка агента" hint="Детальное логирование шагов ReAct" value={s.enableTracing}
                       onChange={(v) => on({enableTracing: v})}/>
        </div>
    )
}

function RAGTab({s, on}: { s: TechSettings['rag']; on: (p: Partial<TechSettings['rag']>) => void }) {
    return (
        <div className="flex flex-col gap-3">
            <SliderField label="Chunk Size" hint="Символов на фрагмент" value={s.chunkSize}
                         onChange={(v) => on({chunkSize: v})} min={200} max={4000} step={100}
                         fmt={(v) => v.toLocaleString()}/>
            <SliderField label="Chunk Overlap" value={s.chunkOverlap} onChange={(v) => on({chunkOverlap: v})} min={0}
                         max={1000} step={50} fmt={(v) => v.toLocaleString()}/>
            <Field label="Batch Size"><TextInput value={s.batchSize} onChange={(v) => on({batchSize: Number(v)})}
                                                 type="number" min={1} max={100}/></Field>
            <Field label="Embedding Batch"><TextInput value={s.embeddingBatchSize}
                                                      onChange={(v) => on({embeddingBatchSize: Number(v)})}
                                                      type="number" min={1} max={50}/></Field>
        </div>
    )
}

function EDMSTab({s, on}: { s: TechSettings['edms']; on: (p: Partial<TechSettings['edms']>) => void }) {
    return (
        <div className="flex flex-col gap-3">
            <Field label="Base URL"><TextInput value={s.baseUrl} onChange={(v) => on({baseUrl: v})} type="url"
                                               mono/></Field>
            <Field label="API Version"><TextInput value={s.apiVersion} onChange={(v) => on({apiVersion: v})}
                                                  mono/></Field>
            <Field label="Timeout (сек)"><TextInput value={s.timeout} onChange={(v) => on({timeout: Number(v)})}
                                                    type="number" min={10} max={600}/></Field>
        </div>
    )
}

type TabDef = { id: SettingsTab; label: string; icon: React.ReactNode }

const USER_TABS: TabDef[] = [
    {id: 'appearance', label: 'Вид', icon: <Palette size={10}/>},
    {id: 'voice', label: 'Голос', icon: <Mic size={10}/>},
    {id: 'documents', label: 'Доки', icon: <FileText size={10}/>},
]
const TECH_TABS: TabDef[] = [
    {id: 'llm', label: 'LLM', icon: <Cpu size={10}/>},
    {id: 'agent', label: 'Agent', icon: <Bot size={10}/>},
    {id: 'rag', label: 'RAG', icon: <Database size={10}/>},
    {id: 'edms', label: 'EDMS', icon: <Globe size={10}/>},
]

function TabBar({tabs, active, onChange}: { tabs: TabDef[]; active: SettingsTab; onChange: (t: SettingsTab) => void }) {
    return (
        <div className="flex gap-0.5 rounded-xl p-[3px]" style={{background: 'rgba(0,0,0,0.06)'}}>
            {tabs.map((t) => {
                const isActive = t.id === active
                return (
                    <button key={t.id} type="button" onClick={() => onChange(t.id)}
                            className="flex-1 flex flex-col items-center gap-0.5 py-1.5 rounded-[9px] text-[8px] font-bold uppercase tracking-wider transition-all duration-150 select-none"
                            style={{
                                background: isActive ? 'white' : 'transparent',
                                color: isActive ? '#4338ca' : '#94a3b8',
                                boxShadow: isActive ? '0 1px 3px rgba(0,0,0,0.12)' : 'none',
                            }}>
                        {t.icon}{t.label}
                    </button>
                )
            })}
        </div>
    )
}

function SaveButton({status, isDirty, onSave}: { status: SaveStatus; isDirty: boolean; onSave: () => void }) {
    const disabled = status === 'saving' || (!isDirty && status === 'idle')
    type Variant = { icon: React.ReactNode; label: string; bg: string; color: string }
    const variants: Record<SaveStatus, Variant> = {
        idle: {
            icon: <Save size={11}/>,
            label: 'Сохранить',
            bg: isDirty ? '#6366f1' : 'rgba(0,0,0,0.06)',
            color: isDirty ? 'white' : '#94a3b8'
        },
        saving: {
            icon: <Loader2 size={11} className="animate-spin"/>,
            label: 'Сохраняем…',
            bg: '#818cf8',
            color: 'white'
        },
        saved: {icon: <Check size={11}/>, label: 'Сохранено', bg: '#10b981', color: 'white'},
        error: {icon: <AlertCircle size={11}/>, label: 'Ошибка', bg: '#ef4444', color: 'white'},
    }
    const v = variants[status]
    return (
        <button type="button" onClick={onSave} disabled={disabled}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-all"
                style={{background: v.bg, color: v.color, cursor: disabled ? 'not-allowed' : 'pointer'}}>
            {v.icon}<span>{v.label}</span>
        </button>
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
        discardDraft
    } = useSettingsStore()
    const [activeTab, setActiveTab] = useState<SettingsTab>('appearance')
    const isUserTab = USER_TABS.some(t => t.id === activeTab)
    const isTechTab = TECH_TABS.some(t => t.id === activeTab)

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center flex-1 gap-2" style={{opacity: 0.5}}>
                <Loader2 size={18} className="animate-spin" style={{color: '#6366f1'}}/>
                <p className="text-[10px]" style={{color: '#64748b'}}>Загрузка…</p>
            </div>
        )
    }

    return (
        <div className="w-full flex flex-col h-full">
            <div className="flex items-center gap-1.5 mb-3 shrink-0">
                <button type="button" onClick={() => {
                    discardDraft();
                    onClose()
                }}
                        className="p-1.5 rounded-lg transition-all shrink-0"
                        style={{color: '#94a3b8'}}
                        title="Назад">
                    <ArrowLeft size={13}/>
                </button>
                <span className="text-[10px] font-bold uppercase tracking-[0.12em] flex-1 truncate"
                      style={{color: '#1e293b'}}>
                    Настройки
                </span>
                {isDirty && <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{background: '#f59e0b'}}
                                  title="Есть несохранённые изменения"/>}
                {isDirty && (
                    <button type="button" onClick={resetAll} title="Сбросить к дефолтам"
                            className="p-1.5 rounded-lg transition-all shrink-0"
                            style={{color: '#94a3b8'}}>
                        <RotateCcw size={11}/>
                    </button>
                )}
            </div>

            {isTechOffline && showTechnical && (
                <div className="flex items-start gap-1.5 px-2 py-1.5 mb-2 rounded-lg shrink-0"
                     style={{background: 'rgba(251,191,36,0.10)', border: '1px solid rgba(251,191,36,0.25)'}}>
                    <WifiOff size={10} className="shrink-0 mt-0.5" style={{color: '#f59e0b'}}/>
                    <p className="text-[9px] leading-relaxed" style={{color: '#92400e'}}>Сервер недоступен — тех.
                        настройки из кэша</p>
                </div>
            )}

            <div className="flex-1 overflow-y-auto scrollbar-thin flex flex-col gap-3 min-h-0">
                {showTechnical && (
                    <p className="text-[8px] font-bold uppercase tracking-[0.15em] px-0.5 shrink-0"
                       style={{color: '#94a3b8'}}>
                        Мои настройки
                    </p>
                )}

                <TabBar tabs={USER_TABS} active={isUserTab ? activeTab : 'appearance'} onChange={setActiveTab}/>

                <div className="shrink-0">
                    {activeTab === 'appearance' &&
                        <AppearanceTab s={draft.user.appearance} on={(p) => updateUser('appearance', p)}/>}
                    {activeTab === 'voice' && <VoiceTab s={draft.user.voice} on={(p) => updateUser('voice', p)}/>}
                    {activeTab === 'documents' &&
                        <DocumentsTab s={draft.user.documents} on={(p) => updateUser('documents', p)}/>}
                </div>

                {showTechnical && (
                    <>
                        <div className="flex items-center gap-2 py-1 shrink-0">
                            <div className="flex-1 h-px" style={{background: 'rgba(0,0,0,0.08)'}}/>
                            <span className="text-[7px] font-bold uppercase tracking-[0.15em]"
                                  style={{color: '#94a3b8'}}>Технические</span>
                            <div className="flex-1 h-px" style={{background: 'rgba(0,0,0,0.08)'}}/>
                        </div>
                        <TabBar tabs={TECH_TABS} active={isTechTab ? activeTab : TECH_TABS[0].id}
                                onChange={setActiveTab}/>
                        <div className="shrink-0">
                            {activeTab === 'llm' && <LLMTab s={draft.tech.llm} on={(p) => updateTech('llm', p)}/>}
                            {activeTab === 'agent' &&
                                <AgentTab s={draft.tech.agent} on={(p) => updateTech('agent', p)}/>}
                            {activeTab === 'rag' && <RAGTab s={draft.tech.rag} on={(p) => updateTech('rag', p)}/>}
                            {activeTab === 'edms' && <EDMSTab s={draft.tech.edms} on={(p) => updateTech('edms', p)}/>}
                        </div>
                    </>
                )}

                <div className="h-2 shrink-0"/>
            </div>

            <div className="flex items-center justify-between pt-2.5 mt-1 shrink-0"
                 style={{borderTop: '1px solid rgba(0,0,0,0.08)'}}>
                {isDirty && saveStatus === 'idle' ? (
                    <button type="button" onClick={discardDraft} className="text-[10px] transition-colors"
                            style={{color: '#94a3b8'}}>
                        Отменить
                    </button>
                ) : <span/>}
                <SaveButton status={saveStatus} isDirty={isDirty} onSave={saveAll}/>
            </div>
        </div>
    )
})