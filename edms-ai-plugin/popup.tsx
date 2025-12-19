import { useState, useEffect } from "react"
import LiquidGlassFilter from "./features/LiquidGlassFilter"
import "./style.css"

function IndexPopup() {
  const [isEnabled, setIsEnabled] = useState(true)

  useEffect(() => {
    chrome.storage.local.get(["assistantEnabled"], (result) => {
      if (result.assistantEnabled !== undefined) {
        setIsEnabled(result.assistantEnabled)
      }
    })
  }, [])

  const toggleAssistant = () => {
    const newValue = !isEnabled
    setIsEnabled(newValue)
    chrome.storage.local.set({ assistantEnabled: newValue })
  }

  const frostedGlass = "bg-white/40 backdrop-blur-md border border-white/20 shadow-lg";
  const liquidGlass = "relative isolation-auto before:content-[''] before:absolute before:inset-0 before:pointer-events-none before:box-shadow-[inset_0_0_10px_rgba(255,255,255,0.5)] after:content-[''] after:absolute after:inset-0 after:pointer-events-none after:bg-white/10 after:backdrop-blur-[4px] after:[filter:url(#liquid-glass-filter)] after:-z-10";

  return (
    <div className="p-2 bg-transparent">
      <LiquidGlassFilter />

      <div className={`w-64 p-5 rounded-2xl flex flex-col gap-4 ${liquidGlass} ${frostedGlass}`}>
        <div className="flex items-center justify-between relative z-10">
          <span className="font-bold text-slate-800 text-sm tracking-tight">
            EDMS Assistant
          </span>

          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              className="sr-only peer"
              checked={isEnabled}
              onChange={toggleAssistant}
            />
            <div className="w-11 h-6 bg-slate-300/50 rounded-full peer peer-focus:ring-2 peer-focus:ring-indigo-300 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600 shadow-sm"></div>
          </label>
        </div>

        <p className="text-[11px] text-slate-600 leading-relaxed relative z-10">
          Управление отображением ассистента в интерфейсе системы.
        </p>

        <div className="pt-2 border-t border-white/20 relative z-10">
            <div className={`text-[10px] font-medium px-2 py-1 rounded-md w-fit ${isEnabled ? 'text-indigo-700 bg-indigo-100/50' : 'text-slate-500 bg-slate-200/50'}`}>
                Статус: {isEnabled ? "Активен" : "Выключен"}
            </div>
        </div>
      </div>
    </div>
  )
}

export default IndexPopup