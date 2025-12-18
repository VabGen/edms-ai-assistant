import { useState, useEffect } from "react"
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

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, {
          type: "toggleAssistant",
          enabled: newValue
        })
      }
    })
  }

  return (
    <div className="w-64 p-4 bg-white flex flex-col gap-4 shadow-xl">
      <div className="flex items-center justify-between">
        <span className="font-bold text-slate-700">EDMS Assistant</span>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            className="sr-only peer"
            checked={isEnabled}
            onChange={toggleAssistant}
          />
          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
        </label>
      </div>
      <p className="text-xs text-slate-500">
        Включите или выключите отображение виджета на страницах системы.
      </p>
    </div>
  )
}

export default IndexPopup