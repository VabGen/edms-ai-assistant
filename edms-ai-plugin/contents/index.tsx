import type { PlasmoCSConfig, PlasmoGetStyle } from "plasmo"
import styleText from "data-text:~style.css"
import { AssistantWidget } from "~features/AssistantWidget"

export const config: PlasmoCSConfig = {
  matches: [
    "http://localhost:3000/*",
    "https://localhost:3000/*",
    "http://localhost:3001/*",
    "https://localhost:3001/*",
    "http://localhost:8080/*",
    "https://localhost:8080/*",
    "http://127.0.0.1:*/*",
    "https://127.0.0.1:*/*"
  ]
}

export const getStyle: PlasmoGetStyle = () => {
  const style = document.createElement("style")
  style.textContent = styleText
  return style
}

export default function PlasmoOverlay() {
  return (
    <div className="edms-assistant-wrapper antialiased text-slate-900">
      <AssistantWidget />
    </div>
  )
}