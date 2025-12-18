// contents\index.ts
import type { PlasmoCSConfig, PlasmoGetStyle } from "plasmo" //
import styleText from "data-text:~style.css"
import { AssistantWidget } from "~features/AssistantWidget"

export const config: PlasmoCSConfig = {
  matches: [
  "http://localhost:3000/*",
  "https://localhost:3000/*",
  "http://localhost:3001/*",
  "http://localhost:8080/*"
]
}

export const getStyle: PlasmoGetStyle = () => {
  const style = document.createElement("style")
  style.textContent = styleText
  return style
}

export default function PlasmoOverlay() {
  return <AssistantWidget />
}