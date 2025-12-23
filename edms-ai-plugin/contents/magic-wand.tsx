// contents\magic-wand.ts
import type {PlasmoCSConfig, PlasmoGetInlineAnchorList} from "plasmo"

export const config: PlasmoCSConfig = {
    matches: ["*://*.your-edms-domain.com/*"]
}

export const getInlineAnchorList: PlasmoGetInlineAnchorList = async () => {
    return document.querySelectorAll('input[maxlength="255"]')
}

const MagicWand = () => {
    return (
        <button
            onClick={() => alert("AI помогает заполнить поле")}
            style={{marginLeft: "-30px", border: "none", background: "none", cursor: "pointer"}}
        >
            ✨
        </button>
    )
}

export default MagicWand