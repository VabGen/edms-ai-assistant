import type {PlasmoCSConfig, PlasmoGetInlineAnchorList, PlasmoGetStyle} from "plasmo"
import React, {useState, useEffect} from "react"

// Конфигурация путей
export const config: PlasmoCSConfig = {
    matches: [
        "http://localhost:3000/*",
        "http://localhost:3001/*",
        "http://localhost:8080/*"
    ]
}

/**
 * КЛЮЧЕВОЙ МОМЕНТ: Переопределение стилей контейнера Plasmo.
 * Это принудительно занижает z-index кнопок-звездочек,
 * чтобы они не "прорезали" основной чат.
 */
export const getStyle: PlasmoGetStyle = () => {
    const style = document.createElement("style")
    style.textContent = `
    :host {
      z-index: 1 !important; 
      position: relative !important;
    }
    #plasmo-shadow-container {
      z-index: 1 !important;
    }
  `
    return style
}

/**
 * Ищем только блоки вложений.
 * Фильтр :has(span.lead) гарантирует, что мы вешаем кнопки
 * только там, где есть имя файла.
 */
export const getInlineAnchorList: PlasmoGetInlineAnchorList = async () => {
    return document.querySelectorAll('div.alert.alert-secondary:has(span.lead)')
}

// --- Вспомогательные функции ---

const extractDocIdFromUrl = (): string => {
    try {
        const pathParts = window.location.pathname.split('/');
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        const foundId = pathParts.find(part => uuidRegex.test(part));
        if (foundId) return foundId;
    } catch (e) {
        console.error("Ошибка парсинга URL:", e);
    }
    return "main_assistant";
};

const getAuthToken = (): string | null => {
    try {
        const directToken = localStorage.getItem('token') || localStorage.getItem('access_token') || sessionStorage.getItem('token');
        if (directToken) return directToken;
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && (key.includes('auth') || key.includes('user') || key.includes('oidc'))) {
                const value = localStorage.getItem(key);
                if (value?.includes('eyJ')) return value.startsWith('{') ? JSON.parse(value).access_token : value;
            }
        }
    } catch (e) {
        console.error("Token error:", e);
    }
    return null;
};

interface AttachmentProps {
    anchor?: { element: HTMLElement }
}

const AttachmentActions = ({anchor}: AttachmentProps) => {
    const [isEnabled, setIsEnabled] = useState(true);
    const [isHovered, setIsHovered] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const element = anchor?.element;

    const fileName = anchor?.element?.querySelector('span.lead')?.textContent || "Документ";

    useEffect(() => {
        chrome.storage.local.get(["assistantEnabled"], (result) => {
            if (result.assistantEnabled !== undefined) setIsEnabled(result.assistantEnabled);
        });

        const handleStorageChange = (changes: { [key: string]: chrome.storage.StorageChange }) => {
            if (changes.assistantEnabled) {
                setIsEnabled(changes.assistantEnabled.newValue);
            }
        };

        chrome.storage.onChanged.addListener(handleStorageChange);
        return () => chrome.storage.onChanged.removeListener(handleStorageChange);
    }, []);

    const getActualFileId = (el: HTMLElement): string => {
        const uuidRegex = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;
        const links = el.querySelectorAll('a');
        for (const link of links) {
            const attrs = ['href', 'onclick', 'data-id', 'id'];
            for (const attr of attrs) {
                const val = link.getAttribute(attr);
                const match = val?.match(uuidRegex);
                if (match) return match[0];
            }
        }
        return fileName;
    };

    const handleAction = (summaryType: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!anchor?.element || isLoading) return;

        setIsLoading(true);
        const fileId = getActualFileId(anchor.element);
        const currentDocId = extractDocIdFromUrl();
        let token = getAuthToken();

        if (token?.startsWith("Bearer ")) token = token.replace("Bearer ", "");
        if (!token) {
            alert("Авторизация не найдена");
            setIsLoading(false);
            return;
        }

        chrome.runtime.sendMessage({
            type: "sendChatMessage",
            payload: {
                message: `Анализ файла: ${fileName}`,
                user_token: token,
                context_ui_id: currentDocId,
                file_path: fileId,
                human_choice: summaryType
            }
        }, (res) => {
            setIsLoading(false);
            setIsHovered(false);

            if (res?.success) {
                window.postMessage({
                    type: "REFRESH_CHAT_HISTORY",
                    messages: [
                        {type: 'human', content: `Запрошен анализ файла: ${fileName} (${summaryType})`},
                        {
                            type: 'ai',
                            content: res.data?.content || res.data?.response || "Файл проанализирован. Результат в чате."
                        }
                    ]
                }, "*");
            } else {
                alert("Ошибка: " + (res?.error || "Неизвестная ошибка"));
            }
        });
    }

    if (!isEnabled) return null;

    return (
        <div
            style={containerStyle}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <div style={{
                ...triggerBtnStyle,
                borderColor: isLoading ? '#007bff' : (isHovered ? '#aaa' : '#ddd'),
                transform: isHovered ? 'scale(1.1)' : 'scale(1)'
            }}>
                {isLoading ? (
                    <span style={spinAnimation}>⏳</span>
                ) : "✨"}
            </div>

            {isHovered && !isLoading && (
                <div style={dropdownStyle}>
                    <div style={menuHeaderStyle}>Анализ документа</div>
                    <div
                        style={menuItemStyle}
                        onClick={(e) => handleAction("extractive", e)}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f0f7ff')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#fff')}
                    >
                        <span style={{fontSize: '16px'}}>📊</span>
                        <span>Факты</span>
                    </div>
                    <div
                        style={menuItemStyle}
                        onClick={(e) => handleAction("abstractive", e)}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f0f7ff')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#fff')}
                    >
                        <span style={{fontSize: '16px'}}>📝</span>
                        <span>Пересказ</span>
                    </div>
                    <div
                        style={menuItemStyle}
                        onClick={(e) => handleAction("thesis", e)}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f0f7ff')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#fff')}
                    >
                        <span style={{fontSize: '16px'}}>📍</span>
                        <span>Тезисы</span>
                    </div>
                </div>
            )}
        </div>
    )
}

// --- Стили ---

const containerStyle: React.CSSProperties = {
    position: 'relative',
    display: 'inline-flex',
    marginLeft: '12px',
    verticalAlign: 'middle',
    zIndex: 1
}

const triggerBtnStyle: React.CSSProperties = {
    fontSize: '16px',
    width: '28px',
    height: '28px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '50%',
    background: '#fff',
    border: '1px solid #ddd',
    boxShadow: '0 2px 4px rgba(0,0,0,0.08)',
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    cursor: 'pointer',
    userSelect: 'none'
}

const dropdownStyle: React.CSSProperties = {
    position: 'absolute',
    top: '50%',
    left: '34px',
    transform: 'translateY(-50%)',
    backgroundColor: '#ffffff',
    border: '1px solid #d1d1d1',
    borderRadius: '10px',
    boxShadow: '0 8px 20px rgba(0,0,0,0.15)',
    zIndex: 10,
    width: '145px',
    display: 'flex',
    flexDirection: 'column',
    padding: '5px',
    overflow: 'hidden'
}

const menuHeaderStyle: React.CSSProperties = {
    padding: '6px 10px',
    fontSize: '10px',
    fontWeight: 700,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    borderBottom: '1px solid #f0f0f0',
    marginBottom: '4px'
}

const menuItemStyle: React.CSSProperties = {
    padding: '8px 10px',
    fontSize: '13px',
    color: '#333',
    borderRadius: '6px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    cursor: 'pointer',
    transition: 'background-color 0.2s ease',
    backgroundColor: '#fff'
}

const spinAnimation: React.CSSProperties = {
    display: 'inline-block',
    fontSize: '14px'
}

export default AttachmentActions;