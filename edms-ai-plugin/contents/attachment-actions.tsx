import type {PlasmoCSConfig, PlasmoGetInlineAnchorList, PlasmoGetStyle} from "plasmo"
import React, {useState, useEffect, useMemo} from "react"

export const config: PlasmoCSConfig = {
    matches: [
        "http://localhost:3000/*",
        "http://localhost:3001/*",
        "http://localhost:8080/*"
    ]
}

export const getStyle: PlasmoGetStyle = () => {
    const style = document.createElement("style")
    style.textContent = `
    :host { z-index: 1000 !important; position: relative !important; }
    #plasmo-shadow-container { z-index: 1000 !important; }
    @keyframes fadeInScale {
      from { opacity: 0; transform: translateY(-50%) scale(0.95); }
      to { opacity: 1; transform: translateY(-50%) scale(1); }
    }
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  `
    return style
}

export const getInlineAnchorList: PlasmoGetInlineAnchorList = async () => {
    return document.querySelectorAll('div.alert.alert-secondary:has(span.lead)')
}

const Icons = {
    Sparkles: () => (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
             strokeLinecap="round" strokeLinejoin="round">
            <path
                d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
        </svg>
    ),
    Chart: () => (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="20" x2="18" y2="10"/>
            <line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/>
        </svg>
    ),
    FileText: () => (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
        </svg>
    ),
    Target: () => (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <circle cx="12" cy="12" r="6"/>
            <circle cx="12" cy="12" r="2"/>
        </svg>
    )
}

const extractDocIdFromUrl = (): string => {
    try {
        const pathParts = window.location.pathname.split('/');
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        const foundId = pathParts.find(part => uuidRegex.test(part));
        if (foundId) return foundId;
    } catch (e) {
        console.error(e);
    }
    return "main_assistant";
};

const getAuthToken = (): string | null => {
    try {
        const directToken = localStorage.getItem('token') || localStorage.getItem('access_token') || sessionStorage.getItem('token');
        if (directToken) return directToken.replace("Bearer ", "");
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && (key.includes('auth') || key.includes('user') || key.includes('oidc'))) {
                const value = localStorage.getItem(key);
                if (value?.includes('eyJ')) {
                    const token = value.startsWith('{') ? JSON.parse(value).access_token : value;
                    return token.replace("Bearer ", "");
                }
            }
        }
    } catch (e) {
        console.error(e);
    }
    return null;
};

const AttachmentActions = ({anchor}: { anchor?: { element: HTMLElement } }) => {
    const [isEnabled, setIsEnabled] = useState(true);
    const [isHovered, setIsHovered] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    const fileName = useMemo(() =>
            anchor?.element?.querySelector('span.lead')?.textContent?.trim() || "Документ"
        , [anchor]);

    useEffect(() => {
        chrome.storage.local.get(["assistantEnabled"], (res) => {
            if (res.assistantEnabled !== undefined) setIsEnabled(res.assistantEnabled);
        });

        const handleStorageChange = (changes: { [key: string]: chrome.storage.StorageChange }) => {
            if (changes.assistantEnabled) setIsEnabled(changes.assistantEnabled.newValue);
        };

        chrome.storage.onChanged.addListener(handleStorageChange);
        return () => chrome.storage.onChanged.removeListener(handleStorageChange);
    }, []);

    const getActualFileId = (el: HTMLElement): string => {
        const uuidRegex = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;
        const links = el.querySelectorAll('a');
        for (const link of links) {
            const attrs = ['href', 'onclick', 'data-id', 'id', 'data-file-id', 'data-attach-id'];
            for (const attr of attrs) {
                const val = link.getAttribute(attr);
                const match = val?.match(uuidRegex);
                if (match) return match[0];
            }
        }

        const parentId = el.closest('[id]')?.id || el.closest('[data-id]')?.getAttribute('data-id');
        const parentMatch = parentId?.match(uuidRegex);
        if (parentMatch) return parentMatch[0];

        const contentMatch = el.innerHTML.match(uuidRegex);
        if (contentMatch) return contentMatch[0];

        return "";
    };

    const handleAction = (summaryType: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!anchor?.element || isLoading) return;

        const token = getAuthToken();
        if (!token) return alert("Авторизация не найдена");

        const fileId = getActualFileId(anchor.element);
        const currentDocId = extractDocIdFromUrl();

        setIsLoading(true);

        chrome.runtime.sendMessage({
            type: "summarizeDocument",
            payload: {
                message: fileName,
                user_token: token,
                context_ui_id: currentDocId,
                file_path: fileId,
                human_choice: summaryType
            }
        }, (res) => {
            setIsLoading(false);
            setIsHovered(false);
            if (res?.success) {
                const finalContent = res.data?.response || "Анализ завершен.";
                window.postMessage({
                    type: "REFRESH_CHAT_HISTORY",
                    messages: [
                        {type: 'human', content: `Запрошен анализ файла: ${fileName}`},
                        {type: 'ai', content: finalContent}
                    ]
                }, "*");
            } else {
                alert("Ошибка: " + (res?.error || "Неизвестная ошибка"));
            }
        });
    };

    if (!isEnabled) return null;

    return (
        <div
            style={s.container}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <button type="button" style={{
                ...s.trigger,
                color: isLoading ? '#6366f1' : (isHovered ? '#4f46e5' : '#94a3b8'),
                borderColor: isHovered || isLoading ? '#e0e7ff' : '#f1f5f9',
                backgroundColor: isHovered ? '#f8faff' : '#fff',
            }}>
                {isLoading ? (
                    <span style={{animation: 'spin 1s linear infinite', display: 'flex'}}>⏳</span>
                ) : (
                    <Icons.Sparkles/>
                )}
            </button>

            {isHovered && !isLoading && (
                <div style={s.dropdown}>
                    <div style={s.header}>Анализ документа</div>
                    <div style={s.item} onClick={(e) => handleAction("extractive", e)}
                         onMouseEnter={e => e.currentTarget.style.backgroundColor = '#f1f5f9'}
                         onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                        <span style={s.iconWrapper}><Icons.Chart/></span>
                        <span style={s.label}>Факты</span>
                    </div>
                    <div style={s.item} onClick={(e) => handleAction("abstractive", e)}
                         onMouseEnter={e => e.currentTarget.style.backgroundColor = '#f1f5f9'}
                         onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                        <span style={s.iconWrapper}><Icons.FileText/></span>
                        <span style={s.label}>Пересказ</span>
                    </div>
                    <div style={s.item} onClick={(e) => handleAction("thesis", e)}
                         onMouseEnter={e => e.currentTarget.style.backgroundColor = '#f1f5f9'}
                         onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                        <span style={s.iconWrapper}><Icons.Target/></span>
                        <span style={s.label}>Тезисы</span>
                    </div>
                </div>
            )}
        </div>
    )
}

const s: Record<string, React.CSSProperties> = {
    container: {
        position: 'relative',
        display: 'inline-flex',
        marginLeft: '12px',
        verticalAlign: 'middle',
        zIndex: 1000
    },
    trigger: {
        all: 'unset',
        width: '28px',
        height: '28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '8px',
        border: '1px solid',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        background: '#fff',
        boxShadow: '0 2px 4px rgba(0,0,0,0.02)'
    },
    dropdown: {
        position: 'absolute', top: '50%', left: '34px', transform: 'translateY(-50%)', backgroundColor: '#fff',
        border: '1px solid #e2e8f0', borderRadius: '12px', boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
        width: '150px', padding: '5px', animation: 'fadeInScale 0.15s ease-out', zIndex: 10001
    },
    header: {
        padding: '6px 10px', fontSize: '10px', fontWeight: 700, color: '#94a3b8',
        textTransform: 'uppercase', borderBottom: '1px solid #f1f5f9', marginBottom: '4px'
    },
    item: {
        display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 10px',
        cursor: 'pointer', borderRadius: '8px', transition: 'all 0.15s ease'
    },
    iconWrapper: {color: '#6366f1', display: 'flex', alignItems: 'center'},
    label: {fontSize: '13px', fontWeight: 500, color: '#334155'}
};

export default AttachmentActions;