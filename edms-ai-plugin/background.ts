export {}

interface ChromeResponse {
    success: boolean;
    data?: any;
    error?: string;
}

const API_BASE_URL = 'http://localhost:8000';

const activeRequests = new Map<string, AbortController>();

/**
 * Основной слушатель сообщений
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    const requestId = message.payload?.requestId || 'default';

    switch (message.type) {
        case 'abortRequest':
            const controller = activeRequests.get(requestId);
            if (controller) {
                controller.abort();
                activeRequests.delete(requestId);
            }
            return false;

        case 'sendChatMessage':
            handleChatMessage(message.payload, sendResponse);
            return true;

        case 'uploadFile':
            handleFileUpload(message.payload, sendResponse);
            return true;

        default:
            return false;
    }
});

/**
 * Обработка чат-сообщений
 */
async function handleChatMessage(payload: any, sendResponse: (res: ChromeResponse) => void) {
    const requestId = payload.requestId || 'default';
    const controller = new AbortController();
    activeRequests.set(requestId, controller);

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
            signal: controller.signal
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `Ошибка сервера: ${response.status}`);

        sendResponse({success: true, data});
    } catch (err: any) {
        if (err.name === 'AbortError') {
            sendResponse({success: false, error: 'Request aborted'});
        } else {
            sendResponse({success: false, error: err.message});
        }
    } finally {
        activeRequests.delete(requestId);
    }
}

/**
 * Обработка загрузки файлов согласно логике main.py
 */
async function handleFileUpload(payload: any, sendResponse: (res: ChromeResponse) => void) {
    try {
        const {fileData, fileName, user_token} = payload;

        const blobRes = await fetch(fileData);
        const blob = await blobRes.blob();

        const formData = new FormData();
        formData.append('file', blob, fileName);
        formData.append('user_token', user_token);

        const response = await fetch(`${API_BASE_URL}/upload-file`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Ошибка сохранения файла на сервере');

        sendResponse({success: true, data});
    } catch (err: any) {
        console.error("Upload error:", err);
        sendResponse({success: false, error: err.message});
    }
}