// public\background.ts
export {} // Необходимо для инициализации файла как модуля в среде Plasmo

interface ChromeResponse {
    success: boolean;
    data?: any;
    error?: string;
}

const API_BASE_URL = 'http://localhost:8000';

/**
 * Слушатель сообщений от Content Script (вашего виджета)
 */
chrome.runtime.onMessage.addListener((request: { type: string; payload: any; }, sender: any, sendResponse: { (response: ChromeResponse): void; (response: ChromeResponse): void; (response: ChromeResponse): void; }) => {
    if (request.type === 'sendChatMessage') {
        handleJsonRequest('/chat', request.payload, sendResponse);
        return true; // Держим канал связи открытым для асинхронного ответа
    }

    if (request.type === 'sendShortSummary') {
        handleJsonRequest('/short-summary', request.payload, sendResponse);
        return true;
    }

    if (request.type === 'uploadFile') {
        handleFileUpload(request.payload, sendResponse);
        return true;
    }
});

/**
 * Универсальный обработчик для JSON запросов
 */
async function handleJsonRequest(
    endpoint: string,
    payload: any, // Измените на any, если Record вызывает сложности с вложенными объектами
    sendResponse: (response: ChromeResponse) => void
) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `Ошибка: ${response.status}`);

        sendResponse({success: true, data});
    } catch (err: any) {
        sendResponse({success: false, error: err.message});
    }
}

async function handleFileUpload(
    payload: any,
    sendResponse: (response: ChromeResponse) => void
) {
    try {
        const {fileData, fileName, user_token} = payload;
        const blobRes = await fetch(fileData);
        const blob = await blobRes.blob();

        const formData = new FormData();
        formData.append('file', blob, fileName);
        if (user_token) formData.append('user_token', user_token);

        const response = await fetch(`${API_BASE_URL}/upload-file`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Ошибка сохранения');

        sendResponse({success: true, data});
    } catch (err: any) {
        sendResponse({success: false, error: err.message});
    }
}