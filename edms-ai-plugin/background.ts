// background.ts
export {}

interface ChromeResponse {
    success: boolean;
    data?: any;
    error?: string;
}

const API_BASE_URL = 'http://localhost:8000';
const activeRequests = new Map<string, AbortController>();

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

        case 'summarizeDocument':
            handleDirectAction(message.payload, '/actions/summarize', sendResponse);
            return true;

        case 'uploadFile':
            handleFileUpload(message.payload, sendResponse);
            return true;

        case 'getChatHistory':
            handleGetHistory(message.payload, sendResponse);
            return true;

        case 'createNewChat':
            handleCreateNewChat(message.payload, sendResponse);
            return true;

        case 'autofillAppeal':
            handleAutofillAppeal(message.payload, sendResponse);
            return true;

        default:
            return false;
    }
});

async function handleChatMessage(payload: any, sendResponse: (res: ChromeResponse) => void) {
    const endpoint = `${API_BASE_URL}/chat`;
    await performFetch(endpoint, payload, sendResponse);
}

async function handleDirectAction(payload: any, path: string, sendResponse: (res: ChromeResponse) => void) {
    const endpoint = `${API_BASE_URL}${path}`;
    await performFetch(endpoint, payload, sendResponse);
}

async function handleCreateNewChat(payload: any, sendResponse: (res: ChromeResponse) => void) {
    const endpoint = `${API_BASE_URL}/chat/new`;
    const simplifiedPayload = {user_token: payload.user_token};
    await performFetch(endpoint, simplifiedPayload, sendResponse);
}

async function handleAutofillAppeal(payload: any, sendResponse: (res: ChromeResponse) => void) {
    const endpoint = `${API_BASE_URL}/appeal/autofill`;

    const requestPayload = {
        message: payload.message || "Заполни обращение",
        user_token: payload.user_token,
        context_ui_id: payload.context_ui_id,
        file_path: payload.file_path || null
    };

    await performFetch(endpoint, requestPayload, sendResponse);
}

async function performFetch(endpoint: string, payload: any, sendResponse: (res: ChromeResponse) => void) {
    const requestId = payload.requestId || 'default';
    const controller = new AbortController();
    activeRequests.set(requestId, controller);

    try {
        const response = await fetch(endpoint, {
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
            console.error("API Error:", err);
            sendResponse({success: false, error: err.message});
        }
    } finally {
        activeRequests.delete(requestId);
    }
}

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
        if (!response.ok) throw new Error(data.detail || 'Ошибка сохранения файла');
        sendResponse({success: true, data});
    } catch (err: any) {
        sendResponse({success: false, error: err.message});
    }
}

async function handleGetHistory(payload: any, sendResponse: (res: ChromeResponse) => void) {
    try {
        const {thread_id} = payload;
        const response = await fetch(`${API_BASE_URL}/chat/history/${thread_id}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Ошибка истории');
        sendResponse({success: true, data});
    } catch (err: any) {
        sendResponse({success: false, error: err.message});
    }
}