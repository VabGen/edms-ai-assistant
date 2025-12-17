// Background script для обхода CORS
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const serverUrl = 'http://localhost:8097, http://localhost:3000, http://localhost:8080, http://localhost:3001';

    if (request.type === 'sendChatMessage') {
        fetch(`${serverUrl}/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(request.payload)
        })
            .then(res => res.json())
            .then(data => sendResponse({success: true, data}))
            .catch(err => sendResponse({success: false, error: err.message}));
        return true;
    }

    if (request.type === 'sendShortSummary') {
        fetch(`${serverUrl}/short-summary`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(request.payload)
        })
            .then(res => res.json())
            .then(data => sendResponse({success: true, data}))
            .catch(err => sendResponse({success: false, error: err.message}));
        return true;
    }
});