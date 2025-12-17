class EDMSContentScript {
    constructor() {
        this.init();
    }

    init() {
        console.log('[EDMS AI] Content Script loaded on:', window.location.href);
        this.injectChatButton();
        this.observeDOM();

        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.type === 'SWITCH_TAB') {
                this.handleTabSwitch(request.tabName, sendResponse);
                return true;
            }
        });
    }

    handleTabSwitch(tabName, sendResponse) {
        const tabs = document.querySelectorAll('.nav-link, [role="tab"], .ant-tabs-tab');
        /** @type {HTMLElement} */
        const target = Array.from(tabs).find(tab =>
            tab.textContent.trim().toLowerCase().includes(tabName.toLowerCase())
        );

        if (target && typeof target.click === 'function') {
            target.click();
            sendResponse({success: true});
        } else {
            sendResponse({success: false});
        }
    }

    injectChatButton() {
        if (document.getElementById('edms-ai-toggle')) return;

        const btn = document.createElement('button');
        btn.id = 'edms-ai-toggle';
        btn.innerHTML = '🤖';
        btn.style.cssText = `
            position: fixed; bottom: 20px; right: 20px;
            width: 60px; height: 60px; border-radius: 30px;
            background: #4f46e5; color: white; border: none;
            cursor: pointer; z-index: 10000; font-size: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        `;

        btn.onmouseover = () => btn.style.transform = 'scale(1.1) rotate(5deg)';
        btn.onmouseout = () => btn.style.transform = 'scale(1) rotate(0deg)';
        btn.onclick = () => this.toggleIframe();
        document.body.appendChild(btn);
    }

    toggleIframe() {
        let iframe = document.getElementById('edms-ai-iframe');
        if (iframe) {
            iframe.style.display = iframe.style.display === 'none' ? 'block' : 'none';
        } else {
            iframe = document.createElement('iframe');
            iframe.id = 'edms-ai-iframe';
            iframe.src = chrome.runtime.getURL('popup.html');
            iframe.style.cssText = `
                position: fixed; bottom: 90px; right: 20px;
                width: 380px; height: 600px; border: none;
                border-radius: 16px; z-index: 10000; display: block;
                box-shadow: 0 12px 40px rgba(0,0,0,0.15);
                color-scheme: light; /* Предотвращает инверсию цветов в темных темах */
            `;
            document.body.appendChild(iframe);
        }
    }

    observeDOM() {
        const observer = new MutationObserver(() => {
            const titleInput = document.querySelector('input[maxlength="255"]');
            if (titleInput && !titleInput.parentElement.querySelector('.magic-wand')) {
                this.addMagicWand(titleInput);
            }
        });
        observer.observe(document.body, {childList: true, subtree: true});
    }

    addMagicWand(input) {
        const container = input.parentElement;
        if (!container) return;

        container.style.position = 'relative';

        const wand = document.createElement('span');
        wand.className = 'magic-wand';
        wand.innerHTML = '✨';
        wand.title = 'Заполнить с помощью ИИ';
        wand.style.cssText = `
            cursor: pointer; position: absolute;
            right: 10px; top: 50%; transform: translateY(-50%);
            z-index: 10; font-size: 16px;
            transition: transform 0.2s ease;
        `;

        wand.onmouseover = () => wand.style.transform = 'translateY(-50%) scale(1.2)';
        wand.onmouseout = () => wand.style.transform = 'translateY(-50%) scale(1)';

        container.appendChild(wand);

        wand.onclick = () => {
            const docId = window.location.pathname.split('/').pop();
            wand.innerHTML = '⏳';

            chrome.runtime.sendMessage({
                type: 'sendShortSummary',
                payload: {
                    documentId: docId,
                    token: localStorage.getItem('token') || ''
                }
            }, (response) => {
                wand.innerHTML = '✨';
                if (response && response.success) {
                    input.value = response.data.summary;
                    input.dispatchEvent(new Event('input', {bubbles: true}));
                    input.dispatchEvent(new Event('change', {bubbles: true}));
                }
            });
        };
    }
}

new EDMSContentScript();