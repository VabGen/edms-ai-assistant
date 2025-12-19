import React, {useState, useRef, useEffect} from 'react';
import {Paperclip, X, Mic, Send, MessageSquare, Square, StopCircle} from 'lucide-react';
import dayjs from "dayjs";
import 'dayjs/locale/ru';

import {ChatMessage} from './ChatMessage';
import ConfirmDialog from './ConfirmDialog';
import LiquidGlassFilter from './LiquidGlassFilter';

dayjs.locale('ru');

const extractDocIdFromUrl = (): string => {
    try {
        const pathParts = window.location.pathname.split('/');
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

        // Ищем UUID в любой части URL, а не только в последней
        const foundId = pathParts.find(part => uuidRegex.test(part));
        if (foundId) return foundId;
    } catch (e) {
        console.error("Ошибка парсинга:", e);
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
        console.error("Ошибка поиска токена:", e);
    }
    return null;
};

const SoundWaveIndicator = () => (
    <div className="flex items-end justify-center space-x-1 h-3 mb-1">
        {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="w-1 bg-indigo-500 rounded-full animate-bounce"
                 style={{animationDuration: '0.6s', animationDelay: `${i * 0.1}s`}}/>
        ))}
    </div>
);

export const AssistantWidget = () => {
    const [isMounted, setIsMounted] = useState(false);
    const [isEnabled, setIsEnabled] = useState(true);
    const [isWidgetVisible, setIsWidgetVisible] = useState(false);
    const [isChatPanelOpen, setIsChatPanelOpen] = useState(false);
    const [messages, setMessages] = useState<any[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [attachedFile, setAttachedFile] = useState<any>(null);
    const [isListening, setIsListening] = useState(false);
    const [recognition, setRecognition] = useState<any>(null);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const currentRequestIdRef = useRef<string | null>(null);

    const frostedGlassClass = "bg-white/30 backdrop-blur-md border border-white/40 shadow-lg hover:bg-white/40 transition-all duration-300";
    const liquidGlassClass = "relative isolation-auto before:content-[''] before:absolute before:inset-0 before:rounded-[32px] before:pointer-events-none before:box-shadow-[inset_0_0_15px_rgba(255,255,255,0.5)] after:content-[''] after:absolute after:inset-0 after:rounded-[32px] after:pointer-events-none after:bg-white/10 after:backdrop-blur-[8px] after:[filter:url(#liquid-glass-filter)] after:-z-10";

    useEffect(() => {
        setIsMounted(true);
        chrome.storage.local.get(["assistantEnabled"], (result) => {
            if (result.assistantEnabled !== undefined) setIsEnabled(result.assistantEnabled);
        });

        const win = window as any;
        const SpeechRecognition = win.SpeechRecognition || win.webkitSpeechRecognition;
        if (SpeechRecognition) {
            const instance = new SpeechRecognition();
            instance.lang = 'ru-RU';
            instance.continuous = true;
            instance.interimResults = true;

            instance.onresult = (e: any) => {
                let finalTranscript = '';
                for (let i = e.resultIndex; i < e.results.length; ++i) {
                    if (e.results[i].isFinal) {
                        finalTranscript += e.results[i][0].transcript;
                    }
                }
                if (finalTranscript) setInputValue(prev => prev + (prev ? ' ' : '') + finalTranscript);
            };

            instance.onend = () => setIsListening(false);
            instance.onerror = () => setIsListening(false);
            setRecognition(instance);
        }
    }, []);

    useEffect(() => {
        if (isMounted) messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
    }, [messages, isLoading, isMounted]);

    const handleAbortRequest = () => {
        if (currentRequestIdRef.current) {
            chrome.runtime.sendMessage({type: 'abortRequest', payload: {requestId: currentRequestIdRef.current}});
            setIsLoading(false);
            currentRequestIdRef.current = null;
            setMessages(prev => [...prev, {role: 'assistant', content: '_Запрос отменен пользователем._'}]);
        }
    };

    const toggleListening = () => {
        if (!recognition) return;
        if (isListening) {
            recognition.stop();
        } else {
            setInputValue('');
            recognition.start();
            setIsListening(true);
        }
    };

    const handleSendMessage = async (e: React.FormEvent) => {
        e.preventDefault();
        if ((!inputValue.trim() && !attachedFile) || isLoading) return;

        if (isListening) {
            recognition.stop();
            setIsListening(false);
        }

        const userToken = getAuthToken() || "no_token_found";
        const currentDocId = extractDocIdFromUrl(); // Получаем актуальный ID
        const requestId = Math.random().toString(36).substring(7);
        currentRequestIdRef.current = requestId;

        setMessages(prev => [...prev, {
            role: 'user',
            content: attachedFile ? `${inputValue} (${attachedFile.name})` : inputValue
        }]);

        setIsLoading(true);
        const text = inputValue;
        setInputValue('');

        try {
            const chatRes: any = await new Promise((resolve, reject) => {
                chrome.runtime.sendMessage({
                    type: 'sendChatMessage',
                    payload: {
                        message: text,
                        user_token: userToken,
                        requestId,
                        context_ui_id: currentDocId
                    }
                }, (res) => {
                    if (res?.success) resolve(res.data);
                    else reject(res?.error === 'aborted' ? 'aborted' : res?.error);
                });
            });
            setMessages(prev => [...prev, {role: 'assistant', content: chatRes.response || chatRes.message}]);
        } catch (err) {
            if (err !== 'aborted' && err !== 'Request aborted') {
                setMessages(prev => [...prev, {role: 'assistant', content: `⚠️ Ошибка: ${err}`}]);
            }
        } finally {
            setIsLoading(false);
            currentRequestIdRef.current = null;
        }
    };

    if (!isMounted || !isEnabled) return null;

    return (
        <div className="fixed bottom-5 right-5 z-[2147483647] font-sans flex flex-col items-end pointer-events-none">
            <LiquidGlassFilter/>

            {!isWidgetVisible && (
                <div className="relative pointer-events-auto group cursor-pointer"
                     onClick={() => setIsWidgetVisible(true)}>
                    <div className="absolute inset-0 m-auto w-full h-full rounded-full bg-cyan-400/30 animate-liquid-ripple"/>
                    <button className={`w-16 h-16 rounded-full flex items-center justify-center ${frostedGlassClass}`}>
                        <MessageSquare size={28} className="text-indigo-600 group-hover:rotate-12 transition-transform"/>
                    </button>
                </div>
            )}

            {isWidgetVisible && (
                <div className={`flex flex-col w-[500px] h-[650px] rounded-[32px] shadow-2xl border border-white/20 overflow-hidden pointer-events-auto animate-in fade-in zoom-in duration-300 origin-bottom-right ${liquidGlassClass}`}>

                    <header className="flex items-center justify-between p-4 border-b border-white/10 shrink-0 relative z-10">
                        <div className="flex items-center gap-3">
                            <button onClick={() => setIsChatPanelOpen(!isChatPanelOpen)}
                                    className={`p-2 rounded-xl ${frostedGlassClass}`}>
                                <MessageSquare size={18} className="text-indigo-700"/>
                            </button>
                            <h3 className="font-bold text-slate-800 text-sm leading-none">EDMS Assistant</h3>
                        </div>
                        <button onClick={() => setIsWidgetVisible(false)}
                                className={`p-2 rounded-xl text-slate-500 hover:text-red-500 ${frostedGlassClass}`}>
                            <X size={20}/>
                        </button>
                    </header>

                    <div className="flex-1 flex overflow-hidden relative z-10">
                        {isChatPanelOpen && (
                            <aside className="w-48 border-r border-white/10 bg-white/10 backdrop-blur-sm p-3">
                                <div className={`p-3 rounded-xl text-[11px] font-bold text-indigo-700 cursor-pointer ${frostedGlassClass}`}>
                                    Новый диалог
                                </div>
                            </aside>
                        )}

                        <main className="flex-1 flex flex-col min-w-0">
                            <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4 custom-scrollbar">
                                {messages.map((msg, idx) => <ChatMessage key={idx} content={msg.content} role={msg.role}/>)}
                                {isLoading && <div className="text-indigo-500 text-[10px] animate-pulse px-2">Печатаю...</div>}
                                <div ref={messagesEndRef}/>
                            </div>

                            <footer className="p-4 shrink-0">
                                {isListening && <SoundWaveIndicator/>}
                                <form onSubmit={handleSendMessage}
                                      className={`flex items-center gap-2 rounded-2xl p-1.5 border transition-all ${frostedGlassClass} focus-within:ring-4 focus-within:ring-indigo-500/20`}>

                                    <button type="button" onClick={() => fileInputRef.current?.click()}
                                            className="p-2 text-slate-500 hover:text-indigo-600">
                                        <Paperclip size={20}/>
                                    </button>

                                    <button
                                        type="button"
                                        onClick={toggleListening}
                                        className={`p-2 rounded-lg transition-all ${isListening ? 'text-red-500 bg-red-100 animate-pulse' : 'text-slate-500 hover:text-indigo-600'}`}
                                    >
                                        {isListening ? <Square size={18} fill="currentColor"/> : <Mic size={20}/>}
                                    </button>

                                    <input
                                        type="text"
                                        value={inputValue}
                                        onChange={(e) => setInputValue(e.target.value)}
                                        placeholder={isListening ? "Слушаю..." : "Спросите AI..."}
                                        className="flex-1 bg-transparent border-none outline-none text-sm text-slate-700 placeholder-slate-400"
                                    />

                                    {isLoading ? (
                                        <button
                                            type="button"
                                            onClick={handleAbortRequest}
                                            className="p-2.5 bg-red-500 text-white rounded-xl shadow-md hover:bg-red-600"
                                            title="Остановить генерацию"
                                        >
                                            <StopCircle size={18}/>
                                        </button>
                                    ) : (
                                        <button
                                            type="submit"
                                            disabled={!inputValue.trim() && !attachedFile}
                                            className="p-2.5 bg-indigo-600 text-white rounded-xl shadow-md hover:bg-indigo-700 disabled:opacity-30 transition-all"
                                        >
                                            <Send size={18}/>
                                        </button>
                                    )}
                                </form>
                            </footer>
                        </main>
                    </div>
                </div>
            )}
            <input type="file" ref={fileInputRef} className="hidden" onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = () => setAttachedFile({path: reader.result, name: file.name});
                    reader.readAsDataURL(file);
                }
            }}/>
        </div>
    );
};