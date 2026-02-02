import React, {useState, useRef, useEffect} from 'react';
import {
    Paperclip,
    X,
    Mic,
    Send,
    MessageSquare,
    Square,
    StopCircle,
    FileText,
    Search,
    List,
    History
} from 'lucide-react';
import dayjs from "dayjs";
import 'dayjs/locale/ru';

import { extractDocIdFromUrl, getAuthToken } from "~utils/edms-helpers"

import {ChatMessage} from './ChatMessage';
import LiquidGlassFilter from './LiquidGlassFilter';

dayjs.locale('ru');

// const extractDocIdFromUrl = (): string => {
//     try {
//         const pathParts = window.location.pathname.split('/');
//         const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
//         const foundId = pathParts.find(part => uuidRegex.test(part));
//         if (foundId) return foundId;
//     } catch (e) {
//         console.error("Ошибка парсинга URL:", e);
//     }
//     return "main_assistant";
// };
//
// const getAuthToken = (): string | null => {
//     try {
//         const directToken = localStorage.getItem('token') || localStorage.getItem('access_token') || sessionStorage.getItem('token');
//         if (directToken) return directToken;
//         for (let i = 0; i < localStorage.length; i++) {
//             const key = localStorage.key(i);
//             if (key && (key.includes('auth') || key.includes('user') || key.includes('oidc'))) {
//                 const value = localStorage.getItem(key);
//                 if (value?.includes('eyJ')) return value.startsWith('{') ? JSON.parse(value).access_token : value;
//             }
//         }
//     } catch (e) {
//         console.error("Ошибка поиска токена:", e);
//     }
//     return null;
// };

const SoundWaveIndicator = () => (
    <div className="flex items-end justify-center space-x-1 h-3 mb-2">
        {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="w-1 bg-indigo-500/60 rounded-full animate-bounce"
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
    const [threads, setThreads] = useState<{ id: string, preview: string, date: string }[]>([]);
    const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [attachedFile, setAttachedFile] = useState<any>(null);
    const [isListening, setIsListening] = useState(false);
    const [recognition, setRecognition] = useState<any>(null);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const currentRequestIdRef = useRef<string | null>(null);
    const currentServerFilePath = useRef<string | null>(null);

    const frostedGlassClass = "bg-white/40 backdrop-blur-xl border border-white/40 shadow-[0_8px_32px_0_rgba(31,38,135,0.15)] transition-all duration-300";
    const liquidGlassClass = "relative isolation-auto before:content-[''] before:absolute before:inset-0 before:rounded-[32px] before:pointer-events-none before:box-shadow-[inset_0_0_20px_rgba(255,255,255,0.5)] after:content-[''] after:absolute after:inset-0 after:rounded-[32px] after:pointer-events-none after:bg-white/20 after:backdrop-blur-[12px] after:[filter:url(#liquid-glass-filter)] after:-z-10 shadow-2xl border border-white/30";

    useEffect(() => {
        setIsMounted(true);
        chrome.storage.local.get(["assistantEnabled"], (result) => {
            if (result.assistantEnabled !== undefined) setIsEnabled(result.assistantEnabled);
        });

        const handleExternalUpdate = (event: MessageEvent) => {
            if (event.data?.type === 'REFRESH_CHAT_HISTORY' && event.data?.messages) {
                const {messages: rawMsgs, thread_id} = event.data;

                const newMsgs = rawMsgs.map((m: any) => ({
                    role: m.type === 'human' ? 'user' : 'assistant',
                    content: m.content
                }));

                if (thread_id) {
                    setCurrentThreadId(thread_id);
                    setMessages(newMsgs);
                } else {
                    setMessages(prev => [...prev, ...newMsgs]);
                }

                setIsWidgetVisible(true);
                setIsChatPanelOpen(false);

                setTimeout(() => {
                    messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
                }, 100);
            }
        };

        window.addEventListener('message', handleExternalUpdate);

        const handleStorageChange = (changes: { [key: string]: chrome.storage.StorageChange }, areaName: string) => {
            if (areaName === 'local' && changes.assistantEnabled) {
                setIsEnabled(changes.assistantEnabled.newValue);
                if (!changes.assistantEnabled.newValue) setIsWidgetVisible(false);
            }
        };
        chrome.storage.onChanged.addListener(handleStorageChange);

        return () => {
            chrome.storage.onChanged.removeListener(handleStorageChange);
            window.removeEventListener('message', handleExternalUpdate);
        };
    }, []);

    useEffect(() => {
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
                    if (e.results[i].isFinal) finalTranscript += e.results[i][0].transcript;
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

    const handleCreateNewChat = async () => {
        const userToken = getAuthToken() || "no_token_found";

        if (messages.length > 0 && currentThreadId) {
            const firstUserMsg = messages.find(m => m.role === 'user')?.content || "Новый диалог";
            const preview = firstUserMsg.length > 40 ? firstUserMsg.substring(0, 40) + "..." : firstUserMsg;

            setThreads(prev => {
                if (prev.find(t => t.id === currentThreadId)) return prev;
                return [{id: currentThreadId, preview, date: dayjs().format('HH:mm')}, ...prev];
            });
        }

        setIsLoading(true);
        try {
            const res: any = await new Promise((resolve) => {
                chrome.runtime.sendMessage({
                    type: 'createNewChat',
                    payload: {user_token: userToken}
                }, (response) => resolve(response));
            });

            if (res?.success) {
                setCurrentThreadId(res.data.thread_id);
                setMessages([]);
                setIsChatPanelOpen(false);
            }
        } catch (err) {
            console.error("New chat error:", err);
        } finally {
            setIsLoading(false);
        }
    };

    const loadThreadHistory = async (threadId: string) => {
        setIsLoading(true);
        setIsChatPanelOpen(false);
        try {
            const res: any = await new Promise((resolve) => {
                chrome.runtime.sendMessage({
                    type: 'getChatHistory',
                    payload: {thread_id: threadId}
                }, (response) => resolve(response));
            });

            if (res?.success) {
                const history = res.data.messages.map((m: any) => ({
                    role: m.type === 'human' ? 'user' : 'assistant',
                    content: m.content
                }));
                setMessages(history);
                setCurrentThreadId(threadId);
            }
        } catch (err) {
            console.error("Load history error:", err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSendMessage = async (e?: React.FormEvent, humanChoice?: string) => {
        if (e) e.preventDefault();
        if ((!inputValue.trim() && !attachedFile && !humanChoice) || isLoading) return;

        if (isListening) {
            recognition.stop();
            setIsListening(false);
        }

        const userToken = getAuthToken() || "no_token_found";
        const currentDocId = extractDocIdFromUrl()
        const requestId = Math.random().toString(36).substring(7);
        currentRequestIdRef.current = requestId;

        const activeThreadId = currentThreadId || `${getAuthToken()?.substring(0, 8)}_${currentDocId}`;
        if (!currentThreadId) setCurrentThreadId(activeThreadId);

        if (humanChoice) {
            const labels: any = {abstractive: 'Пересказ', extractive: 'Факты', thesis: 'Тезисы'};
            setMessages(prev => [...prev, {
                role: 'user',
                content: `Выбран метод: ${labels[humanChoice] || humanChoice}`
            }]);
        } else {
            setMessages(prev => [...prev, {
                role: 'user',
                content: attachedFile ? `${inputValue} (Файл: ${attachedFile.name})` : inputValue
            }]);
        }

        setIsLoading(true);
        const text = inputValue;
        setInputValue('');

        try {
            if (attachedFile && !humanChoice) {
                const uploadRes: any = await new Promise((resolve, reject) => {
                    chrome.runtime.sendMessage({
                        type: 'uploadFile',
                        payload: {fileData: attachedFile.path, fileName: attachedFile.name, user_token: userToken}
                    }, (res) => {
                        if (res?.success) resolve(res.data);
                        else reject(res?.error || 'Ошибка загрузки');
                    });
                });
                currentServerFilePath.current = uploadRes.file_path;
            }

            const chatRes: any = await new Promise((resolve, reject) => {
                chrome.runtime.sendMessage({
                    type: 'sendChatMessage',
                    payload: {
                        message: text,
                        user_token: userToken,
                        requestId,
                        thread_id: activeThreadId,
                        context_ui_id: currentDocId,
                        file_path: currentServerFilePath.current,
                        human_choice: humanChoice
                    }
                }, (res) => {
                    if (res?.success) resolve(res.data);
                    else reject(res?.error || 'Unknown error');
                });
            });

            if (chatRes.status === 'requires_action') {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: chatRes.message || "Пожалуйста, выберите формат анализа:",
                    action_type: chatRes.action_type
                }]);
            } else {
                currentServerFilePath.current = null;
                let finalContent = chatRes.response || chatRes.content || chatRes.message;

                if (chatRes.messages && Array.isArray(chatRes.messages)) {
                    finalContent = chatRes.messages[chatRes.messages.length - 1].content;
                }

                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: finalContent || "Анализ завершен.",
                    action_type: chatRes.action_type
                }]);
            }
        } catch (err: any) {
            if (!String(err).includes('aborted')) {
                setMessages(prev => [...prev, {role: 'assistant', content: `⚠️ Ошибка: ${err}`}]);
            }
        } finally {
            setIsLoading(false);
            currentRequestIdRef.current = null;
            setAttachedFile(null);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    const renderActionButtons = (msg: any) => {
        if (msg.action_type === 'summarize_selection') {
            return (
                <div className="mt-3 flex flex-wrap gap-2 relative z-0">
                    {[
                        {id: 'abstractive', label: 'Пересказ', icon: <FileText size={14}/>},
                        {id: 'extractive', label: 'Факты', icon: <Search size={14}/>},
                        {id: 'thesis', label: 'Тезисы', icon: <List size={14}/>}
                    ].map(btn => (
                        <button
                            key={btn.id}
                            onClick={(e) => {
                                e.stopPropagation();
                                handleSendMessage(undefined, btn.id);
                            }}
                            className="flex items-center gap-2 px-3 py-1.5 bg-white/60 text-indigo-700 backdrop-blur-md border border-white/50 rounded-xl text-xs font-bold hover:bg-indigo-600 hover:text-white transition-all shadow-sm active:scale-95"
                        >
                            {btn.icon} {btn.label}
                        </button>
                    ))}
                </div>
            );
        }
        return null;
    };

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
        if (isListening) recognition.stop();
        else {
            setInputValue('');
            recognition.start();
            setIsListening(true);
        }
    };

    if (!isMounted || !isEnabled) return null;

    return (
        <div className="fixed bottom-5 right-5 z-[2147483647] font-sans flex flex-col items-end pointer-events-none">
            <LiquidGlassFilter/>

            {!isWidgetVisible && (
                <div className="relative pointer-events-auto group cursor-pointer"
                     onClick={() => setIsWidgetVisible(true)}>
                    <div
                        className="absolute inset-0 m-auto w-full h-full rounded-full bg-indigo-400/20 animate-liquid-ripple"/>
                    <button
                        className={`w-16 h-16 rounded-full flex items-center justify-center hover:scale-110 active:scale-95 ${frostedGlassClass}`}>
                        <MessageSquare size={28}
                                       className="text-indigo-600/80 group-hover:rotate-12 transition-transform"/>
                    </button>
                </div>
            )}

            {isWidgetVisible && (
                <div
                    className={`flex flex-col w-[480px] h-[720px] rounded-[32px] overflow-hidden pointer-events-auto animate-in fade-in zoom-in duration-300 origin-bottom-right ${liquidGlassClass}`}>

                    <header
                        className="flex items-center justify-between p-4 border-b border-white/20 shrink-0 relative z-[50] bg-white/10">
                        <div className="flex items-center gap-3">
                            <button
                                onClick={() => setIsChatPanelOpen(!isChatPanelOpen)}
                                className={`p-2.5 rounded-xl transition-all duration-300 group relative overflow-hidden flex items-center justify-center ${isChatPanelOpen ? 'bg-white/40 text-indigo-700 shadow-inner' : 'hover:bg-white/30 text-slate-600'}`}
                            >
                                <div className="flex flex-col gap-1.5 w-5 h-5 justify-center items-center">
                                    <span
                                        className={`h-0.5 bg-current rounded-full transition-all duration-300 ${isChatPanelOpen ? 'absolute rotate-45 w-5' : 'w-5'}`}></span>
                                    <span
                                        className={`h-0.5 bg-current rounded-full transition-all duration-200 ${isChatPanelOpen ? 'opacity-0' : 'w-5'}`}></span>
                                    <span
                                        className={`h-0.5 bg-current rounded-full transition-all duration-300 ${isChatPanelOpen ? 'absolute -rotate-45 w-5' : 'w-5'}`}></span>
                                </div>
                            </button>
                            <h3 className="font-bold text-slate-800/80 text-sm tracking-tight">EDMS Assistant</h3>
                        </div>
                        <button onClick={() => setIsWidgetVisible(false)}
                                className="p-2 rounded-xl text-slate-500 hover:text-red-500 hover:bg-white/30 transition-colors">
                            <X size={20}/>
                        </button>
                    </header>

                    <div className="flex-1 flex overflow-hidden relative z-10 bg-transparent">
                        <aside
                            className={`relative h-full flex flex-col bg-white/10 backdrop-blur-md border-r border-white/20 transition-all duration-300 ease-in-out shrink-0 z-20 ${isChatPanelOpen ? 'w-64 opacity-100' : 'w-0 opacity-0 overflow-hidden'}`}>
                            <div className="p-4 w-64 flex flex-col h-full">
                                <button onClick={handleCreateNewChat}
                                        className="w-full py-2.5 px-4 rounded-2xl bg-indigo-600 text-white font-semibold hover:bg-indigo-700 transition-all text-sm shadow-md flex items-center justify-center gap-2">
                                    + Новый диалог
                                </button>

                                <div className="mt-6 flex-1 overflow-y-auto custom-scrollbar pr-1">
                                    <div className="flex items-center gap-2 px-2 mb-3">
                                        <History size={12} className="text-slate-400"/>
                                        <p className="text-[10px] uppercase tracking-[0.1em] text-slate-500 font-bold">История</p>
                                    </div>

                                    <div className="flex flex-col gap-2">
                                        {threads.length === 0 ? (
                                            <div
                                                className="text-[11px] text-slate-500/70 italic px-2 py-6 text-center bg-white/20 rounded-2xl border border-dashed border-white/30">
                                                История пуста
                                            </div>
                                        ) : (
                                            threads.map((t) => (
                                                <button
                                                    key={t.id}
                                                    onClick={() => loadThreadHistory(t.id)}
                                                    className={`group w-full text-left p-3 rounded-2xl transition-all border ${currentThreadId === t.id ? 'bg-white/50 border-white/60 shadow-sm' : 'hover:bg-white/30 border-transparent'}`}
                                                >
                                                    <p className="text-[11px] text-slate-700 font-medium line-clamp-2 leading-relaxed">{t.preview}</p>
                                                    <span
                                                        className="text-[9px] text-slate-400 mt-1 block">{t.date}</span>
                                                </button>
                                            ))
                                        )}
                                    </div>
                                </div>
                            </div>
                        </aside>

                        <main className="flex-1 flex flex-col min-w-0 bg-transparent relative overflow-hidden">
                            <div
                                className="flex-1 p-4 overflow-y-auto flex flex-col gap-4 custom-scrollbar relative z-10">
                                {messages.length === 0 && !isLoading && (
                                    <div
                                        className="flex-1 flex flex-col items-center justify-center text-slate-400/60 opacity-50 select-none">
                                        <MessageSquare size={48} strokeWidth={1} className="mb-4"/>
                                        <p className="text-sm font-medium">Чем я могу помочь?</p>
                                    </div>
                                )}

                                {messages.map((msg, idx) => (
                                    <div key={idx}
                                         className="flex flex-col animate-in fade-in slide-in-from-bottom-2 duration-300">
                                        <ChatMessage content={msg.content} role={msg.role}/>
                                        {renderActionButtons(msg)}
                                    </div>
                                ))}

                                {isLoading && (
                                    <div
                                        className="flex items-center gap-2 px-4 py-3 bg-white/30 w-fit rounded-2xl border border-white/30 ml-2 animate-pulse">
                                        <div className="h-1.5 w-1.5 bg-indigo-400 rounded-full animate-bounce"
                                             style={{animationDelay: '0ms'}}></div>
                                        <div className="h-1.5 w-1.5 bg-indigo-400 rounded-full animate-bounce"
                                             style={{animationDelay: '150ms'}}></div>
                                        <div className="h-1.5 w-1.5 bg-indigo-400 rounded-full animate-bounce"
                                             style={{animationDelay: '300ms'}}></div>
                                    </div>
                                )}
                                <div ref={messagesEndRef}/>
                            </div>

                            <footer className="p-4 shrink-0 bg-white/5 border-t border-white/20 relative z-[100]">
                                {isListening && <SoundWaveIndicator/>}
                                {attachedFile && (
                                    <div
                                        className="flex items-center gap-2 mb-3 px-3 py-1.5 bg-white/50 border border-white/60 rounded-xl w-fit animate-in slide-in-from-bottom-1">
                                        <Paperclip size={14} className="text-indigo-500"/>
                                        <span
                                            className="text-[11px] font-medium text-slate-700 truncate max-w-[200px]">{attachedFile.name}</span>
                                        <button onClick={() => setAttachedFile(null)}
                                                className="ml-1 p-0.5 hover:bg-red-100 rounded-full text-slate-400 hover:text-red-500 transition-colors">
                                            <X size={14}/></button>
                                    </div>
                                )}

                                <form onSubmit={handleSendMessage}
                                      className={`flex items-center gap-1 rounded-2xl p-1.5 transition-all bg-white/20 backdrop-blur-md border border-white/30 focus-within:bg-white/40 focus-within:ring-4 focus-within:ring-indigo-500/10 shadow-sm`}>
                                    <button type="button" onClick={() => fileInputRef.current?.click()}
                                            className="p-2 text-slate-500 hover:text-indigo-600 hover:bg-white/30 rounded-xl transition-colors">
                                        <Paperclip size={20}/>
                                    </button>
                                    <button type="button" onClick={toggleListening}
                                            className={`p-2 rounded-xl transition-all ${isListening ? 'text-red-500 bg-red-100/50 animate-pulse' : 'text-slate-500 hover:text-indigo-600 hover:bg-white/30'}`}>
                                        {isListening ? <Square size={18} fill="currentColor"/> : <Mic size={20}/>}
                                    </button>
                                    <input
                                        type="text"
                                        value={inputValue}
                                        onChange={(e) => setInputValue(e.target.value)}
                                        placeholder={isListening ? "Слушаю..." : "Спросите AI..."}
                                        className="flex-1 bg-transparent border-none outline-none text-sm text-slate-800 placeholder-slate-500 px-2"
                                    />
                                    {isLoading ? (
                                        <button type="button" onClick={handleAbortRequest}
                                                className="p-2.5 bg-red-500/80 text-white rounded-xl shadow-lg hover:bg-red-600 transition-all active:scale-95">
                                            <StopCircle size={18}/></button>
                                    ) : (
                                        <button type="submit" disabled={!inputValue.trim() && !attachedFile}
                                                className="p-2.5 bg-indigo-600/80 text-white rounded-xl shadow-lg hover:bg-indigo-700 disabled:opacity-30 transition-all active:scale-95">
                                            <Send size={18}/></button>
                                    )}
                                </form>
                            </footer>
                        </main>
                    </div>
                </div>
            )}
            <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = () => setAttachedFile({path: reader.result, name: file.name});
                        reader.readAsDataURL(file);
                    }
                }}
            />
        </div>
    );
};