// features\AssistantWidget.tsx
'use client';

import React, {useState, useRef, useEffect} from 'react';
import {Paperclip, X, Mic, Send, MessageSquare, Loader2} from 'lucide-react';
import dayjs from "dayjs";
import 'dayjs/locale/ru';

import {ChatMessage} from './ChatMessage';
import ConfirmDialog from './ConfirmDialog';
import LiquidGlassFilter from './LiquidGlassFilter';

dayjs.locale('ru');

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

interface Chat {
    chat_id: string;
    preview?: string;
}

// Типизация для ответов от Chrome Runtime
interface ChromeResponse {
    success: boolean;
    data?: any;
    error?: string;
    response?: string;
    message?: string;
}

const SoundWaveIndicator = () => (
    <div className="flex items-end justify-center space-x-0.5 w-5 h-4">
        {[0, 1, 2].map((i) => (
            <div
                key={i}
                className="w-0.5 bg-indigo-500 rounded-full animate-bounce origin-bottom"
                style={{animationDelay: `${i * 0.15}s`}}
            />
        ))}
    </div>
);

export const AssistantWidget = () => {
    const [isMounted, setIsMounted] = useState(false);
    const [isWidgetVisible, setIsWidgetVisible] = useState(false);
    const [isChatPanelOpen, setIsChatPanelOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [chats, setChats] = useState<Chat[]>([]);
    const [activeChatId, setActiveChatId] = useState<string | null>(null);

    const [attachedFile, setAttachedFile] = useState<{ path: string, name: string } | null>(null);
    const [isListening, setIsListening] = useState(false);
    const [recognition, setRecognition] = useState<any>(null);

    const [confirmDialog, setConfirmDialog] = useState<{ isOpen: boolean; chatId: string | null }>({
        isOpen: false,
        chatId: null
    });

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Инициализация и SpeechRecognition (как на вашем скриншоте)
    useEffect(() => {
        setIsMounted(true);
        const win = window as any;
        const SpeechRecognition = win.SpeechRecognition || win.webkitSpeechRecognition;

        if (SpeechRecognition) {
            const instance = new SpeechRecognition();
            instance.continuous = false;
            instance.lang = 'ru-RU';
            instance.onresult = (e: any) => {
                const transcript = e.results[0][0].transcript;
                setInputValue(prev => prev + (prev ? " " : "") + transcript);
            };
            instance.onend = () => setIsListening(false);
            setRecognition(instance);
        }

        if (chats.length === 0) {
            setChats([{chat_id: 'default_chat', preview: 'Новый диалог'}]);
            setActiveChatId('default_chat');
        }
    }, []);

    useEffect(() => {
        if (isMounted) {
            messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
        }
    }, [messages, isLoading, isMounted]);

    if (!isMounted) return null;

    const toggleListening = () => {
        if (isListening) {
            recognition?.stop();
        } else {
            setIsListening(true);
            try {
                recognition?.start();
            } catch {
                setIsListening(false);
            }
        }
    };

    const handleSendMessage = async (e: React.FormEvent) => {
        e.preventDefault();
        if ((!inputValue.trim() && !attachedFile) || !activeChatId) return;

        const userContent = attachedFile ? `${inputValue} (Файл: ${attachedFile.name})`.trim() : inputValue;
        setMessages(prev => [...prev, {role: 'user', content: userContent}]);

        const currentInput = inputValue;
        const fileToUpload = attachedFile;

        setInputValue('');
        setAttachedFile(null);
        setIsLoading(true);

        try {
            let finalFilePath: string | null = null; // Добавили тип string | null
            if (fileToUpload) {
                // Добавляем <any> к Promise, чтобы TypeScript разрешил обращение к .file_path
                const uploadRes = await new Promise<any>((resolve, reject) => {
                    chrome.runtime.sendMessage({
                        type: 'uploadFile',
                        payload: {fileData: fileToUpload.path, fileName: fileToUpload.name}
                    }, (res: ChromeResponse) => {
                        if (res?.success) resolve(res.data);
                        else reject(res?.error);
                    });
                });
                finalFilePath = uploadRes.file_path; // Теперь ошибки не будет
            }

            // Убираем :any здесь, используем типизированный Promise
            const chatRes = await new Promise<any>((resolve, reject) => {
                chrome.runtime.sendMessage({
                    type: 'sendChatMessage',
                    payload: {message: currentInput, file_path: finalFilePath, context_ui_id: "main_assistant"}
                }, (res: ChromeResponse) => (res?.success ? resolve(res.data) : reject(res?.error)));
            });

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: chatRes.response || chatRes.message || "Ответ получен."
            }]);
        } catch (err) {
            setMessages(prev => [...prev, {role: 'assistant', content: `⚠️ Ошибка: ${err}`}]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="fixed bottom-5 right-5 z-[2147483647] font-sans flex flex-col items-end pointer-events-none">
            <LiquidGlassFilter/>

            {!isWidgetVisible && (
                <button
                    onClick={() => setIsWidgetVisible(true)}
                    className="w-16 h-16 bg-indigo-600 rounded-full shadow-2xl flex items-center justify-center hover:scale-110 transition-transform cursor-pointer pointer-events-auto group relative"
                >
                    <MessageSquare size={28} className="text-white group-hover:rotate-12 transition-transform"/>
                    <div className="absolute inset-0 rounded-full bg-indigo-600 animate-ping opacity-20"/>
                </button>
            )}

            {isWidgetVisible && (
                <div
                    className="flex flex-col w-[400px] h-[600px] bg-white rounded-[32px] shadow-[0_20px_50px_rgba(0,0,0,0.15)] border border-slate-200 overflow-hidden pointer-events-auto animate-in fade-in zoom-in duration-200 origin-bottom-right">
                    <header
                        className="flex items-center justify-between p-4 border-b border-slate-100 bg-white/80 backdrop-blur-md shrink-0">
                        <div className="flex items-center gap-3">
                            <button onClick={() => setIsChatPanelOpen(!isChatPanelOpen)}
                                    className="p-2 hover:bg-slate-100 rounded-xl text-slate-500 transition-colors">
                                <MessageSquare size={18}/>
                            </button>
                            <div className="flex flex-col">
                                <h3 className="font-bold text-slate-800 text-sm leading-none">EDMS Assistant</h3>
                                <span className="text-[10px] text-indigo-500 font-medium mt-1">Online</span>
                            </div>
                        </div>
                        <button onClick={() => setIsWidgetVisible(false)}
                                className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all">
                            <X size={20}/>
                        </button>
                    </header>

                    <div className="flex-1 flex overflow-hidden">
                        {isChatPanelOpen && (
                            <aside
                                className="w-64 border-r border-slate-100 bg-slate-50 overflow-y-auto animate-in slide-in-from-left-2">
                                <div className="p-3 space-y-2">
                                    {chats.map((chat) => (
                                        <div key={chat.chat_id} onClick={() => setActiveChatId(chat.chat_id)}
                                             className={`p-3 rounded-xl cursor-pointer text-[11px] transition-all flex justify-between items-center ${activeChatId === chat.chat_id ? 'bg-white shadow-sm ring-1 ring-indigo-100 font-bold text-indigo-700' : 'hover:bg-slate-200 text-slate-600'}`}>
                                            <span className="truncate pr-2">{chat.preview || 'Новый диалог'}</span>
                                        </div>
                                    ))}
                                </div>
                            </aside>
                        )}

                        <main className="flex-1 flex flex-col bg-white min-w-0">
                            <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4">
                                {messages.length === 0 && (
                                    <div
                                        className="flex-1 flex flex-col items-center justify-center text-slate-400 gap-2 opacity-60">
                                        <MessageSquare size={48} strokeWidth={1}/>
                                        <p className="text-xs font-medium text-center">Задайте любой вопрос по EDMS</p>
                                    </div>
                                )}
                                {messages.map((msg, idx) => (
                                    <ChatMessage key={idx} content={msg.content} role={msg.role}/>
                                ))}
                                {isLoading && (
                                    <div
                                        className="flex gap-1.5 p-3 bg-slate-50 w-fit rounded-2xl rounded-tl-none border border-slate-100">
                                        <div
                                            className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]"/>
                                        <div
                                            className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]"/>
                                        <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce"/>
                                    </div>
                                )}
                                <div ref={messagesEndRef}/>
                            </div>

                            <footer className="p-4 bg-white border-t border-slate-50 shrink-0">
                                <form onSubmit={handleSendMessage}
                                      className="flex items-center gap-2 bg-slate-100 rounded-2xl p-1.5 focus-within:bg-white focus-within:ring-4 focus-within:ring-indigo-500/10 border border-transparent focus-within:border-indigo-200 transition-all">
                                    <button type="button" onClick={() => fileInputRef.current?.click()}
                                            className="p-2 text-slate-400 hover:text-indigo-600 transition-colors">
                                        <Paperclip size={20}/>
                                    </button>
                                    <input
                                        type="text"
                                        value={inputValue}
                                        onChange={(e) => setInputValue(e.target.value)}
                                        placeholder="Спросите AI..."
                                        className="flex-1 bg-transparent border-none outline-none text-sm px-1 h-10 text-slate-700"
                                    />
                                    <button type="button" onClick={toggleListening}
                                            className={`p-2 ${isListening ? 'text-red-500' : 'text-slate-400'}`}>
                                        {isListening ? <SoundWaveIndicator/> : <Mic size={20}/>}
                                    </button>
                                    <button type="submit" disabled={isLoading}
                                            className="p-2.5 bg-indigo-600 text-white rounded-xl shadow-md hover:bg-indigo-700 disabled:opacity-30 transition-all">
                                        {isLoading ? <Loader2 size={18} className="animate-spin"/> : <Send size={18}/>}
                                    </button>
                                </form>
                                <input type="file" ref={fileInputRef} className="hidden" onChange={(e) => {
                                    const file = e.target.files?.[0];
                                    if (file) {
                                        const reader = new FileReader();
                                        reader.onload = () => setAttachedFile({
                                            path: reader.result as string,
                                            name: file.name
                                        });
                                        reader.readAsDataURL(file);
                                    }
                                }}/>
                            </footer>
                        </main>
                    </div>
                </div>
            )}

            <ConfirmDialog
                isOpen={confirmDialog.isOpen}
                title="Очистить чат?"
                message="Все сообщения будут удалены."
                onConfirm={() => {
                    setMessages([]);
                    setConfirmDialog({isOpen: false, chatId: null});
                }}
                onCancel={() => setConfirmDialog({isOpen: false, chatId: null})}
            />
        </div>
    );
};

export default AssistantWidget;