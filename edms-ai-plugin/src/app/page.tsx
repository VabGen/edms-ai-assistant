'use client';
import React, {useState, useEffect, useRef} from 'react';
import {Mic, Send, Settings, MessageSquare, Plus, CheckCircle2, Loader2, Bot} from 'lucide-react';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    tools?: Record<string, any>;
}

interface ChatResponse {
    success: boolean;
    data?: {
        message: string;
        toolsExecutionResult?: any;
    };
    error?: string;
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg: Message = {role: 'user', content: input};
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        if (typeof chrome !== 'undefined' && chrome.runtime) {
            interface ChatResponse {
                success: boolean;
                data?: {
                    message: string;
                    toolsExecutionResult?: Record<string, any>;
                };
                error?: string;
            }

            chrome.runtime.sendMessage({
                type: 'sendChatMessage',
                payload: {message: input}
            }, (response: ChatResponse) => {
                setIsLoading(false);
                if (response?.success && response.data) {
                    const assistantData = response.data;
                    setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: assistantData.message,
                        tools: assistantData.toolsExecutionResult
                    }]);

                    if (assistantData.toolsExecutionResult?.navigation) {
                        const navResult = assistantData.toolsExecutionResult.navigation[0];
                        if (navResult?.targetTab) {
                            chrome.tabs.query({active: true, currentWindow: true}, (tabs: chrome.tabs.Tab[]) => {
                                const tabId = tabs?.[0]?.id;

                                if (tabId) {
                                    chrome.tabs.sendMessage(tabId, {
                                        type: 'SWITCH_TAB',
                                        tabName: navResult.targetTab
                                    });
                                }
                            });
                        }
                    }
                } else {
                    console.error('Ошибка плагина:', response?.error);
                }
            });
        } else {
            setTimeout(() => {
                setIsLoading(false);
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: 'Режим разработки: Chrome API недоступен. Настройте расширение для работы с сервером.'
                }]);
            }, 1000);
        }
    };

    return (
        <div className="flex flex-col h-[600px] w-[380px] bg-slate-50 overflow-hidden font-sans">
            {/* Custom Scrollbar Styles */}
            <style jsx global>{`
                ::-webkit-scrollbar {
                    width: 4px;
                }

                ::-webkit-scrollbar-track {
                    background: transparent;
                }

                ::-webkit-scrollbar-thumb {
                    background: #cbd5e1;
                    border-radius: 10px;
                }

                ::-webkit-scrollbar-thumb:hover {
                    background: #94a3b8;
                }
            `}</style>

            {/* Header */}
            <div
                className="bg-gradient-to-br from-indigo-600 via-indigo-700 to-purple-700 p-4 text-white flex justify-between items-center shrink-0 shadow-lg z-10">
                <div className="flex items-center gap-2">
                    <div className="bg-white/20 p-1.5 rounded-lg">
                        <Bot className="w-5 h-5 text-white"/>
                    </div>
                    <div>
                        <h1 className="font-bold text-sm leading-tight tracking-wide uppercase">EDMS Assistant</h1>
                        <div className="flex items-center gap-1">
                            <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"/>
                            <span className="text-[10px] text-indigo-100 font-medium">Online</span>
                        </div>
                    </div>
                </div>
                <div className="flex gap-2">
                    <button onClick={() => setMessages([])}
                            className="p-1.5 hover:bg-white/10 rounded-full transition-colors">
                        <Plus className="w-4 h-4 text-indigo-100"/>
                    </button>
                    <button className="p-1.5 hover:bg-white/10 rounded-full transition-colors">
                        <Settings className="w-4 h-4 text-indigo-100"/>
                    </button>
                </div>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-center space-y-3 opacity-60">
                        <div className="p-4 bg-white rounded-3xl shadow-sm border border-slate-100">
                            <MessageSquare className="w-8 h-8 text-indigo-400"/>
                        </div>
                        <p className="text-xs font-medium text-slate-500 max-w-[200px]">
                            Привет! Я помогу заполнить данные или найти документ.
                        </p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div key={i}
                         className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
                        <div className={`max-w-[88%] p-3.5 rounded-2xl text-[13px] shadow-sm leading-relaxed ${
                            msg.role === 'user'
                                ? 'bg-indigo-600 text-white rounded-br-none'
                                : 'bg-white text-slate-700 border border-slate-200/60 rounded-bl-none'
                        }`}>
                            {msg.content}

                            {msg.tools && Object.keys(msg.tools).length > 0 && (
                                <div className="mt-2.5 pt-2.5 border-t border-slate-100/50 flex flex-wrap gap-1.5">
                                    {Object.keys(msg.tools).map(tool => (
                                        <div key={tool}
                                             className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-50 text-[10px] font-semibold text-emerald-600 border border-emerald-100">
                                            <CheckCircle2 className="w-3 h-3"/>
                                            <span>{tool === 'navigation' ? 'Переход выполнен' : 'Данные получены'}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {isLoading && (
                    <div className="flex justify-start">
                        <div className="bg-white border border-slate-100 p-3 rounded-2xl shadow-sm">
                            <Loader2 className="w-4 h-4 text-indigo-500 animate-spin"/>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef}/>
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white border-t border-slate-100">
                <div
                    className="flex items-center gap-2 bg-slate-50 rounded-2xl p-1.5 border border-slate-200 focus-within:border-indigo-400 focus-within:ring-4 focus-within:ring-indigo-50 transition-all">
                    <button
                        onClick={() => setIsRecording(!isRecording)}
                        className={`p-2 rounded-xl transition-colors ${isRecording ? 'bg-red-500 text-white' : 'text-slate-400 hover:bg-slate-200'}`}
                    >
                        <Mic size={18}/>
                    </button>
                    <input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Задайте вопрос по EDMS..."
                        className="flex-1 bg-transparent border-none focus:outline-none text-[13px] py-2 px-1 text-slate-700 placeholder:text-slate-400"
                    />
                    <button
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                        className="p-2.5 rounded-xl bg-indigo-600 text-white shadow-md shadow-indigo-200 disabled:bg-slate-300 disabled:shadow-none transition-all active:scale-95"
                    >
                        <Send size={16}/>
                    </button>
                </div>
                <p className="text-[9px] text-center text-slate-400 mt-2.5 font-medium tracking-widest uppercase">
                    AI Power • Next.js Edition
                </p>
            </div>
        </div>
    );
}