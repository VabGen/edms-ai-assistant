import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import dayjs from "dayjs";

interface ChatMessageProps {
    content: string;
    role: 'user' | 'assistant';
}

export const ChatMessage: React.FC<ChatMessageProps> = ({content, role}) => {
    const isUser = role === 'user';

    return (
        <div
            className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
            <div
                className={`max-w-[88%] px-4 py-3 rounded-[24px] text-sm leading-relaxed backdrop-blur-md shadow-sm border transition-all ${
                    isUser
                        ? 'bg-indigo-600/70 text-white border-white/20 rounded-tr-none'
                        : 'bg-white/40 text-slate-800 border-white/40 rounded-tl-none'
                }`}
            >
                <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                        code({node, inline, className, children, ...props}: any) {
                            const match = /language-(\w+)/.exec(className || '');

                            if (!inline) {
                                return (
                                    <div className="relative my-3">
                                        <pre
                                            className="overflow-x-auto p-3 rounded-xl bg-slate-900/80 backdrop-blur-sm font-mono text-xs text-indigo-100 shadow-inner border border-white/10">
                                            <code className={className} {...props}>
                                                {children}
                                            </code>
                                        </pre>
                                    </div>
                                );
                            }

                            return (
                                <code
                                    className={`font-mono text-[13px] px-1.5 py-0.5 rounded ${
                                        isUser
                                            ? 'bg-white/20 text-white'
                                            : 'bg-indigo-100/50 text-indigo-700'
                                    }`}
                                    {...props}
                                >
                                    {children}
                                </code>
                            );
                        },

                        p: ({children}) => (
                            <p className="mb-2 last:mb-0 whitespace-pre-wrap break-words font-medium tracking-tight">
                                {children}
                            </p>
                        ),

                        ul: ({children}) => <ul className="list-disc pl-5 mb-2 space-y-1">{children}</ul>,
                        ol: ({children}) => <ol className="list-decimal pl-5 mb-2 space-y-1">{children}</ol>,
                        li: ({children}) => <li className="marker:text-current">{children}</li>,

                        a: ({children, href}) => (
                            <a
                                href={href}
                                className={`${isUser ? 'text-white' : 'text-indigo-600'} hover:underline font-bold transition-all underline-offset-2`}
                                target="_blank"
                                rel="noopener noreferrer"
                            >
                                {children}
                            </a>
                        ),

                        blockquote: ({children}) => (
                            <blockquote
                                className={`border-l-4 pl-3 italic my-2 ${
                                    isUser
                                        ? 'border-white/40 text-indigo-50'
                                        : 'border-indigo-200 text-slate-600'
                                }`}>
                                {children}
                            </blockquote>
                        ),

                        table: ({children}) => (
                            <div
                                className="overflow-x-auto my-3 rounded-xl border border-white/30 bg-white/20 backdrop-blur-sm">
                                <table className="w-full text-xs text-left border-collapse">
                                    {children}
                                </table>
                            </div>
                        ),
                        th: ({children}) => (
                            <th className={`p-2 border-b border-white/30 font-bold ${isUser ? 'bg-white/10' : 'bg-white/40'}`}>
                                {children}
                            </th>
                        ),
                        td: ({children}) => <td className="p-2 border-b border-white/10">{children}</td>,
                    }}
                >
                    {content}
                </ReactMarkdown>

                <div
                    className={`text-[10px] mt-2 opacity-60 flex items-center gap-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
                    <span>{dayjs().format('HH:mm')}</span>
                    {isUser && <span className="text-[8px]">●</span>}
                </div>
            </div>
        </div>
    );

};