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
                className={`max-w-[85%] px-4 py-3 rounded-[20px] text-sm leading-relaxed shadow-sm ${
                    isUser
                        ? 'bg-indigo-600 text-white rounded-tr-none'
                        : 'bg-white text-slate-700 border border-slate-100 rounded-tl-none'
                }`}
            >
                <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                        code({node, inline, className, children, ...props}: any) {
                            const match = /language-(\w+)/.exec(className || '');

                            // В v8 inline — это логический флаг (true для `code`, false для ```code```)
                            if (!inline) {
                                return (
                                    <div className="relative my-3">
                                        <pre
                                            className="overflow-x-auto p-3 rounded-xl bg-slate-900 font-mono text-xs text-indigo-100 shadow-inner">
                                            <code className={className} {...props}>
                                                {children}
                                            </code>
                                        </pre>
                                    </div>
                                );
                            }

                            return (
                                <code
                                    className={`font-mono text-[13px] px-1.5 py-0.5 rounded border ${
                                        isUser
                                            ? 'bg-indigo-500 border-indigo-400 text-white'
                                            : 'bg-slate-100 border-slate-200 text-indigo-600'
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
                                className={`${isUser ? 'text-indigo-100' : 'text-indigo-600'} hover:underline font-bold transition-all`}
                                target="_blank"
                                rel="noopener noreferrer"
                            >
                                {children}
                            </a>
                        ),

                        blockquote: ({children}) => (
                            <blockquote
                                className={`border-l-4 pl-3 italic my-2 ${isUser ? 'border-indigo-300 text-indigo-100' : 'border-slate-200 text-slate-500'}`}>
                                {children}
                            </blockquote>
                        ),

                        table: ({children}) => (
                            <div className="overflow-x-auto my-3 rounded-lg border border-slate-100">
                                <table className="w-full text-xs text-left border-collapse">
                                    {children}
                                </table>
                            </div>
                        ),
                        th: ({children}) => <th
                            className="bg-slate-50 p-2 border border-slate-100 font-bold">{children}</th>,
                        td: ({children}) => <td className="p-2 border border-slate-100">{children}</td>,
                    }}
                >
                    {content}
                </ReactMarkdown>

                <div className={`text-[10px] mt-1.5 opacity-50 flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                    {dayjs().format('HH:mm')}
                </div>
            </div>
        </div>
    );
};