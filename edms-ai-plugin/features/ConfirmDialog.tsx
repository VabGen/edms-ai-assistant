import React from 'react';

interface ConfirmDialogProps {
    isOpen: boolean;
    title: string;
    message: string;
    onConfirm: () => void | Promise<void>;
    onCancel: () => void;
}

export default function ConfirmDialog({
    isOpen,
    title,
    message,
    onConfirm,
    onCancel
}: ConfirmDialogProps) {

    if (!isOpen) return null;

    return (
        /* Контейнер привязан к границам Shadow Host (окна чата) */
        <div className="absolute inset-0 z-[100] flex items-center justify-center p-6 animate-in fade-in duration-200">

            {/* Overlay с легким размытием, ограниченный окном виджета */}
            <div
                className="absolute inset-0 bg-slate-900/60 backdrop-blur-[2px]"
                onClick={onCancel}
            />

            {/* Контент модального окна */}
            <div
                className="relative bg-white rounded-[28px] shadow-2xl border border-slate-200 w-full max-w-[320px] p-6 text-center animate-in zoom-in-95 duration-200"
            >
                {/* Иконка предупреждения */}
                <div
                    className="w-14 h-14 bg-red-50 rounded-full flex items-center justify-center mx-auto mb-4 border border-red-100"
                >
                    <svg className="w-7 h-7 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        />
                    </svg>
                </div>

                <h3 className="text-lg font-bold mb-2 text-slate-800 tracking-tight">
                    {title}
                </h3>
                <p className="text-sm text-slate-500 mb-6 leading-relaxed">
                    {message}
                </p>

                <div className="flex gap-3">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="flex-1 py-3 bg-slate-100 hover:bg-slate-200 rounded-2xl transition-all text-slate-600 font-semibold text-sm border border-transparent active:scale-95"
                    >
                        Отмена
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className="flex-1 py-3 bg-red-500 hover:bg-red-600 rounded-2xl transition-all text-white font-semibold text-sm shadow-lg shadow-red-200 active:scale-95"
                    >
                        Удалить
                    </button>
                </div>
            </div>
        </div>
    );
}