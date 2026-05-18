import { type ReactNode } from 'react'

interface FieldProps {
  label: string
  hint?: string
  children: ReactNode
}

export function Field({ label, hint, children }: FieldProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[9px] font-bold uppercase tracking-[0.12em] text-slate-500">
        {label}
      </span>
      {children}
      {hint && (
        <p className="text-[9px] leading-relaxed text-slate-400">{hint}</p>
      )}
    </div>
  )
}
