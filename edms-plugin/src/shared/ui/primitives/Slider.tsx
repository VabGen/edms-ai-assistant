import { type InputHTMLAttributes } from 'react'
import { cn } from '@shared/lib/cn'

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
  format?: (value: number) => string
  label?: string
  hint?: string
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step = 1,
  format,
  label,
  hint,
  className,
  ...props
}: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <span className="text-[9px] font-bold uppercase tracking-[0.12em] text-slate-500">
          {label}
        </span>
      )}
      <div className="flex items-center gap-2.5">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className={cn('flex-1 h-1 rounded-full appearance-none cursor-pointer', className)}
          style={{
            background: `linear-gradient(to right, #6366f1 ${pct}%, rgba(0,0,0,0.08) ${pct}%)`,
          }}
          {...props}
        />
        <span className="min-w-[40px] text-center text-[10px] font-mono px-2 py-0.5 rounded-md tabular-nums text-indigo-700 bg-indigo-500/6">
          {format ? format(value) : value}
        </span>
      </div>
      {hint && (
        <p className="text-[9px] leading-relaxed text-slate-400">{hint}</p>
      )}
    </div>
  )
}
