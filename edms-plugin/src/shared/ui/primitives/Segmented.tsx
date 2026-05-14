import { cn } from '@shared/lib/cn'

interface SegmentedOption<T> {
  value: T
  label: string
}

interface SegmentedProps<T extends string | number> {
  value: T
  onChange: (value: T) => void
  options: SegmentedOption<T>[]
  className?: string
}

export function Segmented<T extends string | number>({
  value,
  onChange,
  options,
  className,
}: SegmentedProps<T>) {
  return (
    <div
      className={cn(
        'flex gap-0.5 rounded-xl p-[3px] bg-black/4',
        className,
      )}
    >
      {options.map((opt) => {
        const active = String(value) === String(opt.value)
        return (
          <button
            key={String(opt.value)}
            type="button"
            onClick={() => onChange(opt.value)}
            className={cn(
              'flex-1 py-1.5 rounded-lg text-[10px] font-semibold transition-all duration-200 select-none',
              active
                ? 'bg-white text-indigo-700 shadow-[0_1px_4px_rgba(0,0,0,0.08)]'
                : 'bg-transparent text-slate-400',
            )}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}
