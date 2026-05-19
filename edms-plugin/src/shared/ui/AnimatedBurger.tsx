import { cn } from '@shared/lib/cn'

interface AnimatedBurgerProps {
  isOpen: boolean
  className?: string
}

export function AnimatedBurger({ isOpen, className }: AnimatedBurgerProps) {
  return (
    <div className={cn("burger-icon", isOpen && "open", className)}>
      <span></span>
      <span></span>
      <span></span>
    </div>
  )
}
