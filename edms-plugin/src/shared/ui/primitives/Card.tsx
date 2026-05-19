import React from 'react'
import { cn } from '@shared/lib/cn'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  isSelected?: boolean
  isClickable?: boolean
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, isSelected, isClickable, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "bg-white  border border-zinc-100/80 rounded-[20px] transition-all duration-300 ease-out shadow-sm overflow-hidden",
        isClickable && "cursor-pointer hover:shadow-md hover:border-zinc-200 hover:-translate-y-0.5 active:scale-[0.98]",
        isSelected && "border-indigo-500 ring-4 ring-indigo-500/10 shadow-lg bg-indigo-50/5",
        className
      )}
      {...props}
    />
  )
)
Card.displayName = "Card"

export const CardHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-col space-y-1.5 p-4", className)} {...props} />
)
CardHeader.displayName = "CardHeader"

export const CardTitle = ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h3 className={cn("text-base font-semibold leading-none tracking-tight text-zinc-900 ", className)} {...props} />
)
CardTitle.displayName = "CardTitle"

export const CardDescription = ({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p className={cn("text-sm text-zinc-500 ", className)} {...props} />
)
CardDescription.displayName = "CardDescription"

export const CardContent = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("p-4 pt-0", className)} {...props} />
)
CardContent.displayName = "CardContent"

export const CardFooter = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex items-center p-4 pt-0", className)} {...props} />
)
CardFooter.displayName = "CardFooter"
