import React, { useEffect, useRef } from 'react';

interface ParticleEffectProps {
  isActive: boolean;
  onComplete: () => void;
}

export default function ParticleEffect({ isActive, onComplete }: ParticleEffectProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    const container = containerRef.current;
    container.innerHTML = '';

    const particleCount = 150; // Оптимизировано: 1200 частиц в Shadow DOM могут затормозить браузер
    const maxDelay = 0.1;

    for (let i = 0; i < particleCount; i++) {
      const dx = (Math.random() - 0.5) * 400; // Ограничено радиусом виджета
      const dy = (Math.random() - 0.5) * 400;
      const delay = Math.random() * maxDelay;
      const duration = 0.4 + Math.random() * 0.3;

      const particle = document.createElement('div');

      const colors = [
        '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe',
        '#4f46e5', '#3b82f6', '#0ea5e9', '#60a5fa'
      ];

      const randomColor = colors[Math.floor(Math.random() * colors.length)];
      const size = 2 + Math.random() * 3;

      particle.className = 'absolute left-1/2 top-1/2 rounded-full pointer-events-none opacity-0';
      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;

      particle.style.setProperty('--dx', `${dx}px`);
      particle.style.setProperty('--dy', `${dy}px`);

      particle.style.backgroundColor = randomColor;
      particle.style.boxShadow = `0 0 ${size * 2}px ${randomColor}`;
      // Используем анимацию, определенную ниже в теге style
      particle.style.animation = `particleExplode ${duration}s cubic-bezier(0.1, 0.8, 0.3, 1) ${delay}s forwards`;

      container.appendChild(particle);
    }

    const timer = setTimeout(() => {
      onComplete();
    }, 600);

    return () => {
      clearTimeout(timer);
    };
  }, [isActive, onComplete]);

  if (!isActive) return null;

  return (
    <>
      {/* Инъекция анимации специально для Shadow DOM */}
      <style>
        {`
          @keyframes particleExplode {
            0% {
              transform: translate(-50%, -50%) scale(1);
              opacity: 1;
            }
            100% {
              transform: translate(calc(-50% + var(--dx)), calc(-50% + var(--dy))) scale(0);
              opacity: 0;
            }
          }
        `}
      </style>
      <div
        ref={containerRef}
        className="absolute inset-0 pointer-events-none z-[110] overflow-hidden rounded-[28px]"
        aria-hidden="true"
      />
    </>
  );
}