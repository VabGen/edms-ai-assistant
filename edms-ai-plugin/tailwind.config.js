// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./contents/**/*.{ts,tsx}",
    "./features/**/*.{ts,tsx}",
    "./api/**/*.{ts,tsx}",
    "./popup.tsx",
    "./options.tsx"
  ],
  theme: {
    extend: {
      animation: {
        'fade-in-scale': 'fadeInScale 0.35s cubic-bezier(0.22, 1, 0.36, 1) forwards',
        'fade-out-scale': 'fadeOutScale 0.35s cubic-bezier(0.22, 1, 0.36, 1) forwards',
        'pulse-ring': 'pulseRing 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'sound-wave': 'soundWave 0.6s ease-in-out infinite',
      },
      keyframes: {
        fadeInScale: {
          '0%': { opacity: '0', transform: 'scale(0.92)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        fadeOutScale: {
          '0%': { opacity: '1', transform: 'scale(1)' },
          '100%': { opacity: '0', transform: 'scale(0.95)' },
        },
        pulseRing: {
          '0%': { transform: 'scale(0.8)', opacity: '0.5' },
          '100%': { transform: 'scale(1.3)', opacity: '0' },
        },
        soundWave: {
          '0%, 100%': { transform: 'scaleY(0.5)' },
          '50%': { transform: 'scaleY(1.2)' },
        }
      },
    },
  },
  plugins: [
    require("tailwindcss-animate")
  ],
}