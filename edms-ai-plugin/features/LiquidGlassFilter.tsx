import React from 'react';

/**
 * Компонент LiquidGlassFilter для Plasmo/Shadow DOM
 * Применяется к элементам через className="[filter:url(#liquid-glass-filter)]"
 */
const LiquidGlassFilter: React.FC = () => {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="0"
            height="0"
            style={{
                position: 'absolute',
                pointerEvents: 'none',
                userSelect: 'none',
                opacity: 0,
                height: 0,
                width: 0
            }}
            aria-hidden="true"
        >
            <defs>
                <filter id="liquid-glass-filter" x="-20%" y="-20%" width="140%" height="140%"
                        colorInterpolationFilters="sRGB">
                    <feTurbulence
                        type="fractalNoise"
                        baseFrequency="0.012 0.012"
                        numOctaves="3"
                        seed="92"
                        result="noise"
                    />
                    <feGaussianBlur in="noise" stdDeviation="3" result="blurred"/>
                    <feDisplacementMap
                        in="SourceGraphic"
                        in2="blurred"
                        scale="35"
                        xChannelSelector="R"
                        yChannelSelector="G"
                    />
                </filter>
            </defs>
        </svg>
    );
};

export default LiquidGlassFilter;