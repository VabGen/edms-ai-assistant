import React from 'react';

const LiquidGlassFilter: React.FC = () => {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="0"
            height="0"
            style={{
                position: 'absolute',
                pointerEvents: 'none',
                visibility: 'hidden'
            }}
        >
            <defs>
                <filter id="liquid-glass-filter" x="-20%" y="-20%" width="140%" height="140%">
                    <feTurbulence
                        type="fractalNoise"
                        baseFrequency="0.007 0.007"
                        numOctaves="2"
                        seed="92"
                        result="noise"
                    />
                    <feGaussianBlur in="noise" stdDeviation="2" result="blurred" />
                    <feDisplacementMap
                        in="SourceGraphic"
                        in2="blurred"
                        scale="75"
                        xChannelSelector="R"
                        yChannelSelector="G"
                    />
                </filter>
            </defs>
        </svg>
    );
};

export default LiquidGlassFilter;