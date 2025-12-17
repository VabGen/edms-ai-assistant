const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, '../out');

function buildExtension() {
    console.log('Начинаю исправление путей для Chrome...');

    const oldHtml = path.join(outDir, 'index.html');
    const newHtml = path.join(outDir, 'popup.html');
    if (fs.existsSync(oldHtml)) {
        fs.renameSync(oldHtml, newHtml);
    }

    const underscoreNext = path.join(outDir, '_next');
    const dotNext = path.join(outDir, '.next');
    const targetNext = path.join(outDir, 'next');

    if (fs.existsSync(underscoreNext)) {
        if (fs.existsSync(targetNext)) fs.rmSync(targetNext, {recursive: true});
        fs.renameSync(underscoreNext, targetNext);
    } else if (fs.existsSync(dotNext)) {
        if (fs.existsSync(targetNext)) fs.rmSync(targetNext, {recursive: true});
        fs.renameSync(dotNext, targetNext);
    }

    const walkAndReplace = (dir) => {
        const files = fs.readdirSync(dir);
        files.forEach(file => {
            const filePath = path.join(dir, file);
            if (fs.statSync(filePath).isDirectory()) {
                walkAndReplace(filePath);
            } else if (file.endsWith('.html') || file.endsWith('.js') || file.endsWith('.css')) {
                let content = fs.readFileSync(filePath, 'utf8');

                const updatedContent = content
                    .replace(/\/_next\//g, 'next/')
                    .replace(/_next\//g, 'next/')
                    .replace(/\/\.next\//g, 'next/')
                    .replace(/\.next\//g, 'next/');

                fs.writeFileSync(filePath, updatedContent);
            }
        });
    };

    walkAndReplace(outDir);
    console.log('✅ Готово! Теперь пути в popup.html соответствуют папке на диске.');
}

buildExtension();