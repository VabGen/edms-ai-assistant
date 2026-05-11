import './style.css'
import {createRoot, type Root} from 'react-dom/client'
import {AssistantWidget} from './AssistantWidget'

export default defineContentScript({
    matches: [
        'http://localhost:3000/*',
        'http://localhost:3001/*',
        'http://localhost:8080/*',
        'https://next.edo.iba/*',
        'http://127.0.0.1:*/*',
        'https://127.0.0.1:*/*',
    ],
    cssInjectionMode: 'ui',
    runAt: 'document_end',

    async main(ctx) {
        console.log('[EDMS] ✅ Content script started')

        const ui = await createShadowRootUi<Root>(ctx, {
            name: 'edms-assistant',
            position: 'inline',
            anchor: 'body',
            append: 'last',

            onMount(container): Root {
                const wrapper = document.createElement('div')
                container.appendChild(wrapper)
                const root = createRoot(wrapper)
                root.render(<AssistantWidget/>)
                return root
            },

            onRemove(root: Root) {
                root.unmount()
            },
        })

        ui.mount()
    },
})
