import { defineConfig } from 'wxt'
import { loadEnv } from 'vite'
import path from 'path'

const ROOT_DIR = path.resolve(process.cwd(), '..')
const env = loadEnv(process.env.NODE_ENV ?? 'development', ROOT_DIR, '')
const API_URL = new URL(env.VITE_API_URL ?? 'http://localhost:8000')
const EDMS_FRONTEND_URL = env.VITE_EDMS_FRONTEND_URL ?? 'https://next.edo.iba'

export default defineConfig({
  srcDir: 'src',
  modules: ['@wxt-dev/module-react'],
  vite: () => ({
    envDir: ROOT_DIR,
  }),
  manifest: {
    name: 'EDMS AI Assistant',
    description: 'AI-powered assistant for EDMS document management',
    version: '2.0.0',
    permissions: ['activeTab', 'storage', 'tabs'],
    host_permissions: [
      'http://localhost/*',
      'http://127.0.0.1/*',
      `${API_URL.origin}/*`,
      `${EDMS_FRONTEND_URL}/*`,
    ],
  },
})
