// edms-ai-plugin/utils/edms-helpers.ts

/**
 * Общие утилиты для работы с EDMS
 */

/**
 * Извлекает UUID документа из URL страницы
 * @returns UUID документа или null, если не найден
 */
export const extractDocIdFromUrl = (): string | null => {
    try {
        const pathParts = window.location.pathname.split('/')
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
        return pathParts.find(part => uuidRegex.test(part)) || null
    } catch (e) {
        console.error('[EDMS] Ошибка извлечения ID документа:', e)
        return null
    }
}

/**
 * Получает JWT токен авторизации из localStorage/sessionStorage
 * @returns JWT токен (без префикса "Bearer") или null
 */
export const getAuthToken = (): string | null => {
    try {
        // Прямой поиск токена в известных ключах
        const directToken =
            localStorage.getItem('token') ||
            localStorage.getItem('access_token') ||
            sessionStorage.getItem('token')

        if (directToken) {
            return directToken.replace("Bearer ", "").trim()
        }

        // Поиск в других ключах, содержащих 'auth', 'user', 'oidc'
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i)
            if (key && (key.includes('auth') || key.includes('user') || key.includes('oidc'))) {
                const value = localStorage.getItem(key)

                // Проверяем, содержит ли значение JWT (начинается с eyJ)
                if (value?.includes('eyJ')) {
                    let token = value

                    // Если значение — JSON объект, извлекаем access_token
                    if (value.startsWith('{')) {
                        try {
                            const parsed = JSON.parse(value)
                            token = parsed.access_token || parsed.token
                        } catch {
                            // Если не удалось распарсить, используем как есть
                        }
                    }

                    return token.replace("Bearer ", "").trim()
                }
            }
        }
    } catch (e) {
        console.error('[EDMS] Ошибка получения токена:', e)
    }

    return null
}

/**
 * Проверяет, является ли текущая страница страницей обращения
 * @returns true, если это страница обращения
 */
export const isAppealPage = (): boolean => {
    try {
        const url = window.location.pathname.toLowerCase()

        // Проверка URL
        if (url.includes('appeal') || url.includes('обращение')) {
            return true
        }

        // Проверка наличия элементов обращения на странице
        const appealIndicators = [
            'h5:contains("Обращение")',
            'input[name*="fioApplicant"]',
            'input[name*="appeal"]',
            '[data-category="APPEAL"]'
        ]

        for (const selector of appealIndicators) {
            if (document.querySelector(selector)) {
                return true
            }
        }

        return false
    } catch (e) {
        console.error('[EDMS] Ошибка проверки типа страницы:', e)
        return false
    }
}

/**
 * Декодирует JWT payload (без проверки подписи)
 * @param token JWT токен
 * @returns Payload токена или null
 */
export const decodeJwtPayload = (token: string): any | null => {
    try {
        const parts = token.split('.')
        if (parts.length !== 3) {
            return null
        }

        let payload = parts[1]

        // Добавляем padding для base64
        const padding = 4 - (payload.length % 4)
        if (padding < 4) {
            payload += '='.repeat(padding)
        }

        const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'))
        return JSON.parse(decoded)
    } catch (e) {
        console.error('[EDMS] Ошибка декодирования JWT:', e)
        return null
    }
}

/**
 * Получает детальную информацию о токене
 * @returns Информация о токене или null
 */
export interface TokenInfo {
    token: string
    userId?: string
    userName?: string
    expiresAt?: Date
    isExpired?: boolean
}

export const getTokenInfo = (): TokenInfo | null => {
    const token = getAuthToken()
    if (!token) return null

    try {
        const payload = decodeJwtPayload(token)
        if (!payload) return { token }

        const expiresAt = payload.exp ? new Date(payload.exp * 1000) : undefined
        const isExpired = expiresAt ? new Date() > expiresAt : undefined

        return {
            token,
            userId: payload.id || payload.sub,
            userName: payload.name || payload.username,
            expiresAt,
            isExpired
        }
    } catch (e) {
        return { token }
    }
}

/**
 * Логирование с префиксом [EDMS]
 * @param args Аргументы для console.log
 */
export const log = (...args: any[]) => {
    console.log('[EDMS]', ...args)
}

/**
 * Логирование ошибок с префиксом [EDMS]
 * @param args Аргументы для console.error
 */
export const logError = (...args: any[]) => {
    console.error('[EDMS]', ...args)
}

/**
 * Форматирует UUID для отображения (сокращенный вид)
 * @param uuid UUID для форматирования
 * @returns Сокращенный UUID (например, "a1b2c3d4...xyz")
 */
export const formatUuid = (uuid: string): string => {
    if (!uuid) return ''
    if (uuid.length <= 16) return uuid
    return `${uuid.substring(0, 8)}...${uuid.substring(uuid.length - 4)}`
}

/**
 * Проверяет валидность UUID
 * @param uuid Строка для проверки
 * @returns true, если строка является валидным UUID
 */
export const isValidUuid = (uuid: string): boolean => {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
    return uuidRegex.test(uuid)
}
