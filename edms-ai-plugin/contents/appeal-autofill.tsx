// import type {PlasmoCSConfig, PlasmoGetInlineAnchor, PlasmoGetStyle} from "plasmo"
// import React, {useState, useEffect} from "react"
// import { extractDocIdFromUrl, getAuthToken, log, logError } from "~utils/edms-helpers"
//
// export const config: PlasmoCSConfig = {
//     matches: [
//         "http://localhost:3000/document-form/*",
//         "http://localhost:3001/document-form/*",
//         "http://localhost:8080/document-form/*"
//     ]
// }
//
// export const getStyle: PlasmoGetStyle = () => {
//     const style = document.createElement("style")
//     style.textContent = `
//         :host { z-index: 1000 !important; position: relative !important; }
//         #plasmo-shadow-container { z-index: 1000 !important; }
//         @keyframes spin {
//             from { transform: rotate(0deg); }
//             to { transform: rotate(360deg); }
//         }
//         @keyframes pulse {
//             0%, 100% { opacity: 1; }
//             50% { opacity: 0.7; }
//         }
//     `
//     return style
// }
//
// /**
//  * ✅ Единственная привязка к кнопке "Закрыть"
//  * Ищем кнопку один раз при загрузке страницы
//  */
// export const getInlineAnchor: () => Promise<null | HTMLButtonElement> = async () => {
//     // Проверяем что это страница обращения
//     if (!window.location.pathname.includes('/document-form/')) {
//         return null
//     }
//
//     // Ждём загрузки DOM
//     await new Promise(resolve => setTimeout(resolve, 1000))
//
//     log('[AppealFormAutofill] 🔍 Поиск кнопки "Закрыть"...')
//
//     // Ищем кнопку "Закрыть" в правом нижнем углу
//     const buttons = Array.from(document.querySelectorAll('button'))
//     const closeButton = buttons.find(btn =>
//         btn.textContent?.trim() === 'Закрыть' ||
//         btn.textContent?.trim() === 'Отменить'
//     )
//
//     if (closeButton) {
//         log('[AppealFormAutofill] ✅ Найдена кнопка "Закрыть"')
//         return closeButton
//     }
//
//     // Fallback: ищем кнопку "Сохранить"
//     const saveButton = buttons.find(btn => btn.textContent?.trim() === 'Сохранить')
//
//     if (saveButton) {
//         log('[AppealFormAutofill] ✅ Найдена кнопка "Сохранить" (fallback)')
//         return saveButton
//     }
//
//     logError('[AppealFormAutofill] ❌ Кнопки не найдены')
//     return null
// }
//
// const Icons = {
//     Sparkles: () => (
//         <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
//              strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
//             <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
//         </svg>
//     ),
//     Check: () => (
//         <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
//             <polyline points="20 6 9 17 4 12"/>
//         </svg>
//     ),
//     AlertCircle: () => (
//         <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
//             <circle cx="12" cy="12" r="10"/>
//             <line x1="12" y1="8" x2="12" y2="12"/>
//             <line x1="12" y1="16" x2="12.01" y2="16"/>
//         </svg>
//     ),
//     Loader: () => (
//         <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
//             <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
//         </svg>
//     )
// }
//
// const AppealFormAutofillButton = () => {
//     const [isEnabled, setIsEnabled] = useState(true)
//     const [isLoading, setIsLoading] = useState(false)
//     const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle')
//     const [message, setMessage] = useState('')
//
//     useEffect(() => {
//         chrome.storage.local.get(["assistantEnabled"], (res) => {
//             if (res.assistantEnabled !== undefined) {
//                 setIsEnabled(res.assistantEnabled)
//             }
//         })
//
//         const handleStorageChange = (changes: { [key: string]: chrome.storage.StorageChange }) => {
//             if (changes.assistantEnabled) {
//                 setIsEnabled(changes.assistantEnabled.newValue)
//             }
//         }
//
//         chrome.storage.onChanged.addListener(handleStorageChange)
//         return () => chrome.storage.onChanged.removeListener(handleStorageChange)
//     }, [])
//
//     const handleAutofill = async (e: React.MouseEvent) => {
//         e.stopPropagation()
//         e.preventDefault()
//
//         if (isLoading) return
//
//         const token = getAuthToken()
//         const documentId = extractDocIdFromUrl()
//
//         log('[AppealFormAutofill] 🚀 Запуск автозаполнения')
//
//         if (!token) {
//             setStatus('error')
//             setMessage('Токен не найден')
//             setTimeout(() => {
//                 setStatus('idle')
//                 setMessage('')
//             }, 3000)
//             return
//         }
//
//         if (!documentId) {
//             setStatus('error')
//             setMessage('ID документа не найден')
//             setTimeout(() => {
//                 setStatus('idle')
//                 setMessage('')
//             }, 3000)
//             return
//         }
//
//         setIsLoading(true)
//         setStatus('idle')
//         setMessage('')
//
//         try {
//             const response: any = await new Promise((resolve, reject) => {
//                 chrome.runtime.sendMessage({
//                     type: "autofillAppeal",
//                     payload: {
//                         message: "Заполни обращение",
//                         user_token: token,
//                         context_ui_id: documentId
//                     }
//                 }, (res) => {
//                     if (res?.success) resolve(res)
//                     else reject(new Error(res?.error || 'Ошибка'))
//                 })
//             })
//
//             if (response.success) {
//                 const data = response.data
//
//                 if (data.status === 'success' || data.status === 'partial_success') {
//                     setStatus('success')
//
//                     const extracted = data.extracted_data || {}
//                     let msg = `Заполнено ${data.filled_count || 0} полей`
//
//                     if (extracted.fio) {
//                         msg = `Заявитель: ${extracted.fio}`
//                     }
//
//                     setMessage(msg)
//                     log('[AppealFormAutofill] ✅ Успех!')
//
//                     window.postMessage({
//                         type: "REFRESH_CHAT_HISTORY",
//                         messages: [
//                             { type: 'human', content: 'Заполни обращение' },
//                             { type: 'ai', content: data.response || data.message }
//                         ]
//                     }, "*")
//
//                     setTimeout(() => {
//                         window.location.reload()
//                     }, 2000)
//
//                 } else {
//                     setStatus('error')
//                     setMessage(data.message || 'Ошибка')
//                 }
//             }
//
//         } catch (err: any) {
//             logError('[AppealFormAutofill] ❌ Ошибка:', err)
//             setStatus('error')
//             setMessage(err.message || 'Ошибка')
//         } finally {
//             setIsLoading(false)
//
//             if (status !== 'success') {
//                 setTimeout(() => {
//                     setStatus('idle')
//                     setMessage('')
//                 }, 5000)
//             }
//         }
//     }
//
//     if (!isEnabled) {
//         return null
//     }
//
//     return (
//         <div style={s.container}>
//             <button
//                 type="button"
//                 onClick={handleAutofill}
//                 disabled={isLoading}
//                 style={{
//                     ...s.button,
//                     ...(isLoading ? s.buttonLoading : {}),
//                     ...(status === 'success' ? s.buttonSuccess : {}),
//                     ...(status === 'error' ? s.buttonError : {})
//                 }}
//                 title="Автозаполнение обращения"
//             >
//                 <span style={s.iconWrapper}>
//                     {isLoading ? (
//                         <span style={{ animation: 'spin 1s linear infinite', display: 'flex' }}>
//                             <Icons.Loader />
//                         </span>
//                     ) : status === 'success' ? (
//                         <Icons.Check />
//                     ) : status === 'error' ? (
//                         <Icons.AlertCircle />
//                     ) : (
//                         <Icons.Sparkles />
//                     )}
//                 </span>
//
//                 <span style={s.text}>
//                     {isLoading ? 'Заполняю...'
//                         : status === 'success' ? 'Готово!'
//                             : status === 'error' ? 'Ошибка'
//                                 : 'AI Заполнить'}
//                 </span>
//             </button>
//
//             {message && (
//                 <div style={{
//                     ...s.tooltip,
//                     backgroundColor: status === 'error' ? '#ef4444' : '#10b981'
//                 }}>
//                     {message}
//                 </div>
//             )}
//         </div>
//     )
// }
//
// const s: Record<string, React.CSSProperties> = {
//     container: {
//         position: 'relative',
//         display: 'inline-block',
//         marginLeft: '12px'
//     },
//     button: {
//         display: 'flex',
//         alignItems: 'center',
//         gap: '8px',
//         padding: '6px 16px',
//         backgroundColor: '#6366f1',
//         color: '#fff',
//         borderRadius: '6px',
//         fontSize: '14px',
//         fontWeight: 500,
//         cursor: 'pointer',
//         transition: 'all 0.2s ease',
//         boxShadow: '0 2px 4px rgba(99, 102, 241, 0.3)',
//         border: 'none'
//     },
//     buttonLoading: {
//         backgroundColor: '#818cf8',
//         cursor: 'wait',
//         opacity: 0.9,
//         animation: 'pulse 1.5s ease-in-out infinite'
//     },
//     buttonSuccess: {
//         backgroundColor: '#10b981',
//         boxShadow: '0 2px 4px rgba(16, 185, 129, 0.3)'
//     },
//     buttonError: {
//         backgroundColor: '#ef4444',
//         boxShadow: '0 2px 4px rgba(239, 68, 68, 0.3)'
//     },
//     iconWrapper: {
//         display: 'flex',
//         alignItems: 'center',
//         justifyContent: 'center',
//         minWidth: '18px',
//         minHeight: '18px'
//     },
//     text: {
//         whiteSpace: 'nowrap'
//     },
//     tooltip: {
//         position: 'absolute',
//         top: '100%',
//         left: '50%',
//         transform: 'translateX(-50%)',
//         marginTop: '8px',
//         padding: '8px 12px',
//         borderRadius: '6px',
//         fontSize: '12px',
//         fontWeight: 500,
//         color: '#fff',
//         whiteSpace: 'nowrap',
//         boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
//         zIndex: 10001,
//         pointerEvents: 'none'
//     }
// }
//
// export default AppealFormAutofillButton