// // src/types/interrupt.ts
// // Mirror of edms_ai_assistant/agent/interrupt_contract.py
//
// export const INTERRUPT_SCHEMA_VERSION = 1;
//
// // ── Shared building blocks ──────────────────────────────────────────────
//
// export interface InterruptOption {
//   id: string;
//   label: string;
//   description?: string | null;
//   metadata?: Record<string, unknown> | null;
// }
//
// /** A single card in a CardSelectInterrupt */
// export interface InterruptCard {
//   id: string;
//   label: string;
//   description?: string | null;
// }
//
// // ── Outbound payloads (server → frontend) ───────────────────────────────
//
// export interface DisambiguationInterrupt {
//   schema_version: typeof INTERRUPT_SCHEMA_VERSION;
//   kind: 'disambiguation';
//   entity_type: string;
//   prompt: string;
//   options: InterruptOption[];
//   multiple: boolean;
//   search_term?: string | null;
// }
//
// export interface ConfirmationInterrupt {
//   schema_version: typeof INTERRUPT_SCHEMA_VERSION;
//   kind: 'confirmation';
//   prompt: string;
//   danger: boolean;
//   confirm_label: string;
//   cancel_label: string;
// }
//
// export interface TextInputInterrupt {
//   schema_version: typeof INTERRUPT_SCHEMA_VERSION;
//   kind: 'text_input';
//   prompt: string;
//   placeholder?: string | null;
//   secret: boolean;
//   validator_regex?: string | null;
// }
//
// export interface SelectInterrupt {
//   schema_version: typeof INTERRUPT_SCHEMA_VERSION;
//   kind: 'select';
//   prompt: string;
//   options: InterruptOption[];
//   default?: string | null;
// }
//
// /** Beautiful card-select interrupt from ask_user_to_select tool.
//  *  Each card has a label and optional description. */
// export interface CardSelectInterrupt {
//   schema_version?: typeof INTERRUPT_SCHEMA_VERSION;
//   kind: 'card_select';
//   prompt: string;
//   cards: InterruptCard[];
//   multiple: boolean;
//   layout?: 'list' | 'grid';
// }
//
// export interface GenericSelectInterrupt {
//   schema_version?: typeof INTERRUPT_SCHEMA_VERSION;
//   kind: string;
//   prompt: string;
//   options: InterruptOption[];
//   multiple?: boolean;
//   entity_type?: string;
//   [key: string]: unknown;
// }
//
// export type InterruptPayload =
//   | DisambiguationInterrupt
//   | ConfirmationInterrupt
//   | TextInputInterrupt
//   | SelectInterrupt
//   | CardSelectInterrupt
//   | GenericSelectInterrupt;
//
// // ── Inbound resume values (frontend → server) ───────────────────────────
//
// export interface DisambiguationResume {
//   kind: 'disambiguation';
//   selected_ids: [string, ...string[]];
// }
//
// export interface ConfirmationResume {
//   kind: 'confirmation';
//   confirmed: boolean;
// }
//
// export interface TextInputResume {
//   kind: 'text_input';
//   value: string;
// }
//
// export interface SelectResume {
//   kind: 'select';
//   selected_id: string;
// }
//
// /** Resume value for CardSelectInterrupt */
// export interface CardSelectResume {
//   kind: 'card_select';
//   selected_ids: [string, ...string[]];
// }
//
// export interface AbortResume {
//   kind: '__abort__';
//   reason?: string | null;
// }
//
// export type ResumeValue =
//   | DisambiguationResume
//   | ConfirmationResume
//   | TextInputResume
//   | SelectResume
//   | CardSelectResume
//   | AbortResume;
//
// // ── SSE event envelopes ──────────────────────────────────────────────────
//
// export interface InterruptEvent {
//   interrupt_id: string | null;
//   thread_id: string;
//   payload: InterruptPayload;
// }
//
// export interface MessageEvent {
//   role: 'assistant';
//   content: string;
// }
//
// export interface DoneEvent {
//   thread_id: string;
//   paused: boolean;
// }
//
// export interface ErrorEvent {
//   code: string;
//   message: string;
//   thread_id: string;
// }