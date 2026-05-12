// src/types/interrupt.ts
// Mirror of edms_ai_assistant/agent/interrupt_contract.py

export const INTERRUPT_SCHEMA_VERSION = 1;

// ── Shared building blocks ──────────────────────────────────────────────

export interface InterruptOption {
  id: string;
  label: string;
  description?: string | null;
  metadata?: Record<string, unknown> | null;
}

// ── Outbound payloads (server → frontend) ───────────────────────────────

export interface DisambiguationInterrupt {
  schema_version: typeof INTERRUPT_SCHEMA_VERSION;
  kind: 'disambiguation';
  entity_type: string;
  prompt: string;
  options: InterruptOption[];
  multiple: boolean;
  search_term?: string | null;
}

export interface ConfirmationInterrupt {
  schema_version: typeof INTERRUPT_SCHEMA_VERSION;
  kind: 'confirmation';
  prompt: string;
  danger: boolean;
  confirm_label: string;
  cancel_label: string;
}

export interface TextInputInterrupt {
  schema_version: typeof INTERRUPT_SCHEMA_VERSION;
  kind: 'text_input';
  prompt: string;
  placeholder?: string | null;
  secret: boolean;
  validator_regex?: string | null;
}

export interface SelectInterrupt {
  schema_version: typeof INTERRUPT_SCHEMA_VERSION;
  kind: 'select';
  prompt: string;
  options: InterruptOption[];
  default?: string | null;
}

export type InterruptPayload =
  | DisambiguationInterrupt
  | ConfirmationInterrupt
  | TextInputInterrupt
  | SelectInterrupt;

// ── Inbound resume values (frontend → server) ───────────────────────────

export interface DisambiguationResume {
  kind: 'disambiguation';
  selected_ids: [string, ...string[]];
}

export interface ConfirmationResume {
  kind: 'confirmation';
  confirmed: boolean;
}

export interface TextInputResume {
  kind: 'text_input';
  value: string;
}

export interface SelectResume {
  kind: 'select';
  selected_id: string;
}

export interface AbortResume {
  kind: '__abort__';
  reason?: string | null;
}

export type ResumeValue =
  | DisambiguationResume
  | ConfirmationResume
  | TextInputResume
  | SelectResume
  | AbortResume;

// ── SSE event envelopes ──────────────────────────────────────────────────

export interface InterruptEvent {
  interrupt_id: string | null;
  thread_id: string;
  payload: InterruptPayload;
}

export interface MessageEvent {
  role: 'assistant';
  content: string;
}

export interface DoneEvent {
  thread_id: string;
  paused: boolean;
}

export interface ErrorEvent {
  code: string;
  message: string;
  thread_id: string;
}