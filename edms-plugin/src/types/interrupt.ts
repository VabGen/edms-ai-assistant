// shared/types/interrupt.ts
export interface InterruptOption {
  id: string;
  name: string;
  dept: string;
}

export interface InterruptPayload {
  type: 'entity_disambiguation' | 'summary_type_selection' | 'field_correction';
  entity_type?: string;
  options?: InterruptOption[];
  multiple?: boolean;
}

export interface ResumePayload {
  type: 'entity_disambiguation' | 'summary_type_selection' | 'field_correction';
  selected_ids?: string[];
  summary_type?: string;
  field_value?: any;
}