export interface User {
  id: number;
  email: string;
  username: string;
  full_name: string;
  phone?: string;
  role: {
    id: number;
    name: 'admin' | 'doctor' | 'nurse';
    description?: string;
  };
  healthcare_center_id?: number;
  is_active: boolean;
  offline_access: boolean;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface Patient {
  id: number;
  registration_number: string;
  first_name: string;
  last_name: string;
  first_name_ar?: string;
  last_name_ar?: string;
  date_of_birth: string;
  age_months: number;
  sex: 'male' | 'female';
  caregiver_name?: string;
  caregiver_phone?: string;
  caregiver_relation?: string;
  governorate?: string;
  district?: string;
  village?: string;
  residence_type?: 'urban' | 'rural' | 'camp';
  wealth_index?: string;
  maternal_education?: string;
  healthcare_center_id?: number;
  created_at: string;
  measurements?: Measurement[];
  predictions?: PredictionSummary[];
}

export interface Measurement {
  id: number;
  patient_id: number;
  weight_kg: number;
  height_cm: number;
  muac_mm?: number;
  head_circumference_cm?: number;
  oedema: boolean;
  oedema_severity?: string;
  diarrhea_recent: boolean;
  fever_recent: boolean;
  cough_recent: boolean;
  breastfeeding: boolean;
  exclusive_breastfeeding: boolean;
  vitamin_a: boolean;
  haz?: number;
  whz?: number;
  waz?: number;
  bmiz?: number;
  measurement_date: string;
  notes?: string;
}

export interface SeverityResult {
  probability: number;
  risk_percent: number;
  severity: 'normal' | 'mild' | 'moderate' | 'severe';
  confidence: number;
}

export interface Prediction {
  id: number;
  patient_id: number;
  stunting: SeverityResult;
  wasting: SeverityResult;
  underweight: SeverityResult;
  overall_risk: string;
  overall_recommendation: string;
  clinical_query: string;
  rag_evidence: Array<{
    id: number;
    title: string;
    clinical_summary?: string;
    citation?: string;
    relevance_score: number;
  }>;
  recommended_intervention: string;
  referral_needed: boolean;
  referral_urgency?: string;
  doctor_notes?: string;
  doctor_approved: boolean;
  model_version: string;
  created_at: string;
}

export interface PredictionSummary {
  id: number;
  overall_risk: string;
  created_at: string;
}

export interface PredictionRequest {
  patient_id: number;
  measurement_id: number;
  include_explanation?: boolean;
  include_rag?: boolean;
  language?: 'en' | 'ar';
}

export interface NEREntity {
  text: string;
  entity_type: 'DISEASE' | 'SYMPTOM' | 'TREATMENT' | 'MEASUREMENT' | 'NUTRIENT' | 'DEMOGRAPHIC';
  start_pos: number;
  end_pos: number;
  confidence: number;
  language: string;
}

export interface KnowledgeEntry {
  id: number;
  title: string;
  title_ar?: string;
  authors?: string;
  organization?: string;
  year?: number;
  abstract?: string;
  clinical_summary?: string;
  keywords?: string[];
  citation?: string;
  source_type?: string;
  status: string;
}

export interface DashboardStats {
  period_days: number;
  patients: {
    total: number;
    new: number;
  };
  predictions: {
    total: number;
  };
  severity_distribution: Record<string, number>;
  governorate_stats: Array<{
    governorate: string;
    patients: number;
    predictions: number;
  }>;
}
