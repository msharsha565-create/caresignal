export type RiskBand = "high" | "medium" | "low";

export interface PatientRow {
  patient_id: string;
  age: number;
  sex: string;
  charlson_index: number;
  prior_admissions_12m: number;
  discharge_disposition: string;
  predicted_risk: number;
  risk_band: RiskBand;
}

export interface Driver {
  feature: string;
  label: string;
  value: number;
  risk_contribution: number;
}

export interface Interaction {
  drugs: [string, string];
  severity: "major" | "moderate";
  effect: string;
}

export interface PatientDetail {
  patient: PatientRow & {
    length_of_stay: number;
    num_medications: number;
    hba1c: number | null;
    creatinine: number | null;
    systolic_bp: number;
    bmi: number;
    chf: number;
    copd: number;
    diabetes: number;
    ckd: number;
    lives_alone: number;
    medications: string[];
  };
  risk: {
    predicted_30d_readmission_risk: number;
    risk_band: RiskBand;
    cohort_percentile: number;
    top_drivers: Driver[];
  };
  interactions: Interaction[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatReply {
  reply: string;
  tool_calls: { name: string; input: Record<string, unknown> }[];
  model: string;
  refused: boolean;
}

export interface Metrics {
  n_patients: number;
  readmission_rate: number;
  test_auroc: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body && typeof body.detail === "string" ? body.detail : res.statusText;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  metrics: () => request<Metrics>("/api/metrics"),
  patients: (band: RiskBand | "", condition: string) => {
    const q = new URLSearchParams({ limit: "100" });
    if (band) q.set("risk_band", band);
    if (condition) q.set("condition", condition);
    return request<PatientRow[]>(`/api/patients?${q}`);
  },
  patient: (id: string) => request<PatientDetail>(`/api/patients/${encodeURIComponent(id)}`),
  chat: (messages: ChatMessage[]) =>
    request<ChatReply>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    }),
};
