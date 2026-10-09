export interface HealthResponse {
  status: string;
  version: string;
  llm_provider: string;
  llm_model: string;
  llm_configured: boolean;
}

/** Base URL of the backend. Empty in dev, where Vite proxies requests to it. */
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, { signal });
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }
  return (await response.json()) as HealthResponse;
}
