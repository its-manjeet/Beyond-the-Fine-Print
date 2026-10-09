import { useCallback, useEffect, useState } from "react";
import { fetchHealth, type HealthResponse } from "./api/client";

type HealthState =
  | { kind: "loading" }
  | { kind: "ok"; data: HealthResponse }
  | { kind: "error"; message: string };

function useHealth() {
  const [state, setState] = useState<HealthState>({ kind: "loading" });

  const load = useCallback((signal?: AbortSignal) => {
    setState({ kind: "loading" });
    fetchHealth(signal)
      .then((data) => setState({ kind: "ok", data }))
      .catch((error: unknown) => {
        if (signal?.aborted) return;
        setState({ kind: "error", message: error instanceof Error ? error.message : "Unknown error" });
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  return { state, retry: () => load() };
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <dt className="text-slate-500">{label}</dt>
      <dd className="truncate font-medium">{value}</dd>
    </div>
  );
}

function HealthCard() {
  const { state, retry } = useHealth();

  return (
    <section aria-labelledby="status-heading" className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
      <div className="mb-3 flex items-center justify-between">
        <h2 id="status-heading" className="text-lg font-semibold">
          System status
        </h2>
        {state.kind === "ok" && (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-sm font-medium text-emerald-700">
            <span aria-hidden className="h-2 w-2 rounded-full bg-emerald-500" />
            Online
          </span>
        )}
      </div>

      {state.kind === "loading" && (
        <p role="status" className="text-slate-500">
          Checking backend…
        </p>
      )}

      {state.kind === "error" && (
        <div role="alert" className="space-y-3">
          <p className="text-red-700">Can't reach the backend: {state.message}</p>
          <button
            type="button"
            onClick={retry}
            className="min-h-12 w-full rounded-xl bg-blue-900 px-4 text-base font-semibold text-white active:bg-blue-800"
          >
            Try again
          </button>
        </div>
      )}

      {state.kind === "ok" && (
        <dl className="divide-y divide-slate-100 text-base">
          <StatusRow label="Backend" value={state.data.status} />
          <StatusRow label="Version" value={state.data.version} />
          <StatusRow label="AI provider" value={state.data.llm_provider} />
          <StatusRow label="Model" value={state.data.llm_model} />
          <StatusRow label="API key" value={state.data.llm_configured ? "Configured" : "Missing"} />
        </dl>
      )}
    </section>
  );
}

export default function App() {
  return (
    <div className="mx-auto flex min-h-dvh max-w-xl flex-col">
      <header className="bg-blue-900 px-4 pb-6 pt-[max(1.5rem,env(safe-area-inset-top))] text-white">
        <h1 className="text-2xl font-bold">Beyond the Fine Print</h1>
        <p className="mt-1 text-blue-100">Understand your documents in plain language.</p>
      </header>

      <div role="note" className="border-b border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
        This is information, not legal advice. For free legal help, contact your District or State Legal Services
        Authority (NALSA).
      </div>

      <main className="flex-1 space-y-4 px-4 py-6">
        <HealthCard />
      </main>
    </div>
  );
}
