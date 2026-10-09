import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const healthy = {
  status: "ok",
  version: "0.1.0",
  llm_provider: "anthropic",
  llm_model: "claude-sonnet-5-5",
  llm_configured: true,
};

function mockFetch(...responses: Array<Response | Error>) {
  const fn = vi.fn();
  for (const r of responses) {
    if (r instanceof Error) fn.mockRejectedValueOnce(r);
    else fn.mockResolvedValueOnce(r);
  }
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("shows backend status from /health", async () => {
    const fetchMock = mockFetch(Response.json(healthy));

    render(<App />);

    expect(screen.getByRole("status")).toHaveTextContent("Checking backend");
    expect(await screen.findByText("Online")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet-5-5")).toBeInTheDocument();
    expect(screen.getByText("Configured")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/health", expect.anything());
  });

  it("always shows the not-legal-advice notice", () => {
    mockFetch(Response.json(healthy));

    render(<App />);

    expect(screen.getByRole("note")).toHaveTextContent("not legal advice");
  });

  it("shows an error and retries", async () => {
    mockFetch(new Error("Failed to fetch"), Response.json(healthy));

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Failed to fetch");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Online")).toBeInTheDocument();
  });
});
