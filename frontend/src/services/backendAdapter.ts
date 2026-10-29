export type PipelineStage =
  | "LOAD"
  | "SERIALIZE"
  | "ENCRYPT"
  | "RS_ENCODE"
  | "CAPACITY_CHECK"
  | "DWT"
  | "EMBED"
  | "IDWT"
  | "RECEIVER_DWT"
  | "EXTRACT"
  | "RS_DECODE"
  | "AES_VERIFY"
  | "SECRET_RECOVER"
  | "EXACT_MATCH_VERIFY"
  | "ERROR";

export type StageStatus = "waiting" | "processing" | "complete" | "failed" | "skipped";

export interface PipelineEvent {
  stage: PipelineStage;
  status: StageStatus;
  message?: string;
  metadata?: any;
  error?: string;
}

export interface DefaultFixtures {
  secret: {
    session_id: string;
    data_url: string;
    name: string;
    size: number;
    width?: number;
    height?: number;
    format?: string;
    mode?: string;
  };
  cover: {
    session_id: string;
    data_url: string;
    name: string;
    size: number;
    width?: number;
    height?: number;
    format?: string;
    mode?: string;
  };
}

export interface CarrierCapacityInfo {
  carrier_width: number;
  carrier_height: number;
  subband_width: number;
  subband_height: number;
  ecc_profile: string;
  ecc_parity_symbols: number;
  carrier_format_policy: string;
  raw_secret_capacity_bytes: number;
  reference_metadata_capacity_bytes: number;
  max_protected_bytes: number;
  is_qualified_deployment: boolean;
  operational_qualification_note: string;
}

const getDefaultBaseUrl = (): string => {
  try {
    if (typeof import.meta !== "undefined" && (import.meta as any).env && (import.meta as any).env.VITE_API_BASE_URL) {
      return (import.meta as any).env.VITE_API_BASE_URL;
    }
  } catch {
    // Ignore environments where import.meta is unavailable
  }

  if (typeof window !== "undefined" && window.location) {
    const { protocol, hostname, port } = window.location;
    if (port !== "5173" && port !== "3000") {
      return `${protocol}//${hostname}${port ? `:${port}` : ""}`;
    }
  }

  return "http://localhost:8000";
};

export class BackendAdapter {
  private baseUrl: string = getDefaultBaseUrl();
  private abortController: AbortController | null = null;
  private apiToken: string | null = null;

  setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/+$/, "");
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  setApiToken(token: string | null) {
    this.apiToken = token;
  }

  private getHeaders(extraHeaders: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = { ...extraHeaders };
    if (this.apiToken) {
      headers["Authorization"] = `Bearer ${this.apiToken}`;
    }
    return headers;
  }

  async fetchDefaultFixtures(): Promise<DefaultFixtures> {
    const res = await fetch(`${this.baseUrl}/api/fixtures/defaults`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error("Failed to load default fixtures");
    return res.json();
  }

  async uploadMedia(file: File): Promise<{
    session_id: string;
    size: number;
    filename: string;
    width?: number;
    height?: number;
    format?: string;
    mode?: string;
  }> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${this.baseUrl}/api/upload`, {
      method: "POST",
      headers: this.getHeaders(),
      body: formData,
    });
    if (!res.ok) throw new Error("Media upload failed");
    return res.json();
  }

  async fetchCarrierCapacity(width: number, height: number): Promise<CarrierCapacityInfo> {
    const res = await fetch(`${this.baseUrl}/api/carrier/capacity?width=${width}&height=${height}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      let msg = "Failed to fetch carrier capacity";
      try {
        const j = await res.json();
        if (j.detail) msg = j.detail;
      } catch {
        // ignore
      }
      throw new Error(msg);
    }
    return res.json();
  }

  async fetchConfig(width?: number, height?: number): Promise<any> {
    const params = new URLSearchParams();
    if (width) params.set("width", width.toString());
    if (height) params.set("height", height.toString());
    const query = params.toString() ? `?${params.toString()}` : "";
    const res = await fetch(`${this.baseUrl}/api/config${query}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error("Failed to fetch backend configuration");
    return res.json();
  }

  async fetchSessionMedia(sessionId: string): Promise<string> {
    const res = await fetch(`${this.baseUrl}/api/session/${sessionId}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error(`Failed to fetch session media: ${res.status}`);
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }

  async fetchRunSubbands(runId: string): Promise<Record<string, string>> {
    const res = await fetch(`${this.baseUrl}/api/runs/${runId}/subbands`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error(`Failed to fetch run subbands: ${res.status}`);
    const data = await res.json();
    return data.subbands;
  }

  async runMediaExperiment(
    secretSessionId: string,
    coverSessionId: string | null,
    onEvent: (event: PipelineEvent) => void,
    onComplete: () => void,
    keyHex?: string
  ): Promise<void> {
    this.cancel();

    this.abortController = new AbortController();
    const signal = this.abortController.signal;

    try {
      const res = await fetch(`${this.baseUrl}/api/stream`, {
        method: "POST",
        headers: this.getHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          secret_session_id: secretSessionId,
          cover_session_id: coverSessionId || undefined,
          key_hex: keyHex || undefined,
        }),
        signal,
      });

      if (!res.ok) {
        let errDetail = `Stream request failed with status ${res.status}`;
        try {
          const errJson = await res.json();
          if (errJson.detail) errDetail = errJson.detail;
        } catch {
          // Non-JSON error
        }
        onEvent({
          stage: "ERROR",
          status: "failed",
          error: errDetail,
        });
        onComplete();
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        throw new Error("Streaming response body is not readable");
      }

      const onAbort = () => {
        try {
          reader.cancel();
        } catch {
          // ignore
        }
      };
      signal.addEventListener("abort", onAbort, { once: true });

      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let terminalComplete = false;
      let terminalFailed = false;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";

          for (const part of parts) {
            const trimmed = part.trim();
            if (!trimmed) continue;
            for (const line of trimmed.split("\n")) {
              if (line.startsWith("data: ")) {
                try {
                  const data: PipelineEvent = JSON.parse(line.slice(6));
                  onEvent(data);

                  // Only recognize EXACT_MATCH_VERIFY with status "complete" as terminal completion
                  if (data.stage === "EXACT_MATCH_VERIFY" && data.status === "complete") {
                    terminalComplete = true;
                    this.cancel();
                    onComplete();
                    return;
                  }

                  if (data.status === "failed" || data.stage === "ERROR") {
                    terminalFailed = true;
                    this.cancel();
                    onComplete();
                    return;
                  }
                } catch (parseErr) {
                  terminalFailed = true;
                  onEvent({
                    stage: "ERROR",
                    status: "failed",
                    error: "Malformed event data received from pipeline stream.",
                  });
                  this.cancel();
                  onComplete();
                  return;
                }
              }
            }
          }
        }
      } finally {
        signal.removeEventListener("abort", onAbort);
      }

      // If user aborted cleanly, do not dispatch error or completion
      if (signal.aborted) {
        return;
      }

      // If EOF arrives without terminal success or explicit failure
      if (!terminalComplete && !terminalFailed) {
        onEvent({
          stage: "ERROR",
          status: "failed",
          error: "Stream closed prematurely before terminal pipeline completion (unexpected EOF).",
        });
      }

      onComplete();
    } catch (err: any) {
      if (err.name === "AbortError") {
        // Request cleanly aborted
        return;
      }
      console.error("Pipeline streaming error", err);
      onEvent({
        stage: "ERROR",
        status: "failed",
        error: "Backend stream disconnected.",
      });
      onComplete();
    }
  }

  cancel() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }
}

export const backend = new BackendAdapter();
