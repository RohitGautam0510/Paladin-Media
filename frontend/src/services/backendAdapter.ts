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
  };
  cover: {
    session_id: string;
    data_url: string;
    name: string;
    size: number;
  };
}

export class BackendAdapter {
  private eventSource: EventSource | null = null;

  async fetchDefaultFixtures(): Promise<DefaultFixtures> {
    const res = await fetch("http://localhost:8000/api/fixtures/defaults");
    if (!res.ok) throw new Error("Failed to load default fixtures");
    return res.json();
  }

  async uploadMedia(file: File): Promise<{ session_id: string; size: number; filename: string }> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("http://localhost:8000/api/upload", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Media upload failed");
    return res.json();
  }

  runMediaExperiment(
    secretSessionId: string,
    coverSessionId: string | null,
    onEvent: (event: PipelineEvent) => void,
    onComplete: () => void
  ) {
    if (this.eventSource) {
      this.eventSource.close();
    }

    const coverParam = coverSessionId ? `&cover_session_id=${encodeURIComponent(coverSessionId)}` : "";
    const url = `http://localhost:8000/api/stream?secret_session_id=${encodeURIComponent(secretSessionId)}${coverParam}`;
    this.eventSource = new EventSource(url);

    this.eventSource.onmessage = (e) => {
      try {
        const data: PipelineEvent = JSON.parse(e.data);
        onEvent(data);
        if (data.status === "failed" || data.stage === "EXACT_MATCH_VERIFY" || data.stage === "ERROR") {
          this.eventSource?.close();
          onComplete();
        }
      } catch (err) {
        console.error("Failed to parse event", err);
      }
    };

    this.eventSource.onerror = (e) => {
      console.error("SSE Connection Error", e);
      onEvent({
        stage: "ERROR",
        status: "failed",
        error: "Backend stream disconnected.",
      });
      this.eventSource?.close();
      onComplete();
    };
  }

  cancel() {
    if (this.eventSource) {
      this.eventSource.close();
    }
  }
}

export const backend = new BackendAdapter();
