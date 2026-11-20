import assert from "node:assert";
import test from "node:test";
import { BackendAdapter, type PipelineEvent } from "./backendAdapter.ts";

function createMockStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index++]));
      } else {
        controller.close();
      }
    },
  });
}

test("1. Successful terminal completion: EXACT_MATCH_VERIFY with status complete", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");
  adapter.setApiToken("test_token_12345");

  const sseData = [
    'data: {"stage": "LOAD", "status": "complete"}\n\n',
    'data: {"stage": "EMBED", "status": "complete"}\n\n',
    'data: {"stage": "EXACT_MATCH_VERIFY", "status": "complete"}\n\n',
  ];

  // Intercept global fetch
  const originalFetch = globalThis.fetch;
  let sentHeaders: any = null;
  let sentBody: any = null;

  globalThis.fetch = async (_input: any, init?: any) => {
    sentHeaders = init?.headers;
    sentBody = init?.body ? JSON.parse(init.body) : null;
    return new Response(createMockStream(sseData), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  };

  try {
    const events: PipelineEvent[] = [];
    let completed = false;

    await adapter.runMediaExperiment(
      "sec_1",
      "cov_1",
      (ev) => events.push(ev),
      () => { completed = true; },
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    );

    assert.strictEqual(completed, true);
    assert.strictEqual(events.length, 3);
    assert.strictEqual(events[2].stage, "EXACT_MATCH_VERIFY");
    assert.strictEqual(events[2].status, "complete");

    // Verify auth header and post body key
    assert.strictEqual(sentHeaders?.["Authorization"], "Bearer test_token_12345");
    assert.strictEqual(sentBody?.secret_session_id, "sec_1");
    assert.strictEqual(sentBody?.cover_session_id, "cov_1");
    assert.strictEqual(sentBody?.key_hex, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("2. Early EOF without terminal success emits failed ERROR event", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");

  const prematureSse = [
    'data: {"stage": "LOAD", "status": "complete"}\n\n',
    'data: {"stage": "EMBED", "status": "complete"}\n\n',
    // EOF arrives here before EXACT_MATCH_VERIFY
  ];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(createMockStream(prematureSse), {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });

  try {
    const events: PipelineEvent[] = [];
    let completed = false;

    await adapter.runMediaExperiment(
      "sec_1",
      "cov_1",
      (ev) => events.push(ev),
      () => { completed = true; }
    );

    assert.strictEqual(completed, true);
    const lastEvent = events[events.length - 1];
    assert.strictEqual(lastEvent.stage, "ERROR");
    assert.strictEqual(lastEvent.status, "failed");
    assert.match(lastEvent.error || "", /unexpected EOF/i);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("3. Malformed event JSON emits failed ERROR event", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");

  const corruptSse = [
    'data: {"stage": "LOAD", "status": "complete"}\n\n',
    'data: INVALID_CORRUPT_NOT_JSON\n\n',
  ];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(createMockStream(corruptSse), {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });

  try {
    const events: PipelineEvent[] = [];
    let completed = false;

    await adapter.runMediaExperiment(
      "sec_1",
      "cov_1",
      (ev) => events.push(ev),
      () => { completed = true; }
    );

    assert.strictEqual(completed, true);
    const lastEvent = events[events.length - 1];
    assert.strictEqual(lastEvent.stage, "ERROR");
    assert.strictEqual(lastEvent.status, "failed");
    assert.match(lastEvent.error || "", /Malformed event data/i);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("4. Server HTTP failure (e.g. 401 Unauthorized) emits failed ERROR event", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: "Authentication required" }), {
    status: 401,
    headers: { "Content-Type": "application/json" },
  });

  try {
    const events: PipelineEvent[] = [];
    let completed = false;

    await adapter.runMediaExperiment(
      "sec_1",
      "cov_1",
      (ev) => events.push(ev),
      () => { completed = true; }
    );

    assert.strictEqual(completed, true);
    assert.strictEqual(events.length, 1);
    assert.strictEqual(events[0].stage, "ERROR");
    assert.strictEqual(events[0].status, "failed");
    assert.strictEqual(events[0].error, "Authentication required");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("5. User cancellation cleanly aborts the active stream", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");

  let aborted = false;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_input: any, init?: any) => {
    init?.signal?.addEventListener("abort", () => {
      aborted = true;
    });
    // Stream with cancel handler unblocked on reader.cancel()
    const cancellableStream = new ReadableStream({
      pull() {
        // keep open
      },
      cancel() {
        // cleanly unblocks
      },
    });
    return new Response(cancellableStream, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  };

  try {
    const events: PipelineEvent[] = [];
    let completed = false;

    const promise = adapter.runMediaExperiment(
      "sec_1",
      "cov_1",
      (ev) => events.push(ev),
      () => { completed = true; }
    );

    // Cancel after brief yield
    await new Promise((r) => setTimeout(r, 10));
    adapter.cancel();
    await promise;

    assert.strictEqual(completed, false, "Cancelled stream should not trigger onComplete callback");

    assert.strictEqual(aborted, true);
    // User cancellation does not emit spurious error events
    assert.strictEqual(events.length, 0);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("6. extractMedia uploads file and invokes /api/extract with precision and key", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");

  const originalFetch = globalThis.fetch;
  const calls: { url: string; method?: string; body?: any }[] = [];

  globalThis.fetch = async (input: any, init?: any) => {
    const url = typeof input === "string" ? input : input.url;
    calls.push({
      url,
      method: init?.method,
      body: init?.body,
    });

    if (url.includes("/api/upload")) {
      return new Response(JSON.stringify({
        session_id: "upload_stego_session_123",
        size: 5000,
        filename: "stego.png",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.includes("/api/extract")) {
      const parsedBody = JSON.parse(init.body);
      return new Response(JSON.stringify({
        status: "success",
        recovered_session_id: "recovered_session_456",
        recovered_bytes: 576,
        media_type: "IMAGE",
        session_counter: 101,
        ecc_parity: 48,
        protocol_version: 2,
        wire_protocol_version: 2,
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response("Not Found", { status: 404 });
  };

  try {
    const mockFile = new File([new Uint8Array(5000)], "stego.png", { type: "image/png" });
    const result = await adapter.extractMedia(
      mockFile,
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "neural_spatial_onnx",
      "int8"
    );

    assert.strictEqual(calls.length, 2);
    assert.strictEqual(calls[0].url, "http://testserver:8000/api/upload");
    assert.strictEqual(calls[1].url, "http://testserver:8000/api/extract");

    const extractPayload = JSON.parse(calls[1].body);
    assert.strictEqual(extractPayload.stego_session_id, "upload_stego_session_123");
    assert.strictEqual(extractPayload.precision, "int8");
    assert.strictEqual(extractPayload.model_name, "neural_spatial_onnx");
    assert.strictEqual(extractPayload.key_hex, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");

    assert.strictEqual(result.status, "success");
    assert.strictEqual(result.recovered_session_id, "recovered_session_456");
    assert.strictEqual(result.recovered_bytes, 576);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("7. extractMedia correctly propagates fail-closed rejection detail", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: any) => {
    const url = typeof input === "string" ? input : input.url;
    if (url.includes("/api/upload")) {
      return new Response(JSON.stringify({ session_id: "corrupted_stego" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (url.includes("/api/extract")) {
      return new Response(JSON.stringify({
        detail: "Extraction failed closed: AES-GCM GHASH authentication tag mismatch.",
      }), {
        status: 422,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("Not Found", { status: 404 });
  };

  try {
    const mockFile = new File([new Uint8Array(100)], "bad_stego.png", { type: "image/png" });
    await assert.rejects(
      async () => {
        await adapter.extractMedia(mockFile);
      },
      (err: Error) => {
        assert.match(err.message, /AES-GCM GHASH authentication tag mismatch/);
        return true;
      }
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("8. checkSession and login establish session and set CSRF header", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");

  const originalFetch = globalThis.fetch;
  let loginBody: any = null;
  let sessionCredentials: any = null;

  globalThis.fetch = async (input: any, init?: any) => {
    const url = typeof input === "string" ? input : input.url;
    sessionCredentials = init?.credentials;
    if (url.includes("/api/auth/session")) {
      return new Response(JSON.stringify({
        authenticated: false,
        requires_auth: true,
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (url.includes("/api/auth/login")) {
      loginBody = JSON.parse(init?.body);
      return new Response(JSON.stringify({
        status: "authenticated",
        user: "operator",
        csrf_token: "csrf_token_abc123xyz789",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("Not Found", { status: 404 });
  };

  try {
    const initialSession = await adapter.checkSession();
    assert.strictEqual(initialSession.authenticated, false);
    assert.strictEqual(initialSession.requires_auth, true);
    assert.strictEqual(sessionCredentials, "include");

    const loginRes = await adapter.login("secret_token_123");
    assert.strictEqual(loginRes.status, "authenticated");
    assert.strictEqual(loginBody?.token, "secret_token_123");
    assert.strictEqual(adapter.getCsrfToken(), "csrf_token_abc123xyz789");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("9. logout clears CSRF token and invalidates active session", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");
  adapter.setCsrfToken("csrf_token_to_clear");

  const originalFetch = globalThis.fetch;
  let logoutCalled = false;

  globalThis.fetch = async (input: any) => {
    const url = typeof input === "string" ? input : input.url;
    if (url.includes("/api/auth/logout")) {
      logoutCalled = true;
      return new Response(JSON.stringify({ status: "logged_out" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("Not Found", { status: 404 });
  };

  try {
    assert.strictEqual(adapter.getCsrfToken(), "csrf_token_to_clear");
    await adapter.logout();
    assert.strictEqual(logoutCalled, true);
    assert.strictEqual(adapter.getCsrfToken(), null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("10. uploadMedia and mutating requests include X-CSRF-Token and credentials: include", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("http://testserver:8000");
  adapter.setCsrfToken("csrf_active_999");

  const originalFetch = globalThis.fetch;
  let sentHeaders: any = null;
  let sentCredentials: any = null;

  globalThis.fetch = async (input: any, init?: any) => {
    const url = typeof input === "string" ? input : input.url;
    sentHeaders = init?.headers;
    sentCredentials = init?.credentials;
    if (url.includes("/api/upload")) {
      return new Response(JSON.stringify({ session_id: "upload_sid_123" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("Not Found", { status: 404 });
  };

  try {
    const mockFile = new File([new Uint8Array(50)], "test.png", { type: "image/png" });
    const res = await adapter.uploadMedia(mockFile);
    assert.strictEqual(res.session_id, "upload_sid_123");
    assert.strictEqual(sentHeaders?.["X-CSRF-Token"], "csrf_active_999");
    assert.strictEqual(sentCredentials, "include");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("11. getOidcLoginUrl returns full URL and checkSession parses OIDC attributes", async () => {
  const adapter = new BackendAdapter();
  adapter.setBaseUrl("https://paladinmedia.internal:8443");

  assert.strictEqual(adapter.getOidcLoginUrl(), "https://paladinmedia.internal:8443/api/auth/oidc/login");

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    authenticated: false,
    requires_auth: true,
    oidc_configured: true,
    oidc_issuer: "https://auth.internal",
    oidc_login_url: "/api/auth/oidc/login",
    idp_gate_status: "OPEN_PENDING_EXTERNAL_IDP",
  }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

  try {
    const session = await adapter.checkSession();
    assert.strictEqual(session.authenticated, false);
    assert.strictEqual(session.requires_auth, true);
    assert.strictEqual(session.oidc_configured, true);
    assert.strictEqual(session.oidc_issuer, "https://auth.internal");
    assert.strictEqual(session.idp_gate_status, "OPEN_PENDING_EXTERNAL_IDP");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
