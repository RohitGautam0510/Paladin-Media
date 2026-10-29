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
