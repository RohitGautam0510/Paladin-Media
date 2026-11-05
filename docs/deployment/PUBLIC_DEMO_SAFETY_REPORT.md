# PaladinMedia — Phase 9 Public Demonstration Safety & Reliability Qualification Report

**Document Version:** 2.0.0  
**Phase:** Phase 9 (Final Public Demo Safety & Reliability Qualification)  
**Date:** 2026-11-05  
**Repository:** `E:\Paladin_media`  
**Target Infrastructure:** Free Academic Hosting (Vercel Frontend + Render Backend)  
**Safety Verdict:** `SAFE TO PROCEED WITH ACADEMIC DEMO DEPLOYMENT`

---

## 1. Executive Verdict

PaladinMedia has completed the Phase 9 qualification gate for public academic demonstration on Render (backend) and Vercel (frontend). 

The empirical evaluation resolved the carrier scaling vulnerability discovered during deployment audits: while earlier offline environments accepted arbitrary dimensions up to $2048 \times 2048$ ($4\text{ Mpx}$), live benchmarking revealed that a $2048 \times 2048$ carrier allocates **1,250.1 MB resident memory (RSS)** during neural extraction—guaranteeing an immediate Linux cgroup Out-Of-Memory (OOM) termination on Render's 512 MB free tier.

To guarantee zero downtime and continuous operational stability, strict public-demo safety boundaries have been implemented across the full stack:
1. **Dimension Capping:** Public uploads in demo mode are hard-gated to **$512 \times 512$** ($262,144\text{ px}$) and **2 MB**, matching the canonical DIV2K benchmark baseline. At $512 \times 512$, peak RSS is **168.9 MB** for `neural_spatial_onnx` and **99.6 MB** for `acs_sub_inn_onnx` (leaving $> 343\text{ MB}$ headroom).
2. **Concurrency Safety:** Bounded concurrency testing under 1, 2, and 3 simultaneous streams on a live operational FastAPI server demonstrated peak RSS of **320.1 MB** under maximum load (leaving **191.9 MB / 37.5% headroom** below the 512 MB limit, well within the $\le 400\text{ MB}$ target).
3. **CORS Hardening:** Removed permissive origin matching; CORS is strictly confined to `https://paladinmedia.vercel.app` and regex `^https://paladinmedia(-[a-z0-9]+)?\.vercel\.app$` (blanket wildcards fail closed at startup).
4. **Demo Rate Limiting:** Enforced via `DemoRateLimiterMiddleware` at 60 req/min per IP with HTTP 429 backoff and `Retry-After: 60`.
5. **Zero PyTorch Footprint:** The production environment is 100% CPU-only ONNX Runtime. PyTorch is completely excluded from [`requirements.txt`](file:///e:/Paladin_media/requirements.txt) and the production import path.
6. **All Regression Suites Pass:** 436 backend unit tests, 69 backend integration tests, 40 frontend unit tests, 0 linter errors, and 100% clean production bundle build.

---

## 2. Repository and Version Information

- **Repository Root:** `E:\Paladin_media`
- **Git Branch:** `master`
- **Git HEAD Commit:** `9555996` (with Phase 9 qualification modifications in working tree)
- **Runtime Environment:** Python 3.10.11 (AMD64 Windows local / Linux x86_64 target container)
- **Backend Framework:** FastAPI 0.115.x + Uvicorn (Single Worker, `workers=1`)
- **Frontend Framework:** React 19 + TypeScript + Vite 8.3.0
- **Modulation Engines Qualified:**
  - `neural_spatial_onnx` ([`neural_spatial_embedding_opset14.onnx`](file:///e:/Paladin_media/checkpoints/phase5a_onnx/neural_spatial_embedding_opset14.onnx) / [`neural_spatial_extraction_opset14.onnx`](file:///e:/Paladin_media/checkpoints/phase5a_onnx/neural_spatial_extraction_opset14.onnx))
  - `acs_sub_inn_onnx` ([`acs_sub_inn_embedding_opset14.onnx`](file:///e:/Paladin_media/checkpoints/phase5a_onnx/acs_sub_inn_embedding_opset14.onnx) / [`acs_sub_inn_extraction_opset14.onnx`](file:///e:/Paladin_media/checkpoints/phase5a_onnx/acs_sub_inn_extraction_opset14.onnx))
- **Wire Protocol Baseline:** Wire Protocol v2 (Qualified Production Baseline, 16-byte header, RS(24, 16), $2t = 48$ payload parity)
- **Experimental Status:** Wire Protocol v3 remains an offline research prototype (`scripts/experiments/phase8_protocol_experiments.py`); it is rejected by the production API with HTTP 422.

### Modified Source Files in Phase 9
| File | Scope of Modification |
| :--- | :--- |
| [`src/api/server.py`](file:///e:/Paladin_media/src/api/server.py) | Capped upload dimensions to $512 \times 512$ in demo mode; added `DemoRateLimiterMiddleware`; tightened CORS regex; added `model_name` selector to `/api/extract`; gated wire protocol to v2 only |
| [`src/api/security.py`](file:///e:/Paladin_media/src/api/security.py) | Implemented `DemoRateLimiterMiddleware` (60 req/min); configured demo limits (2 MB upload, 25 MB session store, 180s TTL); dual env var resolution (`PALADINMEDIA_MODE` / `PALADIN_DEPLOYMENT_MODE`) |
| [`src/api/launcher.py`](file:///e:/Paladin_media/src/api/launcher.py) | Added dual env var resolution; validated single-worker and loopback invariants |
| [`render.yaml`](file:///e:/Paladin_media/render.yaml) | Specified `PALADINMEDIA_MODE: demo`, `PALADIN_DEPLOYMENT_MODE: demo`, and explicit allowed origins |
| [`requirements.txt`](file:///e:/Paladin_media/requirements.txt) | Stripped `torch` and `torchvision`; verified CPU-only ONNX Runtime dependencies |
| [`frontend/package.json`](file:///e:/Paladin_media/frontend/package.json) | Configured test script to native Node.js runner: `node --test "src/utils/*.test.ts" "src/services/*.test.ts"` |
| [`frontend/src/App.tsx`](file:///e:/Paladin_media/frontend/src/App.tsx) | Bound wire protocol dropdown to backend capabilities; enforced client-side validation on carrier dimensions and upload sizes |
| [`scripts/experiments/run_phase9_safety_benchmark.py`](file:///e:/Paladin_media/scripts/experiments/run_phase9_safety_benchmark.py) | Automated empirical benchmark suite for dimension scaling, concurrency, input safety, and security |

---

## 3. Memory Qualification

### 3.1 Production Architecture & Memory Boundaries
The production backend relies strictly on `onnxruntime` with `CPUExecutionProvider` and NumPy Haar wavelet operations. PyTorch is neither imported nor installed in the deployment container.

Render's free tier imposes an unyielding **512 MB physical memory ceiling**. To protect against kernel OOM kills under traffic bursts or GC jitter, Phase 9 established an engineering acceptance threshold of **$\le 400\text{ MB}$ peak resident set size (RSS)**, ensuring a safety margin of at least $112\text{ MB}$ headroom ($> 20\%$).

### 3.2 Memory Lifecycle Stages (Measured via `psutil`)
For each carrier dimension, isolated measurements tracked the process across seven lifecycle phases:
1. **Baseline RSS:** Process resident memory prior to loading steganography libraries (~28.3 MB).
2. **App Initialization RSS:** Resident memory after instantiating FastAPI and session storage (~84.5 MB).
3. **Model Loaded RSS:** Memory after ONNX Runtime creates inference sessions (~87.4–89.5 MB).
4. **Upload Peak RSS:** Memory during multipart parsing and buffering of carrier and secret payload (~95.4–131.0 MB).
5. **Embed Peak RSS:** Peak memory during DWT, context calculation, affine coupling, and stego reconstruction (~98.2–782.1 MB).
6. **Extraction Peak RSS:** Peak memory during receiver DWT and inverse coupling inference (~99.6–1250.1 MB).
7. **Recovery Peak RSS:** Final memory following Reed–Solomon decoding and AES-GCM verification (~99.6–1250.1 MB).

---

## 4. Carrier Dimension Results

Isolated subprocess benchmarks executed the complete real-world request workflow (POST `/api/upload` $\to$ POST `/api/embed` $\to$ POST `/api/extract` $\to$ exact byte verification) across both qualified neural ONNX engines:

```
Test Command: python scripts/experiments/run_phase9_safety_benchmark.py
Artifact: docs/experiments/phase9_safety_benchmark_results.json
```

### Empirical Dimension Scaling Matrix
| Dimension | Pixels | Modulation Engine | Baseline RSS | App Init RSS | Model Loaded | Upload Peak | Embed Peak | Extract Peak | Max Peak RSS | Recovery Status | Latency | Render 512 MB Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$512 \times 512$** | 262,144 | `neural_spatial_onnx` | 28.3 MB | 84.7 MB | 89.0 MB | 96.4 MB | 138.2 MB | 168.9 MB | **168.9 MB** | `EXACT MATCH` | 0.23 s | **PASS** (343.1 MB Headroom) |
| **$512 \times 512$** | 262,144 | `acs_sub_inn_onnx` | 28.3 MB | 84.4 MB | 87.7 MB | 95.4 MB | 98.2 MB | 99.6 MB | **99.6 MB** | `EXACT MATCH` | 0.27 s | **PASS** (412.4 MB Headroom) |
| **$768 \times 768$** | 589,824 | `neural_spatial_onnx` | 28.3 MB | 84.8 MB | 89.6 MB | 100.9 MB | 210.1 MB | 280.2 MB | **280.2 MB** | `EXACT MATCH` | 0.53 s | **PASS (MARGINAL)** (231.8 MB Headroom) |
| **$768 \times 768$** | 589,824 | `acs_sub_inn_onnx` | 28.3 MB | 84.5 MB | 87.6 MB | 99.2 MB | 100.4 MB | 101.6 MB | **101.6 MB** | `EXACT MATCH` | 0.60 s | **PASS** (410.4 MB Headroom) |
| **$1024 \times 1024$** | 1,048,576 | `neural_spatial_onnx` | 28.3 MB | 84.4 MB | 89.3 MB | 103.7 MB | 268.4 MB | 386.4 MB | **386.4 MB** | `EXACT MATCH` | 0.95 s | **HIGH RISK** (125.6 MB Headroom, Fails Concurrency) |
| **$1024 \times 1024$** | 1,048,576 | `acs_sub_inn_onnx` | 28.4 MB | 84.3 MB | 87.4 MB | 101.8 MB | 105.2 MB | 105.9 MB | **105.9 MB** | `EXACT MATCH` | 1.00 s | **PASS (ISOLATED)** (406.1 MB Headroom) |
| **$2048 \times 2048$** | 4,194,304 | `neural_spatial_onnx` | 28.3 MB | 84.8 MB | 89.5 MB | 131.0 MB | 782.1 MB | 1250.1 MB | **1,250.1 MB** | `EXACT MATCH` | 3.70 s | **FAIL / OOM TERMINATION** (+738.1 MB Over 512 MB Limit) |
| **$2048 \times 2048$** | 4,194,304 | `acs_sub_inn_onnx` | 28.4 MB | 84.3 MB | 87.3 MB | 128.8 MB | 134.2 MB | 132.2 MB | **134.2 MB** | `EXACT MATCH` | 3.87 s | **FAIL (SYSTEM LEVEL)** (Disallowed by Engine Parity) |

### Findings & Dimension Decision
- `neural_spatial_onnx` at $2048 \times 2048$ allocates **1,250.1 MB RSS**, exceeding Render's 512 MB limit by $144\%$.
- While `acs_sub_inn_onnx` isolates subband tensors to $134.2\text{ MB}$, the default engine (`neural_spatial_onnx`) cannot execute $2048 \times 2048$ or concurrent $1024 \times 1024$ requests without guaranteed OOM.
- **Decision:** Public demo carrier dimensions are strictly restricted to **$512 \times 512$** ($262,144\text{ pixels}$). Both backend and frontend enforce this ceiling.

---

## 5. Concurrent Request Results

Live concurrency benchmarks were executed against an active Uvicorn server running under `PALADINMEDIA_MODE=demo` on `127.0.0.1:8021`. Multiple independent clients executed complete end-to-end SSE streaming requests simultaneously.

```
Test Command: Automated in scripts/experiments/run_phase9_safety_benchmark.py
Server Startup: python -m uvicorn src.api.server:app --port 8021
```

### Empirical Concurrency Scaling
| Concurrency Level | Active Concurrent Streams | Initial Process RSS | Peak Process RSS | Headroom below 512 MB | Headroom % | Exact Recovery | SSE Events Emitted | Status Codes | Result |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Request** | 1 Stream (`neural_spatial_onnx`) | 84.0 MB | **193.2 MB** | 318.8 MB | 62.3% | `True` (1/1) | 27 events | 1x 200 OK | **PASS** |
| **2 Requests** | 2 Streams (Spatial + ACS-Sub-INN) | 84.0 MB | **288.3 MB** | 223.7 MB | 43.7% | `True` (2/2) | 54 events | 2x 200 OK | **PASS** |
| **3 Requests** | 3 Streams (Spatial + ACS + Spatial) | 84.0 MB | **320.1 MB** | **191.9 MB** | **37.5%** | `True` (3/3) | 81 events | 3x 200 OK | **PASS** |

### Concurrency Observations
1. **Memory Headroom:** At the maximum planned demo concurrency (3 streams), peak RSS reached **320.1 MB**, maintaining **191.9 MB margin** below Render's 512 MB ceiling. This easily satisfies the $\le 400\text{ MB}$ acceptance target.
2. **Session Isolation:** Each stream operated under an independent cryptographic session ID and session counter. Zero cross-talk, memory leaks, or state collisions occurred.
3. **Resource Cleanup:** Temporary carrier bytearrays in `BoundedSessionStore` are evicted after their TTL (180s) or upon explicit client deletion. Peak memory returns to idle baseline after garbage collection.

---

## 6. Upload and Input Safety

Task B evaluated the complete input validation matrix in `src/api/server.py` and `src/carrier/capacity.py`:

```
Test Command: Automated in scripts/experiments/run_phase9_safety_benchmark.py
Matrix Scope: Upload bounds, image header parsing, corruption rejection, format gates
```

### Input Safety & Boundary Enforcement Results
| Test Condition | Input Data Characteristics | Expected HTTP Code | Actual HTTP Code | Status | Verified Rejection Reason |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Oversized Upload Size** | 2.1 MB arbitrary binary payload | `413` | `413` | **PASS** | `Upload exceeds maximum permitted size of 2097152 bytes` |
| **Oversized Dimensions** | $514 \times 514$ valid PNG carrier (264,196 px) | `413` | `413` | **PASS** | `Image dimensions (514x514=264196 pixels) exceed maximum allowed pixel budget of 262144 pixels (512x512)` |
| **Valid Dimensions** | $512 \times 512$ 8-bit RGB PNG carrier | `200` | `200` | **PASS** | Successfully stored with TTL; session ID generated |
| **Odd Dimensions** | $511 \times 512$ carrier capacity query | `422` | `422` | **PASS** | `Carrier dimensions must be even for DWT decomposition, got 511x512` |
| **Malformed PNG** | Non-image garbage bytes labeled `image/png` | `400` / `422` | `400` | **PASS** | `Malformed or corrupted image content` |
| **Truncated PNG** | 16-byte raw PNG header without IDAT/IEND | `400` / `422` | `400` | **PASS** | `Malformed or corrupted image content` |
| **Unsupported Format** | Lossy baseline JPEG carrier | `422` | `422` | **PASS** | `Carrier format must be lossless PNG, got JPEG` |
| **Payload Capacity Gate** | 10,000 bytes payload on 512x512 carrier | `413` / `422` | `413` | **PASS** | `PAYLOAD_CAPACITY_EXCEEDED: Secret payload data (10000 B) exceeds maximum usable capacity (6525 B with 53B metadata) for ECC parity 48` |

All rejections occur **prior to neural inference**, preventing denial-of-service via computationally expensive tensor operations.

---

## 7. Security and Privacy Verification

### 7.1 Demo Rate Limiting
- **Implementation:** [`DemoRateLimiterMiddleware`](file:///e:/Paladin_media/src/api/security.py#L318-L380) tracks client IPs with a 60-second sliding window.
- **Policy:** Capped at 60 requests per minute per IP when `PALADINMEDIA_MODE=demo`.
- **Exemptions:** `/api/health` health probes and CORS `OPTIONS` preflight requests are exempted.
- **Empirical Test:** 65 sequential rapid requests produced **59 HTTP 200 OK** and **6 HTTP 429 Too Many Requests** responses with headers `Retry-After: 60`.

### 7.2 CORS Hardening
- **Configuration:** [`src/api/server.py`](file:///e:/Paladin_media/src/api/server.py#L82-L115) enforces strict origin checking:
  - Allowed origin `https://paladinmedia.vercel.app` matches and returns header `Access-Control-Allow-Origin: https://paladinmedia.vercel.app`.
  - Preview deployments matching `r"^https://paladinmedia(-[a-z0-9]+)?\.vercel\.app$"` are permitted.
  - Malicious origin `https://evil-site.com` is rejected (no allow-origin header reflected).
  - Wildcard (`*`) is strictly rejected during startup validation.

### 7.3 Sensitive Data Protection & Information Disclosure
- **Cryptographic Keys:** AES-256 keys are verified as 64-character hex strings in transit and never logged, stored in plain text, or emitted in SSE telemetry.
- **Plaintext Payloads:** Secret payloads are never logged or echoed in HTTP error bodies. In the event of AES-GCM authentication failure, extracted plaintext is immediately discarded.
- **Path Sanitization:** All internal file paths (`E:\Paladin_media\...`, `/home/...`) are sanitized via `sanitize_error_message()` before returning to clients.

---

## 8. Engine and Protocol Verification

### 8.1 Modulation Engine Profiles
PaladinMedia exposes two fully qualified ONNX inference engines:
1. **`neural_spatial_onnx` (Default Qualified):** Operates on spatial RGB images; internal opset14 graph computes 2D Haar DWT, context extraction, and inverse DWT.
   - Tested 512x512 RSS: 168.9 MB
   - Average Latency: 0.23 s
   - Terminal Exact Match: `True`
2. **`acs_sub_inn_onnx` (Qualified Transform-Domain Subband):** Computes explicit Python 2D Haar DWT; executes ONNX inference solely on the Red-HH target subband conditioned on Green/Blue context; executes explicit IDWT reconstruction.
   - Tested 512x512 RSS: 99.6 MB
   - Average Latency: 0.27 s
   - Terminal Exact Match: `True`

### 8.2 Wire Protocol Production Invariants
- **Qualified Baseline:** Wire Protocol v2 is the sole qualified production protocol:
  - 16-byte carrier header with RS(24, 16) error protection.
  - Header anchored at spatial row 0.
  - Test split recovery rate: **24/25 (96.0%)** on ACS-Sub-INN.
- **Protocol v3 Handling:** Wire Protocol v3 remains an offline experimental candidate (header anchor relocation at row 32 and coprime symbol interleaving). The production API rejects `version: 3` with HTTP 422 (`Unsupported wire protocol version: 3. Only Wire Protocol v2 is supported for production deployment`).

---

## 9. Sustainability Telemetry Verification

### 9.1 Telemetry Invariants & Reporting
Sustainability instrumentation was audited in [`src/api/telemetry.py`](file:///e:/Paladin_media/src/api/telemetry.py):
- **Monotonic Execution Timing:** Employs `time.perf_counter()` to guarantee non-decreasing latency metrics.
- **Unique Run Identifiers:** Every pipeline execution generates a UUIDv4 `run_id` tying metrics strictly to that request.
- **Transparent Estimation Labeling:** In deployment mode without hardware power meter hardware, telemetry emits:
  - `"source": "estimated_hardware_model"`
  - `"is_estimated": true`
  - `"estimation_confidence": "heuristic_proxy"`
- **Grid Intensity Assumptions:** CO2 emissions calculations explicitly disclose the regional grid intensity factor (EPA eGRID US-East average of $380.0\text{ gCO}_2/\text{kWh}$).
- **Secret Data Isolation:** No secret payloads, keys, or filenames are passed into telemetry metrics.
- **No Fake Zeroes:** Incomplete runs report `None` / `UNAVAILABLE` rather than misleading zero-energy readings.

---

## 10. Frontend and Backend Test Results

All regression suites were executed freshly from clean state with zero failures:

### Comprehensive Test Execution Summary
| Test Suite | Scope / Command | Total Executed | Passed | Failed | Skipped | Warnings | Duration | Exit Code | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frontend Unit Tests** | `node --test "src/utils/*.test.ts" "src/services/*.test.ts"` | 40 | 40 | 0 | 0 | 0 | 174.7 ms | **0** | **PASS** |
| **Frontend Linter** | `npx oxlint` | 14 files | 14 files | 0 errors | 0 | 14 warnings | 156 ms | **0** | **PASS** |
| **Frontend Build** | `tsc -b && vite build` | Production client bundle | Built cleanly | 0 | 0 | 0 | 6.38 s | **0** | **PASS** |
| **Backend Unit Tests** | `pytest tests/unit/` | 436 | 436 | 0 | 0 | 2 warnings | 62.20 s | **0** | **PASS** |
| **Backend Integration Tests**| `pytest tests/integration/` | 69 | 69 | 0 | 0 | 1 warning | 42.59 s | **0** | **PASS** |
| **Total Test Suite** | Full Stack Regression | **545** | **545** | **0** | **0** | **17** | **111.5 s** | **0** | **PASS** |

### Test Count Reconciliation
- Backend unit tests: 436 (matches historical count).
- Backend integration tests: 69 (matches historical count).
- Frontend unit tests: 40 (matches historical count).
- Zero skipped tests; zero regressions across all core components.

---

## 11. Render/Vercel Configuration Validation

The production deployment configurations were verified against all twelve hosting requirements:

1. **Single Backend Worker:** Verified in [`src/api/launcher.py`](file:///e:/Paladin_media/src/api/launcher.py#L52-L57): `workers=1` is strictly enforced; launcher fails closed if `workers != 1`.
2. **0.0.0.0 Binding:** Verified in [`render.yaml`](file:///e:/Paladin_media/render.yaml#L11): `--host 0.0.0.0`.
3. **Dynamic Port Binding:** Verified in [`render.yaml`](file:///e:/Paladin_media/render.yaml#L11): `--port $PORT`.
4. **Valid Start Command:** `python -m src.api.launcher --mode demo --host 0.0.0.0 --port $PORT` is tested and functional.
5. **Python Version Compatibility:** `PYTHON_VERSION: 3.10.11` declared in `render.yaml`; all pinned packages in `requirements.txt` are verified compatible.
6. **No Production PyTorch:** Verified [`requirements.txt`](file:///e:/Paladin_media/requirements.txt) contains only `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `onnxruntime`, `pillow`, `cryptography`, `reedsolo`, `numpy`, and `psutil`.
7. **ONNX Runtime CPU:** `onnxruntime` is pinned and verified functional on CPU without GPU or CUDA dependencies.
8. **Vercel Frontend Directory:** [`frontend/vercel.json`](file:///e:/Paladin_media/frontend/vercel.json) specifies `"framework": "vite"`, `"buildCommand": "npm run build"`, `"outputDirectory": "dist"`.
9. **Clean Dist Output:** Verified `npm run build` outputs `dist/index.html` (0.45 kB), `dist/assets/index-gv4r61ky.css` (23.19 kB), `dist/assets/index-BRCTJNwx.js` (306.75 kB).
10. **Dynamic API Base URL:** Verified [`frontend/src/services/backendAdapter.ts`](file:///e:/Paladin_media/frontend/src/services/backendAdapter.ts#L36-L37) reads `import.meta.env.VITE_API_BASE_URL`.
11. **CORS Alignment:** Verified `PALADINMEDIA_ALLOWED_ORIGINS` in `render.yaml` authorizes `https://paladinmedia.vercel.app`.
12. **Health Check Probe:** Verified `/api/health` returns HTTP 200 with `status: "healthy"` and verified model integrity.

---

## 12. Remaining Limitations

1. **Carrier Dimension Boundary:** Public demo uploads are hard-capped at **$512 \times 512$** ($262,144\text{ pixels}$) to operate safely within Render's 512 MB memory limit. Carrier dimensions of $1024 \times 1024$ and $2048 \times 2048$ are supported in self-hosted or dedicated environments with $\ge 1.5\text{ GB}$ RAM.
2. **Wire Protocol Baseline:** Wire Protocol v2 achieves **96.0% (24/25)** exact recovery on ACS-Sub-INN test carriers. One test carrier (`0715.png`) fails header decoding due to spatial edge clipping ($r < 32$). Wire Protocol v3 resolves this failure but remains an offline experimental candidate.
3. **Render Cold-Start Latency:** Free-tier Render web services spin down after 15 minutes of inactivity. The initial wake-up request may take 45–60 seconds. The frontend features a non-blocking cold-start indicator.
4. **Estimated Telemetry:** Hardware power and carbon emissions are transparently reported as heuristic estimates based on modeled CPU TDP and regional grid factors, not physical inline power metering.

---

## 13. Final Acceptance Decision

```
========================================================================================
FINAL ACCEPTANCE GATE VERDICT:
SAFE TO PROCEED WITH ACADEMIC DEMO DEPLOYMENT
========================================================================================
```

### Acceptance Gate Status Fields
- `MEMORY QUALIFICATION: PASS WITH RESTRICTIONS`
- `UPLOAD SAFETY: PASS`
- `CONCURRENT REQUEST SAFETY: PASS`
- `PUBLIC DEMO SECURITY: PASS`
- `FRONTEND TESTS: PASS`
- `BACKEND UNIT TESTS: PASS`
- `BACKEND INTEGRATION TESTS: PASS`
- `CONFIGURATION CONSISTENCY: PASS`
- `LOCAL END-TO-END WORKFLOW: PASS`

**Terminal Decision:** `SAFE TO PROCEED WITH ACADEMIC DEMO DEPLOYMENT`

---

*Report certified by Antigravity Engineering Runtime — Evidence backed by 545 passing regression tests and empirical memory/concurrency benchmarks.*
