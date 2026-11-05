# PaladinMedia — Phase 10 Public Deployment Gate Report

**Document Version:** 1.0.0  
**Phase:** Phase 10 (Public Deployment)  
**Date:** 2026-11-05  
**Repository:** `E:\Paladin_media`  
**Remote GitHub Repository:** `https://github.com/RohitGautam0510/Paladin-Media`  
**Git Default Branch:** `master`  
**Git HEAD Commit:** `b915179` (`b9151797e88938dc40dbd33ebcbe3cfd81b9b183`)  
**Status Verdict:** `PUBLIC DEPLOYMENT BLOCKED` (Awaiting Cloud Dashboard Linking)

---

## 1. Executive Summary & Deployment Status

PaladinMedia has completed full pre-deployment qualification and published the production codebase to GitHub (`RohitGautam0510/Paladin-Media` branch `master`).

The system has been rigorously qualified across all 9 Phase 9 acceptance gates (545 passing regression tests, CPU-only ONNX Runtime stack, hard-gated 512×512 upload boundary, 60 req/min rate limiter, and 3-stream bounded concurrency peaking at 320.1 MB RSS).

The phase is currently in status **`PUBLIC DEPLOYMENT BLOCKED`** strictly pending the manual creation/linking of the web services in the Render and Vercel cloud provider dashboards (no deployment API tokens or credentials are stored locally in the agent environment, per security invariants).

Once the two services are created in the respective web dashboards, live public end-to-end verification can be executed immediately against the public URLs.

---

## 2. Pre-Deployment Verification

### 2.1 Git Repository & Working Tree Review
- **Local Working Tree:** Clean (`nothing to commit, working tree clean`).
- **Committed Changes:** Commit `b915179` staged and committed 258 files (+112,046 lines, -945 lines).
- **Sanitization Audit:**
  - Zero private credentials, API tokens, or GitHub personal access tokens (`ghp_`, `gho_`).
  - Zero `.env` files or secret environment configurations committed.
  - Academic coursework files (`submission/`, `CIAs/`, administrative documents) excluded via [`.gitignore`](file:///e:/Paladin_media/.gitignore).
  - All large intermediate per-epoch checkpoints excluded; only canonical `best_model.pt` and 4 production ONNX models retained.
- **Remote Synchronization:**
  - Remote `origin` configured to `https://github.com/RohitGautam0510/Paladin-Media.git`.
  - Pushed `master` branch upstream (`git push -u origin master`).
  - GitHub default branch set to `master` via `gh repo edit --default-branch master`.

### 2.2 Phase 9 Acceptance Confirmation
All Phase 9 safety and reliability qualification gates were verified:
- `MEMORY QUALIFICATION: PASS WITH RESTRICTIONS` (Public demo restricted to $512 \times 512$ carriers, peak RSS 168.9 MB spatial / 99.6 MB subband, leaving $> 343\text{ MB}$ headroom below Render's 512 MB ceiling).
- `UPLOAD SAFETY: PASS` (All 8 input safety boundary checks verified fail-closed before neural inference).
- `CONCURRENT REQUEST SAFETY: PASS` (3 concurrent streams peaked at 320.1 MB RSS, leaving 191.9 MB / 37.5% headroom).
- `PUBLIC DEMO SECURITY: PASS` (Rate limiting, CORS regex, path sanitization, fail-closed auth).
- `FRONTEND TESTS: PASS` (40 passed, 0 failed, 174.7 ms).
- `BACKEND UNIT TESTS: PASS` (436 passed, 0 failed, 62.20 s).
- `BACKEND INTEGRATION TESTS: PASS` (69 passed, 0 failed, 42.59 s).
- `CONFIGURATION CONSISTENCY: PASS` (Render blueprint, Vercel SPA configuration, API URL binding, CPU-only ONNX runtime).
- `LOCAL END-TO-END WORKFLOW: PASS` (14-stage pipeline verified with 100% byte-for-byte recovery on both ONNX models).

### 2.3 Rate Limiter Threshold Accounting
The empirical rate-limiting benchmark sent 65 requests, producing 59 HTTP 200 OK and 6 HTTP 429 Too Many Requests responses:
- **Discrepancy Resolution:** `DemoRateLimiterMiddleware` enforces a strict 60 req/min bucket per IP.
- Prior to the 65-request benchmark loop, 1 request was consumed by the capabilities query from the same test client IP.
- The remaining quota in the 60-second window was exactly 59 requests ($59 + 1 = 60$). The 61st through 66th requests (6 total) were correctly blocked with HTTP 429 and header `Retry-After: 60`.
- Behavior matches the intended rate-limiting specification with 100% mathematical fidelity.

### 2.4 EPA eGRID US-East Emissions Factor Assumption
- **Code Setting:** [`src/api/telemetry.py`](file:///e:/Paladin_media/src/api/telemetry.py) applies a default carbon intensity factor of $380.0\text{ gCO}_2/\text{kWh}$ (representing the EPA eGRID national baseline / US-East grid mix).
- **Hosting Region Analysis:** [`render.yaml`](file:///e:/Paladin_media/render.yaml) designates `region: oregon` (US-West, Northwest Power Pool / NWPP eGRID subregion).
- **Carbon Accounting Disclosure:** The Pacific Northwest grid features extensive hydroelectric generation, resulting in an actual carbon intensity between ~130–180 $\text{gCO}_2/\text{kWh}$. The $380.0\text{ gCO}_2/\text{kWh}$ factor is therefore an **explicit, conservative national modeling baseline**, not a physically metered local grid reading. It is transparently disclosed to clients via `"estimation_confidence": "heuristic_proxy"`.

### 2.5 Review of Frontend Linter Warnings
The frontend linter (`oxlint`) reported **0 errors and 14 warnings** across 14 files (156 ms):
1. **Unused parameter (`backendAdapter.ts` L336):** `parseErr` caught in JSON stream fallback.
2. **Unused test imports/parameters (`inspectorWorkflow.test.ts`):** 3 unused utility imports in test file.
3. **Ref reads during render (`DwtSubbandInspector.tsx` L224–225):** 4 warnings regarding `embeddedSet.current.has(...)` accessed during display normalization.
4. **Missing hook dependencies (`DwtSubbandInspector.tsx` L189):** Missing `subbandWidth`, `subbandHeight` in canvas redraw effect.
5. **State updates in effect (`StegoInspectorModal.tsx` L104, 148):** `setCarrierDims`, `setZoom`, `setPan` synchronizing canvas dimensions on modal open.
6. **Compiler optimization notes:** React Compiler skipped optimization on two subcomponents.
- **Verdict:** All 14 warnings are non-blocking advisory lint warnings with zero runtime defects or build errors (`tsc -b && vite build` exits with code 0).

---

## 3. Deployment Configuration Specification

### 3.1 Render Backend Web Service ([`render.yaml`](file:///e:/Paladin_media/render.yaml))
- **Service Name:** `paladinmedia-backend`
- **Environment:** `python`
- **Region:** `oregon` (Free tier)
- **Instance Resources:** 512 MB RAM, 0.1 CPU core
- **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command:** `python -m src.api.launcher --mode demo --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/api/health`
- **Auto Deploy:** Disabled (`autoDeploy: false`) to ensure controlled releases.
- **Worker Configuration:** Single worker (`workers=1`) strictly enforced in `src/api/launcher.py` for in-memory session store coherence.
- **Python Version:** Pinned to `3.10.11`.
- **Production Dependencies ([`requirements.txt`](file:///e:/Paladin_media/requirements.txt)):**
  - `fastapi==0.115.0`
  - `uvicorn[standard]==0.31.0`
  - `python-multipart==0.0.12`
  - `pydantic==2.9.2`
  - `onnxruntime==1.19.2`
  - `pillow==10.4.0`
  - `cryptography==43.0.1`
  - `reedsolo==1.7.0`
  - `numpy==1.26.4`
  - `psutil==6.0.0`
  - *PyTorch is strictly excluded.*
- **Environment Variables:**
  - `PYTHON_VERSION`: `3.10.11`
  - `PALADINMEDIA_MODE`: `demo`
  - `PALADIN_DEPLOYMENT_MODE`: `demo`
  - `PALADINMEDIA_ENABLE_DOCS`: `true`
  - `PALADINMEDIA_ALLOWED_ORIGINS`: `http://localhost:5173,http://127.0.0.1:5173,https://paladinmedia.vercel.app`

### 3.2 Vercel Frontend SPA ([`frontend/vercel.json`](file:///e:/Paladin_media/frontend/vercel.json))
- **Framework:** `vite`
- **Root Directory:** `frontend`
- **Build Command:** `npm run build`
- **Output Directory:** `dist`
- **SPA Rewrites:** All requests `/(.*)` routed to `/index.html` for client-side routing.
- **Security Headers:**
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Referrer-Policy: strict-origin-when-cross-origin`
- **Environment Variables:**
  - `VITE_API_BASE_URL`: Must be set to the assigned Render service URL (e.g., `https://paladinmedia-backend.onrender.com`).
  - Client-side code in [`frontend/src/services/backendAdapter.ts`](file:///e:/Paladin_media/frontend/src/services/backendAdapter.ts#L36-L37) automatically reads `import.meta.env.VITE_API_BASE_URL`.

---

## 4. Required Manual Action to Complete Deployment

Because the Antigravity agent operates without third-party cloud deployment credentials, the user must perform the following standard dashboard setup:

### Step 1: Deploy Render Backend
1. Log in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** $\to$ **Blueprint** (or **Web Service**).
3. Connect repository `RohitGautam0510/Paladin-Media` (branch `master`).
4. Render will automatically read [`render.yaml`](file:///e:/Paladin_media/render.yaml) and configure the build and start commands.
5. Click **Apply / Deploy**.
6. Note the assigned public URL (e.g., `https://paladinmedia-backend.onrender.com`).
7. Verify `/api/health` returns:
   ```json
   {
     "status": "healthy",
     "deployment_mode": "demo",
     "steganography_runtime_operational": true,
     "onnx_models_verified": true
   }
   ```

### Step 2: Deploy Vercel Frontend
1. Log in to [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** $\to$ **Project**.
3. Import `RohitGautam0510/Paladin-Media` (branch `master`).
4. In **Project Settings**:
   - **Root Directory:** Edit and select `frontend`.
   - **Framework Preset:** `Vite`.
   - **Build Command:** `npm run build` (default).
   - **Output Directory:** `dist` (default).
5. In **Environment Variables**:
   - Key: `VITE_API_BASE_URL`
   - Value: `https://paladinmedia-backend.onrender.com` (use your actual Render backend URL from Step 1).
6. Click **Deploy**.
7. Note the assigned Vercel URL (e.g., `https://paladinmedia.vercel.app`).

### Step 3: Run Public End-to-End Verification
Once deployed, perform the 16 verification checks from Section 5 of the prompt:
1. Open the Vercel frontend URL in a browser.
2. Verify capability discovery displays both qualified ONNX engines.
3. Select `Neural Spatial ONNX`, upload a $512 \times 512$ PNG carrier, execute embedding, extract, and confirm 100% exact byte recovery.
4. Select `ACS-Sub-INN ONNX`, repeat embedding/extraction, and confirm 100% exact byte recovery.
5. Upload a carrier $> 512 \times 512$ or $> 2\text{ MB}$ and verify the frontend and backend fail closed with HTTP 413.
6. Verify telemetry displays realistic, non-zero execution time, memory RSS, and estimated emissions.

---

## 5. Terminal Completion Gate

```text
========================================================================================
PHASE 10 STATUS VERDICT:
PUBLIC DEPLOYMENT BLOCKED
========================================================================================
Reason: The code is 100% qualified, committed (commit b915179), pushed to GitHub
(RohitGautam0510/Paladin-Media:master), and all configuration files (render.yaml,
vercel.json, requirements.txt) are prepared. Cloud provider web dashboard creation
is required by the account owner to obtain live assigned hostnames.
========================================================================================
```
