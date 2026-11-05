# PaladinMedia — Vercel & Render Free Deployment Guide

**Target Audience:** Academic Reviewers, Engineers, and System Operators  
**Architecture:** React 19 Frontend (Vercel) + FastAPI Python Backend (Render)  
**Hosting Tier:** 100% Free Tier (No credit card required)  
**Estimated Setup Time:** ~10 minutes

---

## 1. Prerequisites

1. **GitHub Account:** Repository containing the committed code and ONNX models in `checkpoints/phase5a_onnx/`.
2. **Vercel Account:** Free Hobby Tier ([vercel.com](https://vercel.com)).
3. **Render Account:** Free Individual Tier ([render.com](https://render.com)).

---

## 2. Step 1: Deploy Backend to Render

1. Log into your **Render Dashboard** and select **New +** $\to$ **Web Service**.
2. Connect your GitHub repository containing `Paladin_media`.
3. Configure the Web Service settings:
   - **Name:** `paladinmedia-backend`
   - **Region:** Oregon (US West) or Frankfurt (EU Central)
   - **Branch:** `main` (or your active working branch)
   - **Root Directory:** *(leave blank / root)*
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     python -m src.api.launcher --mode demo --host 0.0.0.0 --port $PORT
     ```
   - **Plan:** `Free` (0.1 CPU, 512 MB RAM)
4. Configure **Environment Variables**:
   | Variable Key | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.10.11` | Pinned stable Python runtime |
   | `PALADINMEDIA_MODE` | `demo` | Public academic demonstration mode |
   | `PALADINMEDIA_ENABLE_DOCS` | `true` | Enables interactive Swagger documentation at `/docs` |
5. Click **Create Web Service**.
6. **Wait for Build & Deployment:**
   - Render will install dependencies from `requirements.txt` and launch Uvicorn.
   - Note your public Render URL: `https://paladinmedia-backend.onrender.com`.
7. **Verify Backend Health:**
   Open a browser or terminal and probe:
   ```bash
   curl https://paladinmedia-backend.onrender.com/api/health
   ```
   Expected response:
   ```json
   {"status":"healthy","live":true,"ready":true,"deployment_mode":"demo","is_qualified_deployment":true,"reason":"Operational"}
   ```
   Also verify capabilities:
   ```bash
   curl https://paladinmedia-backend.onrender.com/api/capabilities
   ```

---

## 3. Step 2: Deploy Frontend to Vercel

1. Log into your **Vercel Dashboard** and click **Add New...** $\to$ **Project**.
2. Import the same GitHub repository.
3. Configure the Project settings:
   - **Project Name:** `paladinmedia`
   - **Framework Preset:** `Vite`
   - **Root Directory:** Click **Edit** and select `frontend`.
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Expand **Environment Variables**:
   | Variable Key | Value |
   | :--- | :--- |
   | `VITE_API_BASE_URL` | `https://paladinmedia-backend.onrender.com` *(use your actual Render backend URL)* |
5. Click **Deploy**.
6. Vercel will run `tsc -b && vite build` and deploy the static assets to global edge CDN in ~1 minute.
7. Note your public Vercel URL: `https://paladinmedia.vercel.app`.

---

## 4. Step 3: End-to-End Live Verification

1. Navigate to your live Vercel URL: `https://paladinmedia.vercel.app`.
2. **Verify System Discovery:**
   - Header should display: `PALADINMEDIA` with badge `POST-QUANTUM NEURAL STEGANOGRAPHY`.
   - Backend status indicator should display: `OPERATIONAL` in green.
   - Modulation Engine dropdown should list all 4 engines (`Neural Spatial ONNX`, `ACS-Sub-INN Subband ONNX`, `PyTorch Ref`, `Deterministic Parity`).
3. **Execute Live Steganography Experiment:**
   - Click **Run Steganography Pipeline**.
   - Observe the 14 real stages advancing through the live SSE stream:
     - `LOAD` $\to$ `SERIALIZE` $\to$ `ENCRYPT` $\to$ `RS_ENCODE` $\to$ `CAPACITY_CHECK` $\to$ `DWT` $\to$ `EMBED` $\to$ `IDWT` $\to$ `RECEIVER_DWT` $\to$ `EXTRACT` $\to$ `RS_DECODE` $\to$ `AES_VERIFY` $\to$ `SECRET_RECOVER` $\to$ `EXACT_MATCH_VERIFY`.
   - Stego media container displays with high fidelity ($>54\text{ dB PSNR}$, $1.0000\text{ SSIM}$).
   - Click **Open Stego Inspector (5 Views)** to inspect spatial difference and coefficient heatmaps.
   - Recovered secret media shows **100% BYTE EQUALITY**.
4. **Inspect Green Computing Telemetry:**
   - In the **Green Computing & Sustainability Telemetry** dashboard:
     - Execution Latency: ~200–300 ms
     - CPU Utilization: ~25–35%
     - Energy Consumption: ~$1.5 \mu\text{kWh}$ ($0.0015\text{ mWh}$)
     - $\text{CO}_2$ Emissions: ~$0.7\text{ mg}$ ($0.0007\text{ g}$)
     - Methodology Badge: `HARDWARE-PROFILED (ESTIMATED)`

---

## 5. Free-Tier Operational Notes & Troubleshooting

### Render Cold Starts (Inactivity Sleep)
- **Behavior:** On the free tier, Render suspends the backend container after 15 minutes of inactivity.
- **Impact:** The very first request after sleeping will take ~40–50 seconds to spin up. Subsequent requests run at standard speed (~200ms).
- **Client Handling:** The frontend displays an active spinner and notification if the initial probe is awaiting backend wakeup.

### Render 512 MB RAM Boundary
- The ONNX Runtime inference engine runs with single-thread allocation (`intra_op_num_threads=1`).
- Never upload carriers larger than $2048 \times 2048$ pixels (enforced with HTTP 413 by the API).

---

**Guide Prepared By:** Antigravity Engineering Runtime
