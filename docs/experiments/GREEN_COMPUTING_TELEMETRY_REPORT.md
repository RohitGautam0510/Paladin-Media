# PaladinMedia — Green Computing & Telemetry Report

**Date:** 2026-11-02  
**Status:** MEASURED & INTEGRATED  
**Telemetry Mode:** Hardware-Profiled Transparent Estimation (`is_estimated = True`)  
**Results Source:** `docs/experiments/GREEN_COMPUTING_RESULTS.json`  
**Power Modeling Basis:** Thermal Design Power (TDP) CPU Active Scaling + RAM Energy Density  

---

## 1. Executive Summary

PaladinMedia has implemented an end-to-end sustainable computing telemetry subsystem (`SustainabilityTelemetryTracker` in `src/api/telemetry.py`).

Because the production deployment environment operates without the optional `codecarbon` library (`codecarbon` uninstalled to adhere to repository scope stability), PaladinMedia activates a hardware-profiled power estimation model. 

### Invariant Safeguards Enforced:
- **Zero Fake Metrics:** Strictly no placeholder percentages and no false zeros.
- **Explicit Transparency:** Every telemetry record carries `is_estimated: true` and `telemetry_source: "estimated_hardware_model"`.
- **Thread-Safe Accounting:** Thread-safe, non-blocking start/stop tracking per `run_id`.

---

## 2. Power & Carbon Modeling Formulation

The power modeling equations quantify energy and carbon footprint per steganography inference:

### 2.1 Execution Latency
$$\Delta t = t_{\text{end}} - t_{\text{start}} \quad [\text{seconds}]$$
measured via high-resolution monotonic timer `time.perf_counter()`.

### 2.2 Dynamic Power Consumption
$$P_{\text{avg}} = P_{\text{idle}} + \left(\frac{\text{CPU}\%}{100}\right) \cdot (P_{\text{TDP}} - P_{\text{idle}}) + P_{\text{RAM}} \cdot \left(\frac{\text{RAM}_{\text{RSS}}}{1024}\right) \quad [\text{Watts}]$$
where:
- $P_{\text{TDP}} = 45.0 \text{ W}$ (Nominal CPU TDP baseline)
- $P_{\text{idle}} = 10.0 \text{ W}$ (System idle power)
- $P_{\text{RAM}} = 0.375 \text{ W/GB}$ (Active memory energy consumption)
- $\text{CPU}\%$ = Measured active CPU time divided by $(\Delta t \times N_{\text{cores}})$.

### 2.3 Energy & Carbon Emissions
$$\text{Energy} = \frac{P_{\text{avg}} \cdot \Delta t}{3600 \cdot 1000} \quad [\text{kWh}]$$
$$\text{Emissions} = \text{Energy} \cdot I_{\text{grid}} \quad [\text{g CO}_2\text{-eq}]$$
where $I_{\text{grid}} = 475.0 \text{ g/kWh}$ represents the global average grid carbon intensity.

---

## 3. Empirical Benchmark Results

Benchmark conducted across held-out test carriers ($512 \times 512$ RGB PNGs):

| Modulation Engine | Avg Latency (ms) | Avg CPU (%) | Avg RAM (MB) | Avg Power (W) | Avg Energy / Carrier (kWh) | Avg $\text{CO}_2$ / Carrier (g) | Telemetry Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`parity_classical`** | 149.7 ms | 8.2% | 155.2 MB | 12.92 W | $5.40 \times 10^{-7} \text{ kWh}$ | $0.000258 \text{ g}$ | `estimated_hardware_model` |
| **`neural_spatial_onnx`** | 278.7 ms | 32.4% | 162.8 MB | 21.40 W | $1.65 \times 10^{-6} \text{ kWh}$ | $0.000783 \text{ g}$ | `estimated_hardware_model` |
| **`acs_sub_inn_onnx`** | 264.8 ms | 29.8% | 164.5 MB | 20.49 W | $1.49 \times 10^{-6} \text{ kWh}$ | $0.000708 \text{ g}$ | `estimated_hardware_model` |

### Key Findings:
1. **Low Environmental Footprint:** A single neural wavelet steganography embedding and extraction roundtrip consumes approximately $1.65 \times 10^{-6} \text{ kWh}$ ($0.00078 \text{ g CO}_2\text{-eq}$). Over 1,000 carrier embeddings, total emissions equal approximately $0.78 \text{ g CO}_2\text{-eq}$.
2. **Subband vs Spatial ONNX:** `acs_sub_inn_onnx` exhibits slight efficiency gains in CPU utilization compared to `neural_spatial_onnx` (29.8% vs 32.4%), yielding slightly lower power and emissions ($1.49 \mu\text{kWh}$ vs $1.65 \mu\text{kWh}$).
3. **Classical Parity Baseline:** Classical integer parity is ~1.8x faster in CPU execution than neural inference, consuming $0.54 \mu\text{kWh}$ per carrier roundtrip.

---

## 4. Integration with API & Frontend

1. **API Exposure:**
   - `GET /api/telemetry/{run_id}`: Returns JSON representation of `SustainabilityMetrics`.
   - Streaming SSE events (`/api/stream`): Terminal stage `EXACT_MATCH_VERIFY` embeds the complete telemetry object in its event metadata.
2. **Frontend Presentation:**
   - The React UI displays the live Sustainability Telemetry Dashboard widget with real-time indicators for Energy ($kWh$), Carbon ($\text{g CO}_2\text{-eq}$), Execution Latency ($ms$), and the transparent `ESTIMATED` badge.

---

**Report Approved By:** Antigravity Engineering Runtime  
**Status:** `GREEN TELEMETRY: ESTIMATED`
