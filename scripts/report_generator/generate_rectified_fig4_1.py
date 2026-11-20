import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig_dir = r"E:\Paladin_media\docs\reports\figures"
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "fig_4_1_system_architecture.png")

# Set up figure with comfortable aspect ratio
fig, ax = plt.subplots(figsize=(10.5, 7.8), dpi=300)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Define 5 distinct layers with explicit coordinates
# Top to bottom:
# Layer 0: Presentation Layer (y: 0.81 to 0.93, h=0.12)
# Gap 0-1: 0.74 to 0.81 (gap = 0.07) -> arrow from 0.805 to 0.745
# Layer 1: Gateway Layer (y: 0.62 to 0.74, h=0.12)
# Gap 1-2: 0.55 to 0.62 (gap = 0.07) -> arrow from 0.615 to 0.555
# Layer 2: Cryptographic Core (y: 0.43 to 0.55, h=0.12)
# Gap 2-3: 0.36 to 0.43 (gap = 0.07) -> arrow from 0.425 to 0.365
# Layer 3: Signal & Neural Core (y: 0.24 to 0.36, h=0.12)
# Gap 3-4: 0.17 to 0.24 (gap = 0.07) -> arrow from 0.235 to 0.175
# Layer 4: Hardware & Telemetry Engine (y: 0.05 to 0.17, h=0.12)

layers = [
    {
        "title": "Presentation Layer (Client-Side Laboratory UI)",
        "y": 0.81,
        "h": 0.12,
        "bg": "#F0F7FF",
        "border": "#1976D2",
        "tag": "Tier 1: Client",
        "desc": "React 19 + TypeScript + Vite Laboratory UI | Tailwind CSS Dark Mode (OLED Display Power Conservation)\nFeatures: Drag-and-Drop Carrier Dropzone, Capacity Gauge, Side-by-Side Carrier Inspection, 20x Difference Heatmap"
    },
    {
        "title": "Application & Service Gateway Tier",
        "y": 0.62,
        "h": 0.12,
        "bg": "#F0FDF4",
        "border": "#2E7D32",
        "tag": "Tier 2: API Gateway",
        "desc": "FastAPI 0.104 Asynchronous REST Service | ASGI RequestSizeLimitMiddleware (10MB Hard Gating)\nEndpoints: /api/v1/conceal, /api/v1/extract, /api/v1/telemetry/events (Real-Time SSE Event Stream)"
    },
    {
        "title": "Cryptographic & Data Integrity Foundation Tier",
        "y": 0.43,
        "h": 0.12,
        "bg": "#FFFBEB",
        "border": "#D97706",
        "tag": "Tier 3: Security Core",
        "desc": "SecretSerializer (Canonical b'PLDS' Wire Frames + CRC-32 Trailer) -> AES-256-GCM (96-bit Nonce, 128-bit GHASH Tag)\nReed-Solomon GF(2^8) FEC Encoder (k=223, 2t=32 Parity Symbols, Corrects up to 16 Byte Errors/Block)"
    },
    {
        "title": "Signal Processing & Invertible Neural Carrier Core",
        "y": 0.24,
        "h": 0.12,
        "bg": "#FAF5FF",
        "border": "#7E22CE",
        "tag": "Tier 4: Neural Wavelet",
        "desc": "1-Level 2D Haar DWT (Float64 Precision: LL, LH, HL, HH Subbands) | O(1) Capacity Accounting Hard Gate (8,176 Bytes Max)\nACS-Sub-INN v3 Invertible Architecture (Red-HH Target, Green/Blue-HH Context, Strict Clamping: gamma_min = 1.0)"
    },
    {
        "title": "Hardware Execution & Green Telemetry Tier",
        "y": 0.05,
        "h": 0.12,
        "bg": "#F8FAFC",
        "border": "#475569",
        "tag": "Tier 5: Hardware & Telemetry",
        "desc": "ONNX Runtime 1.16 Engine (Static Compiled Graph, Node Fusion, Session Pooling on Edge x86_64 / ARM CPU)\nSustainabilityTelemetryTracker: Hardware TDP Active Scaling (Joules) + Indian CEA Grid Baseline (0.716 kg CO2/kWh)"
    }
]

# Draw layer boxes
for l in layers:
    # Outer box
    rect = patches.FancyBboxPatch(
        (0.06, l["y"]), 0.88, l["h"],
        boxstyle="round,pad=0.015",
        facecolor=l["bg"],
        edgecolor=l["border"],
        linewidth=1.8,
        zorder=2
    )
    ax.add_patch(rect)
    
    # Tier pill tag on top-right of box
    tag_box = patches.FancyBboxPatch(
        (0.79, l["y"] + l["h"] - 0.032), 0.13, 0.024,
        boxstyle="round,pad=0.006",
        facecolor=l["border"],
        edgecolor="none",
        zorder=3
    )
    ax.add_patch(tag_box)
    ax.text(0.855, l["y"] + l["h"] - 0.020, l["tag"],
            color="#FFFFFF", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=4)

    # Title
    ax.text(0.09, l["y"] + l["h"] - 0.026, l["title"],
            fontsize=10.5, fontweight="bold", color=l["border"], va="center", zorder=3)
    
    # Description
    ax.text(0.09, l["y"] + 0.040, l["desc"],
            fontsize=8.0, color="#1E293B", style="normal", va="center", linespacing=1.35, zorder=3)

# Connectors between layers
connectors = [
    {
        "y_top": 0.81,
        "y_bot": 0.74,
        "label": "HTTP Multipart REST Request  |  Async SSE Telemetry Stream",
        "color": "#1976D2"
    },
    {
        "y_top": 0.62,
        "y_bot": 0.55,
        "label": "In-Memory Byte Buffers  |  Framed Payloads & CRC Trailers",
        "color": "#2E7D32"
    },
    {
        "y_top": 0.43,
        "y_bot": 0.36,
        "label": "Protected Bitstream (Ciphertext + RS Parity)  |  Wavelet Subbands",
        "color": "#D97706"
    },
    {
        "y_top": 0.24,
        "y_bot": 0.17,
        "label": "ONNX Runtime C-API Session  |  Hardware Performance Telemetry (TDP)",
        "color": "#7E22CE"
    }
]

for c in connectors:
    # Arrow coordinates: clear of the borders by 0.006
    y1 = c["y_top"] - 0.005
    y2 = c["y_bot"] + 0.005
    y_mid = (y1 + y2) / 2.0
    
    # Connecting double-headed arrow centered at x = 0.22
    ax.annotate(
        "",
        xy=(0.20, y2),
        xytext=(0.20, y1),
        arrowprops=dict(
            arrowstyle="<->",
            color="#334155",
            lw=1.8,
            shrinkA=2,
            shrinkB=2,
            mutation_scale=12
        ),
        zorder=5
    )
    
    # Connecting double-headed arrow centered at x = 0.80
    ax.annotate(
        "",
        xy=(0.80, y2),
        xytext=(0.80, y1),
        arrowprops=dict(
            arrowstyle="<->",
            color="#334155",
            lw=1.8,
            shrinkA=2,
            shrinkB=2,
            mutation_scale=12
        ),
        zorder=5
    )

    # Clean label pill centered at (0.50, y_mid)
    lbl_box = patches.FancyBboxPatch(
        (0.24, y_mid - 0.020), 0.52, 0.040,
        boxstyle="round,pad=0.008",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.0,
        zorder=4
    )
    ax.add_patch(lbl_box)
    
    ax.text(
        0.50, y_mid,
        c["label"],
        fontsize=7.6,
        fontweight="bold",
        color="#334155",
        ha="center",
        va="center",
        zorder=5
    )

# Overall Diagram Header
ax.text(0.50, 0.970, "Figure 4.1: Multi-Tier System Architecture of the PaladinMedia Steganography Platform",
        fontsize=11.5, fontweight="bold", color="#0F172A", ha="center", va="center")

plt.tight_layout()
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Rectified architecture diagram saved successfully to: {fig_path}")
