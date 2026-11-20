import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig_dir = "E:/Paladin_media/docs/reports/figures"
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ==============================================================================
# 1. RECTIFIED FIGURE 4.2: Data Flow Diagrams (Level 0 and Level 1)
# ==============================================================================
def generate_fig_4_2():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 9.2), dpi=300)
    ax1.axis("off")
    ax2.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # -------------------------------------------------------------
    # (A) Level 0: Context Data Flow Diagram
    # -------------------------------------------------------------
    ax1.text(0.50, 0.96, "(A) Level 0: Context Data Flow Diagram",
             fontsize=11.5, fontweight="bold", ha="center", va="center", color="#0F172A")

    # Left Entity: User / Sender (Alice) [visual: 0.04 to 0.22]
    alice_box = patches.FancyBboxPatch(
        (0.05, 0.32), 0.16, 0.50, boxstyle="round,pad=0.010",
        facecolor="#F0F9FF", edgecolor="#0288D1", linewidth=1.8
    )
    ax1.add_patch(alice_box)
    ax1.text(0.13, 0.62, "User / Sender\n(Alice)", fontsize=9.5, fontweight="bold",
             ha="center", va="center", color="#0369A1")
    ax1.text(0.13, 0.44, "[Originator Node]\n• Secret Payload\n• Carrier Image\n• Passphrase / Key",
             fontsize=7.8, ha="center", va="center", color="#334155", linespacing=1.35)

    # Center Process: 0.0 PaladinMedia Core Platform [center=(0.50, 0.56), rx=0.14 -> left=0.36, right=0.64]
    platform_ellipse = patches.Ellipse(
        (0.50, 0.56), 0.28, 0.50, facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=2.2
    )
    ax1.add_patch(platform_ellipse)
    ax1.text(0.50, 0.65, "0.0", fontsize=12, fontweight="bold", ha="center", va="center", color="#15803D")
    ax1.text(0.50, 0.56, "PaladinMedia\nCore Platform", fontsize=10.5, fontweight="bold",
             ha="center", va="center", color="#166534")
    ax1.text(0.50, 0.43, "Security Before Concealment\nAES-256-GCM + RS ECC\n2D Haar DWT + ACS-Sub-INN",
             fontsize=7.2, ha="center", va="center", color="#334155", style="italic", linespacing=1.25)

    # Right Entity: User / Receiver (Bob) [visual: 0.78 to 0.96]
    bob_box = patches.FancyBboxPatch(
        (0.79, 0.32), 0.16, 0.50, boxstyle="round,pad=0.010",
        facecolor="#FAF5FF", edgecolor="#7E22CE", linewidth=1.8
    )
    ax1.add_patch(bob_box)
    ax1.text(0.87, 0.62, "User / Receiver\n(Bob)", fontsize=9.5, fontweight="bold",
             ha="center", va="center", color="#6B21A8")
    ax1.text(0.87, 0.44, "[Recipient Node]\n• Stego PNG Image\n• 100% Decrypted Secret\n• Energy Telemetry Card",
             fontsize=7.8, ha="center", va="center", color="#334155", linespacing=1.35)

    # Gap 1: [0.22 to 0.36] (width 0.14)
    # Flow 1: Alice -> Process (Secret Payload + Carrier Image)
    ax1.annotate(
        "", xy=(0.355, 0.56), xytext=(0.225, 0.56),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#0288D1", mutation_scale=13)
    )
    lbl_f1 = patches.FancyBboxPatch(
        (0.235, 0.61), 0.110, 0.095, boxstyle="round,pad=0.003",
        facecolor="#FFFFFF", edgecolor="#BAE6FD", lw=0.9
    )
    ax1.add_patch(lbl_f1)
    ax1.text(0.290, 0.658, "Secret Payload\n+ Carrier Image\n(Conceal Req)",
             fontsize=7.2, fontweight="bold", ha="center", va="center", color="#0369A1", linespacing=1.2)

    # Gap 2: [0.64 to 0.78] (width 0.14)
    # Flow 2: Process -> Bob (Lossless Stego Image)
    ax1.annotate(
        "", xy=(0.775, 0.67), xytext=(0.645, 0.67),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#16A34A", mutation_scale=13)
    )
    lbl_f2 = patches.FancyBboxPatch(
        (0.655, 0.72), 0.110, 0.080, boxstyle="round,pad=0.003",
        facecolor="#FFFFFF", edgecolor="#BBF7D0", lw=0.9
    )
    ax1.add_patch(lbl_f2)
    ax1.text(0.710, 0.760, "Lossless Stego\nPNG Container",
             fontsize=7.2, fontweight="bold", ha="center", va="center", color="#15803D", linespacing=1.2)

    # Flow 3: Process -> Bob (100% Extracted Payload + Telemetry)
    ax1.annotate(
        "", xy=(0.775, 0.45), xytext=(0.645, 0.45),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#7E22CE", mutation_scale=13)
    )
    lbl_f3 = patches.FancyBboxPatch(
        (0.655, 0.33), 0.110, 0.085, boxstyle="round,pad=0.003",
        facecolor="#FFFFFF", edgecolor="#E9D5FF", lw=0.9
    )
    ax1.add_patch(lbl_f3)
    ax1.text(0.710, 0.372, "Extracted Secret\n+ Telemetry Card",
             fontsize=7.2, fontweight="bold", ha="center", va="center", color="#6B21A8", linespacing=1.2)

    # -------------------------------------------------------------
    # (B) Level 1: Subsystem Data Flow Pipeline
    # -------------------------------------------------------------
    ax2.text(0.50, 0.96, "(B) Level 1: Subsystem Data Flow Pipeline",
             fontsize=11.5, fontweight="bold", ha="center", va="center", color="#0F172A")

    # 5 distinctly spaced blocks with generous 0.065 gaps:
    # Box visual spans: [0.045, 0.175], [0.240, 0.370], [0.435, 0.565], [0.630, 0.760], [0.825, 0.955]
    # Gaps are strictly 0.065 wide, allowing 0.042 badges with 0.0115 clearance on both sides!
    subsystems = [
        {"vx1": 0.045, "vx2": 0.175, "title": "1.0 Framing", "sub": "SecretSerializer\nb'PLDS' + CRC32",
         "bg": "#FFFBEB", "border": "#D97706", "flow": "Framed\nBytes"},
        {"vx1": 0.240, "vx2": 0.370, "title": "2.0 Crypto", "sub": "AES-256-GCM\n96b IV + 128b Tag",
         "bg": "#FFF7ED", "border": "#EA580C", "flow": "Ciphertext\n+ AAD"},
        {"vx1": 0.435, "vx2": 0.565, "title": "3.0 RS ECC", "sub": "Reed-Solomon\nGF(2^8) 2t=32",
         "bg": "#FEF2F2", "border": "#DC2626", "flow": "ECC\nStream"},
        {"vx1": 0.630, "vx2": 0.760, "title": "4.0 2D DWT", "sub": "Haar Wavelet\nFloat64 Subbands",
         "bg": "#F0FDF4", "border": "#16A34A", "flow": "Red-HH\nBand"},
        {"vx1": 0.825, "vx2": 0.955, "title": "5.0 INN Core", "sub": "ACS-Sub-INN\nONNX + IDWT",
         "bg": "#F0F9FF", "border": "#0288D1", "flow": "Stego\nPNG"}
    ]

    p_pad = 0.006
    for s in subsystems:
        x_draw = s["vx1"] + p_pad
        w_draw = (s["vx2"] - s["vx1"]) - 2*p_pad
        box = patches.FancyBboxPatch(
            (x_draw, 0.28), w_draw, 0.50, boxstyle=f"round,pad={p_pad}",
            facecolor=s["bg"], edgecolor=s["border"], linewidth=1.8
        )
        ax2.add_patch(box)
        x_mid = (s["vx1"] + s["vx2"]) / 2.0
        ax2.text(x_mid, 0.64, s["title"], fontsize=9.2, fontweight="bold",
                 ha="center", va="center", color=s["border"])
        ax2.text(x_mid, 0.44, s["sub"], fontsize=7.6,
                 ha="center", va="center", color="#334155", linespacing=1.35)

    # Arrows strictly inside the 0.065-wide gaps (clear clearance from both box edges):
    for i in range(len(subsystems) - 1):
        s_curr = subsystems[i]
        s_next = subsystems[i+1]
        x1 = s_curr["vx2"] + 0.005
        x2 = s_next["vx1"] - 0.005
        x_mid = (x1 + x2) / 2.0

        # Arrow at y = 0.48
        ax2.annotate(
            "", xy=(x2, 0.48), xytext=(x1, 0.48),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#475569", mutation_scale=11)
        )
        # Flow label badge strictly within the gap (width 0.042, gap is 0.065)
        lbl = patches.FancyBboxPatch(
            (x_mid - 0.021, 0.56), 0.042, 0.12, boxstyle="round,pad=0.002",
            facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.8
        )
        ax2.add_patch(lbl)
        ax2.text(x_mid, 0.62, s_curr["flow"], fontsize=6.8, fontweight="bold",
                 ha="center", va="center", color="#1E293B", linespacing=1.2)

    # Final output arrow
    ax2.annotate(
        "", xy=(0.995, 0.48), xytext=(0.960, 0.48),
        arrowprops=dict(arrowstyle="->", lw=1.6, color="#0288D1", mutation_scale=11)
    )

    plt.suptitle("Figure 4.2: Data Flow Diagrams for PaladinMedia Platform (Level 0 and Level 1)",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_4_2_dfd_levels.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated rectified fig_4_2_dfd_levels.png")


# ==============================================================================
# 2. RECTIFIED FIGURE 4.3: UML Use Case Diagram (Zero Arrow Collisions)
# ==============================================================================
def generate_fig_4_3():
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # System boundary box (dashed) [right visual edge strictly at 0.760]
    sys_box = patches.FancyBboxPatch(
        (0.22, 0.06), 0.53, 0.86, boxstyle="round,pad=0.010",
        facecolor="#F8FAFC", edgecolor="#475569", linewidth=1.8, linestyle="--"
    )
    ax.add_patch(sys_box)
    ax.text(0.49, 0.88, "PaladinMedia System Boundary", fontsize=10.5, fontweight="bold",
             ha="center", va="center", color="#1E293B")

    # 4 distinct Use Cases:
    # UC1 (y=0.74): Conceal Secret in Carrier (DWT + INN)
    # UC2 (y=0.55): Authenticate & Decrypt Payload (AES-256-GCM + RS-ECC)
    # UC3 (y=0.36): Inspect Green Energy Telemetry (Joules / Carrier & CO2e)
    # UC4 (y=0.17): Detect Tampering & Fail-Closed Reject (Tamper Defense)
    ucs = [
        {"y": 0.74, "text": "Conceal Secret in Carrier\n(2D Haar DWT + ACS-Sub-INN)",
         "bg": "#F0FDF4", "border": "#16A34A", "tc": "#166534"},
        {"y": 0.55, "text": "Authenticate & Decrypt Payload\n(AES-256-GCM + RS-ECC)",
         "bg": "#FFFBEB", "border": "#D97706", "tc": "#92400E"},
        {"y": 0.36, "text": "Inspect Green Energy Telemetry\n(Joules / Carrier & CO2e)",
         "bg": "#F0F9FF", "border": "#0288D1", "tc": "#075985"},
        {"y": 0.17, "text": "Detect Tampering &\nFail-Closed Reject",
         "bg": "#FEF2F2", "border": "#DC2626", "tc": "#991B1B"}
    ]

    for uc in ucs:
        el = patches.Ellipse(
            (0.50, uc["y"]), 0.42, 0.13,
            facecolor=uc["bg"], edgecolor=uc["border"], linewidth=1.8, zorder=3
        )
        ax.add_patch(el)
        ax.text(0.50, uc["y"], uc["text"], fontsize=8.6, fontweight="bold",
                ha="center", va="center", color=uc["tc"], linespacing=1.3, zorder=4)

    # Actors:
    # 1. Sender (Alice) - Upper Left (x=0.10, y=0.74)
    alice_circle = patches.Circle((0.10, 0.74), 0.058, facecolor="#E0F2FE", edgecolor="#0288D1", lw=1.8, zorder=3)
    ax.add_patch(alice_circle)
    ax.text(0.10, 0.74, "Sender\n(Alice)", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#0369A1", zorder=4)

    # 2. System Auditor - Lower Left (x=0.10, y=0.36)
    audit_circle = patches.Circle((0.10, 0.36), 0.060, facecolor="#F1F5F9", edgecolor="#64748B", lw=1.8, zorder=3)
    ax.add_patch(audit_circle)
    ax.text(0.10, 0.36, "System Auditor\n/ Observer", fontsize=8.2, fontweight="bold", ha="center", va="center", color="#334155", zorder=4)

    # 3. Receiver (Bob) - Upper Right (x=0.90, y=0.55)
    bob_circle = patches.Circle((0.90, 0.55), 0.058, facecolor="#F3E8FF", edgecolor="#7E22CE", lw=1.8, zorder=3)
    ax.add_patch(bob_circle)
    ax.text(0.90, 0.55, "Receiver\n(Bob)", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#6B21A8", zorder=4)

    # 4. Adversary (Eve) - Lower Right (x=0.90, y=0.17)
    eve_circle = patches.Circle((0.90, 0.17), 0.058, facecolor="#FEE2E2", edgecolor="#EF4444", lw=1.8, zorder=3)
    ax.add_patch(eve_circle)
    ax.text(0.90, 0.17, "Adversary\n(Eve)", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#B91C1C", zorder=4)

    # Association lines (COMPLETELY COLLISION-FREE AND NON-CROSSING):
    # Alice -> UC1 (horizontal line: 0.16 to 0.29, y=0.74)
    ax.plot([0.16, 0.29], [0.74, 0.74], color="#0288D1", lw=1.8, zorder=2)

    # Auditor -> UC3 (horizontal line: 0.162 to 0.29, y=0.36)
    ax.plot([0.162, 0.29], [0.36, 0.36], color="#475569", lw=1.8, zorder=2)

    # Bob -> UC2 (horizontal line: 0.84 to 0.71, y=0.55)
    ax.plot([0.84, 0.71], [0.55, 0.55], color="#7E22CE", lw=1.8, zorder=2)

    # Bob -> UC3 (clean diagonal line: 0.845, 0.51 down-left to 0.71, 0.38) - NEVER crosses Eve!
    ax.plot([0.845, 0.71], [0.51, 0.38], color="#7E22CE", lw=1.5, linestyle="--", zorder=2)

    # Eve -> UC4 (horizontal dashed red attack line: 0.84 to 0.71, y=0.17) - NEVER crosses Bob!
    ax.annotate(
        "", xy=(0.71, 0.17), xytext=(0.84, 0.17),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.8, linestyle="--", mutation_scale=12),
        zorder=2
    )
    # <<attack>> badge positioned strictly outside system boundary (center x=0.805, w=0.060 -> [0.775, 0.835])
    # System boundary right edge is at 0.760, Eve circle left edge is at 0.842 -> completely in the clear!
    lbl_atk = patches.FancyBboxPatch(
        (0.775, 0.20), 0.060, 0.045, boxstyle="round,pad=0.002",
        facecolor="#FFFFFF", edgecolor="#FECACA", lw=0.9, zorder=10
    )
    ax.add_patch(lbl_atk)
    ax.text(0.805, 0.222, "<<attack>>", fontsize=7.2, fontweight="bold", color="#B91C1C", ha="center", va="center", zorder=11)

    # Title
    ax.text(0.50, 0.965, "Figure 4.3: UML Use Case Diagram for PaladinMedia Steganography Platform",
            fontsize=12, fontweight="bold", color="#0F172A", ha="center", va="center")

    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_4_3_use_case.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated rectified fig_4_3_use_case.png")


# ==============================================================================
# 3. RECTIFIED FIGURE 4.4: UML Sequence Diagram (Clean Spacing, No Pierced Lifelines)
# ==============================================================================
def generate_fig_4_4():
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    lifelines = [
        {"name": "User (Client)", "x": 0.12, "color": "#0288D1", "bg": "#F0F9FF"},
        {"name": "FastAPI Server", "x": 0.38, "color": "#16A34A", "bg": "#F0FDF4"},
        {"name": "CryptoCore", "x": 0.64, "color": "#D97706", "bg": "#FFFBEB"},
        {"name": "WaveletINN Engine", "x": 0.90, "color": "#7E22CE", "bg": "#FAF5FF"}
    ]

    for ll in lifelines:
        box = patches.FancyBboxPatch(
            (ll["x"] - 0.08, 0.91), 0.16, 0.050, boxstyle="round,pad=0.008",
            facecolor=ll["bg"], edgecolor=ll["color"], linewidth=1.6
        )
        ax.add_patch(box)
        ax.text(ll["x"], 0.935, ll["name"], fontsize=9.2, fontweight="bold",
                ha="center", va="center", color=ll["color"])
        ax.plot([ll["x"], ll["x"]], [0.04, 0.91], color="#94A3B8", lw=1.2, linestyle="--", zorder=1)

    activations = [
        (0.12, 0.84, 0.40),
        (0.12, 0.31, 0.08),
        (0.38, 0.84, 0.41),
        (0.38, 0.31, 0.08),
        (0.64, 0.75, 0.67),
        (0.64, 0.15, 0.09),
        (0.90, 0.58, 0.50),
        (0.90, 0.23, 0.15)
    ]
    for (x_act, y_top, y_bot) in activations:
        act_box = patches.Rectangle(
            (x_act - 0.008, y_bot), 0.016, (y_top - y_bot),
            facecolor="#E2E8F0", edgecolor="#64748B", linewidth=0.9, zorder=2
        )
        ax.add_patch(act_box)

    messages = [
        {"y": 0.83, "x1": 0.128, "x2": 0.372, "x_txt": 0.250, "type": "call", "color": "#0288D1",
         "msg": "1: POST /api/v1/conceal (secret, carrier, key)"},
        {"y": 0.74, "x1": 0.388, "x2": 0.632, "x_txt": 0.510, "type": "call", "color": "#D97706",
         "msg": "2: frame_and_encrypt(secret, AES_KEY, RS_PARITY)"},
        {"y": 0.66, "x1": 0.632, "x2": 0.388, "x_txt": 0.510, "type": "return", "color": "#16A34A",
         "msg": "3: return protected_payload (ciphertext + RS blocks)"},
        {"y": 0.57, "x1": 0.388, "x2": 0.892, "x_txt": 0.510, "type": "call", "color": "#7E22CE",
         "msg": "4: embed_coefficients(Red_HH, protected_payload)"},
        {"y": 0.49, "x1": 0.892, "x2": 0.388, "x_txt": 0.770, "type": "return", "color": "#16A34A",
         "msg": "5: return stego_carrier_png + execution_metrics"},
        {"y": 0.40, "x1": 0.372, "x2": 0.128, "x_txt": 0.250, "type": "return", "color": "#0288D1",
         "msg": "6: 200 OK: stego PNG + Green Telemetry (Joules, CO2e)"},
        {"y": 0.30, "x1": 0.128, "x2": 0.372, "x_txt": 0.250, "type": "call", "color": "#0288D1",
         "msg": "7: POST /api/v1/extract (stego_png, key)"},
        {"y": 0.22, "x1": 0.388, "x2": 0.892, "x_txt": 0.510, "type": "call", "color": "#7E22CE",
         "msg": "8: extract_coefficients() -> IDWT -> raw_payload_bytes"},
        {"y": 0.14, "x1": 0.892, "x2": 0.388, "x_txt": 0.770, "type": "return", "color": "#16A34A",
         "msg": "9: return extracted_payload (verify RS parity)"},
        {"y": 0.07, "x1": 0.372, "x2": 0.128, "x_txt": 0.250, "type": "return", "color": "#0288D1",
         "msg": "10: 200 OK: 100% Verified Decrypted Original Secret"}
    ]

    for m in messages:
        y = m["y"]
        x1, x2 = m["x1"], m["x2"]
        x_txt = m["x_txt"]
        is_call = (m["type"] == "call")
        ls = "-" if is_call else "--"

        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(
                arrowstyle="->", color=m["color"], lw=1.6, linestyle=ls,
                mutation_scale=11
            ),
            zorder=3
        )

        ax.text(
            x_txt, y + 0.016, m["msg"],
            fontsize=7.4, fontweight="bold" if is_call else "normal",
            color=m["color"], ha="center", va="bottom", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.7, alpha=1.0)
        )

    ax.text(0.50, 0.98, "Figure 4.4: UML Sequence Diagram: End-to-End Concealment and Extraction Workflow",
            fontsize=11.5, fontweight="bold", color="#0F172A", ha="center", va="center")

    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_4_4_sequence.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated rectified fig_4_4_sequence.png")


# ==============================================================================
# 4. RECTIFIED FIGURE 5.1: Deployment Architecture (Generous Gaps, Zero Overlaps)
# ==============================================================================
def generate_fig_5_1():
    fig, ax = plt.subplots(figsize=(11.0, 7.0), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    host = patches.FancyBboxPatch(
        (0.03, 0.05), 0.94, 0.88, boxstyle="round,pad=0.015",
        facecolor="#F8FAFC", edgecolor="#334155", linewidth=2.0
    )
    ax.add_patch(host)
    ax.text(0.06, 0.89, "Deployment Environment: Edge / Workstation Node (Windows 11 / Ubuntu 22.04 LTS)",
            fontsize=10.5, fontweight="bold", color="#0F172A")
    ax.text(0.06, 0.86, "Local Offline Execution Boundary | Zero Outbound Network Traffic | Zero Cloud Telemetry Leaks",
            fontsize=7.8, color="#64748B", style="italic")

    p_pad = 0.008
    boxes = [
        {
            "vx1": 0.050, "vx2": 0.285, "y": 0.12, "h": 0.69,
            "title": "Client Runtime Tier\n(Browser / OLED Display)",
            "bg": "#F0F9FF", "border": "#0288D1", "tc": "#0369A1",
            "body": "React 19 + TypeScript 5.3\nVite 5.1 Module Bundler\nSSE Stream Listener\nTailwind CSS Dark Mode\n(OLED Power Conservation)\nPre-Flight Capacity Gauge\n20x Difference Heatmap"
        },
        {
            "vx1": 0.375, "vx2": 0.635, "y": 0.12, "h": 0.69,
            "title": "Application Gateway Tier\n(Uvicorn / FastAPI)",
            "bg": "#F0FDF4", "border": "#16A34A", "tc": "#15803D",
            "body": "Python 3.11 CPython Runtime\nFastAPI 0.104 Server Daemon\nASGI RequestSizeLimit\nMiddleware (10MB Hard Gating)\nSecretSerializer + AES-256-GCM\nReed-Solomon GF(2^8) ECC\nThread-Safe Telemetry Bus"
        },
        {
            "vx1": 0.725, "vx2": 0.960, "y": 0.12, "h": 0.69,
            "title": "Inference Engine Tier\n(ONNX Runtime Session)",
            "bg": "#FFFBEB", "border": "#D97706", "tc": "#92400E",
            "body": "ONNX Runtime 1.16 Engine\nStatic FP32 Compiled Graph\nNode Fusion & Constant Folding\nDedicated Session Pooling\nZero Redundant RAM Realloc\n1-Level 2D Haar DWT Engine\nACS-Sub-INN v3 Model"
        }
    ]

    for b in boxes:
        x_draw = b["vx1"] + p_pad
        w_draw = (b["vx2"] - b["vx1"]) - 2*p_pad
        box = patches.FancyBboxPatch(
            (x_draw, b["y"]), w_draw, b["h"], boxstyle=f"round,pad={p_pad}",
            facecolor=b["bg"], edgecolor=b["border"], linewidth=1.8
        )
        ax.add_patch(box)
        x_mid = (b["vx1"] + b["vx2"]) / 2.0
        ax.text(x_mid, b["y"] + b["h"] - 0.08, b["title"],
                fontsize=9.2, fontweight="bold", ha="center", va="center",
                color=b["tc"], linespacing=1.25)
        ax.text(x_mid, b["y"] + 0.27, b["body"],
                fontsize=7.6, ha="center", va="center", color="#334155", linespacing=1.4)

    # Gap 1 Connector: strictly between Box 1 visual right (0.285) and Box 2 visual left (0.375)
    ax.annotate(
        "", xy=(0.370, 0.44), xytext=(0.290, 0.44),
        arrowprops=dict(arrowstyle="<->", lw=1.8, color="#0288D1", mutation_scale=12)
    )
    lbl1 = patches.FancyBboxPatch(
        (0.297, 0.485), 0.066, 0.095, boxstyle="round,pad=0.003",
        facecolor="#FFFFFF", edgecolor="#BAE6FD", lw=0.9
    )
    ax.add_patch(lbl1)
    ax.text(0.330, 0.532, "HTTP / SSE\nLoopback\nPort 8000", fontsize=6.8, fontweight="bold",
             ha="center", va="center", color="#0369A1", linespacing=1.2)

    # Gap 2 Connector: strictly between Box 2 visual right (0.635) and Box 3 visual left (0.725)
    ax.annotate(
        "", xy=(0.720, 0.44), xytext=(0.640, 0.44),
        arrowprops=dict(arrowstyle="<->", lw=1.8, color="#16A34A", mutation_scale=12)
    )
    lbl2 = patches.FancyBboxPatch(
        (0.647, 0.485), 0.066, 0.095, boxstyle="round,pad=0.003",
        facecolor="#FFFFFF", edgecolor="#BBF7D0", lw=0.9
    )
    ax.add_patch(lbl2)
    ax.text(0.680, 0.532, "In-Memory\nC-API\nBinding", fontsize=6.8, fontweight="bold",
             ha="center", va="center", color="#15803D", linespacing=1.2)

    # Title
    ax.text(0.50, 0.965, "Figure 5.1: Edge Deployment Setup and Component Runtime Topology",
            fontsize=12, fontweight="bold", color="#0F172A", ha="center", va="center")

    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_5_1_deployment.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated rectified fig_5_1_deployment.png")


if __name__ == "__main__":
    generate_fig_4_2()
    generate_fig_4_3()
    generate_fig_4_4()
    generate_fig_5_1()
